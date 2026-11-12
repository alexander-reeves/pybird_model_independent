"""The DESI DR1 FS+BAO model-independent pipeline, one step per subcommand. What is fitted is defined in config.py.

    python run.py [--platform gpu|cpu] [--cpus N] <step> [options]

Steps, in the order they depend on each other (pipeline.sh submits them all as dependent SLURM jobs):
    reference --model lcdm|w0wa|ede|mnu [--theta-s]
                                           fit a cosmological model directly with the full likelihood (the reference)
    knots                                  data weights of the map cosmology -> MI parameters (already in data/mi_chains)
    chain --template T [--seed K] [--draws N] [--chains N] [--full-shape-only]
                                           MI best fit + one NUTS chain per seed; more seeds = more draws
    recovery --template T                  project the MI posterior onto lcdm, w0wa, ede and compare with the references
                                           (three density estimates: flow, Gaussian, score regression)
    project --template T [--density D] [--models lcdm,w0wa,ede,mnu]
                                           cosmological models from DESI alone, without and with the theta_s prior
    sectors --template T [--theta-s yes]   lcdm from parts of the MI posterior: shape | evolution, background | perturbations
    planck                                 sigma_8 of the Planck PR4 lcdm samples, to overplot them
    fisher                                 Fisher forecast of the templates (always on the CPU)
    bands                                  the bands of the master figure
    export                                 output/desi_dr1/ -> data/mi_chains/ (float32), what the notebooks read
    notebooks [--execute]                  write (and run) notebooks/1_fisher ... 10_y6_pk_bk.ipynb

Hardware: by default JAX takes every GPU of the node and spreads the chains over them (several GPUs), or runs them all on
the one device it finds (one GPU, or the CPU). --platform cpu forces the CPU; --cpus N splits the CPU into N JAX devices so
that N groups of chains run in parallel (keep --chains a multiple of N). Validation that is not pipeline: checks.py.
Outputs: output/desi_dr1/{reference/<model>, <template>/{chain_seed<K>, recovery, projections, sectors}, fisher, master}.
A chain or a reference that does not fit in one job: --warmup W --draws N first, then --extend --draws N as often as needed.
MI_MOCK=<name> in the environment: the same steps on a noiseless mock (config.MOCKS), in output/desi_dr1_mock_<name>/.
"""
import os, sys, glob, json, time, pickle, shutil, argparse


def _hardware():
    """--platform and --cpus have to act before jax is imported."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--platform', choices=['auto', 'gpu', 'cpu'], default='auto'); p.add_argument('--cpus', type=int)
    a, _ = p.parse_known_args()
    if a.platform == 'cpu' or a.cpus or 'fisher' in sys.argv[1:]:          # the Fisher needs clean derivatives: CPU
        os.environ['JAX_PLATFORMS'] = 'cpu'
    elif a.platform == 'gpu':
        os.environ['JAX_PLATFORMS'] = 'cuda'
    if a.cpus:
        os.environ['XLA_FLAGS'] = os.environ.get('XLA_FLAGS', '') + f' --xla_force_host_platform_device_count={a.cpus}'
_hardware()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import config as C
from model import COSMO_MODELS, C_LIGHT, sym_H, sym_DM, true_loglkl, theory_sky
from cosmology import growth, theta_s
from templates import Frame, batched
from inference import (log, Tee, todo, vbatch, minimize_lbfgs, fisher_at, precond_cov, chain_diagnostics, WhitenedNUTS,
                       BoxTransform, run_nuts, train_flow, density_logpost, projected_logpost, gaussian_estimate, load_or_fit,
                       load_or_fit_score)

S, THETA_S = C.SAMPLER, C.THETA_S          # THETA_S: Planck's anchor; on a mock (MI_MOCK) centred on the mock's truth


def start_log(d, name='run.log'):
    os.makedirs(d, exist_ok=True); sys.stdout = Tee(os.path.join(d, name)); sys.stderr = sys.stdout
    log(f"run.py {' '.join(sys.argv[1:])}; JAX devices {jax.devices()}")


def sigma8_of(M, th, B=500):
    f = jax.jit(jax.vmap(lambda t: M.sigma8_z0(M.theta_to_full(t))))
    return np.concatenate([np.asarray(f(jnp.array(th[i:i + B]))) for i in range(0, len(th), B)])


def split(s):
    return [v for v in s.replace('+', ',').split(',') if v]


# ================================================================================================= reference
def step_reference(a):
    """The true likelihood of a cosmological model sampled directly (the reference of the recovery test):
    reference/<model>/bestfit.npz (L-BFGS-B + Hessian), chain.npz (NUTS), meta.json. w0wa and ede start from lcdm's best fit.
    --theta-s: with 100 theta_s ~ N(THETA_S) in the prior -> reference/<model>_theta_s/. --sound-horizon: the likelihood in the
    alpha_rs = 1 frame, EFT priors in units of the sound horizon (config.build_reference) -> reference/<model>_rd/."""
    d = C.reference_dir(a.model, a.theta_s, a.sound_horizon); start_log(d); m = C.MODELS[a.model]
    M = C.build_reference(a.model, with_theta_s=a.theta_s, sound_horizon=a.sound_horizon)
    with open(os.path.join(d, 'meta.json'), 'w') as fh: json.dump(C.describe(M, kind='true likelihood, r_d from rd_h_fixed'), fh, indent=1)
    x0 = M.x_direct_fid(C.eft_start(M)); f_lcdm = os.path.join(C.reference_dir('lcdm'), 'bestfit.npz')
    if C.MOCK: x0 = np.asarray(C.truth()['x'])                    # a mock starts from its truth
    elif a.model != 'lcdm' and os.path.exists(f_lcdm):
        b0 = np.load(f_lcdm)['x_bf']; c = dict(M.cosmo_fixed); c.update(zip(COSMO_MODELS['lcdm'], b0[M.n_eft:])); c.update(m.get('start') or {})
        x0 = np.concatenate([b0[:M.n_eft], [c[k] for k in M.cosmo_keys]])
    lp_d = jax.jit(M.logpost_direct); n_d = M.n_eft + M.n_cosmo
    bt = BoxTransform([M.cosmo_box[k][0] for k in M.cosmo_keys], [M.cosmo_box[k][1] for k in M.cosmo_keys], n_free=M.n_eft) \
        if m.get('box_coordinates') else None                      # unbounded coordinates: no -inf walls for NUTS
    lp = jax.jit(bt.wrap(lp_d)) if bt else lp_d
    f_bf, f_ch = os.path.join(d, 'bestfit.npz'), os.path.join(d, 'chain.npz')
    if todo(f_bf, False, n_d):
        if bt:
            y_bf, _ = minimize_lbfgs(lp, np.asarray(bt.to_y(x0)), name='reference (box coordinates)',
                                     scales=np.concatenate([np.full(M.n_eft, 0.5), np.full(M.n_cosmo, 0.3)]))
            np.savez(f_bf, x_bf=np.asarray(bt.to_x(jnp.array(y_bf))), y_bf=y_bf, F_y=fisher_at(lp, y_bf, name='reference'))
        else:
            x_bf, _ = minimize_lbfgs(lp, x0, name='reference', scales=M.min_scales('direct'), bounds=M.min_bounds('direct'))
            np.savez(f_bf, x_bf=x_bf, F=fisher_at(lp, x_bf, name='reference'))
    b = np.load(f_bf)
    log(f"best fit ({a.model}): chi2 = {-2 * float(M.loglkl_direct(jnp.array(b['x_bf']))):.2f}; "
        f"{dict(zip(M.cosmo_names, np.round(b['x_bf'][M.n_eft:], 5)))}")
    old = dict(np.load(f_ch)) if a.extend and os.path.exists(f_ch) else None       # --extend: more draws from where the chain stopped
    if old is not None or todo(f_ch, False, n_d):
        center, cov = (b['y_bf'], precond_cov(b['F_y'], floor_prec=0.25)) if bt else (b['x_bf'], precond_cov(b['F']))
        if old is not None and bt: old['x'] = old['y']                              # the sampler's own coordinates
        r = WhitenedNUTS(lp, center, cov, name='reference').run(
            jax.random.key(m['seed']), a.warmup or S['n_warmup'], a.draws or S['reference_draws'], n_chains=S['n_chains'], jitter=S['jitter_reference'],
            max_num_doublings=S['max_doublings'], target_accept=S['target_accept'], initial_step_size=S['initial_step_size'], resume=old)
        if bt: r['y'] = r['x']; r['x'] = np.asarray(jax.vmap(jax.vmap(bt.to_x))(jnp.array(r['y'])))
        np.savez(f_ch, **r); chain_diagnostics(r['x'], M.eft_labels + M.cosmo_names)
    log("done")


# ================================================================================================= knots
def step_knots(a):
    """W = J^T C^-1 J, J = d(multipoles + BAO alphas)/d ln P on the 80 emulator knots at the fiducial spectrum (EFT at the
    lcdm reference best fit, the analytically marginalized EFT parameters profiled): the weights with which a cosmology's
    spectrum is fitted by the MI parameters in every projection. The committed file was made at the best fit of lcdm with
    omega_b and n_s fixed."""
    start_log(os.path.join(C.OUT, 'logs'), 'knots.log')
    if os.path.exists(C.KNOT_FISHER) and not a.force: log(f"{C.KNOT_FISHER} exists (--force to redo)"); return
    MB = C.build_template('agnostic', verbose=False); eft = jnp.array(C.eft_start(MB))
    def model_vector(dlnP):
        d = [dict(di, pk_lin=di['pk_lin'] * jnp.exp(dlnP)) for di in MB.cosmo_dicts(jnp.array(MB.phi_fid))]
        L = MB.L; L.loglkl(eft, MB.eft_names_flat, need_cosmo_update=True, cosmo_dict=d, cosmo_module=None, cosmo_engine=None); out = []
        for i in range(L.nsky):
            v = theory_sky(L, i)
            if L.c.get('with_bao_rec'): v = jnp.concatenate([v, jnp.atleast_1d(jnp.asarray(L.alpha_sky[i]))])
            out.append(v)
        return jnp.concatenate(out)
    J = np.asarray(jax.jit(jax.jacfwd(model_vector))(jnp.zeros(len(MB.knots_h))))
    W = J.T @ np.asarray(MB.L.p_all) @ J; W = 0.5 * (W + W.T); ev = np.linalg.eigvalsh(W)[::-1]
    os.makedirs(os.path.dirname(C.KNOT_FISHER), exist_ok=True)
    np.savez(C.KNOT_FISHER, W=W, J=J, eig=ev, knots_h=MB.knots_h); log(f"saved {C.KNOT_FISHER}")


# ================================================================================================= MI chain
def step_chain(a):
    """MI best fit (shared by all seeds: <template>/bestfit.npz, with its Gauss-Newton Fisher) and one NUTS chain per seed
    (<template>/chain_seed<K>.npz), whitened with that Fisher; --whiten-with-chains uses the sample covariance of the seeds
    already there instead (fine for bao_template; for agnostic a 500-draw covariance in 95 dimensions is too noisy: warmup
    then picked 2x larger steps and 12% of the transitions diverged)."""
    d = C.template_dir(a.template, a.full_shape_only); start_log(d, f'chain_seed{a.seed}.log')
    M = C.build_template(a.template, full_shape_only=a.full_shape_only)
    with open(os.path.join(d, 'meta.json'), 'w') as fh: json.dump(C.describe(M, template=a.template), fh, indent=1)
    lp = jax.jit(M.logpost_mi); f_bf, f_ch = os.path.join(d, 'bestfit.npz'), os.path.join(d, f'chain_seed{a.seed}.npz')
    if todo(f_bf, a.redo, M.n_mi):
        x_bf, _ = minimize_lbfgs(lp, M.x_mi_fid(C.eft_start(M)), maxiter=3000, name='MI', scales=M.min_scales('mi'), bounds=M.min_bounds('mi'))
        np.savez(f_bf, x_bf=x_bf, F_gn=M.gn_fisher('mi', x_bf))
    b = np.load(f_bf); log(f"MI best fit: chi2 = {-2 * float(M.loglkl_mi(jnp.array(b['x_bf']))):.2f}")
    if a.draws == 0: log("--draws 0: best fit only"); return
    old = dict(np.load(f_ch)) if a.extend and os.path.exists(f_ch) else None       # --extend: more draws from where the chain stopped
    if old is not None or todo(f_ch, a.redo, M.n_mi):
        cov = precond_cov(b['F_gn'], floor_prec=float(np.linalg.eigvalsh(M.prior_prec).min()))
        others = [f for f in C.chain_files(a.template, a.full_shape_only) if f != f_ch] if a.whiten_with_chains else []
        if others:
            x = np.concatenate([np.load(f)['x'].reshape(-1, M.n_mi) for f in others]); cov = np.cov(x, rowvar=False)
            log(f"whitened with the sample covariance of {[os.path.basename(f) for f in others]}")
        r = WhitenedNUTS(lp, b['x_bf'], cov, name='MI').run(
            jax.random.key(a.seed), a.warmup or S['n_warmup'], a.draws or C.TEMPLATES[a.template]['draws'], n_chains=a.chains or S['n_chains'],
            jitter=S['jitter_mi'], max_num_doublings=S['max_doublings'], target_accept=S['target_accept'], initial_step_size=S['initial_step_size'],
            resume=old)
        np.savez(f_ch, **r); chain_diagnostics(r['x'], M.names, max_print=8)
    log("done")


def flow_of(M, template):
    """The flow ensemble of a template's MI posterior (all seeds), fitted once: <template>/flow.pkl."""
    ch = C.load_chain(template); x = ch['x'].reshape(-1, ch['x'].shape[-1])
    log(f"MI chains of {template}: {[os.path.basename(f) for f in ch['files']]}, {len(x)} draws")
    return x, load_or_fit(M, x, os.path.join(C.template_dir(template), 'flow.pkl'), chains=ch['files'], n_seeds=10, steps=8000,
                          groups=np.repeat(np.arange(ch['x'].shape[0]), ch['x'].shape[1]), **C.FLOW)      # early stopping on whole held-out chains


def score_of(M, template):
    """The score regression of a template's MI likelihood (all seeds), fitted once: <template>/score.pkl (+ the gradients at
    the draws, score_gradients.npz)."""
    ch = C.load_chain(template)
    return load_or_fit_score(M, ch['x'], os.path.join(C.template_dir(template), 'score.pkl'), chains=ch['files'], **C.SCORE)


def density_of(M, template, density):
    return score_of(M, template) if density == 'score' else flow_of(M, template)[1]


# ================================================================================================= recovery
def stats(th_ref, th, w=None):
    w = np.full(len(th), 1. / len(th)) if w is None else w / w.sum()
    mu = w @ th; sd = np.sqrt(w @ (th - mu)**2)
    return (mu - th_ref.mean(0)) / th_ref.std(0), sd / th_ref.std(0)


def step_recovery(a):
    """The MI posterior projected onto each model, p(theta|d) ~ p_MI(phi(theta)|d) / pi_MI(phi(theta)) pi(theta) (DESI-alone
    priors), against its reference chain. A miss splits into representation (the reference chain importance-weighted by
    L_MI(phi(theta)) / L_true(theta)) and the density estimate. Writes <template>/recovery/: <model>.npz (flow),
    <model>_gauss.npz (Gaussian estimate) and <model>_score.npz (score regression) projected chains, reweight_<model>.npz,
    run.log. --redo recomputes them."""
    d = os.path.join(C.template_dir(a.template), 'recovery'); start_log(d)
    M = C.build_template(a.template); x, fl = flow_of(M, a.template); sc = score_of(M, a.template); M0 = C.build_reference('lcdm', verbose=False)
    for model in split(a.models):
        M.set_cosmo_model(model, gauss=C.desi_priors(model)); M0.set_cosmo_model(model, gauss=C.desi_priors(model))
        keys, fx = COSMO_MODELS[model], dict(M0.cosmo_fixed)
        xr = np.load(os.path.join(C.reference_dir(model), 'chain.npz'))['x']; xr = xr.reshape(-1, xr.shape[-1]); th_r = xr[:, M.n_eft:]
        f_rw = os.path.join(d, f'reweight_{model}.npz')
        if os.path.exists(f_rw) and not a.redo: rw = dict(np.load(f_rw))
        else:
            sel = np.random.default_rng(1).choice(len(xr), min(2000, len(xr)), replace=False); xs = xr[sel]; fw = M.phi_ln_keys(keys)
            L0 = vbatch(lambda e, t: true_loglkl(M0, e, M0.theta_to_full(t, keys, fx)), xs[:, :M.n_eft], xs[:, M.n_eft:])
            Lw = vbatch(lambda e, t: M.loglkl_phi(e, fw(t)), xs[:, :M.n_eft], xs[:, M.n_eft:])
            dl = Lw - L0; rw = dict(sel=sel, d=dl, w=np.exp(dl - dl.max()), th=xs[:, M.n_eft:]); np.savez(f_rw, **rw)
        shw, raw = stats(th_r, rw['th'], rw['w'])
        for est, dens in (('flow', fl), ('gauss', gaussian_estimate(M, x)), ('score', sc)):
            f_pr = os.path.join(d, f"{model}{'' if est == 'flow' else '_' + est}.npz")
            if os.path.exists(f_pr) and not a.redo: th_p = np.load(f_pr)['x'].reshape(-1, M.n_cosmo)
            else:
                lo, hi = [M.cosmo_box[k][0] for k in M.cosmo_keys], [M.cosmo_box[k][1] for k in M.cosmo_keys]
                r = run_nuts(projected_logpost(M, dens), lo, hi, np.median(th_r, 0), f'{model} ({est})')
                np.savez(f_pr, **r, keys=np.array(keys)); th_p = r['x'].reshape(-1, M.n_cosmo)
            sh, ra = stats(th_r, th_p)
            log(f"=== {a.template} -> {model} ({est}) vs the reference: max |shift| {np.abs(sh).max():.2f} sigma, widths [{ra.min():.2f}, {ra.max():.2f}]")
            for i, n in enumerate(M.cosmo_names):
                log(f"   {n:11s} reference {th_r[:, i].mean():.4f} +- {th_r[:, i].std():.4f}   MI {th_p[:, i].mean():.4f} +- {th_p[:, i].std():.4f}"
                    f"   shift {sh[i]:+.2f} sigma (representation {shw[i]:+.2f}), width {ra[i]:.3f}")
    log("done")


# ================================================================================================= projections
def step_project(a):
    """The MI posterior projected onto lcdm and w0wa with the DESI-alone priors, without and with the theta_s prior
    100 theta_s ~ N(THETA_S): <template>/projections/<model>.npz and <model>_theta_s.npz (chains, keys, sigma_8 per draw,
    the density estimate used). --density: flow (the default, config.DENSITY) or score (score regression, the back-up)."""
    d = os.path.join(C.template_dir(a.template), 'projections'); start_log(d)
    M = C.build_template(a.template); fl = density_of(M, a.template, a.density); log(f"density estimate: {a.density}")
    for model in split(a.models):
        for with_theta in {'no': [False], 'yes': [True], 'both': [False, True]}[a.theta_s]:
            name = model + ('_theta_s' if with_theta else ''); f_out = os.path.join(d, f'{name}.npz')
            if os.path.exists(f_out):
                old = str(np.load(f_out)['density']) if 'density' in np.load(f_out).files else 'flow'
                if old == a.density: log(f"{f_out} exists ({old})"); continue
                os.makedirs(os.path.join(d, old), exist_ok=True); shutil.move(f_out, os.path.join(d, old, f'{name}.npz'))
                log(f"{f_out} was made with the {old} density: moved to {old}/, redoing it with {a.density}")
            M.set_cosmo_model(model, gauss=C.desi_priors(model)); keys = list(M.cosmo_keys)
            lo, hi = [M.cosmo_box[k][0] for k in keys], [M.cosmo_box[k][1] for k in keys]
            base = density_logpost(M, fl, M.phi_ln_keys(keys), lo, hi, M.logprior_theta_keys(keys))
            lp = (lambda t: base(t) - 0.5 * ((theta_s(M.theta_to_full(t)) - THETA_S[0]) / THETA_S[1])**2) if with_theta else base
            th0 = np.array([M.cosmo_fixed[k] for k in keys]); th0[keys.index('omega_cdm')] = 0.118
            if 'w0' in keys: th0[keys.index('w0')], th0[keys.index('wa')] = -0.8, -0.5
            r = run_nuts(lp, lo, hi, th0, name); th = r['x'].reshape(-1, len(keys)); s8 = sigma8_of(M, th)
            np.savez(f_out, **r, keys=np.array(keys), sigma8=s8, density=a.density)
            log(f"{name}: " + ", ".join(f"{n} {th[:, i].mean():.4f} +- {th[:, i].std():.4f}" for i, n in enumerate(M.cosmo_names))
                + f", sigma8 {s8.mean():.4f} +- {s8.std():.4f}")
    log("done")


# ================================================================================================= sectors
def step_sectors(a):
    """lcdm (DESI-alone priors, no theta_s) from one sector of the MI parameters at a time:
        shape_vs_evolution          shape: ln a(k) on every node = P(k) relative to the template [+ A, dSigma^2]
                                    evolution: ln f, ln D ratios, ln alphas [+ ln alpha_rs]
        background_vs_perturbations background: ln alphas [+ ln alpha_rs]; perturbations: ln a, ln f, ln D ratios [+ A, dSigma^2]
    ([...] = bao_template only). A sector's posterior is the chains' marginal divided by its marginal prior, after
    marginalizing the directions it cannot see alone (posterior/prior width > 0.9). <template>/sectors/<sector>.npz.
    --theta-s yes: the same with the theta_s prior 100 theta_s ~ N(THETA_S) on each sector (the cached sector flows are
    reused): <sector>_theta_s.npz, log run_theta_s.log."""
    d = os.path.join(C.template_dir(a.template), 'sectors'); with_theta = a.theta_s == 'yes'; suffix = '_theta_s' if with_theta else ''
    start_log(d, f'run{suffix}.log')
    M = C.build_template(a.template); P, n = M.pidx, M.n_phys
    ch = C.load_chain(a.template); phi = ch['x'].reshape(-1, ch['x'].shape[-1])[:, M.n_eft:]
    chain_of = np.repeat(np.arange(ch['x'].shape[0]), ch['x'].shape[1])                          # early stopping on whole held-out chains
    C0, mu0 = np.linalg.inv(M.prior_prec), np.asarray(M.prior_mean)

    def sel(*blocks):
        idx = np.concatenate([P[b] for b in blocks if b in P] or [np.zeros(0, int)]).astype(int)    # agnostic: no env, no rs
        Q = np.zeros((n, len(idx))); Q[idx, np.arange(len(idx))] = 1.0; return Q
    SECTORS = {'shape': sel('lna', 'env'), 'evolution': sel('f', 'D', 'apar', 'aper', 'rs'),
               'background': sel('apar', 'aper', 'rs'), 'perturbations': sel('lna', 'f', 'D', 'env')}

    def sector_flow(name, Q0):
        f = os.path.join(d, f'flow_{name}.pkl')
        if os.path.exists(f):
            with open(f, 'rb') as fh: fl = pickle.load(fh)
            return jax.tree_util.tree_map(lambda v: jnp.asarray(v) if isinstance(v, np.ndarray) else v, fl)
        L = np.linalg.cholesky(Q0.T @ C0 @ Q0); Li = np.linalg.inv(L)
        w, V = np.linalg.eigh(Li @ np.cov(phi @ Q0, rowvar=False) @ Li.T); keep = np.sqrt(np.clip(w, 0, None)) < 0.9
        Q = Q0 @ Li.T @ V[:, keep]; log(f"[{name}] {Q0.shape[1]} coordinates, {keep.sum()} measured beyond the prior")
        y = phi @ Q; mu = y.mean(0); wv, Vv = np.linalg.eigh(np.linalg.inv(np.cov(y, rowvar=False)))
        Linv = (Vv * np.sqrt(np.clip(wv, 1e-8 * wv.max(), None))).T
        members = [train_flow((y - mu) @ Linv.T, seed=k, steps=8000, groups=chain_of, **C.FLOW) for k in range(10)]
        fl = {'members': members, 'mu': mu, 'Linv': Linv, 'block': np.arange(n), 'Q': Q, 'prior_prec_y': np.linalg.inv(Q.T @ C0 @ Q),
              'prior_mean_y': Q.T @ mu0}
        with open(f, 'wb') as fh: pickle.dump(jax.tree_util.tree_map(lambda v: np.asarray(v) if hasattr(v, 'shape') else v, fl), fh)
        return fl

    keys = list(M.cosmo_keys); lo, hi = [M.cosmo_box[k][0] for k in keys], [M.cosmo_box[k][1] for k in keys]
    pri, fmap = M.logprior_theta_keys(keys), M.phi_ln_keys(keys)
    th0 = np.array([M.cosmo_fixed[k] for k in keys]); th0[keys.index('omega_cdm')] = 0.118
    for name, Q0 in SECTORS.items():
        f = os.path.join(d, f'{name}{suffix}.npz')
        if os.path.exists(f): r = dict(np.load(f))
        else:
            base = density_logpost(M, sector_flow(name, Q0), fmap, lo, hi, pri)
            lp = (lambda t: base(t) - 0.5 * ((theta_s(M.theta_to_full(t)) - THETA_S[0]) / THETA_S[1])**2) if with_theta else base
            r = run_nuts(lp, lo, hi, th0, name + suffix)
            r['sigma8'] = sigma8_of(M, r['x'].reshape(-1, len(keys))); r['keys'] = np.array(keys); np.savez(f, **r)
        th = r['x'].reshape(-1, len(keys)); Om = (th[:, 0] + th[:, 1]) / th[:, 4]**2
        log(f"=== {name + suffix:13s} Omega_m {Om.mean():.4f} +- {Om.std():.4f}   h {th[:, 4].mean():.4f} +- {th[:, 4].std():.4f}"
            f"   sigma8 {r['sigma8'].mean():.4f} +- {r['sigma8'].std():.4f}")
    log("done")


# ================================================================================================= Planck sigma_8
def step_planck(a):
    """sigma_8 of the Planck PR4 early-universe lcdm samples (data/mi_chains/planck_early_lcdm_samples.npz has none), from the
    same engine as the chains so Planck can be overplotted on them: data/mi_chains/planck_early_lcdm_sigma8.npz (one per sample);
    and the lcdm chain of the theta_s prior alone (DESI-alone priors x theta_s, no data): theta_s_alone_lcdm.npz."""
    start_log(os.path.join(C.OUT, 'master'), 'planck.log')
    pl = np.load(os.path.join(C.PRODUCTS, 'planck_early_lcdm_samples.npz')); M = C.build_template('agnostic', verbose=False)
    th = np.stack([pl['omega_cdm'], pl['omega_b'], pl['n_s'], pl['ln10As'], pl['h']], 1); s8 = sigma8_of(M, th)
    np.savez(os.path.join(C.PRODUCTS, 'planck_early_lcdm_sigma8.npz'), sigma8=s8.astype(np.float32))
    log(f"Planck PR4 lcdm: {len(s8)} samples, sigma8 {s8.mean():.4f} +- {s8.std():.4f}, Omega_m {pl['Omega_m'].mean():.4f}, h {pl['h'].mean():.4f}")
    # the theta_s prior alone (no DESI data): the DESI-alone priors of every sector fit times 100 theta_s ~ N(THETA_S), to show
    # where theta_s adds information: data/mi_chains/theta_s_alone_lcdm.npz
    f = os.path.join(C.PRODUCTS, 'theta_s_alone_lcdm.npz')
    if not os.path.exists(f):
        keys = list(M.cosmo_keys); lo, hi = [M.cosmo_box[k][0] for k in keys], [M.cosmo_box[k][1] for k in keys]; pri = M.logprior_theta_keys(keys)
        th0 = np.array([M.cosmo_fixed[k] for k in keys]); th0[keys.index('omega_cdm')] = 0.118
        r = run_nuts(lambda t: pri(t) - 0.5 * ((theta_s(M.theta_to_full(t)) - THETA_S[0]) / THETA_S[1])**2, lo, hi, th0, 'theta_s alone')
        th = r['x'].reshape(-1, len(keys)); s8 = sigma8_of(M, th)
        np.savez(f, x=r['x'].astype(np.float32), sigma8=s8.astype(np.float32), keys=np.array(keys))
        log("theta_s alone: " + ", ".join(f"{k} {th[:, i].mean():.4f} +- {th[:, i].std():.4f}" for i, k in enumerate(keys)) + f", sigma8 {s8.mean():.3f} +- {s8.std():.3f}")


# ================================================================================================= Fisher
def step_fisher(a):
    """Fisher forecast of both templates at their MI best fits: F = J^T C^-1 J + Pi (bestfit.npz). C_post = (F^-1)_phi (EFT
    marginalized), F_data = C_post^-1 - Pi; propagated to ln P~(k) (sound-horizon frame), ln f sigma_s8, ln alpha_BAO, and to
    lcdm / w0wa: F_theta = (dphi/dtheta)^T F_data (dphi/dtheta) + the DESI-alone priors, at the lcdm reference best fit."""
    start_log(os.path.join(C.OUT, 'fisher')); res = {}
    f_ref = os.path.join(C.reference_dir('lcdm'), 'bestfit.npz')
    th_bf = np.load(f_ref)['x_bf'][-5:] if os.path.exists(f_ref) else np.array([C.FIDUCIAL['ln10^{10}A_s' if k == 'lnAs' else k] for k in COSMO_MODELS['lcdm']])
    for t in C.TEMPLATES:
        if C.TEMPLATES[t].get('growth_from_h'): continue           # no f, D, alpha blocks: not part of the Fisher comparison
        if not os.path.exists(os.path.join(C.template_dir(t), 'bestfit.npz')): continue      # a tree with the baseline template only
        M = C.build_template(t, verbose=False); P, ne = M.pidx, M.n_eft; Fr = Frame(M)
        b = np.load(os.path.join(C.template_dir(t), 'bestfit.npz')); Fgn = 0.5 * (b['F_gn'] + b['F_gn'].T); Pi = np.asarray(M.prior_prec)
        C_post = np.linalg.inv(Fgn)[ne:, ne:]; C_prior = np.linalg.inv(Pi)
        F_data = np.linalg.inv(C_post) - Pi; F_data = 0.5 * (F_data + F_data.T); ph = jnp.array(b['x_bf'][ne:])

        def derived(p):
            fs = jnp.exp(p[P['f']]) * Fr.sigma_s8(Fr.mi(p, Fr.k_int)) * jnp.concatenate([jnp.exp(p[P['D']]), jnp.ones(1)])
            return jnp.log(Fr.mi(p, Fr.k_plot)), jnp.log(fs), jnp.log(M.bao_alphas_mi(p))
        for key, Jk, v in zip(('lnP', 'lnfsig', 'lnbao'), jax.jacfwd(derived)(ph), derived(ph)):
            Jk = np.asarray(Jk); res[f'{t}_{key}'] = np.asarray(v)
            res[f'{t}_{key}_sd_post'] = np.sqrt(np.einsum('ij,jk,ik->i', Jk, C_post, Jk))
            res[f'{t}_{key}_sd_prior'] = np.sqrt(np.einsum('ij,jk,ik->i', Jk, C_prior, Jk))
        res.update({f'{t}_sd_post': np.sqrt(np.diag(C_post)), f'{t}_sd_prior': np.sqrt(np.diag(C_prior)), f'{t}_names': np.array(M.phys_names)})
        for model in ('lcdm', 'w0wa'):
            M.set_cosmo_model(model, gauss=C.desi_priors(model)); keys = list(M.cosmo_keys)
            th = np.array(list(th_bf) + ([-1.0, 0.0] if model == 'w0wa' else []))
            Jt = np.asarray(jax.jacfwd(M.phi_ln_keys(keys))(jnp.array(th))); Ft = Jt.T @ F_data @ Jt
            for k, (mu, sd) in M.cosmo_gauss.items(): Ft[keys.index(k), keys.index(k)] += sd**-2
            res.update({f'{t}_{model}_C': np.linalg.inv(Ft), f'{t}_{model}_theta': th, f'{t}_{model}_keys': np.array(keys)})
            log(f"[{t}] {model} Fisher sigma: " + ", ".join(f"{k} {v:.4f}" for k, v in zip(keys, np.sqrt(np.diag(np.linalg.inv(Ft))))))
    res.update(kh=Fr.kh, zi=np.array(M.z))
    np.savez(os.path.join(C.OUT, 'fisher', 'fisher.npz'), **res); log("done")


# ================================================================================================= master bands
def step_bands(a):
    """68% / 95% bands, in sound-horizon units (templates.Frame), of what each template's MI posterior measures and of its MI
    prior, of lcdm and w0wa projected from the bao_template posterior (DESI alone, without / with theta_s), and of Planck
    PR4 lcdm: P~(k~, z_N); f sigma_s8(z); D(a)/a / [D(a_N)/a_N]; D_V, D_M, D_H / r_d -> master/master_bands.npz."""
    d = os.path.join(C.OUT, 'master'); start_log(d)
    rng = np.random.default_rng(3); ND, NT, PCT = 4000, 1500, [2.5, 16, 50, 84, 97.5]; res = {}
    for t in C.TEMPLATES:
        if C.TEMPLATES[t].get('growth_from_h'): continue
        M = C.build_template(t, verbose=False); P = M.pidx; Fr = Frame(M)
        ch = C.load_chain(t); x = ch['x'].reshape(-1, ch['x'].shape[-1])[:, M.n_eft:]
        draws = {'post': x[rng.choice(len(x), ND, replace=False)], 'prior': rng.multivariate_normal(M.prior_mean, np.linalg.inv(M.prior_prec), ND)}
        fbao = jax.jit(jax.vmap(M.bao_alphas_mi))

        def one(p):
            f, D = jnp.exp(p[P['f']]), jnp.concatenate([jnp.exp(p[P['D']]), jnp.ones(1)])
            return Fr.mi(p, Fr.k_plot) * Fr.hf**3, f * Fr.sigma_s8(Fr.mi(p, Fr.k_int)) * D, D * (1 + jnp.array(M.z)) / (1 + M.z_N)
        for lab, ph in draws.items():
            Pk, fs, G = batched(one, ph); ok = np.isfinite(Pk).all(1) & (Pk > 0).all(1) & np.isfinite(fs).all(1)
            res.update({f'{t}_{lab}_Pk': np.percentile(Pk[ok], PCT, 0), f'{t}_{lab}_fsig': np.percentile(fs[ok], PCT, 0),
                        f'{t}_{lab}_G': np.percentile(G[ok], PCT, 0)})
            al = np.asarray(fbao(jnp.array(ph[ok]))); j = 0; bao = {'V': [], 'M': [], 'H': []}
            for bb, z in zip(M.bao, M.z):
                if bb['iso']: bao['V'].append((z, al[:, j] * (z * bb['DM_over_rd_fid']**2 * bb['DH_over_rd_fid'])**(1 / 3))); j += 1
                else: bao['H'].append((z, al[:, j] * bb['DH_over_rd_fid'])); bao['M'].append((z, al[:, j + 1] * bb['DM_over_rd_fid'])); j += 2
            for q, lst in bao.items():
                res[f'{t}_{lab}_{q}_z'] = np.array([z for z, _ in lst]); res[f'{t}_{lab}_{q}'] = np.array([np.percentile(v, PCT) for _, v in lst])
            log(f"[{t}] {lab}: {ok.sum()} of {len(ok)} draws finite"
                + ("   WARNING: dropping non-finite draws biases the bands" if ok.mean() < 0.99 else ""))
    zi, zN, zg = np.array(M.z), M.z_N, np.linspace(0.05, 1.7, 60)          # M, Fr: bao_template (the last)

    def theory(th, keys):
        M.set_cosmo_model(keys); fx = dict(M.cosmo_fixed)
        def one(t):
            c = M.theta_to_full(t, keys, fx); c2, pl = M.engine(c); s = (c['h'] / M.h_fid) / (c2['rd_h'] / M.rdh_fid)
            Om, w0, wa, rdh = c2['Omega_m'], c['w0'], c['wa'], c2['rd_h']
            Dz, f = growth(Om, w0, wa, zg); DN = growth(Om, w0, wa, [zN])[0][0]       # redshifts must be concrete
            DH = jnp.array([C_LIGHT / (100. * sym_H(Om, z, w0, wa)) for z in zg]) / rdh
            DM = jnp.array([sym_DM(Om, z, w0, wa) for z in zg]) / rdh
            return (pl(Fr.k_plot * s) * s**3 * Fr.hf**3, f * Fr.sigma_s8(pl(Fr.k_int * s) * s**3) * Dz / DN, Dz / DN * (1 + zg) / (1 + zN),
                    DH, DM, (zg * DM**2 * DH)**(1 / 3))
        return dict(zip(('Pk', 'fsig', 'G', 'H', 'M', 'V'), batched(one, th, B=100)))

    def store(tag, th, keys, best=None):
        for k, v in theory(th, keys).items(): res[f'{tag}_{k}'] = np.percentile(v, PCT, 0)
        if best is not None:
            for k, v in theory(best[None, :], keys).items(): res[f'{tag}_best_{k}'] = v[0]
        log(f"[{tag}] {len(th)} draws")
    proj = os.path.join(C.template_dir('bao_template'), 'projections')
    for model in ('lcdm', 'w0wa'):
        for name in (model, model + '_theta_s'):
            r = np.load(os.path.join(proj, f'{name}.npz')); keys = [str(k) for k in r['keys']]; th = r['x'].reshape(-1, len(keys))
            store(name, th[rng.choice(len(th), NT, replace=False)], keys, best=th[np.argmax(r['logp'].reshape(-1))])
    pl = np.load(os.path.join(C.PRODUCTS, 'planck_early_lcdm_samples.npz')); sel = rng.choice(len(pl['h']), NT, replace=False)
    store('planck', np.stack([pl['omega_cdm'], pl['omega_b'], pl['n_s'], pl['ln10As'], pl['h']], 1)[sel], COSMO_MODELS['lcdm'])
    res.update(kh=Fr.kh, zg=zg, zi=zi, zN=zN, pct=np.array(PCT), data_range=np.array([0.02, 0.2]),
               T_fid=np.exp(np.interp(np.log(np.asarray(Fr.k_plot)), np.asarray(Fr.lnk_tab), np.asarray(Fr.lnT))) * Fr.hf**3)
    np.savez(os.path.join(d, 'master_bands.npz'), **res); log("done")


# ================================================================================================= export
def step_export(a):
    """What the notebooks read, as float32, in data/mi_chains/: chain_<template> (all seeds), reference_<model>,
    recovery_<template>, projections_<template>, sectors_<template>, fisher, master_bands, map_test."""
    f32 = lambda v: np.asarray(v, dtype=np.float32)

    def save(name, **kw):
        np.savez_compressed(os.path.join(C.PRODUCTS, name), **kw); print(f"{name:32s} {os.path.getsize(os.path.join(C.PRODUCTS, name)) / 1e6:6.1f} MB")
    os.makedirs(C.PRODUCTS, exist_ok=True)
    for t in C.TEMPLATES:
        if not C.chain_files(t): continue                       # a mock tree has the baseline template only
        td = C.template_dir(t); ch = C.load_chain(t); M = C.build_template(t, verbose=False)
        # growth_from_h: what its H(z) spline implies at the data redshifts, per draw: f, alpha_par, alpha_per, D ratios (linear)
        kw = {'derived': f32(batched(lambda ph: jnp.concatenate(M.expansion(ph[M.pidx['hz']], ph[M.pidx['m']][0])),
                                     ch['x'].reshape(-1, M.n_mi)[:, M.n_eft:]))} if M.gfh else {}
        if C.MOCK:                                              # the MI parameters of the mock's truth
            tr = C.truth(); kw['truth'] = np.asarray(M.set_cosmo_model(str(tr['model'])).phi_ln(jnp.array(tr['x'][M.n_eft:])))
        elif C.DATASET != 'dr1': kw['truth'] = np.asarray(M.prior_mean)   # the Y6 mock was made at the fiducial
        save(f'chain_{t}.npz', x=f32(ch['x']), logp=f32(ch['logp']), names=np.array(M.names), n_eft=M.n_eft, prior_mean=np.asarray(M.prior_mean),
             prior_sd=M.prior_widths(), nodes_h=np.asarray(M.nodes_h), seeds=np.array([os.path.basename(f) for f in ch['files']]),
             divergent=ch['divergent'].mean(), z=np.array(M.z), **kw)
        for sub in ('recovery', 'projections', 'sectors'):
            kw = {}
            for f in sorted(glob.glob(os.path.join(td, sub, '*.npz'))):
                if os.path.basename(f).startswith('reweight'): continue
                r = np.load(f); n = os.path.basename(f)[:-4]; kw[f'{n}_x'] = f32(r['x'])
                for k in ('sigma8', 'logp', 'keys'):
                    if k in r.files: kw[f'{n}_{k}'] = r[k] if k == 'keys' else f32(r[k])
            if kw: save(f'{sub}_{t}.npz', **kw)
    for f in sorted(glob.glob(os.path.join(C.OUT, 'reference', '*', 'chain.npz'))):          # <model> and <model>_theta_s
        name = os.path.basename(os.path.dirname(f)); r = np.load(f); M0 = C.build_reference(name.replace('_theta_s', '').replace('_rd', ''), verbose=False)
        ths = batched(lambda t: theta_s(M0.theta_to_full(t)), r['x'].reshape(-1, r['x'].shape[-1])[:, M0.n_eft:])    # to weight the draws by the theta_s prior
        save(f'reference_{name}.npz', x=f32(r['x']), names=np.array(M0.eft_labels + M0.cosmo_names), n_eft=M0.n_eft, theta_s=ths, theta_s_prior=np.array(THETA_S))
    if C.MOCK: shutil.copy2(os.path.join(C.OUT, 'truth.npz'), os.path.join(C.PRODUCTS, 'truth.npz'))
    elif C.DATASET != 'dr1': np.savez(os.path.join(C.PRODUCTS, 'truth.npz'), x=np.array([C.FIDUCIAL['ln10^{10}A_s' if k == 'lnAs' else k] for k in COSMO_MODELS['lcdm']]), model='lcdm')
    for src, dst in (('checks/map_test.npz', 'map_test.npz'), ('fisher/fisher.npz', 'fisher.npz'), ('master/master_bands.npz', 'master_bands.npz'),
                     ('checks/degeneracy_mock.npz', 'degeneracy_mock.npz'), ('checks/degeneracy_data.npz', 'degeneracy_data.npz'),
                     ('checks/exact_repr.npz', 'exact_repr.npz'), ('checks/exact_repr_wide.npz', 'exact_repr_wide.npz'),
                     ('checks/growth_from_h.npz', 'growth_from_h.npz'),
                     *((f'checks/bestfit_{t}.npz', f'bestfit_{t}.npz') for t in C.TEMPLATES)):
        f = os.path.join(C.OUT, src)
        if os.path.exists(f): shutil.copy2(f, os.path.join(C.PRODUCTS, dst)); print(f"{dst:32s} copied")


# ================================================================================================= notebooks
SETUP = "import sys; sys.path.insert(0, '../mi')\nimport plots as P"
PRIORS_LINE = "DESI-alone priors: BBN ω_b, n_s ~ N(0.9649, 0.042)."
NOTEBOOKS = {
    '1_fisher': [
        ('md', "# Fisher forecast: agnostic vs BAO-template P(k), DESI DR1 FS+BAO\nGauss-Newton Fisher at each MI best fit (`run.py fisher`), against the chains."),
        ('code', SETUP),
        ('md', "## Fisher vs chain, every MI parameter"), ('code', "P.fisher_vs_chain();"),
        ('md', "## Linear power spectrum\nSound-horizon units; dotted: MI prior; dashed: the chain's 68% half-width."), ('code', "P.fisher_pk();"),
        ('md', "## Growth and BAO"), ('code', "P.fisher_growth_bao();"),
        ('md', f"## ΛCDM\n{PRIORS_LINE} Dashed: Fisher, solid: projected chain."), ('code', "P.fisher_cosmo('lcdm');"),
        ('md', "## w0waCDM\nFisher at w0 = −1, wa = 0."), ('code', "P.fisher_cosmo('w0wa');")],
    '2_sampling': [
        ('md', f"# Sampling: the two templates on DESI DR1 FS+BAO\n{PRIORS_LINE}"),
        ('code', SETUP),
        ('md', "## The MI chains"), ('code', "P.chain_table()"), ('code', "P.posterior_prior();"),
        ('md', "## Can the MI model represent the true likelihood?\nThe map cosmology → MI before any sampling (`checks.py map`); "
               "legacy = the original 60-node model with h_conv."), ('code', "P.map_table()"),
        ('md', "## Recovery of direct fits\nMax |shift| of the mean in reference σ, and width ratios. flow: normalizing flows of the MI "
               "posterior, divided by the MI prior (used for the projections); score: score regression of the MI likelihood "
               "(`inference.fit_score`), the back-up."), ('code', "P.recovery_table()"),
        ('md', "### ΛCDM"), ('code', "P.recovery('lcdm');"), ('md', "### w0waCDM"), ('code', "P.recovery('w0wa');"),
        ('md', "w0waCDM with score regression (the back-up), for comparison"), ('code', "P.recovery('w0wa', est='score');"),
        ('md', "### EDE\nn_s fixed."), ('code', "P.recovery('ede');"),
        ('md', "## Without and with θ_s\nSolid: + 100θ_s ~ N(1.040268, 0.000246)."), ('code', "P.projections('lcdm');"), ('code', "P.projections('w0wa');")],
    '3_sectors': [
        ('md', f"# Sector splits: ΛCDM from parts of the MI posterior\nDESI FS+BAO alone, no θ_s. {PRIORS_LINE}"),
        ('code', SETUP),
        ('md', "## Shape | evolution\nP(k) relative to the template | (f, D, α)."), ('code', "P.sectors('shape_vs_evolution', 'agnostic');"),
        ('code', "P.sectors('shape_vs_evolution', 'bao_template');"),
        ('md', "## Background | perturbations\nα | (P(k) bands, fσ8, D)."), ('code', "P.sectors('background_vs_perturbations', 'agnostic');"),
        ('code', "P.sectors('background_vs_perturbations', 'bao_template');")],
    '4_master_plot': [
        ('md', f"# Master plot\nSound-horizon units: k̃ = k r_d/r_d^fid, fσ_s8 on R = 8 (r_d/r_d^fid) Mpc/h_fid. {PRIORS_LINE}"),
        ('code', SETUP),
        ('md', "## Reconstructed P(k): agnostic vs BAO template"), ('code', "P.pk_compare();"),
        ('md', "## DESI FS+BAO alone"), ('code', "P.master(theta_s=False);"),
        ('md', "## DESI FS+BAO + θ_s\n100θ_s ~ N(1.040268, 0.000246)."), ('code', "P.master(theta_s=True);")],
    '6_alpha_rs_degeneracy': [
        ('md', "# Is α_rs degenerate with an isotropic dilation? BAO template, DESI DR1 FS+BAO\n"
               "Model: $P_{\\rm lin}(k)=a(k)\\,T_{\\rm nw}(k)\\,[1+(1+A)\\,e^{-k^2\\Delta\\Sigma^2/2}\\,O(\\alpha_{r_s}k)]$, "
               "AP from $\\alpha_\\parallel(z)=D_H/D_H^{\\rm fid}$, $\\alpha_\\perp(z)=D_M/D_M^{\\rm fid}$ (Mpc/h, no $r_d$), "
               "post-recon BAO points $=\\alpha/\\alpha_{r_s}$.\n\n"
               "Gauge $T_\\lambda$: $\\alpha_{r_s}\\to\\lambda\\alpha_{r_s}$, every $\\alpha\\to\\lambda\\alpha$, "
               "$a(k)\\to\\lambda^3a(\\lambda k)\\,T_{\\rm nw}(\\lambda k)/T_{\\rm nw}(k)$, $\\Delta\\Sigma^2\\to\\lambda^2\\Delta\\Sigma^2$, "
               "EFT $c\\to\\lambda^n c$ ($n$ = k-dimension of the term). In the continuum it changes no observable "
               "(the loops are dilation covariant, the BAO points see $\\alpha/\\alpha_{r_s}$).\n\n"
               "$\\Delta\\chi^2=-2\\,\\Delta\\ln L$ of the analysis likelihood (12 EFT parameters fixed or re-fitted, 42 profiled with their "
               "Gaussian priors), no MI prior, from the MI best fit. Mock: noiseless, true ΛCDM at the data's best fit. "
               "`checks.py degeneracy`, CPU, float64."),
        ('code', SETUP),
        ('md', "## 1. α_rs alone vs an isotropic BAO dilation\n"
               "Moves, $\\lambda=e^t$. A: $\\alpha_{r_s}\\to\\lambda\\alpha_{r_s}$. B: every $\\alpha\\to\\alpha/\\lambda$ (same BAO "
               "$\\alpha/\\alpha_{r_s}$ as A, broadband AP-dilated too). Bc: B with the broadband dilated back ($=T_{1/\\lambda}$ applied to A).\n"
               "Expected if $T$ is a symmetry: BAO-alone identical for A, B, Bc; $\\chi^2$(Bc) = $\\chi^2$(A); B differs at fixed broadband; "
               "A and B agree once ln a, A, ΔΣ², b1, c2 are re-fitted."),
        ('code', "P.deg_moves('mock');"), ('code', "P.deg_moves('data');"),
        ('md', "## 2. The flat direction $T_\\lambda$, and what breaks it\n"
               "$\\Delta\\chi^2$ along $T_\\lambda$ with the exact transformed spectrum, one ingredient added at a time. "
               "Exact one-loop without IR resummation is pure SPT: 0 up to numerics. Marginalizing the 42 EFT parameters instead of profiling "
               "them adds the Jacobian $\\lambda^{\\Sigma n}$ through log det F₂: $-2\\Sigma n\\ln\\lambda$."),
        ('code', "P.deg_ladder('mock');"), ('code', "P.deg_ladder('data');"),
        ('md', "Each of the 42 EFT templates in observed space at $T_{1.02}$ against the best fit: $T'=\\lambda^{-n}T$ measured bin by bin."),
        ('code', "P.deg_eft_powers('mock')"),
        ('md', "## 3. Fisher\n"
               "$F=X^\\top X$: Gauss–Newton Fisher of the analysis likelihood at the best fit over the 59 MI parameters (data: P0, P2 and "
               "the BAO α), the 42 profiled EFT parameters eliminated with their priors (QR projection, $F$ never formed), no MI prior.\n"
               "Curvature along a direction $d$ (per unit $\\ln\\lambda$): $\\kappa=d^\\top Fd$, so $\\Delta\\chi^2\\simeq\\kappa\\,(\\ln\\lambda)^2$. "
               "Gauge vector $v=dT_\\lambda/d\\ln\\lambda$ (broadband part: the dilation projected on the 21 nodes).\n"
               "Tests: $\\kappa(v)\\ll\\kappa(A)$; A and Bc fully correlated ($\\rho\\to1$); where $v$ sits among the eigenvectors of the "
               "prior-whitened Fisher (eigenvalue = data/prior information); the data's own best compensation of an α_rs move "
               "(minimum χ² with $d\\ln\\alpha_{r_s}=1$; directions the data cannot see fixed by $10^{-3}\\times$ the MI prior) against $v$."),
        ('code', "P.deg_fisher('mock');"), ('code', "P.deg_fisher('data');"),
        ('md', "The same with exact one-loop (FFTLog) instead of the loop emulator, wide EFT priors."),
        ('code', "P.deg_fisher('mock', exact=True);"), ('code', "P.deg_fisher('data', exact=True);"),
        ('md', "## 4. Every draw mapped to α_rs = 1\n"
               "$T_{1/\\alpha_{r_s}}$ on 400 chain draws: $\\alpha_{r_s}\\to1$, $\\alpha\\to\\alpha/\\alpha_{r_s}$ (the BAO α), broadband dilated "
               "(the α_rs = 1 template with BAO-parametrized AP). On 80 of them also with wide EFT priors, and with exact loops."),
        ('code', "P.deg_map();"),
        ('md', "## 5. The posterior along the ridge\n"
               "ln α_rs of the chains against the MI prior along $v$, $\\sigma=(v^\\top\\Pi v)^{-1/2}$, and the likelihood alone along $v$, "
               "$\\kappa(v)^{-1/2}$."),
        ('code', "P.deg_ridge();")],
    '7_alpha_rs_comparison': [
        ('md', "# α_rs free vs α_rs = 1: the BAO template on DESI DR1 FS+BAO\n"
               "`bao_template`: $P_{\\rm lin}=a(k)\\,T_{\\rm nw}(k)\\,[1+(1+A)\\,e^{-k^2\\Delta\\Sigma^2/2}\\,O(\\alpha_{r_s}k)]$, "
               "AP from $\\alpha=D/D^{\\rm fid}$ (Mpc/h), BAO points $\\alpha/\\alpha_{r_s}$.\n"
               "`bao_template_no_rs`: the same with $\\alpha_{r_s}=1$: the AP α are the BAO α, $P_{\\rm lin}$ is in sound-horizon units. "
               "The two are related by the gauge $T_\\lambda$ of notebook 6; they differ only where the likelihood is not dilation "
               "covariant (loop emulator, EFT priors in h/Mpc). Same data, priors, map, sampler and pipeline. " + PRIORS_LINE),
        ('code', SETUP + "\nP.OPTS = ('bao_template', 'bao_template_no_rs')"),
        ('md', "## The MI chains"), ('code', "P.chain_table()"), ('code', "P.posterior_prior();"),
        ('md', "## The MI posterior on what both measure\nBAO α (α/α_rs for `bao_template`), f, D, A, ΔΣ²: shift of the mean in "
               "`bao_template` σ, and width ratio."), ('code', "P.compare_mi();"),
        ('md', "## Representation of the true likelihood\nThe map cosmology → MI before any sampling (`checks.py map`): implied shift "
               "of the posterior mean, direct σ."), ('code', "P.map_table()"),
        ('md', "## Recovery of the direct fits\nMax |shift| of the projected mean vs the reference chain (full likelihood), reference σ, "
               "and width ratios."), ('code', "P.recovery_table()"),
        ('md', "### ΛCDM"), ('code', "P.recovery('lcdm');"), ('md', "### w0waCDM"), ('code', "P.recovery('w0wa');"),
        ('md', "### EDE\nn_s fixed."), ('code', "P.recovery('ede');"),
        ('md', "## Why α_rs = 1 moves w0waCDM: the units of the EFT priors\n"
               "$T_{1/\\alpha_{r_s}}$ (notebook 6) takes the point of a cosmology in `bao_template` to $\\alpha_{r_s}=1$: $\\alpha\\to\\alpha/\\alpha_{r_s}$, "
               "$P_{\\rm lin}(k)\\to\\alpha_{r_s}^{-3}P_{\\rm lin}(k/\\alpha_{r_s})$, EFT $c\\to\\alpha_{r_s}^{-n}c$ ($n=0$ for $b_3$; 2 for "
               "$c_{ct},c_{r1},c_{r2}$; 3 for $c_{\\epsilon0}$; 5 for $c_{\\epsilon1},c_{\\epsilon2}$), $\\alpha_{r_s}=h r_d/(h r_d)_{\\rm fid}$ of the cosmology. "
               "The data term does not change; the Gaussian priors $N(\\mu,\\sigma)$ of the 42 profiled $c$ stay where they are:\n\n"
               "$$\\Delta\\chi^2=\\sum\\frac{(\\alpha_{r_s}^{-n}c-\\mu)^2-(c-\\mu)^2}{\\sigma^2}\\simeq-2\\ln\\alpha_{r_s}\\sum\\frac{n\\,c\\,(c-\\mu)}{\\sigma^2}.$$\n\n"
               "So `bao_template_no_rs` is the same likelihood with the EFT priors in units of $r_d$ instead of $h^{-1}$Mpc.\n\n"
               "Test (`checks.py exact_repr`): 2000 draws of the w0waCDM direct chain, importance-weighted to each likelihood. `oracle` = the exact "
               "spectrum of the draw in the $\\alpha_{r_s}=1$ frame, no template. If the priors are the cause: oracle − truth equals the formula draw by "
               "draw (evaluated at the truth's profiled $c$; re-profiling can only lower it), the same with the loop emulator and with exact loops; "
               "it vanishes with the priors × 1000; and `bao_template_no_rs` − oracle ≈ 0.\n\n"
               "The converse (`checks.py adapted`): for each draw take its own $\\alpha_{r_s}$ and give the $\\alpha_{r_s}=1$ frame the h/Mpc "
               "priors, $N(\\alpha_{r_s}^{-n}\\mu,\\alpha_{r_s}^{-n}\\sigma)$ on the frame's coefficients. Rows `_adapted`: the shift against "
               "the normal-prior likelihood drops to the 0.1σ that α_rs free and the priors × 1000 also leave."),
        ('code', "P.frame_table()"), ('code', "P.frame_figure();"),
        ('md', "The same as posteriors: the direct fit, and its draws weighted to the α_rs = 1 template with the EFT priors as they are "
               "and adapted per cosmology."),
        ('code', "P.prior_adapted();"),
        ('md', "### The direct fit with the EFT priors in sound-horizon units\n"
               "`run.py reference --model w0wa --sound-horizon` (`templates.sound_horizon_loglkl`): the true likelihood of the cosmology, no "
               "template, written in the α_rs = 1 frame. pybird gets the cosmology's own spectrum with every length in units of its sound "
               "horizon, $\\tilde P(k)=\\rho^3P(\\rho k)$ with $\\rho=r_{d,\\rm fid}/r_d$ (the BAO wiggles at their fiducial positions), and the "
               "BAO α, $(D_H/r_d)/(D_H/r_d)_{\\rm fid}$ and $(D_M/r_d)/(D_M/r_d)_{\\rm fid}$, as its AP dilations; f and D as before. The "
               "likelihood object, the EFT parameters and their priors are not touched. That prediction equals the standard one with "
               "$c\\to\\alpha_{r_s}^{-n}c$, so the unchanged Gaussian priors now sit on EFT coefficients whose scales ($k_M$, $k_R$, $\\bar n$) "
               "are in units of $r_d$.\n\n"
               "All together: both direct fits and both templates projected. α_rs free follows the h/Mpc fit, α_rs = 1 the sound-horizon "
               "fit."),
        ('code', "P.prior_units();"),
        ('md', "The baseline against its like-for-like direct fit; second table: + θ_s (the direct draws weighted by the θ_s prior)."),
        ('code', "P.direct_vs_mi('w0wa', rd=True);"), ('code', "P.direct_vs_mi('w0wa', rd=True, theta_s=True);"),
        ('md', "## ΛCDM and w0waCDM from DESI alone, without and with θ_s\nSolid: + 100θ_s ~ N(1.040268, 0.000246)."),
        ('code', "P.projections('lcdm');"), ('code', "P.projections('w0wa');"),
        ('md', "## Sector splits"), ('code', "P.sectors('shape_vs_evolution', 'bao_template');"),
        ('code', "P.sectors('shape_vs_evolution', 'bao_template_no_rs');"), ('code', "P.sectors('background_vs_perturbations', 'bao_template');"),
        ('code', "P.sectors('background_vs_perturbations', 'bao_template_no_rs');"),
        ('md', "## Reconstructed P(k)\nSound-horizon units; dotted: MI prior."), ('code', "P.pk_compare();"),
        ('md', "## Fisher forecast at each MI best fit"), ('code', "P.fisher_growth_bao();"),
        ('code', "P.fisher_cosmo('lcdm');"), ('code', "P.fisher_cosmo('w0wa');"),
        ('md', "## All posteriors\nEvery MI parameter with the same meaning in both (filled: α_rs free; line: α_rs = 1); ln a(k) is "
               "frame-dependent, see the P(k) figure above."), ('code', "P.compare_mi_1d();"),
        ('md', "BAO α at each redshift."), ('code', "P.compare_bao_2d();"),
        ('md', "Every cosmological parameter, DESI alone (grey: direct fit with the true likelihood)."),
        ('code', "P.compare_cosmo('lcdm');"), ('code', "P.compare_cosmo('w0wa');"), ('code', "P.compare_cosmo('ede');"),
        ('md', "DESI + θ_s."), ('code', "P.compare_cosmo('lcdm', theta_s=True);"), ('code', "P.compare_cosmo('w0wa', theta_s=True);")],
    '8_validation': [
        ('md', "# Validation: extended cosmologies, θ_s, noiseless mocks, neutrino mass\n"
               "Template `bao_template_no_rs` (the baseline). **Direct fit**: the full likelihood sampled in the cosmological parameters "
               "(`run.py reference`). **MI**: the model-independent posterior projected onto the same model with the same priors (`run.py project`, "
               "normalizing flow). A reconstruction that loses nothing gives shift 0 and width ratio 1; shifts are in direct-fit σ. " + PRIORS_LINE + "\n\n"
               "θ_s: 100θ_s ~ N(1.040268, 0.000246). Where no direct chain was sampled with it, the DESI-alone direct draws are weighted by that "
               "prior (effective number printed). EDE and neutrino mass: θ_s = the ΛCDM value × the ede-v2 emulator's ratio to its ΛCDM limit "
               "(`cosmology.theta_s`); for EDE the prior is a test of the projection, not a CMB constraint.\n\n"
               "w0waCDM is compared with the direct fit that has the EFT priors in sound-horizon units, which is what α_rs = 1 means "
               "(7_alpha_rs_comparison.ipynb); the h/Mpc-prior fit differs from it by 0.5σ."),
        ('code', SETUP + "\nP.OPTS = ('bao_template_no_rs',)"),
        ('md', "## 1. DESI DR1\n### ΛCDM"), ('code', "P.direct_vs_mi('lcdm');"), ('code', "P.direct_vs_mi('lcdm', theta_s=True);"),
        ('md', "### w0waCDM"), ('code', "P.direct_vs_mi('w0wa', rd=True);"), ('code', "P.direct_vs_mi('w0wa', rd=True, theta_s=True);"),
        ('md', "### EDE\nn_s fixed."), ('code', "P.direct_vs_mi('ede');"), ('code', "P.direct_vs_mi('ede', theta_s=True);"),
        ('md', "## 2. Noiseless mocks with a known truth\n"
               "Data vector = the true likelihood's prediction at the truth (`config.MOCKS`: the cosmology and EFT parameters of the data's ΛCDM best "
               "fit with the extension switched on), DR1 covariance, priors and sampler unchanged; the θ_s prior is centred on the truth's θ_s. "
               "Dotted lines: the truth. A noiseless mock tests bias, not coverage: (direct − truth)/σ is the prior-volume and prior-centre effect of "
               "the standard analysis itself, MI − direct is what the reconstruction adds; if the projection reproduces the direct posterior, its "
               "coverage is that of the direct analysis.\n### w0waCDM truth: w0 = −0.75, wa = −0.9"),
        ('code', "P.use('mock_w0wa'); P.chain_table()"),
        ('md', "The MI posterior against the MI parameters of the truth. The α are unbiased. a(k) and f sit 1.4–2σ from the truth in opposite "
               "directions: b1² a and f/b1 are measured, so a, b1 and f move together, and the marginals follow the prior volume along that "
               "direction. The measured combination f D a^1/2 is 3–5 times tighter and within 0.5–0.8σ, the size of the direct fit's own "
               "offset in ln A_s below. Read the a(k) and f marginals of the reconstruction with this in mind; the projections do not use them."),
        ('code', "P.mi_truth();"),
        ('md', "Against the direct fit with the same EFT-prior units (the comparison that tests the reconstruction), without and with θ_s:"),
        ('code', "P.direct_vs_mi('w0wa', rd=True);"), ('code', "P.direct_vs_mi('w0wa', rd=True, theta_s=True);"),
        ('md', "Both direct fits and the reconstruction: the 0.6–0.7σ between the two direct fits is the prior-units effect of "
               "7_alpha_rs_comparison.ipynb, here with a known truth. On this truth the sound-horizon units land closer in (w0, wa); one "
               "truth does not rank the two conventions."),
        ('code', "P.prior_units(opts=('bao_template_no_rs',));"),
        ('md', "### EDE truth: f_EDE = 0.1, log10 z_c = 3.56, θ_i = 2.8"),
        ('code', "P.use('mock_ede'); P.chain_table()"), ('code', "P.mi_truth();"),
        ('md', "DESI alone does not constrain EDE: the direct fit follows the prior volume of the EDE parameters and sits about 1σ from the "
               "truth in ω_cdm, h, f_EDE and θ_i. What is tested is that the reconstruction reproduces that posterior, offsets included. "
               "With θ_s the direct fit is the DESI-alone chain weighted by the prior (few effective draws)."),
        ('code', "P.direct_vs_mi('ede');"), ('code', "P.direct_vs_mi('ede', theta_s=True);"),
        ('md', "## 3. Neutrino mass\n"
               "`mnu`: ΛCDM + m_ν, the mass of each of three degenerate neutrinos (Σm_ν = 3 m_ν), flat in [0, 0.33] eV. Direct fit: the cdm + baryon "
               "spectrum of the emulator at each redshift, so with the scale-dependent growth. MI: one spectrum at z = 1.49 times scale-independent "
               "growth ratios, f and the distances with the neutrinos counted as matter. A bias of the reconstruction shows as MI − direct.\n"
               "### DESI DR1"),
        ('code', "P.use(''); P.direct_vs_mi('mnu');"), ('code', "P.direct_vs_mi('mnu', theta_s=True);"),
        ('md', "The reconstruction reproduces both direct fits (shifts ≤ 0.16σ, the mass ≤ 0.07σ, widths within 6%): at DR1 precision the "
               "scale-independent growth of the reconstruction adds no bias and no degeneracy with the mass. With θ_s the direct fit is "
               "sampled with the prior.\n### Mock: Σm_ν = 0.06 eV"),
        ('code', "P.use('mock_mnu'); P.chain_table()"), ('code', "P.mi_truth();"),
        ('md', "The mass is bounded at zero and the truth (0.02 eV per species) sits next to the bound: the direct fit's mean is 1σ above it, "
               "and ω_cdm follows through its correlation with the mass. What is tested is that the reconstruction reproduces that posterior."),
        ('code', "P.direct_vs_mi('mnu');"), ('code', "P.direct_vs_mi('mnu', theta_s=True);")],
    '9_growth_from_expansion': [
        ('md', "# Growth from the expansion: an H(z) spline instead of free f, D and α\n"
               "`growth_from_h` = `bao_template_no_rs` with the 23 numbers f(z_i), α_∥(z_i), α_⊥(z_i), D(z_i)/D(z_N) replaced by 9: "
               "$s(z)=\\ln[H r_d/(H r_d)_{\\rm fid}]$ on 8 nodes (z = 0, 0.15 and the six data redshifts; cubic spline in ln(1+z)) and "
               "$\\mu=\\ln[\\Omega_m(H_0r_d)^2/{\\rm fid}]$. Then $\\alpha_\\parallel=D_H/r_d$ over its fiducial, $\\alpha_\\perp$ from "
               "$D_M/r_d=\\int_0^z D_H/r_d\\,dz$, and $D$, $f$ from\n\n"
               "$$D''+\\Big(2+\\frac{d\\ln H}{d\\ln a}\\Big)D'-\\frac32\\Omega_m(a)D=0,\\qquad \\Omega_m(a)=\\frac{\\Omega_mH_0^2a^{-3}}{H^2},$$\n\n"
               "matter domination at a = 10⁻³; above z_N = 1.49 the dark-energy density $H^2-\\Omega_mH_0^2(1+z)^3$ keeps its value at z_N.\n\n"
               "Valid for: GR, scale-independent growth (no massive-neutrino or modified-gravity scale dependence), smooth dark energy, a flat "
               "universe, and a dark-energy density that does not grow above z = 1.49. Same P(k) template, EFT priors, data and sampler as the baseline. "
               + PRIORS_LINE),
        ('code', SETUP + "\nP.OPTS = ('bao_template_no_rs', 'growth_from_h')"),
        ('md', "## Does it contain ΛCDM, w0waCDM and EDE?\n`checks.py growth`: at the map of 300 direct-fit draws of each model, the spline + growth "
               "equation against the cosmology's own f, α, D; and the shift of the posterior mean implied by the likelihood difference."),
        ('code', "P.growth_validity()"),
        ('md', "## The chains"), ('code', "P.chain_table()"),
        ('md', "## The data's test of the growth equation\nχ² at the maximum of the posterior, `growth_from_h` against the baseline, for 14 fewer "
               "parameters. `checks.py bestfit`: L-BFGS-B from 9 starting points (the best fit the chains start from, the chain mean, the "
               "chain's best draw, 6 random draws), each restarted until it stops improving. The posterior has shallow local maxima about 1σ "
               "apart, so the end points differ (χ² by up to 18 for the baseline, 9 for `growth_from_h`); the highest of each is reached "
               "from two starts. Between those, Δχ² = 23 for 14 parameters (p = 0.06): the free growth is not required. The best fit that "
               "`run.py chain` starts from is one such local maximum, not the highest."),
        ('code', "P.bestfit_table()"),
        ('md', "## f, α and D at the data redshifts: derived from H(z) against free\n"
               "Filled: chain mean and 68%. Open: best fit. Grey: fiducial ΛCDM. f is degenerate with the amplitude and the biases, so its "
               "marginal is set by the prior volume in both templates: chain means above the best fits. The projections below evaluate the "
               "likelihood on each model's own f(z) and do not use these marginals."),
        ('code', "P.growth_from_h();"),
        ('md', "## Cosmology through the reconstruction\nBoth templates projected, against the direct fits.\n### ΛCDM"),
        ('code', "P.direct_vs_mi('lcdm', opts=P.OPTS);"), ('code', "P.direct_vs_mi('lcdm', theta_s=True, opts=P.OPTS);"),
        ('md', "### w0waCDM"), ('code', "P.direct_vs_mi('w0wa', rd=True, opts=P.OPTS);"), ('code', "P.direct_vs_mi('w0wa', rd=True, theta_s=True, opts=P.OPTS);")],
    '10_y6_pk_bk': [
        ('md', "# DESI-Y6 mock: the reconstruction with and without the bispectrum\n"
               "The noiseless mock of 5_fisher_pk_bk.ipynb (7 samples, Y6-like volumes; truth = the fiducial ΛCDM): P0, P2, P4 for "
               "0.01 < k < 0.2 h/Mpc, alone (`MI_DATA=y6_p`) or with the tree-level bispectrum monopole on closed triangles 0.02 ≤ k ≤ 0.10 h/Mpc "
               "(`MI_DATA=y6_pb`, pybird-dev). Matched: the same P(k) data and covariance, template (`bao_template_no_rs`), MI and EFT priors "
               "and map; the bispectrum adds its data and its two EFT parameters (profiled). No BAO points, no window. " + PRIORS_LINE + "\n\n"
               "Recovered parameters: both MI best fits and the direct ΛCDM best fit return the truth, χ² = 0.00 (0.01 for the direct fit). "
               "Uncertainties: the Gauss–Newton Fisher at each best fit (`run.py fisher`), MI prior included, propagated to ΛCDM and w0waCDM.\n\n"
               "Not done: chains. NUTS through pybird-dev costs 0.3–0.5 s per leapfrog step even without the bispectrum (13 ms on DR1 with "
               "the loop emulator), and short warm-ups leave 74–80% divergent steps; weighting P-alone draws to P + B has an effective "
               "fraction of 10⁻²¹ (from the two Fishers). So the prior-volume shifts of the marginals, seen on the DR1 mocks of "
               "8_validation.ipynb, are not quantified here."),
        ('code', SETUP + "\nP.OPTS = ('bao_template_no_rs',)"),
        ('md', "## What the bispectrum adds\nsd(P + B) / sd(P) of every MI parameter. Blue: measured by P alone."), ('code', "P.y6_fisher();")],
}


def step_notebooks(a):
    """Write notebooks/1_fisher ... 10_y6_pk_bk.ipynb (outputs cleared); --execute runs them in place (errors stay in the cells)."""
    nb_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks')
    for stem, cells in NOTEBOOKS.items():
        if a.only and stem not in split(a.only): continue
        nb = {'cells': [dict({'cell_type': 'markdown', 'metadata': {}, 'source': t} if k == 'md' else
                             {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': t}, id=f'c{i:02d}')
                        for i, (k, t) in enumerate(cells)],
              'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
              'nbformat': 4, 'nbformat_minor': 5}
        path = os.path.join(nb_dir, f'{stem}.ipynb')
        with open(path, 'w') as fh: json.dump(nb, fh, indent=1)
        if a.execute:
            import nbformat
            from nbclient import NotebookClient
            nb = nbformat.read(path, as_version=4)
            NotebookClient(nb, timeout=1800, kernel_name='python3', allow_errors=True, resources={'metadata': {'path': nb_dir}}).execute()
            nbformat.write(nb, path)
            print(f"{stem}: {sum(1 for c in nb.cells for o in c.get('outputs', []) if o.get('output_type') == 'error')} cells with errors", flush=True)
        print(f"{'executed' if a.execute else 'wrote'} {path}", flush=True)


# =================================================================================================
if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--platform', choices=['auto', 'gpu', 'cpu'], default='auto', help='auto: GPUs if there are any')
    p.add_argument('--cpus', type=int, help='split the CPU into this many JAX devices (parallel chains)')
    sp = p.add_subparsers(dest='step', required=True, metavar='step')
    q = sp.add_parser('reference', help='direct fit of a cosmological model'); q.add_argument('--model', choices=list(C.MODELS), required=True)
    q.add_argument('--theta-s', action='store_true', help='with the theta_s prior (-> reference/<model>_theta_s)')
    q.add_argument('--sound-horizon', action='store_true', help='EFT priors in units of the sound horizon (-> reference/<model>_rd)')
    q.add_argument('--draws', type=int, help='per chain (default 2000)'); q.add_argument('--warmup', type=int, help='default 500')
    q.add_argument('--extend', action='store_true', help='add --draws draws to the existing chain (no warmup)')
    q = sp.add_parser('knots', help='data weights of the cosmology -> MI map'); q.add_argument('--force', action='store_true')
    q = sp.add_parser('chain', help='MI best fit + one NUTS chain'); q.add_argument('--template', choices=list(C.TEMPLATES), required=True)
    q.add_argument('--seed', type=int, default=1); q.add_argument('--draws', type=int, help='per chain (default: config.TEMPLATES; 0: the best fit only)')
    q.add_argument('--chains', type=int, help='default 32'); q.add_argument('--warmup', type=int, help='default 500')
    q.add_argument('--full-shape-only', action='store_true', help='leave the BAO points out of the likelihood')
    q.add_argument('--redo', action='store_true', help='redo the best fit and this seed')
    q.add_argument('--extend', action='store_true', help='add --draws draws to this seed (no warmup)')
    q.add_argument('--whiten-with-chains', action='store_true', help='whiten NUTS with the sample covariance of the other seeds')
    for name, hlp in (('recovery', 'project onto lcdm, w0wa, ede vs the references'), ('project', 'lcdm, w0wa without / with theta_s'),
                      ('sectors', 'lcdm from shape | evolution, background | perturbations')):
        q = sp.add_parser(name, help=hlp); q.add_argument('--template', choices=list(C.TEMPLATES), required=True)
        if name == 'recovery':
            q.add_argument('--models', default='lcdm,w0wa,ede'); q.add_argument('--redo', action='store_true')
        if name == 'project':
            q.add_argument('--models', default='lcdm,w0wa'); q.add_argument('--theta-s', choices=['no', 'yes', 'both'], default='both')
            q.add_argument('--density', choices=['flow', 'score'], default=C.DENSITY, help=f'density estimate of the MI posterior (default {C.DENSITY})')
        if name == 'sectors':
            q.add_argument('--theta-s', choices=['no', 'yes'], default='no', help='add the theta_s prior (writes <sector>_theta_s.npz)')
    sp.add_parser('fisher', help='Fisher forecast of the templates (CPU)'); sp.add_parser('bands', help='master-figure bands')
    sp.add_parser('planck', help='sigma_8 of the Planck PR4 lcdm samples')
    sp.add_parser('export', help='-> data/mi_chains/')
    q = sp.add_parser('notebooks', help='write the notebooks'); q.add_argument('--execute', action='store_true')
    q.add_argument('--only', help='comma-separated notebook stems (default: all)')
    a = p.parse_args()
    globals()[f'step_{a.step}'](a)
