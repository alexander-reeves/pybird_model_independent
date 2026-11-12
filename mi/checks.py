"""Validation that is not part of the pipeline (run.py), one check per subcommand. SLURM: SCRIPT=checks.py ARGS="map" sbatch run.sbatch

    map    the representation test of the projection: on 300 draws of each true-likelihood chain, Delta ln L =
           ln L_model(phi(theta)) - ln L_true(theta) for the original 60-node MI model ('legacy': node read-off + h_conv),
           the templates and the exact spectrum in the sound-horizon frame ('oracle'); exp(Delta ln L) gives the
           shift of the posterior mean they imply -> output/desi_dr1/checks/map_test.npz (read by 2_sampling.ipynb)
    theta  theta_s of cosmology.py against CLASS (LCDM off the polynomial's grid; w0wa as a CLASS fluid); builds the
           polynomial data/mi_chains/theta_s_poly.npz from a CLASS grid if it is absent
    ede    the CosmoPower ede-v2 emulators against CLASS in their LCDM limit, the size of the EDE effect, JAX gradients
    growth growth_from_h (growth and distances from a spline of H(z)) against the lcdm, w0wa and ede reference draws: how well
           it contains them -> checks/growth_from_h.npz
    bestfit [templates]  is the MI best fit of `run.py chain` (one L-BFGS-B run from the fiducial; the chains start there) the
           maximum of the posterior? restarts from it, from the chain mean and from the chain's best draw -> checks/bestfit_<template>.npz
    truth  MI_MOCK=<name> checks.py truth: the data vector and truth of the noiseless mock config.MOCKS[name] -> its output
           tree's truth.npz; every run.py step with MI_MOCK=<name> then runs on it
    mock   [seed [draws]] bao_template best fit + one chain on the noiseless mock: the data vector replaced by the true LCDM
           likelihood's prediction at the real data's LCDM best fit (covariance, priors, sampler unchanged) -> checks/mock/
    exact_repr [models] [wide]  representation of the true likelihood by bao_template and bao_template_no_rs (alpha_rs = 1) with the
           loop emulator and with exact (FFTLog) loops: the reference draws importance-weighted to each -> checks/exact_repr.npz
    adapted [models]  alpha_rs = 1 with the EFT priors adapted to its frame per cosmology (the h/Mpc priors on the h/Mpc
           coefficients): the oracle and bao_template_no_rs against the true likelihood on the draws of exact_repr -> checks/exact_repr.npz
    degeneracy  <mock|data|smoke> the alpha_rs gauge of bao_template (alpha_rs and every alpha x lam with the broadband
           dilated leaves the observables unchanged): moves, profiles, what breaks it, Fisher, the chains
           -> checks/degeneracy_<case>.npz (read by 6_alpha_rs_degeneracy.ipynb); CPU, x64
"""
import os, sys, time, glob
if sys.argv[1:2] in (['ede'], ['degeneracy']): os.environ.setdefault('JAX_PLATFORMS', 'cpu')
if sys.argv[1:2] == ['loop_matrices']:
    # pybird's exact one-loop (FFTLog) matrices, needed by the exact-loop variants of `degeneracy`: computed once with numpy and
    # cached in ~/.cache/pybird/loop_matrices (under JAX the same computation takes hours), before anything switches JAX on
    import yaml
    from pybird.likelihood import Likelihood
    d = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'desi_dr1_fs_bao')
    cfg = yaml.full_load(open(os.path.join(d, 'likelihood_config', 'desi_dr1_fs_bao.yaml')))
    cfg.update(data_path=d, with_emu=False, write={'save': False, 'fake': False, 'plot': False, 'show': False})
    t0 = time.time(); Likelihood(cfg); print(f"loop matrices ready ({time.time() - t0:.0f}s)"); sys.exit()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import config as C
import cosmology
from model import MIModel, COSMO_MODELS, PkLinCPJ, rd_h, sym_D_early, Omega_m, fix_rd, true_loglkl, _pb_interp1d
from templates import TemplateModel, sound_horizon_loglkl, batched
from inference import log, vbatch, minimize_lbfgs


# ------------------------------------------------------------------------------------------------- map
def implied(th, d):
    w = np.exp(d - d.max()); w /= w.sum()
    return (w @ th - th.mean(0)) / th.std(0), 1.0 / np.sum(w**2) / len(w)


def check_map(names):
    names = names or list(C.TEMPLATES); t0 = time.time()
    OUT = os.path.join(C.OUT, 'checks'); os.makedirs(OUT, exist_ok=True)
    WM = {n: C.build_template(n, verbose=False) for n in names}
    MB = WM.get('agnostic') or C.build_template('agnostic', verbose=False)              # the sound-horizon frame of the oracle
    M0 = C.build_reference('lcdm', verbose=False)                                        # the true likelihood
    M0legacy = fix_rd(MIModel(C.load_data(), C.FIDUCIAL, z_early=C.Z_EARLY, verbose=False))   # the original 60-node model's own map

    res = {}
    for cv in C.MODELS:
        keys = COSMO_MODELS[cv]; f = os.path.join(C.reference_dir(cv), 'chain.npz')
        x = np.load(f)['x']; x = x.reshape(-1, x.shape[-1]); x = x[np.random.default_rng(0).choice(len(x), 300, replace=False)]
        e, th = x[:, :M0.n_eft], x[:, M0.n_eft:]
        for m in [M0, MB, M0legacy] + list(WM.values()): m.set_cosmo_model(cv)
        fx = dict(M0.cosmo_fixed); fl0 = M0legacy.phi_ln_keys(keys)
        Lt = vbatch(lambda a, t: true_loglkl(M0, a, M0.theta_to_full(t, keys, fx)), e, th)
        rows = {'oracle-truth': vbatch(lambda a, t: sound_horizon_loglkl(MB, a, MB.theta_to_full(t, keys, fx)), e, th) - Lt,
                'legacy-truth': vbatch(lambda a, t: M0legacy.loglkl_phi(a, fl0(t)), e, th) - Lt}
        for n, M in WM.items():
            fw = M.phi_ln_keys(keys); rows[f'{n}-truth'] = vbatch(lambda a, t: M.loglkl_phi(a, fw(t)), e, th) - Lt
        for k, d in rows.items():
            sh, ess = implied(th, d); res[f'{cv}_{k}'] = d
            log(f"[{cv}] {k:15s}: Delta ln L std {d.std():.3f}; ESS/N {ess:.2f}; implied shift "
                + ", ".join(f"{kk} {v:+.2f}" for kk, v in zip(keys, sh)) + " sigma")
        res[f'{cv}_th'] = th
        np.savez(os.path.join(OUT, 'map_test.npz'), **res)
    log(f"saved {OUT}/map_test.npz ({(time.time() - t0) / 60:.1f} min)")


# ------------------------------------------------------------------------------------------------- theta_s
def _class_theta_s(a):
    from classy import Class
    ob, oc, h = a; c = Class(); c.set({'omega_b': ob, 'omega_cdm': oc, 'h': h}); c.compute()
    d = c.get_current_derived_parameters(['z_rec', 'rs_rec', 'da_rec'])
    v = 100.0 * d['rs_rec'] / (d['da_rec'] * (1.0 + d['z_rec'])); c.struct_cleanup(); c.empty()
    return v


def build_poly(n_proc=32):
    """Fit the theta_s polynomial on a CLASS grid and cache it (~2 min)."""
    import multiprocessing
    from itertools import product
    os.environ['OMP_NUM_THREADS'] = '1'
    grid = list(product(np.linspace(0.0195, 0.0245, 9), np.linspace(0.095, 0.150, 13), np.linspace(0.55, 0.82, 15)))
    with multiprocessing.get_context('spawn').Pool(n_proc) as pool: vals = pool.map(_class_theta_s, grid, chunksize=8)
    X, y = np.array(grid), np.array(vals)
    coef, *_ = np.linalg.lstsq(cosmology._feats(*X.T), y, rcond=None); res = y - cosmology._feats(*X.T) @ coef
    os.makedirs(os.path.dirname(cosmology.POLY_FILE), exist_ok=True)
    np.savez(cosmology.POLY_FILE, coef=coef, grid=X, theta=y, rms=res.std(), max=np.abs(res).max())
    log(f"theta_s polynomial built: rms residual {res.std():.2e}, max {np.abs(res).max():.2e}")


def check_theta():
    from classy import Class
    if not os.path.exists(cosmology.POLY_FILE): build_poly()
    sig = cosmology.THETA_S[1]

    def class_theta(ob, oc, h, w0=None, wa=None):
        Cl = Class(); p = {'omega_b': ob, 'omega_cdm': oc, 'h': h}
        if w0 is not None: p.update({'Omega_Lambda': 0., 'w0_fld': w0, 'wa_fld': wa})
        Cl.set(p); Cl.compute(); d = Cl.get_current_derived_parameters(['z_rec', 'rs_rec', 'da_rec'])
        v = 100 * d['rs_rec'] / (d['da_rec'] * (1 + d['z_rec'])); Cl.struct_cleanup(); Cl.empty(); return v
    rng = np.random.default_rng(1)
    pts = [(rng.uniform(0.020, 0.024), rng.uniform(0.10, 0.14), rng.uniform(0.60, 0.78)) for _ in range(6)]
    d = [float(cosmology.theta_s_lcdm(*p)) - class_theta(*p) for p in pts]
    log(f"LCDM, off-grid points: max |ours - CLASS| = {np.abs(d).max():.2e} ({np.abs(d).max() / sig:.3f} sigma of the anchor)")
    for w0, wa in [(-1., 0.), (-0.8, -0.6), (-0.6, -1.4), (-1.2, 0.5), (-0.45, -1.8)]:
        c = {'omega_b': 0.0223, 'omega_cdm': 0.119, 'h': 0.66, 'w0': w0, 'wa': wa}
        ours, cl = float(cosmology.theta_s(c)), class_theta(0.0223, 0.119, 0.66, w0, wa)
        log(f"w0wa ({w0:+.2f}, {wa:+.2f}): ours {ours:.6f}  CLASS {cl:.6f}  diff {ours - cl:+.2e} ({(ours - cl) / sig:+.3f} sigma)")


# ------------------------------------------------------------------------------------------------- EDE
def check_ede():
    from classy import Class
    E = cosmology.EDEEmulator()
    log("PKL training box: " + ", ".join(f"{k} [{a:.4g}, {b:.4g}]" for k, (a, b) in E.pkl.training_box().items()))
    fid = {'omega_cdm': 0.12, 'omega_b': 0.02237, 'h': 0.6736, 'lnAs': 3.044, 'n_s': 0.9649, 'w0': -1., 'wa': 0.}
    lcdm = dict(fid, fEDE=0.001, log10z_c=3.562, thetai_scf=2.83); zN = 1.491
    k = np.geomspace(1e-3, 0.5, 200); P_e = np.asarray(E.plin(jnp.array(k), lcdm, zN))
    Cl = Class()
    Cl.set({'omega_b': fid['omega_b'], 'omega_cdm': fid['omega_cdm'], 'h': fid['h'], 'ln10^{10}A_s': fid['lnAs'], 'n_s': fid['n_s'],
            'N_ncdm': 1, 'deg_ncdm': cosmology.DEG_NCDM, 'm_ncdm': cosmology.DEFAULTS['m_ncdm'], 'N_ur': cosmology.DEFAULTS['N_ur'],
            'tau_reio': cosmology.DEFAULTS['tau_reio'], 'output': 'mPk', 'P_k_max_1/Mpc': 12., 'z_max_pk': 3.})
    Cl.compute(); P_c = np.array([Cl.pk_lin(kk, zN) for kk in k]); r = P_e / P_c
    log(f"LCDM limit (fEDE=0.001) vs CLASS at z_N={zN}: P ratio in [{r.min():.4f}, {r.max():.4f}]; r_d emu {float(E.rs_drag(lcdm)):.3f} vs "
        f"CLASS {Cl.rs_drag():.3f} Mpc; sigma8 emu {float(E.derived(lcdm)[1]):.4f} vs CLASS {Cl.sigma8():.4f}")
    g = sym_D_early(Omega_m(fid), 1 / (1 + zN)) / sym_D_early(Omega_m(fid), 1 / 6.)
    r2 = P_e / (np.asarray(PkLinCPJ()(jnp.array(k), fid, 5.0)) * float(g)**2)
    log(f"LCDM limit vs our CosmoPower-LCDM route (z=5 x D^2): ratio in [{r2.min():.4f}, {r2.max():.4f}]; analytic r_d {float(rd_h(fid)) / fid['h']:.3f} Mpc")
    for fe in (0.05, 0.1, 0.2):
        c = dict(lcdm, fEDE=fe); re = np.asarray(E.plin(jnp.array(k), c, zN)) / P_e
        log(f"fEDE={fe}: P(k,z_N)/P_LCDM in [{re.min():.3f}, {re.max():.3f}]; r_d {float(E.rs_drag(c)):.2f} Mpc "
            f"({100 * (float(E.rs_drag(c)) / float(E.rs_drag(lcdm)) - 1):+.2f}%)")
    gfe = jax.grad(lambda fe: jnp.log(E.plin(jnp.array([0.1]), dict(lcdm, fEDE=fe), zN))[0])(0.1)
    log(f"d ln P(k=0.1/Mpc)/d fEDE at fEDE=0.1: {float(gfe):+.3f}")


# ------------------------------------------------------------------------------------------------- mock
MOCK = os.path.join(C.OUT, 'checks', 'mock')          # the noiseless mock: truth.npz, bao_template/{bestfit, chain_seed*}.npz


def mock_vector(model='lcdm', change=None):
    """The noiseless DESI DR1 FS+BAO data vector, per sky (masked multipoles, then the BAO alphas): the true likelihood's
    prediction at the real data's LCDM best fit (its 12 sampled EFT parameters; the 42 profiled ones at their profile). For
    another model: the parameters of `change` set, the ones the model fixes at their fixed values. Returns (y, x = [EFT, theta])."""
    b = np.load(os.path.join(C.DATA_OUT, 'reference', 'lcdm', 'bestfit.npz'))['x_bf']
    ref = C.build_reference(model, verbose=False, mock=False); ne = ref.n_eft; bf = dict(zip(COSMO_MODELS['lcdm'], b[ne:]), **(change or {}))
    x = np.concatenate([b[:ne], [bf.get(k, ref.cosmo_fixed[k]) for k in ref.cosmo_keys]])
    ref.loglkl_direct(jnp.array(x)); L = ref.L
    y = [np.concatenate([np.asarray(L.correlator_sky[i].get(L.b_sky[i])).reshape(-1)[L.m_sky[i]], np.asarray(L.alpha_sky[i])])
         for i in range(L.nsky)]
    assert all(a.shape == b.shape for a, b in zip(y, L.y_sky))
    return y, x


def check_truth():
    """The mock of MI_MOCK (config.MOCKS) -> <its output tree>/truth.npz: y (all skies), x = [EFT, theta], the model and its
    keys, 100 theta_s of the truth (the centre of the mock's theta_s prior), chi2 of the truth on its own mock (the EFT priors)."""
    model, change = C.MOCKS[C.MOCK]; y, x = mock_vector(model, change)
    ref = set_data(C.build_reference(model, verbose=False, mock=False), y); c = ref.theta_to_full(jnp.array(x[ref.n_eft:]))
    os.makedirs(C.OUT, exist_ok=True); th = float(cosmology.theta_s(c)); chi2 = -2 * float(ref.loglkl_direct(jnp.array(x)))
    np.savez(os.path.join(C.OUT, 'truth.npz'), y=np.concatenate(y), x=x, keys=np.array(ref.cosmo_keys), model=model, n_eft=ref.n_eft,
             theta_s=th, sigma8=float(ref.sigma8_z0(c)))
    log(f"mock {C.MOCK} ({model}): " + ", ".join(f"{k} {v:.5g}" for k, v in zip(ref.cosmo_keys, x[ref.n_eft:]))
        + f"; 100 theta_s {th:.6f} (Planck anchor {cosmology.THETA_S[0]}); -2 ln L of the truth {chi2:.3f} -> {C.OUT}/truth.npz")


def set_data(M, y):
    """Replace the data vector of M's likelihood (the covariance stays)."""
    M.L.y_sky = [np.array(a) for a in y]; M.L.y_all = np.concatenate(M.L.y_sky)
    return M


def check_mock(args):
    """bao_template MI best fit + one chain on the noiseless mock (run.step_chain with the data vector replaced) ->
    checks/mock/bao_template/. args: [seed [draws]], default 1 700 (32 x 700 fits the 1h30 debug limit on 4 GH200)."""
    import argparse, run
    seed, draws = (int(args[0]) if args else 1), (int(args[1]) if len(args) > 1 else 700)
    y, x_bf = mock_vector(); build = C.build_template; os.makedirs(MOCK, exist_ok=True)
    M = set_data(build('bao_template', verbose=False), y)
    np.savez(os.path.join(MOCK, 'truth.npz'), x_bf=x_bf, ln_phi=np.asarray(M.phi_ln(jnp.array(x_bf[M.n_eft:]))),
             names=np.array(M.names[M.n_eft:]), n_eft=M.n_eft)
    C.build_template = lambda t, model='lcdm', full_shape_only=False, verbose=True: set_data(build(t, model, full_shape_only, verbose), y)
    C.eft_start = lambda M: x_bf[:M.n_eft]; C.OUT = MOCK
    run.step_chain(argparse.Namespace(template='bao_template', seed=seed, draws=draws, chains=None, warmup=None,
                                      full_shape_only=False, redo=False, whiten_with_chains=False, extend=False))


# ------------------------------------------------------------------------------------------------- growth from the expansion
def check_growth():
    """growth_from_h against the cosmologies it should contain. On 300 draws of each reference chain: f, the alphas and the D
    ratios that the H(z) spline and the growth equation give at the map of the cosmology (TemplateModel.expansion), against the
    cosmology's own (the map of bao_template_no_rs), and ln L of one minus the other with the same P(k) and EFT parameters;
    exp(Delta ln L) gives the shift of the posterior mean it implies -> checks/growth_from_h.npz"""
    A, B = C.build_template('bao_template_no_rs', verbose=False), C.build_template('growth_from_h', verbose=False); out = {}; n = A.N
    for model in ('lcdm', 'w0wa', 'ede'):
        keys = COSMO_MODELS[model]; A.set_cosmo_model(model); B.set_cosmo_model(model); fa, fb, P = A.phi_ln_keys(keys), B.phi_ln_keys(keys), A.pidx
        x = np.load(os.path.join(C.reference_dir(model), 'chain.npz'))['x']; x = x.reshape(-1, x.shape[-1])
        x = x[np.random.default_rng(0).choice(len(x), 300, replace=False)]; e, th = x[:, :A.n_eft], x[:, A.n_eft:]

        def both(a, t):
            pa, pb = fa(t), fb(t); q = B.unpack(pb)
            return jnp.concatenate([jnp.concatenate([jnp.log(q[k]) - pa[P[k]] for k in ('f', 'apar', 'aper', 'D')]),
                                    jnp.atleast_1d(B.loglkl_phi(a, pb) - A.loglkl_phi(a, pa))])
        r = vbatch(both, e, th, B=25); d, dl = r[:, :-1], r[:, -1]; sh, ess = implied(th, dl)
        out.update({f'{model}_dln': d, f'{model}_dlnL': dl, f'{model}_th': th})
        log(f"[{model}] max |Delta ln|: f {np.abs(d[:, :n]).max():.2e}, alpha_par {np.abs(d[:, n:2*n]).max():.2e}, alpha_per {np.abs(d[:, 2*n:3*n]).max():.2e}, "
            f"D ratio {np.abs(d[:, 3*n:]).max():.2e}; Delta chi2 mean {-2 * dl.mean():+.3f} sd {2 * dl.std():.3f}; implied shift "
            + ", ".join(f"{k} {v:+.2f}" for k, v in zip(keys, sh)) + f" sigma (ESS/N {ess:.2f})")
        np.savez(os.path.join(C.OUT, 'checks', 'growth_from_h.npz'), **out)


def check_bestfit(names):
    """Is the MI best fit of `run.py chain` the maximum of the posterior? L-BFGS-B restarted from it, from the chain mean, from the
    chain's best draw and from 6 random draws, each run repeated from its own end point until ln posterior gains < 0.01
    -> checks/bestfit_<template>.npz: x, lnpost, chi2 of the stored best fit (row 0) and of the end points; growth_from_h: what
    each implies (`derived`)."""
    for t in names or ('bao_template_no_rs', 'growth_from_h'):
        M = C.build_template(t, verbose=False); lp, ll = jax.jit(M.logpost_mi), jax.jit(M.loglkl_mi)
        ch = C.load_chain(t); x = ch['x'].reshape(-1, M.n_mi); X = [np.load(os.path.join(C.template_dir(t), 'bestfit.npz'))['x_bf']]
        log(f"[{t}] stored best fit: ln posterior {float(lp(jnp.array(X[0]))):.2f}, chi2 {-2 * float(ll(jnp.array(X[0]))):.2f}")
        for lab, x0 in (('the stored best fit', X[0]), ('the chain mean', x.mean(0)), ('the best chain draw', x[np.argmax(ch['logp'].reshape(-1))]),
                        *((f'draw {i}', x[i]) for i in np.random.default_rng(0).choice(len(x), 6, replace=False))):
            v = float(lp(jnp.array(x0)))
            for it in range(10):
                x0, _ = minimize_lbfgs(lp, x0, maxiter=3000, name=lab, scales=M.min_scales('mi'), bounds=M.min_bounds('mi'), verbose=False)
                v, v0 = float(lp(jnp.array(x0))), v
                if v - v0 < 0.01: break
            X.append(x0); log(f"[{t}] from {lab}: ln posterior {v:.2f}, chi2 {-2 * float(ll(jnp.array(x0))):.2f} after {it + 1} runs")
            Xa = np.array(X); out = dict(x=Xa, lnpost=batched(lp, Xa), chi2=-2 * batched(ll, Xa), names=np.array(M.names), n_eft=M.n_eft)
            if M.gfh: out['derived'] = batched(lambda ph: jnp.concatenate(M.expansion(ph[M.pidx['hz']], ph[M.pidx['m']][0])), Xa[:, M.n_eft:])
            np.savez(os.path.join(C.OUT, 'checks', f'bestfit_{t}.npz'), **out)


# ------------------------------------------------------------------------------------------------- exact loops
def _widen(cfg):
    for pr in cfg['eft_prior'].values():
        if pr['type'] == 'marg_gauss': pr['range'] = [1e3 * r for r in pr['range']]


def loop_models(emu, wide=False):
    """(true likelihood, {template: MI model}) on the pybird likelihood with the loop emulator (emu) or exact FFTLog loops;
    wide: the priors of the 42 profiled EFT parameters x 1e3."""
    if not emu and not getattr(build_variant, 'matrices', False):
        import subprocess; subprocess.run([sys.executable, os.path.abspath(__file__), 'loop_matrices'], check=True); build_variant.matrices = True
    data = C.load_data(); data['cfg']['with_emu'] = emu
    if wide: _widen(data['cfg'])
    M0 = fix_rd(MIModel(data, C.FIDUCIAL, z_early=C.Z_EARLY, verbose=False)); mi = {}
    for t in ('bao_template', 'bao_template_no_rs'):
        d = C.load_data(); d['cfg']['with_emu'] = emu; c = C.TEMPLATES[t]
        if wide: _widen(d['cfg'])
        mi[t] = TemplateModel(d, C.FIDUCIAL, prior=C.MI_PRIOR, nodes={'spacing': c['node_spacing']}, bao_template=c['bao_template'],
                              alpha_rs=c.get('alpha_rs'), envelope=c.get('envelope'), ls_weight=C.KNOT_FISHER, z_early=C.Z_EARLY, verbose=False)
    return M0, mi


def check_exact_repr(models):
    """The representation term of the recovery test (run.py recovery: the reference chain importance-weighted by
    L_MI(phi(theta)) / L_true(theta)) with exact loops. The 2000 reference draws of run.py recovery (sampled with the
    emulator truth) are weighted to the truth with each kind of loops (L_true / L_true,emu) and to each MI model with the same
    loops (L_MI / L_true,emu); shift = MI mean - truth mean (same loops), in the reference sigma; ESS/N of each weighting.
    Also the 'oracle' (templates.sound_horizon_loglkl: the exact spectrum in the alpha_rs = 1 frame), which splits bao_template_no_rs - truth
    into the frame (oracle - truth) and the template (no_rs - oracle), and the Delta chi2 the EFT priors alone predict for
    the frame. With 'wide' the draws are still those of the analysis priors: a test of L_MI / L_true, not a wide-prior posterior."""
    wide = 'wide' in models; models = [m for m in models if m != 'wide'] or ['lcdm', 'w0wa']; t0 = time.time()
    f_out = os.path.join(C.OUT, 'checks', f"exact_repr{'_wide' if wide else ''}.npz")
    out = dict(np.load(f_out)) if os.path.exists(f_out) else {}                     # models run earlier stay in the file
    T, MI = {}, {}
    for emu in (True, False):
        T[emu], m = loop_models(emu, wide); MI.update({(t, emu): v for t, v in m.items()})
    log(f"EFT priors of the 42 profiled parameters: {'x 1e3 (wide)' if wide else 'as in the analysis'}")
    log(f"likelihoods built ({time.time() - t0:.0f}s)")
    for model in models:
        keys = COSMO_MODELS[model]
        for m in list(T.values()) + list(MI.values()): m.set_cosmo_model(model, gauss=C.desi_priors(model))
        fx = dict(T[True].cosmo_fixed); ne = T[True].n_eft
        xr = np.load(os.path.join(C.reference_dir(model), 'chain.npz'))['x']; xr = xr.reshape(-1, xr.shape[-1])
        sel = np.random.default_rng(1).choice(len(xr), min(2000, len(xr)), replace=False); e, th = xr[sel, :ne], xr[sel, ne:]
        L = {}
        for emu, M0 in T.items():
            L[('true', emu)] = vbatch(lambda a, s, M0=M0: true_loglkl(M0, a, M0.theta_to_full(s, keys, fx)), e, th, B=25)
            log(f"[{model}] true likelihood, {'emulator' if emu else 'exact loops'}: done ({time.time() - t0:.0f}s)")
        for (t, emu), M in MI.items():
            fw = M.phi_ln_keys(keys); L[(t, emu)] = vbatch(lambda a, s, M=M, fw=fw: M.loglkl_phi(a, fw(s)), e, th, B=25)
            log(f"[{model}] {t}, {'emulator' if emu else 'exact loops'}: done ({time.time() - t0:.0f}s)")
        # 'oracle': the truth's exact spectrum in the alpha_rs = 1 frame. oracle - truth is the frame alone (zero if the gauge
        # T_lam of notebook 6 were exact); bao_template_no_rs - oracle is what the 21 nodes + envelope cannot represent
        for emu in (True, False):
            MB = MI[('bao_template_no_rs', emu)]
            L[('oracle', emu)] = vbatch(lambda a, s, MB=MB: sound_horizon_loglkl(MB, a, MB.theta_to_full(s, keys, fx)), e, th, B=25)
        # prediction for oracle - truth from the EFT priors alone: alpha_rs = 1 rescales the 42 profiled parameters,
        # c -> alpha_rs^-n c (EFT_POWER), under priors that stay N(mu, sigma): Delta chi2 = sum [(alpha_rs^-n c - mu)^2 -
        # (c - mu)^2] / sigma^2 at the truth's profiled c (exact to first order in ln alpha_rs; re-profiling can only lower it)
        M0, MB = T[True], MI[('bao_template_no_rs', True)]; Lk = M0.L
        npow = jnp.array([float(EFT_POWER[p]) for p in Lk.bg_name]); mu, sg = jnp.asarray(Lk.bg_prior_mean).T, jnp.asarray(Lk.bg_prior_sigma).T

        def prior_pred(a, s):
            c = M0.theta_to_full(s, keys, fx); true_loglkl(M0, a, c)                       # leaves the profiled values in L.b_sky
            bg = jnp.array([[Lk.b_sky[i][p] for p in Lk.bg_name] for i in range(Lk.nsky)])
            lnars = jnp.log(MB.engine(c)[0]['rd_h'] / MB.rdh_fid)
            return jnp.stack([lnars, jnp.sum(((jnp.exp(-npow * lnars) * bg - mu)**2 - (bg - mu)**2) / sg**2)])
        lnars, pred = vbatch(prior_pred, e, th, B=25).T
        log(f"[{model}] oracle and EFT-prior prediction: done ({time.time() - t0:.0f}s); ln alpha_rs of the draws {lnars.mean():+.4f} +- {lnars.std():.4f}")
        for emu in (True, False):
            dc = -2 * (L[('oracle', emu)] - L[('true', emu)]); dn = -2 * (L[('bao_template_no_rs', emu)] - L[('oracle', emu)])
            log(f"=== {model} Delta chi2 vs ln alpha_rs, {'emulator   ' if emu else 'exact loops'}: oracle - truth slope {np.polyfit(lnars, dc, 1)[0]:+.1f} "
                f"(sd {dc.std():.2f}), predicted from the EFT priors {np.polyfit(lnars, pred, 1)[0]:+.1f} (sd {pred.std():.2f}), corr(measured, predicted) "
                f"{np.corrcoef(dc, pred)[0, 1]:+.2f}; no_rs - oracle slope {np.polyfit(lnars, dn, 1)[0]:+.1f} (sd {dn.std():.2f})")
        out.update({f'{model}_lnars': lnars, f'{model}_prior_pred': pred})
        base = L[('true', True)]

        def weighted(dl):
            w = np.exp(dl - dl.max()); w /= w.sum(); mu = w @ th
            return mu, np.sqrt(w @ (th - mu)**2), 1. / np.sum(w**2) / len(w)
        tr = {emu: weighted(L[('true', emu)] - base) for emu in (True, False)}
        log(f"[{model}] exact-loop truth vs emulator truth on the draws: Delta chi2 std {np.std(-2 * (L[('true', False)] - base)):.3f}, "
            f"ESS/N {tr[False][2]:.2f}; mean shift " + ", ".join(f"{k} {v:+.2f}" for k, v in zip(keys, (tr[False][0] - tr[True][0]) / tr[True][1])) + " sigma")
        for (t, emu) in list(MI) + [('oracle', True), ('oracle', False)]:
            mu, sd, ess = weighted(L[(t, emu)] - base); sh, wr = (mu - tr[emu][0]) / tr[emu][1], sd / tr[emu][1]
            out.update({f'{model}_{t}_{"emu" if emu else "exact"}_{q}': v for q, v in (('shift', sh), ('width', wr), ('ess', ess))})
            log(f"=== {model} {t:20s} {'emulator   ' if emu else 'exact loops'}: max |shift| {np.abs(sh).max():.2f} sigma (ESS/N {ess:.2f}); "
                + ", ".join(f"{k} {s:+.2f}" for k, s in zip(keys, sh)) + "; widths [" + ", ".join(f"{v:.2f}" for v in wr) + "]")
        for emu in (True, False):                                 # the template against the truth in its own frame
            (mo, so, _), (mn, _, _) = weighted(L[('oracle', emu)] - base), weighted(L[('bao_template_no_rs', emu)] - base)
            out[f'{model}_no_rs_vs_oracle_{"emu" if emu else "exact"}_shift'] = (mn - mo) / so
            log(f"=== {model} bao_template_no_rs vs oracle, {'emulator   ' if emu else 'exact loops'}: " + ", ".join(f"{k} {s:+.2f}" for k, s in zip(keys, (mn - mo) / so)))
        out.update({f'{model}_L_{k[0]}_{"emu" if k[1] else "exact"}': v for k, v in L.items()}, **{f'{model}_th': th, f'{model}_keys': np.array(keys)})
        np.savez(f_out, **out)
    log(f"-> checks/exact_repr{'_wide' if wide else ''}.npz ({(time.time() - t0) / 60:.1f} min)")


def check_adapted(models):
    """alpha_rs = 1 with the EFT priors adapted to its frame. The frame relabels the profiled EFT parameters, c -> alpha_rs^-n c
    (EFT_POWER), so the h/Mpc prior N(mu, sigma) on c is N(alpha_rs^-n mu, alpha_rs^-n sigma) on the frame's coefficient, with
    alpha_rs of the cosmology. With that prior set per draw, on the draws of exact_repr (loop emulator): the oracle (exact spectrum
    in the frame) and bao_template_no_rs against the true likelihood -> shifts added to checks/exact_repr.npz"""
    f_out = os.path.join(C.OUT, 'checks', 'exact_repr.npz'); out = dict(np.load(f_out)); MB = loop_models(True)[1]['bao_template_no_rs']; Lk = MB.L
    # the prior enters twice: the profile of the EFT parameters (F1, F2) and the penalty at the profiled values (mean, sigma)
    attr = ('F1_bg_prior_mean', 'F2_bg_prior_matrix', 'bg_prior_mean', 'bg_prior_sigma'); F1, F2, mu, sg = saved = [getattr(Lk, k) for k in attr]
    npow = jnp.array([float(EFT_POWER[p]) for p in Lk.bg_name])
    for model in models or ['w0wa']:
        keys = COSMO_MODELS[model]; MB.set_cosmo_model(model, gauss=C.desi_priors(model)); fx, fw, ne = dict(MB.cosmo_fixed), MB.phi_ln_keys(keys), MB.n_eft
        xr = np.load(os.path.join(C.reference_dir(model), 'chain.npz'))['x']; xr = xr.reshape(-1, xr.shape[-1])
        sel = np.random.default_rng(1).choice(len(xr), min(2000, len(xr)), replace=False); e, th = xr[sel, :ne], xr[sel, ne:]
        assert np.allclose(th, out[f'{model}_th']), 'not the draws of exact_repr'
        for t, f in (('oracle', lambda a, s, c: sound_horizon_loglkl(MB, a, c)), ('bao_template_no_rs', lambda a, s, c: MB.loglkl_phi(a, fw(s)))):
            def g(a, s):
                c = MB.theta_to_full(s, keys, fx); sp = jnp.exp(npow * jnp.log(MB.engine(c)[0]['rd_h'] / MB.rdh_fid)); sc = jnp.tile(sp, Lk.nsky)   # alpha_rs^n
                for k, v in zip(attr, (F1 * sc, F2 * sc[:, None] * sc[None, :], mu / sp[:, None], sg / sp[:, None])): setattr(Lk, k, v)   # mu, sigma / alpha_rs^n
                return f(a, s, c)
            try: l = vbatch(g, e, th, B=25)
            finally: [setattr(Lk, k, v) for k, v in zip(attr, saved)]
            dl = l - out[f'{model}_L_true_emu']; sh, ess = implied(th, dl)
            out.update({f'{model}_{t}_adapted_emu_shift': sh, f'{model}_L_{t}_adapted_emu': l})
            log(f"=== {model} {t}, EFT priors adapted to the frame: Delta chi2 vs truth mean {-2 * dl.mean():+.3f} sd {2 * dl.std():.3f}; max |shift| "
                f"{np.abs(sh).max():.2f} sigma (ESS/N {ess:.2f}); " + ", ".join(f"{k} {v:+.2f}" for k, v in zip(keys, sh)))
        np.savez(f_out, **out)


# ------------------------------------------------------------------------------------------------- degeneracy
# The alpha_rs gauge of bao_template (notebooks/6_alpha_rs_degeneracy.ipynb). A point x is moved by s = (s_rs, s_a, mu):
#     ln alpha_rs += s_rs,   every ln alpha_par(z), ln alpha_per(z) += s_a,   broadband and damping dilated by mu:
#     P_lin(k) = mu^3 a(mu k) T_nw(mu k) [1 + (1 + A) exp(-mu^2 k^2 dSigma^2 / 2) O(alpha_rs' k)]
# 'exact': this P_lin on the 80 knots; 'nodes': ln a' projected on the 21 nodes by the template's least-squares map and
# dSigma^2 -> mu^2 dSigma^2 (a point of the model itself). The moves, lam = e^t:
#     A  = (t, 0, 1)        alpha_rs -> lam alpha_rs
#     B  = (0, -t, 1)       every alpha -> alpha / lam: the same BAO alphas alpha / alpha_rs as A, broadband AP-dilated too
#     Bc = (0, -t, e^-t)    B with the broadband dilated back = T_{1/lam} applied to A: chi^2(Bc) = chi^2(A) if T is a symmetry
#     T  = (t, t, e^t)      the gauge T_lam: alpha_rs and every alpha x lam, broadband dilated; nothing observable should move
# T_lam also needs the 42 profiled EFT parameters rescaled, c -> lam^n c (EFT_POWER, the k-dimension of their term); the
# profile does that by itself, but their Gaussian priors do not move with them.
MOVES = {'A': lambda t: (t, 0. * t, 1. + 0. * t), 'B': lambda t: (0. * t, -t, 1. + 0. * t),
         'Bc': lambda t: (0. * t, -t, jnp.exp(-t)), 'T': lambda t: (t, t, jnp.exp(t))}
EFT_POWER = {'b3': 0, 'cct': 2, 'cr1': 2, 'cr2': 2, 'ce0': 3, 'ce1': 5, 'ce2': 5}
LAMS = np.array([0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 1.0, 1.005, 1.01, 1.02, 1.03, 1.04, 1.05])
LAMS_PROFILE = np.array([0.95, 0.97, 0.99, 1.0, 1.01, 1.03, 1.05])
LAMS_LADDER = np.array([0.95, 0.98, 0.99, 1.0, 1.01, 1.02, 1.05])
# pybird likelihood variants: (loop emulator, IR resummation, priors of the 42 profiled EFT parameters, profiled / marginalized)
VARIANTS = {'analysis': (True, True, 'analysis', True),            # the likelihood of every fit
            'wide': (True, True, 'wide', True),                    # their priors x 1e3
            'marg_wide': (True, True, 'wide', False),              # marginalized analytically (log det F2 kept), wide priors
            'exact_wide': (False, True, 'wide', True),             # exact one-loop (FFTLog), IR-resummed
            'exact_noresum_wide': (False, False, 'wide', True)}    # exact one-loop, no IR resummation: SPT is dilation covariant


def build_variant(name, y=None):
    """bao_template (as config.build_template) on a pybird likelihood variant (VARIANTS); y: replace the data vector."""
    emu, resum, prior, profiled = VARIANTS[name]; t = C.TEMPLATES['bao_template']
    if not emu and not getattr(build_variant, 'matrices', False):     # the container's ~/.cache does not outlive the job
        import subprocess; subprocess.run([sys.executable, os.path.abspath(__file__), 'loop_matrices'], check=True); build_variant.matrices = True
    data = C.load_data(); cfg = data['cfg']; cfg['with_emu'], cfg['with_resum'] = emu, resum
    if prior == 'wide':
        for pr in cfg['eft_prior'].values():
            if pr['type'] == 'marg_gauss': pr['range'] = [1e3 * r for r in pr['range']]
    M = TemplateModel(data, C.FIDUCIAL, prior=C.MI_PRIOR, nodes={'spacing': t['node_spacing']}, bao_template=t['bao_template'],
                      ls_weight=C.KNOT_FISHER, z_early=C.Z_EARLY, get_maxlkl=profiled, verbose=False)
    return M if y is None else set_data(M, y)


class Gauge:
    """chi^2 = -2 ln L of M's pybird likelihood (EFT priors included, no MI prior) at points moved by s = (s_rs, s_a, mu)."""

    def __init__(self, M):
        self.M, self.L, self.n = M, M.L, M.n_eft; p = M.pidx
        self.i_lna, self.i_env, self.i_rs = np.asarray(p['lna']), np.asarray(p['env']), int(p['rs'][0])
        self.i_a = np.concatenate([p['apar'], p['aper']])
        self.lnT = jnp.log(M.T_nw_knots)
        self.bao = []                                  # per sky: (number of BAO alphas, inverse of their block of the covariance)
        for i in range(self.L.nsky):
            nb = 1 if M.bao_iso[M.sky_to_z[i]] else 2
            self.bao.append((nb, jnp.array(np.linalg.inv(np.linalg.inv(np.asarray(self.L.p_sky[i]))[-nb:, -nb:]))))

    def lna_at(self, lna, k):
        M = self.M
        return _pb_interp1d(M.nodes_lnk, lna, kind='cubic')(jnp.clip(jnp.log(k), M.nodes_lnk[0], M.nodes_lnk[-1]))

    def spectrum(self, x, k, s):
        """P_lin(k [1/Mpc], z_N) [Mpc^3] of the moved point, exact."""
        M, phi = self.M, x[self.n:]; env, q = phi[self.i_env], s[2] * k
        E = (1. + env[0]) * jnp.exp(-0.5 * q**2 * env[1])
        return s[2]**3 * jnp.exp(self.lna_at(phi[self.i_lna], q)) * M.T_nw(q) * (1. + E * M.O_of_k(jnp.exp(phi[self.i_rs] + s[0]) * k))

    def phi(self, x, s):
        """ln phi of the moved point on the nodes (ln a' by the template's least-squares projection)."""
        M, phi = self.M, x[self.n:]; k = M.knots_mpc
        lna = 3. * jnp.log(s[2]) + self.lna_at(phi[self.i_lna], s[2] * k) + jnp.log(M.T_nw(s[2] * k)) - self.lnT
        phi = phi.at[self.i_rs].add(s[0]).at[self.i_a].add(s[1]).at[self.i_env[1]].multiply(s[2]**2)
        return phi.at[self.i_lna].set(M.Pi_ls @ lna)

    def x(self, x, s):
        return jnp.concatenate([x[:self.n], self.phi(x, s)])

    def dicts(self, phi, kk_h, pk):
        """M.cosmo_dicts with an explicit P_lin(kk_h h_fid, z_N) in Mpc^3."""
        M = self.M; p = M.unpack(phi); H, DA, bpar, bper = M.background(p)
        D = jnp.concatenate([p['D'], jnp.ones(1)]); pk = pk * M.h_fid**3; out = []
        for i in range(M.num_skies):
            j = M.sky_to_z[i]; b = M.bao[j]
            out.append({'H': H[j], 'DA': DA[j], 'f': p['f'][j], 'kk': kk_h, 'pk_lin': pk * D[j]**2,
                        'DH': bpar[j] * b['DH_over_rd_fid'], 'DM': bper[j] * b['DM_over_rd_fid'], 'rd': 1.0})
        return out

    def lnl(self, x, d):
        return self.L.loglkl(x[:self.n], self.M.eft_names_flat, need_cosmo_update=True, cosmo_dict=d, cosmo_module=None, cosmo_engine=None)

    def chi2(self, x, s, exact=False, kk_h=None):
        phi = self.phi(x, s)
        if not exact: return -2. * self.lnl(x, self.M.cosmo_dicts(phi))
        kk = self.M.kk if kk_h is None else jnp.asarray(kk_h)
        return -2. * self.lnl(x, self.dicts(phi, kk, self.spectrum(x, kk * self.M.h_fid, s)))

    def chi2_bao(self, x, s):
        """chi^2 of the BAO alphas alone (their block of the joint covariance)."""
        M = self.M; _, _, bpar, bper = M.background(M.unpack(self.phi(x, s))); out = 0.
        for i, (nb, Pb) in enumerate(self.bao):
            j = M.sky_to_z[i]
            th = jnp.atleast_1d(bpar[j]**(1 / 3.) * bper[j]**(2 / 3.)) if nb == 1 else jnp.stack([bpar[j], bper[j]])
            r = th - jnp.asarray(self.L.y_sky[i][-nb:]); out = out + r @ Pb @ r
        return out

    def profiler(self):
        """chi^2 minimized over psi = (12 sampled EFT, 21 ln a nodes, A, dSigma^2) from a starting point xm, the alphas, alpha_rs,
        f and D fixed: L-BFGS-B, one compile. Returns run(xm) -> (chi2, x, iterations)."""
        from scipy.optimize import minimize
        M = self.M; idx = np.r_[np.arange(self.n), self.n + self.i_lna, self.n + self.i_env]
        sc = np.asarray(M.min_scales('mi'))[idx]; b = M.min_bounds('mi'); scj = jnp.array(sc)
        vg = jax.jit(jax.value_and_grad(lambda u, xm: -2. * M.loglkl_mi(xm.at[idx].add(scj * u))))

        def run(xm, maxiter=5000):
            xm = jnp.asarray(xm); xn = np.asarray(xm)
            ub = [(None if b[i][0] is None else (b[i][0] - xn[i]) / sc[k], None if b[i][1] is None else (b[i][1] - xn[i]) / sc[k])
                  for k, i in enumerate(idx)]

            def fun(u):
                v, g = vg(jnp.array(u), xm); v, g = float(v), np.asarray(g, float)
                return (v if np.isfinite(v) else 1e30), np.where(np.isfinite(g), g, 0.)
            r = minimize(fun, np.zeros(len(idx)), jac=True, method='L-BFGS-B', bounds=ub,
                         options={'maxiter': maxiter, 'maxcor': 30, 'ftol': 1e-15, 'gtol': 1e-7})
            return float(r.fun), np.asarray(xm.at[idx].add(scj * jnp.array(r.x))), int(r.nit)
        return run

    def eft_templates(self, x, s):
        """Observed-space templates of the 42 profiled EFT parameters at the moved point (exact spectrum on the knots)."""
        M, L = self.M, self.L; phi = self.phi(x, s)
        self.lnl(x, self.dicts(phi, M.kk, self.spectrum(x, M.kk * M.h_fid, s)))
        return [np.asarray(L.correlator_sky[i].getmarg(L.b_sky[i], L.bg_name))[:, np.asarray(L.m_sky[i])] for i in range(L.nsky)]

    def fisher_jacobian(self, x):
        """Whitened Jacobian Xp (rows: data, priors) of the likelihood over the 59 MI parameters at x with the 42 profiled EFT
        parameters eliminated (their Gaussian priors kept; c2's prior kept): F = Xp^T Xp is the Gauss-Newton Fisher of the
        profile likelihood, never formed. Also returns -2 ln L rebuilt from the residuals (must equal the likelihood's)."""
        M, L = self.M, self.L; ng, ns = len(L.bg_name), L.nsky; x = jnp.asarray(x)
        lnl0 = float(M.loglkl_mi(x))
        bg0 = np.array([[float(L.b_sky[i][p]) for p in L.bg_name] for i in range(ns)]).reshape(-1)

        def theory(z):
            xx, bg = z[:M.n_mi], z[M.n_mi:].reshape(ns, ng); M.loglkl_mi(xx); out = []
            for i in range(ns):
                bi = dict(L.b_sky[i]); bi.update({p: bg[i, k] for k, p in enumerate(L.bg_name)})
                v = L.correlator_sky[i].get(bi).reshape(-1)[L.m_sky[i]]
                out.append(jnp.concatenate([v, jnp.atleast_1d(jnp.asarray(L.alpha_sky[i]))]))
            return jnp.concatenate(out)
        z0 = jnp.concatenate([x, jnp.array(bg0)])
        th0 = np.asarray(theory(z0)); J = np.asarray(jax.jacfwd(theory)(z0))
        P = np.asarray(L.p_all, float); r = th0 - np.asarray(L.y_all, float)
        sig_bg = np.asarray(L.bg_prior_sigma, float).T.reshape(-1); mu_bg = np.asarray(L.bg_prior_mean, float).T.reshape(-1)
        ic2 = np.array([i for i, nm in enumerate(M.eft_names_flat) if nm == 'c2'])
        sig_c2, mu_c2 = np.asarray(L.bng_prior_sigma, float)[0], np.asarray(L.bng_prior_mean, float)[0]
        chi2 = r @ P @ r + np.sum(((bg0 - mu_bg) / sig_bg)**2) + np.sum(((np.asarray(x)[ic2] - mu_c2) / sig_c2)**2)
        W = np.linalg.cholesky(P); Jd = W.T @ J; nx = M.n_mi
        Xc2 = np.zeros((len(ic2), nx)); Xc2[np.arange(len(ic2)), ic2] = 1. / sig_c2
        X = np.vstack([Jd[:, :nx], np.zeros((len(sig_bg), nx)), Xc2])
        Cm = np.vstack([Jd[:, nx:], np.diag(1. / sig_bg), np.zeros((len(ic2), len(sig_bg)))])
        Q, _ = np.linalg.qr(Cm)
        return X - Q @ (Q.T @ X), chi2, -2. * lnl0


def _jit(f):
    """jax.jit of f(x, s) called with s as a float64 array (one compile whatever the move)."""
    fj = jax.jit(f)
    return lambda x, s: fj(jnp.asarray(x), jnp.array([float(v) for v in s]))


def _eft_powers(g, x0j, lam=1.02):
    """Measured n of the 42 profiled EFT templates in observed space, T' = lam^-n T at T_lam vs x0: median over the bins where
    the template is > 1% of its maximum, and the 5-95% spread over those bins."""
    L = g.L; Ta, Tb = g.eft_templates(x0j, (0., 0., 1.)), g.eft_templates(x0j, MOVES['T'](np.log(lam)))
    npow, nspread = np.zeros((L.nsky, len(L.bg_name))), np.zeros((L.nsky, len(L.bg_name)))
    for i in range(L.nsky):
        for k in range(len(L.bg_name)):
            a, b = Ta[i][k], Tb[i][k]; w = np.abs(a) > 1e-2 * np.abs(a).max()
            if not w.any(): npow[i, k] = nspread[i, k] = np.nan; continue
            rr = -np.log(np.abs(b[w] / a[w])) / np.log(lam); npow[i, k], nspread[i, k] = np.median(rr), np.percentile(rr, 95) - np.percentile(rr, 5)
    return {'eft_n': npow, 'eft_n_spread': nspread}


def _fisher(g, x0):
    """The Fisher tests at x0 (section 3 of the notebook): curvatures kappa = d^T F d per unit ln lam along the moves, the
    correlation of A and Bc, alpha_rs with everything else free, the gauge in the spectrum of the prior-whitened Fisher."""
    M = g.M; x0j = jnp.asarray(x0); Xp, chi2_rebuilt, chi2_lkl = g.fisher_jacobian(x0)
    e = {m: np.asarray(jax.jacfwd(lambda t, m=m: g.x(x0j, MOVES[m](t)))(0.)) for m in MOVES}
    kap = {m: float(np.sum((Xp @ e[m])**2)) for m in e}
    FA = np.array([[(Xp @ e[a]) @ (Xp @ e[b]) for b in ('A', 'Bc')] for a in ('A', 'Bc')])
    # the data's own best compensation of an alpha_rs step: min |Xp d|^2 with d_lnrs = 1, ties along directions the data cannot
    # see (the lowest-k nodes, ...) broken by 1e-3 x the MI prior (stacked least squares, F never formed)
    Pi = np.zeros((M.n_mi, M.n_mi)); Pi[:g.n, :g.n] = np.diag(np.r_[[12. / 100. if nm == 'b1' else 1. / 25. for nm in M.eft_names_flat]])
    Pi[g.n:, g.n:] = M.prior_prec; Aa = np.vstack([Xp, np.sqrt(1e-3) * np.linalg.cholesky(Pi).T])
    j = g.n + g.i_rs; oth = np.array([i for i in range(M.n_mi) if i != j])
    u, *_ = np.linalg.lstsq(Aa[:, oth], Aa[:, j], rcond=None); d_opt = np.zeros(M.n_mi); d_opt[j] = 1.; d_opt[oth] = -u
    res = Xp @ d_opt
    Sig = np.zeros((M.n_mi, M.n_mi)); Sig[:g.n, :g.n] = np.diag(np.r_[[10. / np.sqrt(12.) if nm == 'b1' else 5. for nm in M.eft_names_flat]]**2)
    Sig[g.n:, g.n:] = np.linalg.inv(M.prior_prec); S = np.linalg.cholesky(Sig)
    _, sv, Vt = np.linalg.svd(Xp @ S, full_matrices=False)
    vz = {m: np.linalg.solve(S, e[m]) for m in ('T', 'A')}; vz = {m: v / np.linalg.norm(v) for m, v in vz.items()}
    return dict({f'kappa_{m}': v for m, v in kap.items()}, F_A_Bc=FA, rho_A_Bc=FA[0, 1] / np.sqrt(FA[0, 0] * FA[1, 1]),
                kappa_min_rs=float(res @ res), d_opt=d_opt, gauge=e['T'], e_A=e['A'], e_Bc=e['Bc'], names=np.array(M.names),
                mu=sv**2, overlap_T=(Vt @ vz['T'])**2, overlap_A=(Vt @ vz['A'])**2, cos_soft_T=float(abs(Vt[-1] @ vz['T'])),
                rayleigh_T=float(np.sum((Xp @ S @ vz['T'])**2)), rayleigh_A=float(np.sum((Xp @ S @ vz['A'])**2)),
                fisher_chi2_check=np.array([chi2_rebuilt, chi2_lkl]))


def degeneracy_extra(case, root, y):
    """Appended to checks/degeneracy_<case>.npz: the alpha_rs -> 1 map of 80 of the 400 draws with the emulator and wide EFT
    priors ('wide') and with exact loops ('exact_wide'), and the Fisher tests and EFT powers with exact loops at the best fit."""
    f = os.path.join(C.OUT, 'checks', f'degeneracy_{case}.npz'); out = dict(np.load(f)); t0 = time.time()
    x0 = out['x0']; fs = sorted(glob.glob(os.path.join(root, 'bao_template', 'chain_seed*.npz'))); ch = [np.load(q)['x'] for q in fs]
    n = min(c.shape[1] for c in ch); xs = np.concatenate([c[:, :n] for c in ch]).reshape(-1, len(x0))
    sel = xs[np.linspace(0, len(xs) - 1, 400).astype(int)]; sub = np.arange(0, 400, 5); out['map_sub'] = sub
    for name in ('wide', 'exact_wide'):
        Mv = build_variant(name, y); gv = Gauge(Mv); fv = _jit(lambda x, s, gv=gv: gv.chi2(x, s, exact=True)); j = gv.n + gv.i_rs
        out[f'map_{name}'] = np.array([float(fv(sel[i], MOVES['T'](-sel[i, j]))) - float(fv(sel[i], (0., 0., 1.))) for i in sub])
        log(f"[{case}] alpha_rs -> 1, {len(sub)} draws, {name}: |Delta chi2| median {np.median(np.abs(out[f'map_{name}'])):.4f}, "
            f"95% {np.percentile(np.abs(out[f'map_{name}']), 95):.4f} (analysis on the same draws: {np.median(np.abs(out['map_exact'][sub])):.4f}, "
            f"{np.percentile(np.abs(out['map_exact'][sub]), 95):.4f}); {time.time() - t0:.0f}s")
        if name == 'exact_wide':
            out.update({f'exact_{k}': v for k, v in _eft_powers(gv, jnp.asarray(x0)).items()})
            out.update({f'exact_{k}': v for k, v in _fisher(gv, x0).items()})
            log(f"[{case}] exact loops, wide priors: EFT powers median {np.round(np.nanmedian(out['exact_eft_n'], 0), 3)}; Fisher curvature "
                + ", ".join(f"{m} {out[f'exact_kappa_{m}']:.4g}" for m in MOVES) + f"; rho(A, Bc) = {out['exact_rho_A_Bc']:.6f}; data/prior "
                f"along the gauge {out['exact_rayleigh_T']:.3g} vs alpha_rs alone {out['exact_rayleigh_A']:.3g}; {time.time() - t0:.0f}s")
    np.savez(f, **out); log(f"-> {f}")


def degeneracy_fisher(case, y):
    """Recompute only the Fisher tests (analysis and exact loops) in checks/degeneracy_<case>.npz."""
    f = os.path.join(C.OUT, 'checks', f'degeneracy_{case}.npz'); out = dict(np.load(f))
    out.update(_fisher(Gauge(build_variant('analysis', y)), out['x0']))
    out.update({f'exact_{k}': v for k, v in _fisher(Gauge(build_variant('exact_wide', y)), out['x0']).items()})
    np.savez(f, **out); log(f"[{case}] Fisher tests recomputed -> {f}: alpha_rs best compensation, data-only curvature "
                            f"{out['kappa_min_rs']:.4g} (exact loops {out['exact_kappa_min_rs']:.4g})")


def case_paths(case):
    """(directory with bao_template/{bestfit, chain_seed*}.npz, data vector or None) of a case."""
    if case == 'mock': return MOCK, mock_vector()[0]
    return C.OUT, None


def check_degeneracy(args):
    case = args[0] if args else 'smoke'
    root, y = case_paths('data' if case == 'smoke' else case); t0 = time.time()
    if args[1:2] == ['extra']: return degeneracy_extra(case, root, y)
    if args[1:2] == ['fisher']: return degeneracy_fisher(case, y)
    x0 = np.load(os.path.join(root, 'bao_template', 'bestfit.npz'))['x_bf']; x0j = jnp.array(x0)
    M = build_variant('analysis', y); g = Gauge(M)
    log(f"[{case}] bao_template analysis likelihood built ({time.time() - t0:.0f}s); base point = MI best fit of {root}/bao_template")
    f_nodes, f_exact, f_bao = _jit(lambda x, s: g.chi2(x, s)), _jit(lambda x, s: g.chi2(x, s, exact=True)), _jit(g.chi2_bao)
    one = (0., 0., 1.)
    c0 = float(f_nodes(x0j, one)); out = {'chi2_base': c0}
    # identities at lam = 1 (the code paths agree before anything moves)
    ident = {'chi2_eager_minus_jit': float(-2 * M.loglkl_mi(x0j)) - c0, 'chi2_exact_minus_nodes': float(f_exact(x0j, one)) - c0,
             'spectrum_rel': float(jnp.max(jnp.abs(g.spectrum(x0j, M.knots_mpc, one) / M.spectrum_knots(x0j[g.n:][g.i_lna], x0[g.n + g.i_rs],
                                                                                                          x0j[g.n:][g.i_env]) - 1.))),
             'phi_nodes_abs': float(jnp.max(jnp.abs(g.phi(x0j, one) - x0j[g.n:])))}
    log(f"[{case}] chi2(base) = {c0:.4f}; identities at lam = 1: {ident}")
    out.update({f'ident_{k}': v for k, v in ident.items()})
    if case == 'smoke':
        for f, nm in ((f_nodes, 'nodes'), (f_exact, 'exact')):
            t1 = time.time(); [float(f(x0j, MOVES['T'](jnp.log(1.02)))) for _ in range(5)]; log(f"{nm}: {(time.time() - t1) / 5 * 1e3:.0f} ms per chi2")
        run = g.profiler(); t1 = time.time(); c, _, nit = run(np.asarray(g.x(x0j, MOVES['A'](jnp.log(1.01))))); log(f"profile A(1.01): chi2 {c:.3f}, {nit} it, {time.time() - t1:.0f}s")
        t1 = time.time(); Me = build_variant('exact_wide'); ge = Gauge(Me); log(f"exact loops built in {time.time() - t1:.0f}s")
        fe = _jit(lambda x, s: ge.chi2(x, s, exact=True)); t1 = time.time()
        a, b = float(fe(x0j, one)), float(fe(x0j, MOVES['T'](jnp.log(1.02))))
        log(f"exact loops: chi2 {a:.3f} (emulator, analysis priors {c0:.3f}), T(1.02) - base {b - a:+.4f} ({time.time() - t1:.0f}s)")
        return

    # 1. the moves at fixed broadband / EFT, exact and on the nodes, and the BAO alphas alone
    tl = np.log(LAMS)
    for m, mv in MOVES.items():
        out[f'fixed_{m}_nodes'] = np.array([float(f_nodes(x0j, mv(t))) for t in tl]) - c0
        out[f'fixed_{m}_exact'] = np.array([float(f_exact(x0j, mv(t))) for t in tl]) - c0
        out[f'bao_{m}'] = np.array([float(f_bao(x0j, mv(t))) for t in tl]) - float(f_bao(x0j, one))
        log(f"[{case}] move {m:2s} lam {LAMS[[0, 4, 8, -1]]}: nodes {np.round(out[f'fixed_{m}_nodes'][[0, 4, 8, -1]], 3)}, "
            f"exact {np.round(out[f'fixed_{m}_exact'][[0, 4, 8, -1]], 3)}, BAO alone {np.round(out[f'bao_{m}'][[0, 4, 8, -1]], 3)}")
    # 2. profiles: psi re-fitted at the moved alphas / alpha_rs; starts mu = 1 and mu = e^{s_a} (the T-image of an A move)
    run = g.profiler(); c1, _, _ = run(x0); out['prof_base'] = c1
    for m in ('A', 'B', 'T'):
        v, it = [], []
        for t in np.log(LAMS_PROFILE):
            s = [float(u) for u in MOVES[m](t)]; starts = {1.0, float(np.exp(s[1]))}
            r = [run(np.asarray(g.x(x0j, (s[0], s[1], mu)))) for mu in starts]
            best = min(r, key=lambda q: q[0]); v.append(best[0] - c1); it.append(best[2])
        out[f'prof_{m}'], out[f'prof_it_{m}'] = np.array(v), np.array(it)
        log(f"[{case}] profiled {m}: {np.round(out[f'prof_{m}'], 3)} (lam {LAMS_PROFILE}); {time.time() - t0:.0f}s")
    # 3. what breaks T: the likelihood variants (exact spectrum along T) and the measured EFT rescaling powers
    tl = np.log(LAMS_LADDER)
    for name in VARIANTS:
        if name == 'analysis': Mv, gv = M, g
        else: Mv = build_variant(name, y); gv = Gauge(Mv)
        fv = _jit(lambda x, s, gv=gv: gv.chi2(x, s, exact=True))     # the exact path of pybird needs P_lin on the 80 knots too
        b0 = float(fv(x0j, one)); out[f'ladder_{name}'] = np.array([float(fv(x0j, MOVES['T'](t))) for t in tl]) - b0
        out[f'ladder_base_{name}'] = b0
        log(f"[{case}] T along {name:20s}: {np.round(out[f'ladder_{name}'], 4)}  (chi2 at lam=1: {b0:.3f})")
    out['ladder_nodes'] = np.array([float(f_nodes(x0j, MOVES['T'](t))) for t in tl]) - c0
    out.update(_eft_powers(g, x0j), eft_names=np.array(M.L.bg_name), eft_n_pred=np.array([EFT_POWER[p] for p in M.L.bg_name]),
               skies=np.array(M.skies)); npow = out['eft_n']
    log(f"[{case}] EFT rescaling powers (median over bins; predicted {dict(EFT_POWER)}):\n" + "\n".join(
        f"   {s:5s} " + " ".join(f"{p}={npow[i, k]:+.3f}" for k, p in enumerate(M.L.bg_name)) for i, s in enumerate(M.skies)))
    # 4. Fisher of the analysis likelihood at the base point (CPU, x64): curvatures along the moves, the gauge in its spectrum
    out.update(_fisher(g, x0)); j = g.n + g.i_rs; e = {m: out[k] for m, k in (('T', 'gauge'), ('A', 'e_A'))}; kap = {'T': out['kappa_T']}
    log(f"[{case}] Fisher: -2 ln L rebuilt from the residuals {out['fisher_chi2_check'][0]:.6f} vs likelihood {out['fisher_chi2_check'][1]:.6f}; "
        f"curvature per unit ln lam: " + ", ".join(f"{m} {out[f'kappa_{m}']:.4g}" for m in MOVES) + f"; alpha_rs with all else free "
        f"{out['kappa_min_rs']:.4g}; rho(A, Bc) = {out['rho_A_Bc']:.6f}; cos(gauge, softest) = {out['cos_soft_T']:.4f}")
    # 5. the chains: posterior along the ridge (ln alpha_rs), and the map alpha_rs -> 1 (lam = 1/alpha_rs) of 400 draws
    fs = sorted(glob.glob(os.path.join(root, 'bao_template', 'chain_seed*.npz'))); ch = [np.load(f)['x'] for f in fs]
    n = min(c.shape[1] for c in ch); xs = np.concatenate([c[:, :n] for c in ch]).reshape(-1, M.n_mi)
    Pi = np.zeros((M.n_mi, M.n_mi)); Pi[g.n:, g.n:] = M.prior_prec
    out.update(chain_lnrs=xs[:, j].astype(np.float32), map_lnrs=x0[j], sigma_prior_T=float(1 / np.sqrt(e['T'] @ Pi @ e['T'])),
               sigma_prior_A=float(1 / np.sqrt(e['A'] @ Pi @ e['A'])), sigma_like_T=float(1 / np.sqrt(kap['T'])),
               sigma_post_T=float(1 / np.sqrt(e['T'] @ Pi @ e['T'] + kap['T'])), chains=np.array([os.path.basename(f) for f in fs]))
    if case == 'mock':
        tr = np.load(os.path.join(MOCK, 'truth.npz')); out['truth_lnrs'] = float(tr['ln_phi'][list(tr['names']).index('ln alpha_rs')])
    sel = xs[np.linspace(0, len(xs) - 1, 400).astype(int)]; dn, de, lr = [], [], []
    for x in sel:
        xj = jnp.array(x); s = MOVES['T'](-x[j]); base = float(f_nodes(xj, one))
        dn.append(float(f_nodes(xj, s)) - base); de.append(float(f_exact(xj, s)) - base); lr.append(x[j])
    out.update(map_draw_lnrs=np.array(lr), map_nodes=np.array(dn), map_exact=np.array(de))
    log(f"[{case}] alpha_rs -> 1 on 400 draws: Delta chi2 exact {np.median(de):+.4f} [{np.min(de):+.4f}, {np.max(de):+.4f}], "
        f"nodes {np.median(dn):+.4f} [{np.min(dn):+.4f}, {np.max(dn):+.4f}]; ridge: chain sd {xs[:, j].std():.4f}, prior along T "
        f"{out['sigma_prior_T']:.4f}, likelihood along T {out['sigma_like_T']:.4g}")
    out.update(lams=LAMS, lams_profile=LAMS_PROFILE, lams_ladder=LAMS_LADDER, variants=np.array(list(VARIANTS)), case=case, x0=x0)
    f = os.path.join(C.OUT, 'checks', f'degeneracy_{case}.npz'); np.savez(f, **out); log(f"-> {f} ({time.time() - t0:.0f}s)")


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'map'
    {'map': lambda: check_map(sys.argv[2:]), 'theta': check_theta, 'ede': check_ede, 'mock': lambda: check_mock(sys.argv[2:]), 'truth': check_truth, 'growth': check_growth, 'bestfit': lambda: check_bestfit(sys.argv[2:]), 'adapted': lambda: check_adapted(sys.argv[2:]),
     'degeneracy': lambda: check_degeneracy(sys.argv[2:]), 'exact_repr': lambda: check_exact_repr(sys.argv[2:])}[cmd]()
