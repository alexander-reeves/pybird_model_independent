"""Recovery test of a wiggle/no-wiggle MI model: its MI chain projected onto each direct model,

    p(theta | d)  proportional to  p_MI(phi_w(theta) | d) / pi_MI(phi_w(theta)) * pi(theta),

against the EXACT direct chain of that model (output/desi_fsbao/<cosmo>/, the full likelihood with the
CosmoPower/ede-v2 spectrum, not the wiggle model). For the models with omega_b free on the CosmoPower engine (lcdm5,
w0wa7) the reference is the chain with the corrected sound horizon (wiggle_direct.py, output/desi_fsbao/rdfix_<cosmo>/):
pybird.symbolic.rs_drag has the wrong sign on omega_b, which no model that ties the full-shape wiggles to the BAO can
reproduce. Any miss is split in two:
  representation  the direct chain importance-weighted by L_wiggle(phi_w(theta)) / L_exact(theta), i.e. the
                  direct posterior the wiggle model would give if it were sampled directly ("direct-W");
  compression     the projection against direct-W (the density estimate of the MI chain).

    SCRIPT=wiggle_recovery.py ARGS="--wiggle B --est gauss,c5" sbatch exec_wiggle.sbatch
Writes wiggle_<w>/{chain_projected_<cosmo>_<est>.npz, reweight_<cosmo>.npz, recovery_<est>.npz, recovery.log}.
"""
import os, sys, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import settings as S
import wiggle_settings as W
from model import COSMO_MODELS
from sample import Tee
from sampling import log, WhitenedNUTS, minimize_lbfgs, fisher_at, precond_cov, BoxTransform
from compress import load_or_fit, projected_logpost, gaussian_estimate
from wiggle_direct import fix_rd, rdfix_files, true_loglkl

p = argparse.ArgumentParser(); p.add_argument('--wiggle', default='B'); p.add_argument('--est', default='gauss')
p.add_argument('--cosmo', default=','.join(W.COSMO_VARIANTS)); p.add_argument('--n-rw', type=int, default=2000)
p.add_argument('--force', action='store_true')
a = p.parse_args()
s = W.resolve(a.wiggle); OUT = s['out']; os.makedirs(OUT, exist_ok=True)
sys.stdout = Tee(os.path.join(OUT, 'recovery.log')); sys.stderr = sys.stdout
log(f"wiggle_recovery.py {' '.join(sys.argv[1:])}; JAX devices {jax.devices()}")
M, _ = W.build(s); F = W.files(s)
M0, _ = S.build(S.resolve('baseline'), verbose=False)
M0fix = fix_rd(S.build(S.resolve('baseline'), verbose=False)[0])

def reference(cv, keys, fixed):
    """(direct chain file, its log-likelihood (eft, theta) -> ln L) for cosmo variant cv: the true-likelihood chain
    (wiggle_direct.py exact) if it exists, else the corrected-r_d one, else the original (lcdm3 and EDE do not depend on
    the r_d fix; lcdm5/w0wa7 do, and get a warning)."""
    for kind in ('exact', 'exactrw'):              # sampled true-likelihood chain, else the reweighted original (wiggle_reweight_ref.py)
        f = rdfix_files(cv, kind)[1]['chain_direct']
        try: np.load(f)['x'].shape                 # a chain still being written (np.savez in place) fails here -> next kind
        except Exception: continue
        return f, (lambda e, t: true_loglkl(M0fix, e, M0fix.theta_to_full(t, keys, fixed)))
    f = rdfix_files(cv, 'rdfix')[1]['chain_direct']; f0 = M0fix.phi_ln_keys(keys)
    if os.path.exists(f): return f, (lambda e, t: M0fix.loglkl_phi(e, f0(t)))
    if cv in ('lcdm5', 'w0wa7'): log(f"WARNING: no corrected-r_d reference for {cv}: comparing with the original chain")
    f0 = M0.phi_ln_keys(keys)
    return S.files(S.resolve(cv))['chain_direct'], (lambda e, t: M0.loglkl_phi(e, f0(t)))
xm = np.load(F['chain_mi'])['x']; xm = xm.reshape(-1, xm.shape[-1])
log(f"MI chain {F['chain_mi']}: {len(xm)} draws x {M.n_mi}")


def batched(f, *args, B=50):
    fb = jax.jit(jax.vmap(f)); n = len(args[0]); out = []
    for i in range(0, n, B):
        a_ = [np.asarray(x[i:i + B]) for x in args]; m = len(a_[0])
        if m < B: a_ = [np.concatenate([x, np.repeat(x[-1:], B - m, 0)]) for x in a_]
        out.append(np.asarray(fb(*[jnp.array(x) for x in a_]))[:m])
    return np.concatenate(out)


def stats(th_ref, th, w=None):
    w = np.full(len(th), 1. / len(th)) if w is None else w / w.sum()
    mu = w @ th; sd = np.sqrt(w @ (th - mu)**2)
    return (mu - th_ref.mean(0)) / th_ref.std(0), sd / th_ref.std(0)


# ---- representation: the exact direct chain reweighted to the wiggle model's likelihood --------------
direct, rw = {}, {}
for cv in a.cosmo.replace('+', ',').split(','):
    sv = S.resolve(cv); keys = COSMO_MODELS[sv['cosmo_model']]
    for m in (M, M0, M0fix): m.set_cosmo_model(sv['cosmo_model'], gauss=sv['cosmo_gauss'])
    f_d, lnL_ref = reference(cv, keys, dict(M0fix.cosmo_fixed))
    x = np.load(f_d)['x']; x = x.reshape(-1, x.shape[-1]); direct[cv] = x[:, M.n_eft:]; log(f"{cv}: direct chain {f_d}")
    f_rw = os.path.join(OUT, f'reweight_{cv}.npz')
    if os.path.exists(f_rw) and not a.force and str(np.load(f_rw).get('direct_file', '')) == f_d:
        rw[cv] = dict(np.load(f_rw)); log(f"loaded {f_rw}"); continue
    sel = np.random.default_rng(1).choice(len(x), min(a.n_rw, len(x)), replace=False); xs = x[sel]
    fw = M.phi_ln_keys(keys)
    t0 = time.time()
    L0 = batched(lnL_ref, xs[:, :M.n_eft], xs[:, M.n_eft:])
    Lw = batched(lambda e, t: M.loglkl_phi(e, fw(t)), xs[:, :M.n_eft], xs[:, M.n_eft:])
    d = Lw - L0; w = np.exp(d - d.max())
    rw[cv] = dict(sel=sel, d=d, w=w, th=xs[:, M.n_eft:], direct_file=f_d); np.savez(f_rw, **rw[cv])
    sh, ra = stats(direct[cv], xs[:, M.n_eft:], w)
    log(f"[direct-{a.wiggle}] {cv}: {len(d)} draws in {time.time()-t0:.0f}s; Delta ln L mean {d.mean():+.3f} std {d.std():.3f}; "
        f"ESS/N {w.sum()**2 / (w**2).sum() / len(w):.2f}; shifts {np.round(sh, 3)} sigma, width ratios {np.round(ra, 3)}")


# ---- compression: the projected chains ---------------------------------------------------------------
# M variants: the density (and the MI prior divided out) is the marginal over the wiggle envelope (A, dSigma^2), so the
# envelope is a nuisance measured by the data alone, as in a BAO fit; theta then predicts only the rest of phi.
blk = dict(block=np.setdiff1d(np.arange(M.n_phys), M.pidx['env'])) if s['marg_env'] else {}
if blk: log(f"marginalizing the wiggle envelope {M.env_names()} in the projection")
for est in a.est.replace('+', ',').split(','):
    fl = (gaussian_estimate(M, xm, **blk) if est == 'gauss' else
          load_or_fit(M, xm, os.path.join(OUT, f'flow_{est}.pkl'), n_seeds=10, steps=8000, **blk, **S.FLOW_CONFIGS[est]))
    summary = {}
    for cv in a.cosmo.replace('+', ',').split(','):
        sv = S.resolve(cv); M.set_cosmo_model(sv['cosmo_model'], gauss=sv['cosmo_gauss'])
        th_d = direct[cv]; f_pr = os.path.join(OUT, f'chain_projected_{cv}_{est}.npz')
        if os.path.exists(f_pr) and not a.force:
            th_p = np.load(f_pr)['x'].reshape(-1, M.n_cosmo); log(f"loaded {f_pr}")
        else:
            bt = BoxTransform([M.cosmo_box[k][0] for k in M.cosmo_keys], [M.cosmo_box[k][1] for k in M.cosmo_keys])
            lp_y = jax.jit(bt.wrap(projected_logpost(M, fl)))
            y_bf, _ = minimize_lbfgs(lp_y, np.asarray(bt.to_y(jnp.array(np.median(th_d, 0)))), name=f'projected {cv}', scales=np.full(M.n_cosmo, 0.3))
            r = WhitenedNUTS(lp_y, y_bf, precond_cov(fisher_at(lp_y, y_bf, name='projected'), floor_prec=0.25), name=f'projected {cv}').run(
                jax.random.key(41), 1000, 5000, n_chains=8, jitter=0.5, max_num_doublings=7)
            r['y'] = r['x']; r['x'] = np.asarray(jax.vmap(jax.vmap(bt.to_x))(jnp.array(r['y'])))
            np.savez(f_pr, **r); th_p = r['x'].reshape(-1, M.n_cosmo)
        sh, ra = stats(th_d, th_p)
        shw, raw = stats(th_d, rw[cv]['th'], rw[cv]['w'])
        summary[cv] = dict(names=M.cosmo_names, shift=sh, ratio=ra, shift_rep=shw, ratio_rep=raw, samples=th_p)
        log(f"=== {cv} ({M.cosmo_model}) [{a.wiggle}, {est}]: MI -> model vs the exact direct chain")
        for i, n in enumerate(M.cosmo_names):
            log(f"   {n:11s} direct {th_d[:, i].mean():.4f} +- {th_d[:, i].std():.4f}   MI {th_p[:, i].mean():.4f} +- {th_p[:, i].std():.4f}   "
                f"shift {sh[i]:+.2f} sigma (representation {shw[i]:+.2f}), width {ra[i]:.3f} ({raw[i]:.3f})")
        ok = bool((np.abs(sh) < 0.3).all() and (np.abs(ra - 1) < 0.2).all())
        log(f"   T3-{M.cosmo_model} [{a.wiggle}, {est}]: max |shift| {np.abs(sh).max():.2f} sigma, widths [{ra.min():.2f}, {ra.max():.2f}] -> "
            f"{'PASS' if ok else 'FAIL'} (|shift|<0.3, |ratio-1|<0.2)")
    f_sum = os.path.join(OUT, f'recovery_{est}.npz')                 # merged: runs may cover different cosmo variants
    old = {k: v for k, v in np.load(f_sum).items()} if os.path.exists(f_sum) else {}
    old = {k: v for k, v in old.items() if k.split('_')[0] not in summary and not any(k.startswith(cv + '_') for cv in summary)}
    np.savez(f_sum, **old, **{f'{cv}_{k}': v for cv, d in summary.items() for k, v in d.items()})
log("done")
