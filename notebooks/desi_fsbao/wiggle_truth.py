"""Which direct likelihood is "exact"? The direct model of model.MIModel is the MI likelihood composed with phi(theta):
P_lin(theta)/T sampled on the 60 ln a nodes, cubic onto the 80 knots in 1/Mpc, relabelled to h/Mpc by h_conv = h and
interpolated again by the emulator onto its own knots. The TRUE direct likelihood evaluates the spectrum of the engine
(CosmoPower at z_early grown to z_N, or ede-v2) directly at the emulator's knots, kk = knots_h, P = h^3 P(knots_h h);
growth, AP and BAO as model.MIModel.

At draws of the direct chains (the corrected-r_d ones for lcdm5/w0wa7), this prints Delta ln L against the truth for
  direct    the existing direct model (60-node composition)
  <wiggle>  the wiggle models' direct maps (wiggle_model.py)
so the representation error of each is measured against the same reference.

    SCRIPT=wiggle_truth.py ARGS="Be2 B" sbatch exec_wiggle.sbatch
Writes output/desi_fsbao/wiggle_gates/truth.npz.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import settings as S
import wiggle_settings as W
from model import COSMO_MODELS, Omega_m
from wiggle_direct import fix_rd, rdfix_files, true_loglkl
from sampling import log

NAMES = sys.argv[1:] or ['Be2']
N_DRAW, BATCH = 300, 50
OUT = os.path.join(S.OUT_ROOT, 'wiggle_gates')
t00 = time.time()
M0 = fix_rd(S.build(S.resolve('baseline'), verbose=False)[0])        # r_d corrected: lcdm3 and EDE are unaffected by it


def batched(f, *args):
    fb = jax.jit(jax.vmap(f)); out = []
    for i in range(0, len(args[0]), BATCH):
        a = [np.asarray(v[i:i + BATCH]) for v in args]; m = len(a[0])
        if m < BATCH: a = [np.concatenate([v, np.repeat(v[-1:], BATCH - m, 0)]) for v in a]
        out.append(np.asarray(fb(*[jnp.array(v) for v in a]))[:m])
    return np.concatenate(out)


def implied(th, d):
    w = np.exp(d - d.max()); w /= w.sum()
    return (w @ th - th.mean(0)) / th.std(0), 1.0 / np.sum(w**2) / len(w)


WM = {n: W.build(W.resolve(n), verbose=False)[0] for n in NAMES}
res = {}
for cv in W.COSMO_VARIANTS:
    sv = S.resolve(cv); keys = COSMO_MODELS[sv['cosmo_model']]
    f = rdfix_files(cv)[1]['chain_direct'] if cv in ('lcdm5', 'w0wa7') else S.files(sv)['chain_direct']
    if not os.path.exists(f): f = S.files(sv)['chain_direct']
    x = np.load(f)['x']; x = x.reshape(-1, x.shape[-1]); x = x[np.random.default_rng(0).choice(len(x), N_DRAW, replace=False)]
    eft, th = x[:, :M0.n_eft], x[:, M0.n_eft:]
    M0.set_cosmo_model(sv['cosmo_model']); fx = dict(M0.cosmo_fixed); f0 = M0.phi_ln_keys(keys)
    Lt = batched(lambda e, t: true_loglkl(M0, e, M0.theta_to_full(t, keys, fx)), eft, th)
    L0 = batched(lambda e, t: M0.loglkl_phi(e, f0(t)), eft, th)
    rows = {'direct': L0 - Lt}
    for n, M in WM.items():
        M.set_cosmo_model(sv['cosmo_model']); fw = M.phi_ln_keys(keys)
        rows[n] = batched(lambda e, t: M.loglkl_phi(e, fw(t)), eft, th) - Lt
    for n, d in rows.items():
        sh, ess = implied(th, d); res[f'{cv}_{n}'] = d
        log(f"[{cv}] {n:7s} vs truth: Delta ln L mean {d.mean():+.3f}, std {d.std():.3f}; ESS/N {ess:.2f}; implied shift "
            + ", ".join(f"{k} {a:+.2f}" for k, a in zip(keys, sh)) + f" sigma   ({f.split('/')[-2]})")
    res[f'{cv}_th'] = th
    np.savez(os.path.join(OUT, os.environ.get('TRUTH_OUT', 'truth.npz')), **res)
log(f"saved {os.path.join(OUT, os.environ.get('TRUTH_OUT', 'truth.npz'))} ({(time.time() - t00) / 60:.1f} min)")
