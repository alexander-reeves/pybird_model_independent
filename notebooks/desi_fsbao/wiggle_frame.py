"""Where does the remaining representation error of the wiggle models come from? At draws of the true-likelihood
chains (wiggle_direct.py exact; the original chain where it is missing), compare

  truth    the engine's spectrum at the emulator knots, h frame (wiggle_direct.true_loglkl)
  oracle   model B's likelihood (sound-horizon frame: alphas = BAO alphas, kk = knots_h) with its own phi(theta) but the
           EXACT spectrum of that frame, P_B(knots_h) = h_fid^3 rho^3 P(knots_mpc rho), rho = r_d,fid / r_d
  <model>  the wiggle model's direct map (least squares of ln a [+ envelope] on the knots)

oracle - truth is what the frame costs (dimensionful EFT priors, BAO D_M factor); <model> - oracle is the spectrum map.

    SCRIPT=wiggle_frame.py ARGS="Be2 B" sbatch exec_wiggle.sbatch
Writes output/desi_fsbao/wiggle_gates/frame.npz.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import settings as S
import wiggle_settings as W
from model import COSMO_MODELS
from wiggle_direct import fix_rd, rdfix_files, true_loglkl
from sampling import log

NAMES = sys.argv[1:] or ['Be2', 'B']
N_DRAW, BATCH = 300, 50
OUT = os.path.join(S.OUT_ROOT, 'wiggle_gates')
t00 = time.time()
M0 = fix_rd(S.build(S.resolve('baseline'), verbose=False)[0])
WM = {n: W.build(W.resolve(n), verbose=False)[0] for n in dict.fromkeys(['B'] + NAMES)}
MB = WM['B']


def oracle_loglkl(eft, c):
    """Model B's likelihood with the exact sound-horizon-frame spectrum in place of a(k) T(k)."""
    phi = MB.phi_ln_full(c); d = MB.cosmo_dicts(phi)
    c2, plin = MB.engine(c)
    rho = (c['h'] / MB.h_fid) / (c2['rd_h'] / MB.rdh_fid)
    P = plin(MB.knots_mpc * rho) * rho**3 * MB.h_fid**3
    D = jnp.concatenate([jnp.exp(phi[MB.pidx['D']]), jnp.ones(1)])
    d = [dict(di, pk_lin=P * D[MB.sky_to_z[i]]**2) for i, di in enumerate(d)]
    return MB.L.loglkl(eft, MB.eft_names_flat, need_cosmo_update=True, cosmo_dict=d, cosmo_module=None, cosmo_engine=None)


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


res = {}
for cv in ('baseline', 'lcdm5', 'ede7'):
    sv = S.resolve(cv); keys = COSMO_MODELS[sv['cosmo_model']]
    f = rdfix_files(cv, 'exact')[1]['chain_direct']
    if not os.path.exists(f): f = S.files(sv)['chain_direct']
    x = np.load(f)['x']; x = x.reshape(-1, x.shape[-1]); x = x[np.random.default_rng(0).choice(len(x), N_DRAW, replace=False)]
    eft, th = x[:, :M0.n_eft], x[:, M0.n_eft:]
    for m in [M0] + list(WM.values()): m.set_cosmo_model(sv['cosmo_model'])
    fx = dict(M0.cosmo_fixed)
    Lt = batched(lambda e, t: true_loglkl(M0, e, M0.theta_to_full(t, keys, fx)), eft, th)
    Lo = batched(lambda e, t: oracle_loglkl(e, MB.theta_to_full(t, keys, fx)), eft, th)
    rows = {'oracle-truth': Lo - Lt}
    for n in NAMES:
        fw = WM[n].phi_ln_keys(keys)
        Lw = batched(lambda e, t: WM[n].loglkl_phi(e, fw(t)), eft, th)
        rows[f'{n}-oracle'] = Lw - Lo; rows[f'{n}-truth'] = Lw - Lt
    for k, d in rows.items():
        sh, ess = implied(th, d); res[f'{cv}_{k}'] = d
        log(f"[{cv}] {k:13s}: Delta ln L mean {d.mean():+.3f}, std {d.std():.3f}; ESS/N {ess:.2f}; implied shift "
            + ", ".join(f"{kk} {a:+.2f}" for kk, a in zip(keys, sh)) + f" sigma   ({f.split('/')[-2]})")
    res[f'{cv}_th'] = th
    np.savez(os.path.join(OUT, 'frame.npz'), **res)
log(f"saved {os.path.join(OUT, 'frame.npz')} ({(time.time() - t00) / 60:.1f} min)")
