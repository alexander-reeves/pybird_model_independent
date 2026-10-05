"""Representation of the direct spectra with a free wiggle envelope (wiggle_model.py n_env): wiggle_gates.py W4/W5
for a list of wiggle models, on the same direct-chain draws (seed 0, 300 per model).

  W1  the MI likelihood at the fiducial vs the exact direct one
  W4  Delta ln L = ln L_wiggle(phi_w(theta)) - ln L_exact(theta) at the draws: spread, ESS, implied shift
  W5  ln P_MI - ln P_direct after the fit, in 0.01 < k < 0.3 h/Mpc (rms, max over 100 draws)
  W6  the envelope coefficients the direct spectra need (mean, std over the draws)

    SCRIPT=wiggle_envelope.py ARGS="B Be1 Be2 Be3 Be290 Be2120" sbatch exec_wiggle.sbatch
Writes output/desi_fsbao/wiggle_gates/envelope.npz. With RDFIX=1 the exact likelihood uses the corrected sound horizon
(wiggle_direct.fix_rd; the wiggle models always do since rd_fix exists) -> envelope_rdfix.npz.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import settings as S
import wiggle_settings as W
from model import COSMO_MODELS
from sampling import log

OUT = os.path.join(S.OUT_ROOT, 'wiggle_gates'); os.makedirs(OUT, exist_ok=True)
NAMES = sys.argv[1:] or ['B', 'Be1', 'Be2', 'Be3']
N_DRAW, BATCH = 300, 50
KEYS7, KEYS_EDE = COSMO_MODELS['w0wa7'], COSMO_MODELS['ede7']
t00 = time.time()
s0 = S.resolve('baseline'); M0, _ = S.build(s0, verbose=False)
RDFIX = bool(int(os.environ.get('RDFIX', '0')))
if RDFIX:
    from wiggle_direct import fix_rd
    M0 = fix_rd(M0); log("exact likelihood with the corrected sound horizon (RDFIX=1)")
FOUT = os.path.join(OUT, 'envelope_rdfix.npz' if RDFIX else 'envelope.npz')
eft_bf = np.load(S.files(s0)['bestfit_direct'])['x_bf'][:M0.n_eft]


def draws(variant, n=N_DRAW, seed=0):            # as wiggle_gates.py
    s = S.resolve(variant); x = np.load(S.files(s)['chain_direct'])['x']; x = x.reshape(-1, x.shape[-1])
    x = x[np.random.default_rng(seed).choice(len(x), n, replace=False)]
    keys = COSMO_MODELS[s['cosmo_model']]; eft, th = x[:, :M0.n_eft], x[:, M0.n_eft:]
    if s['cosmo_model'] == 'ede7': return eft, th, keys, KEYS_EDE, th
    full = np.array([[dict(zip(keys, t)).get(kk, M0.cosmo_fid[kk]) for kk in KEYS7] for t in th])
    return eft, th, keys, KEYS7, full

def snapshot(M, keys):
    M.set_cosmo_model('ede7' if keys == KEYS_EDE else 'w0wa7'); f, fx = M.phi_ln_keys(keys), dict(M.cosmo_fixed); M.set_cosmo_model('lcdm3')
    return f, fx

def batched(f, *args):
    fb = jax.jit(jax.vmap(f)); n = len(args[0]); out = []
    for i in range(0, n, BATCH):
        a = [np.asarray(x[i:i + BATCH]) for x in args]; m = len(a[0])
        if m < BATCH: a = [np.concatenate([x, np.repeat(x[-1:], BATCH - m, 0)]) for x in a]
        out.append(np.asarray(fb(*[jnp.array(x) for x in a]))[:m])
    return np.concatenate(out)

def implied(th, d):
    w = np.exp(d - d.max()); w /= w.sum(); mu, sd = th.mean(0), th.std(0); mw = w @ th
    return (mw - mu) / sd, np.sqrt(w @ (th - mw)**2) / sd, 1.0 / np.sum(w**2) / len(w)

VAR = {v: draws(v) for v in W.COSMO_VARIANTS}
L0 = {}
for v, (eft, th, keys, kfull, full) in VAR.items():
    f0, _ = snapshot(M0, kfull); L0[v] = batched(lambda e, t: M0.loglkl_phi(e, f0(t)), eft, full)
Lfid = float(jax.jit(M0.loglkl_direct)(jnp.array(M0.x_direct_fid(eft_bf))))
log(f"exact likelihoods at {N_DRAW} draws of {list(VAR)} ({time.time() - t00:.0f}s)")
res = {}
for name in NAMES:
    M = W.build(W.resolve(name), verbose=True)[0]
    log(f"W1 [{name}] fiducial: diff {float(jax.jit(M.loglkl_mi)(jnp.array(M.x_mi_fid(eft_bf)))) - Lfid:+.2e}")
    data_k = (M.knots_h > 0.01) & (M.knots_h < 0.3)
    for v, (eft, th, keys, kfull, full) in VAR.items():
        fw, fx = snapshot(M, kfull)
        d = batched(lambda e, t: M.loglkl_phi(e, fw(t)), eft, full) - L0[v]; res[f'W4_{v}_{name}'] = d
        sh, ra, ess = implied(th, d)
        log(f"W4 [{name}] {v}: Delta ln L mean {d.mean():+.3f}, std {d.std():.3f}; ESS/N {ess:.2f}; implied shift "
            + ", ".join(f"{k} {a:+.2f}" for k, a in zip(keys, sh)) + f" sigma; width ratio {np.round(ra, 2)}")
        if v != 'baseline':
            r = batched(lambda t: M.spectrum_residual(M.theta_to_full(t, kfull, fx)), full[:100])[:, data_k]
            res[f'W5_{name}_{v}'] = np.array([np.sqrt((r**2).mean()), np.abs(r).max()])
            msg = f"W5 [{name}] {v}: ln P residual in 0.01-0.3 h/Mpc rms {np.sqrt((r**2).mean()):.1e}, max {np.abs(r).max():.1e}"
            if M.n_env:
                e = batched(lambda t: fw(t)[M.pidx['env']], full); res[f'W6_{name}_{v}'] = e
                msg += f"; envelope e = {np.round(e.mean(0), 3)} +- {np.round(e.std(0), 3)} over the posterior"
            log(msg)
    log(f"   ({time.time() - t00:.0f}s)")
    np.savez(FOUT, **res)
log(f"saved {FOUT} ({(time.time() - t00)/60:.1f} min)")
