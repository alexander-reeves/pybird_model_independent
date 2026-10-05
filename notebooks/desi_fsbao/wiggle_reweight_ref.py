"""True-likelihood reference posteriors by importance reweighting, when the sampled ones (wiggle_direct.py exact) cannot
finish (the preemptable partition preempts after ~1 h; w0wa7 needs ~100 min). The existing direct chain of a variant
was sampled from L_orig(theta) pi(theta), L_orig = model.MIModel's direct likelihood with pybird's r_d; it is
reweighted by L_true / L_orig (wiggle_direct.true_loglkl, corrected r_d) on N draws and resampled to equal weights.

Writes output/desi_fsbao/exactrw_<variant>/chain_direct_<model>.npz (x of shape (1, n, dim), the resampled draws, plus
the weights) and a log line with the ESS and the shift from the original chain. Variants with a sampled exact chain are
done too, as a check of the method.

    SCRIPT=wiggle_reweight_ref.py ARGS="ede7 w0wa7 baseline lcdm5" sbatch exec_wiggle.sbatch
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import settings as S
from model import COSMO_MODELS
from wiggle_direct import fix_rd, true_loglkl, rdfix_files
from sampling import log

N, BATCH = int(os.environ.get('N_RW', 8000)), 50
t00 = time.time()


def batched(f, *args):
    fb = jax.jit(jax.vmap(f)); out = []
    for i in range(0, len(args[0]), BATCH):
        a = [np.asarray(v[i:i + BATCH]) for v in args]; m = len(a[0])
        if m < BATCH: a = [np.concatenate([v, np.repeat(v[-1:], BATCH - m, 0)]) for v in a]
        out.append(np.asarray(fb(*[jnp.array(v) for v in a]))[:m])
    return np.concatenate(out)


for cv in sys.argv[1:] or ['ede7', 'w0wa7']:
    sv = S.resolve(cv); keys = COSMO_MODELS[sv['cosmo_model']]
    Morig, _ = S.build(sv, verbose=False)                       # the likelihood the chain was sampled with
    Mtrue = fix_rd(S.build(sv, verbose=False)[0])
    x = np.load(S.files(sv)['chain_direct'])['x']; x = x.reshape(-1, x.shape[-1])
    xs = x[np.random.default_rng(7).choice(len(x), min(N, len(x)), replace=False)]
    fx = dict(Mtrue.cosmo_fixed)
    Lt = batched(lambda e, t: true_loglkl(Mtrue, e, Mtrue.theta_to_full(t, keys, fx)), xs[:, :Morig.n_eft], xs[:, Morig.n_eft:])
    Lo = batched(lambda z: Morig.loglkl_direct(z), xs)
    lw = Lt - Lo; w = np.exp(lw - lw.max()); w /= w.sum(); ess = 1.0 / np.sum(w**2)
    idx = np.random.default_rng(8).choice(len(xs), len(xs), p=w)
    th, thw = xs[:, Morig.n_eft:], xs[idx, Morig.n_eft:]
    sh, ra = (thw.mean(0) - th.mean(0)) / th.std(0), thw.std(0) / th.std(0)
    out = os.path.join(S.OUT_ROOT, f'exactrw_{cv}'); os.makedirs(out, exist_ok=True)
    f = os.path.join(out, os.path.basename(S.files(sv)['chain_direct']))
    np.savez(f, x=xs[idx][None], weights=w, draws=xs, lnw=lw)
    log(f"[{cv}] {len(xs)} draws: ln(L_true/L_orig) std {lw.std():.3f}; ESS {ess:.0f} ({ess / len(xs):.2f}); true vs original direct: "
        + ", ".join(f"{k} {a:+.2f}σ ×{b:.2f}" for k, a, b in zip(keys, sh, ra)) + f"  -> {f} ({time.time() - t00:.0f}s)")
    fe = rdfix_files(cv, 'exact')[1]['chain_direct']
    if os.path.exists(fe):
        xe = np.load(fe)['x']; the = xe.reshape(-1, xe.shape[-1])[:, Morig.n_eft:]
        log(f"[{cv}] check against the sampled true-likelihood chain: reweighted - sampled = "
            + ", ".join(f"{k} {a:+.2f}σ ×{b:.2f}" for k, a, b in zip(keys, (thw.mean(0) - the.mean(0)) / the.std(0), thw.std(0) / the.std(0))))
