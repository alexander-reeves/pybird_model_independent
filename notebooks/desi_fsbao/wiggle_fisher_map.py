"""A data-weighted direct map for the wiggle models. wiggle_frame.py showed that the remaining representation error of Be2
is all in the least-squares map theta -> (ln a, e) (uniform in ln k), not in the sound-horizon frame. Here the map is
weighted by the data instead: W = J^T C^-1 J with J = d(multipoles + BAO alphas)/d ln P on the 80 emulator knots at the
fiducial (EFT at the true-likelihood LCDM best fit; the analytically marginalized EFT parameters profiled, as in the
likelihood). Writes output/desi_fsbao/wiggle_gates/knot_fisher.npz (W, J, eigenvalues), then repeats the frame test
(truth / oracle / map) for the maps of NAMES on the same draws -> wiggle_gates/fisher_map.npz.

    SCRIPT=wiggle_fisher_map.py ARGS="Be2 Be2F B BF" sbatch exec_wiggle.sbatch
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

NAMES = sys.argv[1:] or ['Be2', 'Be2F']
N_DRAW, BATCH = 300, 50
OUT = os.path.join(S.OUT_ROOT, 'wiggle_gates'); FOUT = os.environ.get('FISHER_OUT', 'fisher_map.npz')
t00 = time.time()

# ---- the knot Fisher -----------------------------------------------------------------------------------
MB = W.build(W.resolve('B'), verbose=False)[0]
eft = jnp.array(np.load(rdfix_files('baseline', 'exact')[1]['bestfit_direct'])['x_bf'][:MB.n_eft])

def model_vector(dlnP):
    d = [dict(di, pk_lin=di['pk_lin'] * jnp.exp(dlnP)) for di in MB.cosmo_dicts(jnp.array(MB.phi_fid))]
    L = MB.L
    L.loglkl(eft, MB.eft_names_flat, need_cosmo_update=True, cosmo_dict=d, cosmo_module=None, cosmo_engine=None)
    out = []
    for i in range(L.nsky):
        v = L.correlator_sky[i].get(L.b_sky[i]).reshape(-1)[L.m_sky[i]]
        if L.c['with_bao_rec']: v = jnp.concatenate([v, jnp.atleast_1d(jnp.asarray(L.alpha_sky[i]))])
        out.append(v)
    return jnp.concatenate(out)

if not os.path.exists(W.KNOT_FISHER) or os.environ.get('FORCE'):
    J = np.asarray(jax.jit(jax.jacfwd(model_vector))(jnp.zeros(len(MB.knots_h))))
    Wd = J.T @ np.asarray(MB.L.p_all) @ J; Wd = 0.5 * (Wd + Wd.T)
    ev = np.linalg.eigvalsh(Wd)[::-1]
    np.savez(W.KNOT_FISHER, W=Wd, J=J, eig=ev, knots_h=MB.knots_h)
    log(f"knot Fisher: J {J.shape}; eigenvalues (largest 12) {np.round(ev[:12], 1)}; number above 1: {(ev > 1).sum()}, above 1e-2 x max: "
        f"{(ev > 1e-2 * ev[0]).sum()}; sqrt(diag) per knot in 0.02-0.2 h/Mpc: "
        f"{np.round(np.sqrt(np.diag(Wd))[(MB.knots_h > 0.02) & (MB.knots_h < 0.2)][::6], 1)} ({time.time() - t00:.0f}s)")
else:
    log(f"loaded {W.KNOT_FISHER}")

# ---- truth / oracle / maps ------------------------------------------------------------------------------
M0 = fix_rd(S.build(S.resolve('baseline'), verbose=False)[0])
WM = {n: W.build(W.resolve(n), verbose=False)[0] for n in NAMES}

def oracle_loglkl(eft, c):
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
for cv in W.COSMO_VARIANTS:
    sv = S.resolve(cv); keys = COSMO_MODELS[sv['cosmo_model']]
    f = next((g for g in (rdfix_files(cv, k)[1]['chain_direct'] for k in ('exact', 'exactrw')) if os.path.exists(g)), S.files(sv)['chain_direct'])
    x = np.load(f)['x']; x = x.reshape(-1, x.shape[-1]); x = x[np.random.default_rng(0).choice(len(x), N_DRAW, replace=False)]
    e, th = x[:, :M0.n_eft], x[:, M0.n_eft:]
    for m in [M0, MB] + list(WM.values()): m.set_cosmo_model(sv['cosmo_model'])
    fx = dict(M0.cosmo_fixed)
    Lt = batched(lambda a, t: true_loglkl(M0, a, M0.theta_to_full(t, keys, fx)), e, th)
    Lo = batched(lambda a, t: oracle_loglkl(a, MB.theta_to_full(t, keys, fx)), e, th)
    rows = {'oracle-truth': Lo - Lt}
    for n, M in WM.items():
        fw = M.phi_ln_keys(keys)
        Lw = batched(lambda a, t: M.loglkl_phi(a, fw(t)), e, th)
        rows[f'{n}-truth'] = Lw - Lt
    for k, d in rows.items():
        sh, ess = implied(th, d); res[f'{cv}_{k}'] = d
        log(f"[{cv}] {k:13s}: Delta ln L std {d.std():.3f}; ESS/N {ess:.2f}; implied shift "
            + ", ".join(f"{kk} {a:+.2f}" for kk, a in zip(keys, sh)) + f" sigma   ({f.split('/')[-2]})")
    res[f'{cv}_th'] = th
    np.savez(os.path.join(OUT, FOUT), **res)
log(f"saved {os.path.join(OUT, FOUT)} ({(time.time() - t00) / 60:.1f} min)")
