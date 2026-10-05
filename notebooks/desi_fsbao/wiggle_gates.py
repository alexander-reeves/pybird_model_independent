"""Gates and design study of the wiggle/no-wiggle MI models (wiggle_model.py), before any chain.

  W0  the fiducial split T = T_nw (1 + O)
  W1  the MI likelihood at the fiducial = the exact direct likelihood at the fiducial cosmology
  W2  the BAO alphas the MI parameters predict = the direct model's, at direct-chain draws (LCDM, w0wa, EDE)
  W3  the alpha_rs gauge of model A against the node spacing: chi2 along (alpha_rs, every alpha) x lambda
      with ln a refitted, against alpha_rs alone; and the fraction of a broadband dilation the basis cannot hold
  W4  representation of the direct spectra: Delta ln L = ln L_wiggle(phi_w(theta)) - ln L_exact(theta) at
      draws of the exact direct chains -> importance weights, ESS, the shift they imply
  W5  the same at spectrum level: least-squares residual of ln a on the knots against the node spacing

Writes output/desi_fsbao/wiggle_gates/gates.npz (+ the job log).
    SCRIPT=wiggle_gates.py sbatch exec_wiggle.sbatch
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import settings as S
import wiggle_settings as W
from model import COSMO_MODELS
from wiggle_model import node_grid, ls_projector
from sampling import log

OUT = os.path.join(S.OUT_ROOT, 'wiggle_gates'); os.makedirs(OUT, exist_ok=True)
N_DRAW, BATCH = int(os.environ.get('N_DRAW', 300)), 50
KEYS7, KEYS_EDE = COSMO_MODELS['w0wa7'], COSMO_MODELS['ede7']
res = {}
t00 = time.time()

s0 = S.resolve('baseline'); M0, _ = S.build(s0, verbose=False)
eft_bf = np.load(S.files(s0)['bestfit_direct'])['x_bf'][:M0.n_eft]
MODELS = {n: W.build(W.resolve(n), verbose=True)[0] for n in ('A', 'B', 'B30', 'B120')}
MA, MB = MODELS['A'], MODELS['B']

# ---- W0 ---------------------------------------------------------------------------------------
sg = MA.split_grid; k, O = sg['k_mpc'], sg['O']
ipk = np.argmax(np.abs(O) * (k < 0.5))
log(f"W0 split: max |O| = {np.abs(O).max():.4f} at k = {k[ipk]:.4f}/Mpc; |O_raw| below the taper (k < 4e-3/Mpc) "
    f"{sg['O_at_taper']:.2e}; |O| above 0.5/Mpc {np.abs(O[k > 0.5]).max():.2e}; CPJ modes [{sg['k_cpj'][0]:.2e}, {sg['k_cpj'][-1]:.1f}]/Mpc")
dk = (k > 0.01) & (k < 0.5)
log(f"W0 EH+filter vs DST (pybird): max |O| in 0.01-0.5/Mpc {np.abs(O[dk]).max():.4f} vs {np.abs(sg['O_dst'][dk]).max():.4f}; "
    f"below 0.01/Mpc {np.abs(sg['O_raw'][k < 0.01]).max():.2e} vs {np.abs(sg['O_dst'][(k < 0.01) & (k > 1e-3)]).max():.2e}")
res.update(split_k=k, split_O=O, split_O_raw=sg['O_raw'], split_O_dst=sg['O_dst'], split_P=sg['P'])

# ---- W1 ---------------------------------------------------------------------------------------
L0 = float(jax.jit(M0.loglkl_direct)(jnp.array(M0.x_direct_fid(eft_bf))))
for n in ('A', 'B'):
    M = MODELS[n]; Lw = float(jax.jit(M.loglkl_mi)(jnp.array(M.x_mi_fid(eft_bf))))
    log(f"W1 [{n}] fiducial: ln L wiggle {Lw:.6f}, exact {L0:.6f}, diff {Lw - L0:+.2e}"); res[f'W1_{n}'] = Lw - L0


# ---- draws of the exact direct chains, as full parameter vectors of the two engine families ------
def draws(variant, n=N_DRAW, seed=0):
    s = S.resolve(variant); x = np.load(S.files(s)['chain_direct'])['x']; x = x.reshape(-1, x.shape[-1])
    x = x[np.random.default_rng(seed).choice(len(x), n, replace=False)]
    keys = COSMO_MODELS[s['cosmo_model']]; eft, th = x[:, :M0.n_eft], x[:, M0.n_eft:]
    if s['cosmo_model'] == 'ede7': return eft, th, keys, KEYS_EDE, th
    full = np.array([[dict(zip(keys, t)).get(kk, M0.cosmo_fid[kk]) for kk in KEYS7] for t in th])
    return eft, th, keys, KEYS7, full

VARIANTS = {v: draws(v) for v in W.COSMO_VARIANTS}

def phimap(M, keys):
    M.set_cosmo_model('ede7' if keys == KEYS_EDE else 'w0wa7'); f = M.phi_ln_keys(keys); M.set_cosmo_model('lcdm3')
    return f

def batched(f, *args):
    fb = jax.jit(jax.vmap(f)); n = len(args[0]); out = []
    for i in range(0, n, BATCH):
        a = [np.asarray(x[i:i + BATCH]) for x in args]; m = len(a[0])
        if m < BATCH: a = [np.concatenate([x, np.repeat(x[-1:], BATCH - m, 0)]) for x in a]
        out.append(np.asarray(fb(*[jnp.array(x) for x in a]))[:m])
    return np.concatenate(out)

# ---- W2 ---------------------------------------------------------------------------------------
for v in ('lcdm5', 'w0wa7', 'ede7'):
    eft, th, keys, kfull, full = VARIANTS[v]; full = full[:40]
    f0 = phimap(M0, kfull); a0 = batched(lambda t: jnp.exp(f0(t)[M0.pidx['bao']]), full)
    phi_ref = batched(f0, full)
    for n in ('A', 'B'):
        M = MODELS[n]; fw = phimap(M, kfull)
        aw = batched(lambda t: M.bao_alphas_mi(fw(t)), full)
        HDA = batched(lambda t: jnp.concatenate(M.background(M.unpack(fw(t)))[:2]), full)
        rel = np.abs(aw / a0 - 1).max()
        H0, DA0 = np.exp(phi_ref[:, M0.pidx['H']]), np.exp(phi_ref[:, M0.pidx['DA']])
        log(f"W2 [{n}] {v}: BAO alphas predicted vs direct, max |rel diff| {rel:.1e}; FS AP inputs vs the direct model's: "
            f"H/H0 {np.abs(HDA[:, :M.N] / H0 - 1).max():.1e}, D_A H0 {np.abs(HDA[:, M.N:] / DA0 - 1).max():.1e}"
            + ("  (B: both differ by alpha_rs by construction)" if n == 'B' else ""))
        res[f'W2_{n}_{v}'] = rel

# ---- W3 ---------------------------------------------------------------------------------------
lams = np.array([0.97, 0.985, 1.015, 1.03])
for n in ('A', 'A120', 'A180'):
    M = MODELS.get(n) or W.build(W.resolve(n), verbose=False)[0]
    llk = jax.jit(M.loglkl_mi); x0 = M.x_mi_fid(eft_bf); l0 = float(llk(jnp.array(x0)))
    def chi2(phi): return -2 * (float(llk(jnp.concatenate([jnp.array(eft_bf), jnp.asarray(phi)]))) - l0)
    def prior(phi): d = np.asarray(phi) - M.prior_mean; return float(d @ M.prior_prec @ d)
    g, rs, al, pg = [], [], [], []
    for lam in lams:
        pf = jnp.array(M.phi_fid); ll = np.log(lam)
        pgauge = M.gauge_shift(pf, lam); g.append(chi2(pgauge)); pg.append(prior(pgauge))
        rs.append(chi2(pf.at[M.pidx['rs']].add(ll)))
        al.append(chi2(pf.at[M.pidx['apar']].add(ll).at[M.pidx['aper']].add(ll)))
    log(f"W3 [{n}, {M.n_amp} nodes, spacing {M.node_spacing:.2f}] Delta chi2 at lambda = {lams}: gauge {np.round(g, 3)} "
        f"(prior along it {np.round(pg, 3)}); alpha_rs alone {np.round(rs, 1)}; alphas alone {np.round(al, 1)}")
    res[f'W3_{n}'] = np.array([g, rs, al, pg])

# fraction of a broadband dilation d ln T_nw / d ln k the node basis cannot represent, against the spacing
eps = 1e-3
v = np.asarray(jnp.log(MA.T_nw(MA.knots_mpc * (1 + eps)) / MA.T_nw(MA.knots_mpc * (1 - eps)))) / (2 * eps)
data_k = (MA.knots_h > 0.01) & (MA.knots_h < 0.3)
spac = np.array([0.15, 0.3, 0.45, 0.6, 0.9, 1.2, 1.8, 2.5]); fr = []
for sp in spac:
    g = node_grid((1e-4, 0.7), sp, MA.h_fid); Sm, Pi = ls_projector(g, MA.lnk_knots, MA.ls_w)
    r = v - Sm @ (Pi @ v); fr.append([len(g), np.sqrt(np.sum(MA.ls_w * r**2) / np.sum(MA.ls_w * (v - v.mean())**2)), np.abs(r[data_k]).max()])
fr = np.array(fr)
for sp, (nn, f1, f2) in zip(spac, fr):
    log(f"W3 dilation not representable: spacing {sp:.2f} ({int(nn)} nodes): rms fraction {f1:.1e}, max |residual| in 0.01-0.3 h/Mpc {f2:.1e} (x lambda-1 in ln P)")
res.update(W3_spacings=spac, W3_dilation=fr, W3_lams=lams)

# ---- W4 ---------------------------------------------------------------------------------------
def implied(th, d):
    w = np.exp(d - d.max()); w /= w.sum()
    mu, sd = th.mean(0), th.std(0); mw = w @ th
    return (mw - mu) / sd, np.sqrt(w @ (th - mw)**2) / sd, 1.0 / np.sum(w**2) / len(w)

for v, (eft, th, keys, kfull, full) in VARIANTS.items():
    f0 = phimap(M0, kfull)
    L_ex = batched(lambda e, t: M0.loglkl_phi(e, f0(t)), eft, full); res[f'W4_{v}_th'] = th; res[f'W4_{v}_Lexact'] = L_ex
    for n in ('A', 'B', 'B30', 'B120'):
        M = MODELS[n]; fw = phimap(M, kfull)
        d = batched(lambda e, t: M.loglkl_phi(e, fw(t)), eft, full) - L_ex
        sh, ra, ess = implied(th, d); res[f'W4_{v}_{n}'] = d
        lin = np.array([np.cov(th[:, i], d)[0, 1] / th[:, i].std() for i in range(th.shape[1])])
        log(f"W4 [{n}] {v}: Delta ln L mean {d.mean():+.3f}, std {d.std():.3f}, range [{d.min():+.2f}, {d.max():+.2f}]; ESS/N {ess:.2f}; "
            f"implied shift {dict(zip(keys, np.round(sh, 2)))} sigma (linear {np.round(lin, 2)}), width ratio {np.round(ra, 2)}")
    log(f"   ({time.time() - t00:.0f}s)")

# ---- W5 ---------------------------------------------------------------------------------------
for n in ('A', 'B'):
    M = MODELS[n]
    for v in ('lcdm5', 'ede7'):
        eft, th, keys, kfull, full = VARIANTS[v]
        M.set_cosmo_model('ede7' if kfull == KEYS_EDE else 'w0wa7'); fx = dict(M.cosmo_fixed); M.set_cosmo_model('lcdm3')
        tg = batched(lambda t: M.lna_target_knots(*M.engine(M.theta_to_full(t, kfull, fx))), full[:100])
        rows = []
        for sp in spac:
            g = node_grid((1e-4, 0.7), sp, M.h_fid); Sm, Pi = ls_projector(g, M.lnk_knots, M.ls_w)
            r = tg - tg @ (Sm @ Pi).T
            rows.append([np.sqrt((r[:, data_k]**2).mean()), np.abs(r[:, data_k]).max()])
        rows = np.array(rows); res[f'W5_{n}_{v}'] = rows
        log(f"W5 [{n}] {v}: ln P residual in 0.01-0.3 h/Mpc, rms / max over 100 draws vs spacing {spac}: "
            + ", ".join(f"{a:.1e}/{b:.1e}" for a, b in rows))

np.savez(os.path.join(OUT, 'gates.npz'), **res)
log(f"saved {os.path.join(OUT, 'gates.npz')} ({(time.time() - t00)/60:.1f} min)")
