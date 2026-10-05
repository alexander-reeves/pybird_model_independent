"""P(k) band recovery in the MODEL THE HMC ACTUALLY SAMPLES, P(k) alone vs P(k)+B(k).

WHY. robust_marg_pb.py freed all 80 emulator knots with no prior at all. That model is
degenerate: the knots outside the data range are constrained only through the loop integrals,
and marginalizing them inflates every band to O(100%). It is not a numerical failure (the SVD
marginals are converged to five digits, and v3's own Fisher gives the same wildness: median
per-knot sigma 0.72 with growth and geometry KNOWN), but it is not the model behind
output/hmc_v3_p5/mock_pk_reconstruction.png either. That figure comes from mi_model.py, which
samples 60 ln-a nodes with

    s_lna = 0.5 per node,  smooth_lambda = 200 on second differences of ln a along the nodes,
    s_lng = 0.3 on every (f, H, D_A, D-ratio),  s_lnh = 0.05 on h_conv

(setups.COMMON), the nodes being log-spaced over [1e-4, 0.7] h/Mpc at spacing 0.15 in ln k and
cubic-interpolated onto the 80 knots. This script puts exactly that model into the Fisher.

WHAT IT DOES
  1. Builds the node -> knot cubic map S (80 x 60) of mi_model.lna_to_knots.
  2. VALIDATION: takes v3's own P(k)-only Fisher (same mock, same EFT setup as the HMC run),
     maps it into the node basis, adds the four priors above, and compares the predicted
     per-node sigma(ln a) with the STANDARD DEVIATION OF THE HMC CHAIN itself
     (output/hmc_v3_p5/chain_mi.npz, x = [21 EFT, 60 ln a, 25 ln growth]). If those agree, the
     Fisher machinery is sound and the earlier wildness was the prior-free parametrization.
  3. PRODUCTION: applies the same priors to the P and P+B systems of jacobian_pb.py and writes
     the band-recovery figure in the same form as the HMC figure (P_lin and a = P/P_template
     with 68% bands), plus the per-redshift geometry with these priors.

Outputs: ../../output/fisher_pb/{hmc_prior_bands.png, hmc_prior_bands.npz}
"""
import os
import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'output', 'fisher_pb')
V3 = os.path.join(HERE, '..', '..', 'output', 'fisher_v3', 'fisher_v3_results.npz')
HMC = os.path.join(HERE, '..', '..', 'output', 'hmc_v3_p5', 'chain_mi.npz')

R = np.load(os.path.join(OUT, 'jacobians_pb.npz'), allow_pickle=True)
knots_h = R['knots_h']; nk = int(R['n_knots']); ng = int(R['n_growth'])
nz = int(R['n_z_unique']); nsky = int(R['num_skies']); zz = R['zeff_unique']
gfid = R['growth_fid']; h_fid = float(R['h_true'])
iF, iH, iD = R['idx_f'], R['idx_H'], R['idx_DA']
iDr = 3 * nz + np.arange(nz); ihc = ng - 1
gamma = 3.0 + R['dlnT_dlnk']
T_ = R['template_mpc']
Tnw = R['template_nw_mpc'] if 'template_nw_mpc' in R.files else None

# the HMC model's priors (setups.COMMON)
S_LNA = float(os.environ.get('S_LNA', '0.5'))
S_LNG = float(os.environ.get('S_LNG', '0.3'))
S_LNH = float(os.environ.get('S_LNH', '0.05'))   # prior on the ruler h_conv: see the scan below
LAM = float(os.environ.get('SMOOTH_LAMBDA', '200'))
NODE_RANGE, NODE_SPACING = (1e-4, 0.7), 0.15
Z_REF = int(np.argmin(np.abs(zz - 0.93)))


def log(m=''):
    print(m, flush=True)


# ---- node basis: mi_model.lna_to_knots -------------------------------------------------
lnk_knots = np.log(knots_h * h_fid)
lo, hi = np.log(NODE_RANGE[0] * h_fid), np.log(NODE_RANGE[1] * h_fid)
n_nodes = int(np.round((hi - lo) / NODE_SPACING)) + 1
nodes_lnk = np.linspace(lo, hi, n_nodes)
q = np.clip(lnk_knots, nodes_lnk[0], nodes_lnk[-1])
S = np.zeros((nk, n_nodes))
for i in range(n_nodes):
    e = np.zeros(n_nodes); e[i] = 1.0
    S[:, i] = CubicSpline(nodes_lnk, e, bc_type='not-a-knot')(q)
log(f"node basis: {n_nodes} nodes over [{NODE_RANGE[0]}, {NODE_RANGE[1]}] h/Mpc, spacing "
    f"{NODE_SPACING} in ln k -> {nk} knots; row sums of S (should be 1): "
    f"{S.sum(1).min():.4f}..{S.sum(1).max():.4f}")

D2 = np.zeros((n_nodes - 2, n_nodes))
for i in range(n_nodes - 2):
    D2[i, i:i + 3] = (1.0, -2.0, 1.0)


def mi_prior_precision(n_amp):
    """Prior precision on [ln a nodes, ln growth] exactly as mi_model.prior_prec_mat."""
    n = n_amp + ng
    Pi = np.zeros((n, n))
    Pi[:n_amp, :n_amp] = np.eye(n_amp) / S_LNA**2 + LAM * (D2.T @ D2)
    idx_g = n_amp + np.arange(ng)
    Pi[idx_g[:4 * nz], idx_g[:4 * nz]] = 1.0 / S_LNG**2      # f, H, D_A, D-ratios
    Pi[idx_g[ihc], idx_g[ihc]] = 1.0 / S_LNH**2              # h_conv
    return Pi


# ---- 1. validation against the HMC chain -----------------------------------------------
log("\n" + "=" * 96)
log("VALIDATION: v3's own P(k) Fisher in the HMC's node basis + the HMC priors, versus the "
    "standard deviation of the HMC chain itself.")
v3 = np.load(V3)
F_v3, pf_v3, gfid_v3 = v3['F_full'], v3['params_fid'], v3['growth_fid']
n_eft_v3 = len(pf_v3) - nk - ng
sc = np.concatenate([np.abs(pf_v3[:n_eft_v3]), np.ones(nk), gfid_v3])   # -> d ln theta
F_ln = (0.5 * (F_v3 + F_v3.T)) * np.outer(sc, sc)
Tm = np.zeros((len(pf_v3), n_eft_v3 + n_nodes + ng))
Tm[:n_eft_v3, :n_eft_v3] = np.eye(n_eft_v3)
Tm[n_eft_v3:n_eft_v3 + nk, n_eft_v3:n_eft_v3 + n_nodes] = S
Tm[n_eft_v3 + nk:, n_eft_v3 + n_nodes:] = np.eye(ng)
F_nodes = Tm.T @ F_ln @ Tm
Pi = np.zeros_like(F_nodes)
Pi[n_eft_v3:, n_eft_v3:] = mi_prior_precision(n_nodes)
F_tot = 0.5 * (F_nodes + F_nodes.T) + Pi
w_ = np.linalg.eigvalsh(F_tot)
C_v3 = np.linalg.inv(F_tot)
sig_node_fisher = np.sqrt(np.diag(C_v3)[n_eft_v3:n_eft_v3 + n_nodes])
C_nodes = C_v3[n_eft_v3:n_eft_v3 + n_nodes, n_eft_v3:n_eft_v3 + n_nodes]
sig_knot_fisher = np.sqrt(np.clip(np.diag(S @ C_nodes @ S.T), 0, None))
log(f"   v3 Fisher + MI priors: min eigenvalue {w_.min():.3e} (must be > 0)")

nodes_h = np.exp(nodes_lnk) / h_fid
in_data = (nodes_h >= 0.01) & (nodes_h <= 0.20)
if os.path.exists(HMC):
    ch = np.load(HMC)['x']
    n_eft_ch = ch.shape[-1] - n_nodes - ng
    lna = ch[..., n_eft_ch:n_eft_ch + n_nodes].reshape(-1, n_nodes)
    sig_node_chain = lna.std(axis=0)
    log(f"   HMC chain: {ch.shape[0]} chains x {ch.shape[1]} draws, {ch.shape[-1]} params "
        f"= {n_eft_ch} EFT + {n_nodes} nodes + {ng} growth")
    r = sig_node_fisher / sig_node_chain
    log(f"   sigma(ln a) per node, Fisher / chain:  median {np.median(r):.3f}, "
        f"in the data range {np.median(r[in_data]):.3f}  (range {r.min():.2f}-{r.max():.2f})")
    log(f"   median sigma(ln a) in the data range:  Fisher {np.median(sig_node_fisher[in_data]):.4f}"
        f"   chain {np.median(sig_node_chain[in_data]):.4f}")
else:
    sig_node_chain = None
    log(f"   chain not found at {HMC}")

# ---- 2. the same priors applied to P and P+B -------------------------------------------
log("\n" + "=" * 96)
log("P(k) vs P(k)+B(k) in the HMC's model (60 nodes, smoothness + amplitude + growth priors).")
DATASETS = [('P', 'p'), ('P+B', 'pb')]
res = {}
for tag, key in DATASETS:
    J, P, prior, x0 = R[f'J_{key}'], R[f'P_{key}'], R[f'prior_{key}'], R[f'x0_{key}']
    n_e = int(R[f'n_e_{key}']); o_a = n_e * nsky; o_g = o_a + nk
    Lc = np.linalg.cholesky(0.5 * (P + P.T))
    s = np.ones(len(x0)); e = np.arange(o_a); has = prior[e] > 0
    s[e] = np.where(has, 1 / np.sqrt(np.where(has, prior[e], 1.0)), np.maximum(np.abs(x0[e]), 1.0))
    s[o_a:o_g] = x0[o_a:o_g]; s[o_g:] = gfid
    Wd = (Lc.T @ J) * s
    Wn = np.hstack([Wd[:, :o_a], Wd[:, o_a:o_g] @ S, Wd[:, o_g:]])      # knots -> nodes
    pr = np.where(has)[0]
    Wp = np.zeros((len(pr), Wn.shape[1])); Wp[np.arange(len(pr)), pr] = 1.0
    F = Wn.T @ Wn + Wp.T @ Wp
    Pi = np.zeros_like(F); Pi[o_a:, o_a:] = mi_prior_precision(n_nodes)
    F = 0.5 * (F + F.T) + Pi
    C = np.linalg.inv(F)
    Cn = C[o_a:o_a + n_nodes, o_a:o_a + n_nodes]
    res[key] = dict(C=C, o_a=o_a, sig_node=np.sqrt(np.diag(Cn)),
                    sig_knot=np.sqrt(np.clip(np.diag(S @ Cn @ S.T), 0, None)),
                    min_eig=np.linalg.eigvalsh(F).min())
    log(f"   {tag:4s}: min eigenvalue {res[key]['min_eig']:.3e}; median sigma(ln a) per node in "
        f"the data range {np.median(res[key]['sig_node'][in_data]):.4f}; per knot "
        f"{np.median(res[key]['sig_knot'][(knots_h >= 0.01) & (knots_h <= 0.20)]):.4f}")

# geometry with these priors (no symmetry projection: the priors make every direction proper)
log("\n   Per-redshift geometry and growth with the HMC priors (fractional 1 sigma):")
QT = {'f': lambda o, iz: (o + iF[iz], 1.0), 'lnDA': lambda o, iz: (o + iD[iz], 1.0),
      'lnH': lambda o, iz: (o + iH[iz], 1.0)}
geo = {}
for tag, key in DATASETS:
    C, o_a = res[key]['C'], res[key]['o_a']
    o_g = o_a + n_nodes
    def sig(cvec):
        return float(np.sqrt(cvec @ C @ cvec))
    rows = {}
    for nm in ('f', 'F_AP', 'lnDA', 'lnH'):
        vals = []
        for iz in range(nz):
            c = np.zeros(C.shape[0])
            if nm == 'f':
                c[o_g + iF[iz]] = 1.0
            elif nm == 'F_AP':
                c[o_g + iH[iz]] = 1.0; c[o_g + iD[iz]] = 1.0
            elif nm == 'lnDA':
                c[o_g + iD[iz]] = 1.0
            else:
                c[o_g + iH[iz]] = 1.0
            vals.append(sig(c))
        rows[nm] = np.array(vals)
    geo[key] = rows
log(f"      {'':10s}" + "".join(f"{'z='+format(z,'.3f'):>13s}" for z in zz))
for nm in ('f', 'F_AP', 'lnDA', 'lnH'):
    for tag, key in DATASETS:
        log(f"      {nm:6s}{tag:4s}" + "".join(f"{v:13.4f}" for v in geo[key][nm]))

np.savez(os.path.join(OUT, os.environ.get('NPZNAME', 'hmc_prior_bands.npz')), knots_h=knots_h, nodes_h=nodes_h,
         sig_node_fisher_v3=sig_node_fisher, sig_knot_fisher_v3=sig_knot_fisher,
         **({'sig_node_chain': sig_node_chain} if sig_node_chain is not None else {}),
         **{f'sig_node_{k}': res[k]['sig_node'] for _, k in DATASETS},
         **{f'sig_knot_{k}': res[k]['sig_knot'] for _, k in DATASETS},
         **{f'geo_{k}_{nm}': geo[k][nm] for _, k in DATASETS for nm in geo[k]})

# ---- figure ------------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from make_fisher_pb_fig import C_P, C_PB, C_MUTED, C_INK
except Exception:                                                    # noqa: BLE001
    C_P, C_PB, C_MUTED, C_INK = '#E8710A', '#1A73E8', '#80868B', '#202124'
COL = {'p': C_P, 'pb': C_PB}; LAB = {'p': 'P(k)', 'pb': 'P(k)+B(k)'}

fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True, layout='constrained',
                         gridspec_kw={'height_ratios': [2, 1.4]})
kk = knots_h * h_fid                                     # 1/Mpc, as in the HMC figure
Pk = T_                                                  # P_lin(k, z_ref=5) in Mpc^3
for ax, ratio in ((axes[0], False), (axes[1], True)):
    ax.axvspan(0.01 * h_fid, 0.20 * h_fid, color=C_MUTED, alpha=0.12, lw=0)
    y = np.ones(nk) if ratio else Pk
    sp, spb = res['p']['sig_knot'], res['pb']['sig_knot']
    ax.fill_between(kk, y * np.exp(-spb), y * np.exp(spb), color=C_PB, alpha=0.35, lw=0,
                    label=r'P(k)+B(k)  $\pm1\sigma$')
    for sgn in (-1, 1):
        ax.plot(kk, y * np.exp(sgn * sp), color=C_P, lw=1.7,
                label=r'P(k)  $\pm1\sigma$' if sgn > 0 else None)
    ax.plot(kk, y, color=C_INK, lw=1.6, ls='--',
            label='truth' if not ratio else None, zorder=5)
    ax.set_xscale('log'); ax.grid(alpha=0.2, which='both')
if sig_node_chain is not None:
    sig_knot_chain = np.sqrt(np.clip(np.diag(S @ np.diag(sig_node_chain**2) @ S.T), 0, None))
    axes[1].plot(kk, np.exp(sig_knot_fisher_v3 if False else sig_knot_fisher), color=C_MUTED,
                 lw=1.4, ls=':', label=r'v3 P(k) Fisher $+1\sigma$ (validation)')
    axes[1].plot(kk, np.exp(-sig_knot_fisher), color=C_MUTED, lw=1.4, ls=':')
axes[0].set_yscale('log')
axes[0].set_ylabel(r'$P_{\rm lin}(k^*,\,z_{\rm ref}=5)\ [{\rm Mpc}^3]$', fontsize=13)
axes[0].set_title('Linear power recovered in the model the HMC samples '
                  '(60 nodes, smoothness + amplitude + growth priors)', fontsize=13, color=C_INK)
axes[0].legend(frameon=False, fontsize=10, loc='lower left')
axes[1].axhline(1.0, color=C_INK, lw=1.2, ls='--')
axes[1].axhline(np.exp(S_LNA), color='#C5221F', lw=1.1, ls=':', label=r'prior $\pm1\sigma$')
axes[1].axhline(np.exp(-S_LNA), color='#C5221F', lw=1.1, ls=':')
axes[1].set_ylim(0.45, 1.75)
axes[1].set_ylabel(r'$a_i = P/P_{\rm template}$', fontsize=13)
axes[1].set_xlabel(r'$k^*\ [1/{\rm Mpc}]$', fontsize=13)
axes[1].legend(frameon=False, fontsize=9, loc='upper left', ncol=2)
p = os.path.join(OUT, os.environ.get('FIGNAME', 'hmc_prior_bands.png'))
fig.savefig(p, dpi=170); plt.close(fig)
log(f"\nsaved {p}")
