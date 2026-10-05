# %% [markdown]
# # Two α per redshift: a wiggle/no-wiggle model-independent parametrization
#
# DESI DR1 full shape + post-reconstruction BAO, the same data, EFT model and priors as the lean pipeline
# (`model.py`, `analysis.ipynb`). Only the parametrization of the cosmology-dependent block changes.
#
# **The proposal.** Split the fiducial linear spectrum at $z_N=1.491$ into a smooth part and wiggles,
# $T(k)=T_{\rm nw}(k)\,[1+O(k)]$. Model the broadband with a spline on a few nodes and keep the wiggles as
# the fiducial template, rescaled by the sound horizon:
#
# $$P_{\rm lin}(k,z_N)=a(k)\,T_{\rm nw}(k)\,\big[1+O(k\,\alpha_{r_s})\big],\qquad \alpha_{r_s}=\frac{h\,r_d}{(h\,r_d)_{\rm fid}} .$$
#
# Each redshift gets two AP dilations $\alpha_\parallel(z),\alpha_\perp(z)$ that apply to the whole full shape,
# and the post-reconstruction BAO is *predicted*: $\alpha^{\rm BAO}=\alpha/\alpha_{r_s}$. That replaces
# the baseline's $H, D_A$ per redshift + $h_{\rm conv}$ + ten free BAO dilations, and its 60 nodes by 16.
# This is **model A** (`wiggle_model.py`, `rs_free=True`).
#
# **A near-symmetry.** With the broadband free, scaling $\alpha_{r_s}$ and every $\alpha$ by the same
# $\lambda$ and dilating $a(k)$ leaves the observed multipoles unchanged. The wiggles see
# $\alpha/\alpha_{r_s}$, which is unchanged. The broadband dilates, and the spline absorbs that. The AP volume
# factor $1/(q_\parallel q_\perp^2)$ cancels the $\lambda^3$. So $\alpha_{r_s}$ is fixed by its prior unless
# the node basis is too coarse to follow a dilation of $T_{\rm nw}$. **Model B** removes the direction by
# setting $\alpha_{r_s}=1$. Its two α per redshift *are* the BAO α's, its template is the spectrum in
# sound-horizon units, $P(k\,r_d)/r_d^3$, and it needs no split.
#
# **The test** is the one of meeting item B. Each MI chain is projected onto a cosmological model,
# $p(\theta|d)\propto p_{\rm MI}(\phi(\theta)|d)/\pi_{\rm MI}(\phi(\theta))\,\pi(\theta)$, and compared with
# the *exact* direct chain of that model: CosmoPower/ede-v2 spectra, not the wiggle model. The models are
# ΛCDM (lcdm3), ΛCDM with ω_b (BBN) and n_s free (lcdm5), w0waCDM (w0wa7) and EDE (ede7). A miss splits into
# *representation*, where the direct chain is reweighted by $L_{\rm wiggle}/L_{\rm exact}$, and *compression*.
#
# Scripts: `wiggle_model.py`, `wiggle_settings.py`, `wiggle_gates.py`, `wiggle_sample.py`, `wiggle_recovery.py`
# (launcher `exec_wiggle.sbatch`). This notebook is built from `wiggle_analysis.py` by `build_wiggle_nb.py`
# and executed by `exec_wiggle_nb.sbatch`. It only reads their outputs, plus light evaluations of the model.

# %%
import os, sys, re, glob
import numpy as np
import matplotlib, matplotlib.ticker
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp
sys.path.insert(0, os.path.abspath('.'))
import settings as S
import wiggle_settings as W
from model import COSMO_MODELS, COSMO_TEX, COSMO_BOX, EDE_BOX
from sampling import log, chain_diagnostics
from getdist import MCSamples, plots

RUNS = [w for w in ('A', 'B', 'Be2', 'Be245', 'Ae245', 'Ag245') if os.path.exists(os.path.join(S.OUT_ROOT, f'wiggle_{w}', 'chain_mi.npz'))]
REC_RUNS = [w for w in RUNS if w not in ('Ae245', 'Ag245')] + [w for w in ('Be2F', 'Be245F', 'Ae245F', 'Ag245F') if os.path.exists(os.path.join(S.OUT_ROOT, f'wiggle_{w}', 'recovery_gauss.npz'))]
OUT = {w: os.path.join(S.OUT_ROOT, f'wiggle_{w}') for w in dict.fromkeys(RUNS + REC_RUNS)}       # Be2F: Be2's chain, data-weighted direct map
GATES = os.path.join(S.OUT_ROOT, 'wiggle_gates', 'gates.npz')
FIG = os.path.join(S.OUT_ROOT, 'wiggle_figs'); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'figure.dpi': 110})
C_D, C_OLD = '#D84315', '#757575'                                       # direct, baseline 106-parameter MI
COL = {'A': '#E8710A', 'B': '#1E88E5', 'Be2': '#2E7D32', 'Be2F': '#9CCC65', 'Be245': '#6A1B9A', 'Be245F': '#BA68C8', 'Ae245': '#00838F', 'Ae245F': '#00ACC1', 'Ag245': '#AD1457', 'Ag245F': '#F06292'}; C_A, C_B = COL['A'], COL['B']
def savefig(name):
    plt.savefig(os.path.join(FIG, f'{name}.pdf'), bbox_inches='tight'); plt.savefig(os.path.join(FIG, f'{name}.png'), dpi=140, bbox_inches='tight')
    plt.show()
G = dict(np.load(GATES)); ENV = dict(np.load(os.path.join(S.OUT_ROOT, 'wiggle_gates', 'envelope.npz')))
MW = {w: W.build(W.resolve(w), verbose=False)[0] for w in dict.fromkeys(['A', 'B'] + RUNS)}
MA, MB = MW['A'], MW['B']
s0 = S.resolve('baseline'); M0, _ = S.build(s0, verbose=False)
log("parameters: " + ", ".join(f"model {w} {M.n_mi}" for w, M in MW.items()) + f", baseline {M0.n_mi}; MI chains found for {RUNS}")

# %% [markdown]
# ## 1. The split
#
# $T_{\rm nw}$ is the Eisenstein–Hu no-wiggle shape times the ratio $T/T_{\rm EH}$ smoothed by a Gaussian of
# 0.25 dex in $k$ (Vlah et al. 2016). Below $3\times10^{-3}\,{\rm Mpc}^{-1}$, $O$ is tapered to exactly zero, and
# $T_{\rm nw}\equiv T/(1+O)$, so model A at $\alpha_{r_s}=1$ is exactly the template. The DST split that pybird
# uses (Hamann et al. 2010) leaves a ~10% non-oscillatory residual near the turnover. Model A would rescale
# that residual as if it were BAO, so it is shown only for comparison. Model B does not use the split.

# %%
k = G['split_k']; m = (k > 2e-3) & (k < 0.6)
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].semilogx(k[m], G['split_O'][m], color='k', label='EH + Gaussian filter (used)')
ax[0].semilogx(k[m], G['split_O_dst'][m], color='0.6', ls='--', label='DST (pybird), for comparison')
for lam, c in ((0.97, C_D), (1.03, C_B)):
    ax[0].semilogx(k[m], np.asarray(MA.O_of_k(jnp.array(k[m] * lam))), color=c, lw=0.8, label=rf'$O(k\,\alpha_{{r_s}})$, $\alpha_{{r_s}}={lam}$')
ax[0].axvspan(0.02 * MA.h_fid, 0.2 * MA.h_fid, color='0.92', zorder=0)
ax[0].set_xlabel(r'$k\ [{\rm Mpc}^{-1}]$'); ax[0].set_ylabel(r'$O(k)=T/T_{\rm nw}-1$'); ax[0].legend(fontsize=8)
ax[1].loglog(k[m], G['split_P'][m], color='k', label=r'$T$'); ax[1].loglog(k[m], G['split_P'][m] / (1 + G['split_O'][m]), color=C_A, label=r'$T_{\rm nw}$')
for x in MA.nodes_mpc: ax[1].axvline(float(x), color='0.85', lw=0.6, zorder=0)
ax[1].set_xlabel(r'$k\ [{\rm Mpc}^{-1}]$'); ax[1].set_ylabel(r'$P_{\rm lin}(k,z_N)\ [{\rm Mpc}^3]$'); ax[1].legend()
ax[1].set_title(f'{MA.n_amp} broadband nodes (grey), spacing {MA.node_spacing:.2f} in ln k', fontsize=10)
savefig('split'); plt.close('all')
log(f"max |O| = {np.abs(G['split_O']).max():.3f}; data range shaded (0.02-0.2 h/Mpc at h_fid)")

# %% [markdown]
# ## 2. Gates, and whether $\alpha_{r_s}$ is a gauge
#
# The log of `wiggle_gates.py` is printed below. W1: the MI likelihood at the fiducial against the exact direct one.
# W2: the BAO α's that the MI parameters predict against the direct model's α's at direct-chain draws.
# W3 moves model A along the near-symmetry and, for contrast, moves $\alpha_{r_s}$ alone and the α's alone. It
# also gives the fraction of a broadband dilation $d\ln T_{\rm nw}/d\ln k$ that each node basis cannot hold.

# %%
glog = sorted(glob.glob(os.path.join(S.OUT_ROOT, 'logs', 'wiggle_*.log')), key=os.path.getmtime)
glog = [f for f in glog if 'W3 dilation' in open(f).read()][-1]
print(f"(from {glog})")
print(''.join(l for l in open(glog) if re.search(r'\] W[0-5]', l)))

# %%
lams = G['W3_lams']
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for n, ls in (('A', '-'), ('A120', '--'), ('A180', ':')):
    g, rs, al, pg = G[f'W3_{n}']
    lab = {'A': '16 nodes (0.6)', 'A120': '8 nodes (1.2)', 'A180': '6 nodes (1.8)'}[n]
    ax[0].plot(lams, g, 'o' + ls, color=C_A, label=f'gauge move, {lab}')
    if n == 'A':
        ax[0].plot(lams, rs, 's-', color=C_D, ms=4, label=r'$\alpha_{r_s}$ alone')
        ax[0].plot(lams, al, 'v-', color='#6A1B9A', ms=4, label=r'every $\alpha$ alone')
        ax[0].plot(lams, pg, '^-', color='0.5', ms=4, label='prior along the gauge move')
ax[0].set_yscale('symlog', linthresh=0.3); ax[0].axhline(0, color='0.8', lw=0.8)
ax[0].set_xlabel(r'$\lambda$ (from the fiducial)'); ax[0].set_ylabel(r'$\Delta\chi^2$ (data)'); ax[0].legend(fontsize=8)
sp, fr = G['W3_spacings'], G['W3_dilation']
ax[1].loglog(fr[:, 0], fr[:, 1], 'o-', color='k', label='rms over all knots')
ax[1].loglog(fr[:, 0], fr[:, 2], 's--', color=C_A, label=r'max over $0.01<k<0.3\,h/$Mpc')
ax[1].set_xlabel('number of broadband nodes'); ax[1].set_ylabel('fraction of a dilation the basis cannot hold')
ax[1].legend(); savefig('gauge'); plt.close('all')
for n in ('A', 'A120', 'A180'):
    g, rs, al, pg = G[f'W3_{n}']
    log(f"{n}: at lambda = 1.03 the gauge move costs chi2 {g[-1]:.3f} (prior {pg[-1]:.2f}); alpha_rs alone {rs[-1]:.1f}; alphas alone {al[-1]:.1f}")

# %% [markdown]
# ## 3. How few nodes, and what the wiggles need: representing real spectra
#
# For each direct model, 300 draws of its exact chain are mapped into the wiggle model: weighted least squares of
# $\ln a$ (and, with an envelope, of the $e_j$) on the 80 emulator knots, then the α's, $f$ and $D$ ratios.
# W5 is the leftover in $\ln P$ inside $0.01<k<0.3\,h/$Mpc. W4 is what the leftover does to the likelihood: the
# spread of $\Delta\ln L=\ln L_{\rm wiggle}-\ln L_{\rm exact}$ over the posterior, and the posterior shift implied by
# importance weights. That shift is reliable only when ESS/N is not small.
#
# **Without an envelope** (models A, B) the leftover is about 0.4% rms in ln P. It hardly changes between 31 and 6
# nodes, and drops only at 60 nodes, where the spline can move the wiggles. So it is the BAO wiggles themselves:
# their amplitude and damping follow $\omega_b/\omega_m$ and the Silk scale, which a template rescaled only by
# $\alpha_{r_s}$ cannot do. **With an envelope**, $[1+(1+e_0+e_1x+\dots)\,O(k\alpha_{r_s})]$ with $x=\ln(k/0.1\,{\rm Mpc}^{-1})$,
# the model carries the standard BAO-fit amplitude/damping freedom: one to three global parameters.
#
# The first table compares with the existing direct likelihood, whose $r_d$ has the wrong $\omega_b$ dependence (section 6).
# That is why lcdm5 and w0wa7 stay off by ~1.5σ in $\omega_b$ even with the envelope. The second table compares with the
# corrected $r_d$. The draws of all tables are the same 300 per model, from the original chains.

# %%
sp = G['W3_spacings']
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for key, c in (('W5_B_lcdm5', C_B), ('W5_A_lcdm5', C_A), ('W5_B_ede7', C_B), ('W5_A_ede7', C_A)):
    r = G[key]; ls = '-' if 'lcdm5' in key else '--'
    ax[0].loglog(sp, r[:, 0], 'o' + ls, color=c, label=f"{key[3]}, {key[5:]} draws (no envelope)")
for n, mk in (('Be1', 's'), ('Be2', 'D'), ('Be3', '^')):
    for v, ls in (('lcdm5', '-'), ('ede7', '--')):
        if f'W5_{n}_{v}' in ENV: ax[0].plot([0.6], [ENV[f'W5_{n}_{v}'][0]], mk, color=COL['Be2'], ms=7, label=f'{n}, {v}' if v == 'lcdm5' else None)
ax[0].axvline(0.6, color='0.7', lw=0.8); ax[0].set_xlabel('node spacing in ln k'); ax[0].set_ylabel(r'rms residual in $\ln P_{\rm lin}$'); ax[0].legend(fontsize=7)
ax[0].xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}')); ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].set_xticks(sp)
names = [n for n in ('B', 'Be1', 'Be2', 'Be3', 'Be290', 'Be2120') if f'W4_baseline_{n}' in ENV]
for i, n in enumerate(names):
    ax[1].bar(np.arange(4) + (i - (len(names) - 1) / 2) * 0.8 / len(names), [ENV[f'W4_{v}_{n}'].std() for v in W.COSMO_VARIANTS],
              0.8 / len(names), label=n, color=COL['B'] if n == 'B' else COL['Be2'], alpha=0.35 + 0.65 * (i + 1) / len(names))
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(W.COSMO_VARIANTS); ax[1].set_ylabel(r'std of $\Delta\ln L$ over the posterior'); ax[1].legend(fontsize=8)
ax[1].set_title('against the existing direct likelihood (pybird $r_d$, 60-node composition); see 3b, 3c', fontsize=9)
savefig('representation'); plt.close('all')

def implied(th, d):
    w = np.exp(d - d.max()); w /= w.sum(); return (w @ th - th.mean(0)) / th.std(0), 1.0 / np.sum(w**2) / len(w)
print(f"{'model':7s} " + " ".join(f"{v:>26s}" for v in W.COSMO_VARIANTS))
print(f"{'':7s} " + " ".join(f"{'std dlnL  ESS/N  max|shift|':>26s}" for v in W.COSMO_VARIANTS))
TH = {v: G[f'W4_{v}_th'] for v in W.COSMO_VARIANTS}
for src, n in [(G, 'A'), (G, 'B30'), (G, 'B120')] + [(ENV, n) for n in names]:
    cells = []
    for v in W.COSMO_VARIANTS:
        d = src[f'W4_{v}_{n}']; sh, ess = implied(TH[v], d); cells.append(f"{d.std():8.2f} {ess:6.2f} {np.abs(sh).max():8.2f}σ")
    print(f"{n:7s} " + " ".join(f"{c:>26s}" for c in cells))
ENVF = dict(np.load(os.path.join(S.OUT_ROOT, 'wiggle_gates', 'envelope_rdfix.npz')))
print("\nthe same against the exact likelihood with the corrected sound horizon (envelope_rdfix.npz):")
for n in [n for n in ('B', 'Be2', 'Be245', 'Be230') if f'W4_baseline_{n}' in ENVF]:
    cells = []
    for v in W.COSMO_VARIANTS:
        d = ENVF[f'W4_{v}_{n}']; sh, ess = implied(TH[v], d); cells.append(f"{d.std():8.2f} {ess:6.2f} {np.abs(sh).max():8.2f}σ")
    print(f"{n:7s} " + " ".join(f"{c:>26s}" for c in cells))
for n in names:
    for v in ('lcdm5', 'w0wa7', 'ede7'):
        if f'W6_{n}_{v}' in ENV:
            e = ENV[f'W6_{n}_{v}']; log(f"{n} {v}: envelope the direct spectra need: e = {np.round(e.mean(0), 3)} +- {np.round(e.std(0), 3)}")

# %% [markdown]
# ### 3b. Against the *true* likelihood
#
# With the corrected $r_d$ and an envelope, the spectrum residual in the data range is below 0.1%. The likelihood
# difference to the existing direct model still has a spread of ~0.5 (lcdm5) to ~0.9 (w0wa7), and it *grows* with more
# nodes. So part of it is in the reference. The existing direct model does not hand the emulator the CosmoPower spectrum:
# it samples $P/T$ on the baseline's 60 nodes, interpolates to the knots in 1/Mpc, and the emulator re-interpolates
# after the $h_{\rm conv}$ relabelling. `wiggle_truth.py` evaluates the engine's spectrum directly at the emulator's
# knots, with the same growth, AP and BAO and the corrected $r_d$, and measures every direct map against that, on the
# same 300 draws.

# %%
f_tr = os.path.join(S.OUT_ROOT, 'wiggle_gates', 'truth.npz')
if os.path.exists(f_tr):
    T = dict(np.load(f_tr)); models = [n for n in ('direct', 'A', 'B', 'Be2') if f'baseline_{n}' in T]
    print(f"{'vs truth':9s} " + " ".join(f"{m:>24s}" for m in models)); print(f"{'':9s} " + " ".join(f"{'std dlnL ESS/N max|sh|':>24s}" for m in models))
    for cv in W.COSMO_VARIANTS:
        if f'{cv}_th' not in T: continue
        cells = []
        for m in models:
            d = T[f'{cv}_{m}']; sh, ess = implied(T[f'{cv}_th'], d); cells.append(f"{d.std():7.2f} {ess:6.2f} {np.abs(sh).max():7.2f}σ")
        print(f"{cv:9s} " + " ".join(f"{c:>24s}" for c in cells))
else:
    log(f"{f_tr} missing: SCRIPT=wiggle_truth.py ARGS='Be2 B A' sbatch exec_wiggle.sbatch")

# %% [markdown]
# ### 3c. Where the rest comes from: the frame or the map?
#
# `wiggle_frame.py` gives model B's likelihood the *exact* spectrum of its own sound-horizon frame (the "oracle")
# and compares it with the truth: oracle − truth is what the frame costs (α's = BAO α's, $P$ in $r_d$ units, the
# dimensionful EFT priors expressed in that frame). The rest, map − oracle, is the direct map's least-squares fit of ln a and
# the envelope. `wiggle_fisher_map.py` then weights that fit by the data, $W=J^TC^{-1}J$, with $J$ the response of the
# multipoles and BAO points to ln P on the 80 knots at the fiducial (variant suffix F). That changes only the direct map,
# so the same MI chain is projected with it.

# %%
for fname, title in (('frame.npz', 'frame vs map'), ('fisher_map.npz', 'data-weighted map'), ('truth_spacing.npz', 'node spacing, vs truth'),
                     ('fisher_map_spacing.npz', 'data-weighted map, 21 and 31 nodes'),
                     ('fisher_map_frame.npz', 'explicit alpha_rs (A frame) vs alpha_rs = 1 (B frame)'),
                     ('fisher_map_e1.npz', 'envelope e0 only (Ae145F) vs e0 + e1 (Ae245F)'),
                     ('fisher_map_e1_nodes.npz', 'e0 only at 21 / 31 / 60 nodes, and e0 + e1 at 31 nodes'),
                     ('fisher_map_bao.npz', 'BAO-fit envelope (1+A) exp(-k^2 dSigma^2/2): Ag245F; damping only: Ag145F')):
    f = os.path.join(S.OUT_ROOT, 'wiggle_gates', fname)
    if not os.path.exists(f): log(f"{f} missing"); continue
    T = dict(np.load(f)); print(f"--- {title} ({fname}): std of Delta ln L, max implied |shift|")
    labels = sorted({k.split('_', 1)[1] for k in T if not k.endswith('_th')})
    print(f"{'':9s} " + " ".join(f"{l:>20s}" for l in labels))
    for cv in W.COSMO_VARIANTS:
        if f'{cv}_th' not in T: continue
        cells = []
        for l in labels:
            if f'{cv}_{l}' not in T: cells.append(f"{'':>20s}"); continue
            d = T[f'{cv}_{l}']; sh, ess = implied(T[f'{cv}_th'], d); cells.append(f"{d.std():9.2f} {np.abs(sh).max():8.2f}σ")
        print(f"{cv:9s} " + " ".join(cells))

# %% [markdown]
# ## 4. The MI chains

# %%
CH = {}
for w in RUNS:
    M = MW[w]; r = dict(np.load(os.path.join(OUT[w], 'chain_mi.npz'))); CH[w] = r['x'].reshape(-1, r['x'].shape[-1])
    xb = np.load(os.path.join(OUT[w], 'bestfit_mi.npz'))['x_bf']
    log(f"model {w}: {M.n_mi} parameters; {r['x'].shape[0]} chains x {r['x'].shape[1]} draws in {float(r['walltime'])/60:.1f} min; accept "
        f"{r['accept'].mean():.3f}, divergent {r['divergent'].mean():.4f}, leapfrog/iter {r['n_leapfrog'].mean():.1f}; best-fit chi2 "
        f"{-2 * float(jax.jit(M.loglkl_mi)(jnp.array(xb))):.2f}, -ln posterior {-float(jax.jit(M.logpost_mi)(jnp.array(xb))):.2f}")
    chain_diagnostics(r['x'], M.names, max_print=4)
xo = np.load(S.files(s0)['chain_mi'])['x']; CH['old'] = xo.reshape(-1, xo.shape[-1])
log(f"baseline MI: {M0.n_mi} parameters, best-fit chi2 {-2 * float(jax.jit(M0.loglkl_mi)(jnp.array(np.load(S.files(s0)['bestfit_mi'])['x_bf']))):.2f}; "
    f"direct LCDM best fit chi2 {-2 * float(jax.jit(M0.loglkl_direct)(jnp.array(np.load(S.files(s0)['bestfit_direct'])['x_bf']))):.2f}")

# %% [markdown]
# ## 5. What the two α's per redshift measure
#
# The baseline model has ten free BAO dilations that only the post-reconstruction BAO points constrain. Its
# full-shape AP ($H$, $D_A$, $h_{\rm conv}$) is separate. In the wiggle models the two α's per redshift are shared by
# the full shape and the BAO. The BAO-equivalent α's are $\alpha$ (B, Be2) or $\alpha/\alpha_{r_s}$ (A). The width
# ratio against the baseline's free BAO α's measures what the full shape adds to post-reconstruction BAO. If
# $\alpha_{r_s}$ is a near-gauge, A and B must agree on these, and A's $\alpha_{r_s}$ must stay near its prior
# width (s_lnrs = 0.05).

# %%
def bao_alphas(w, x):
    M = MW[w]; phi = x[:, M.n_eft:]; ln = lambda b: phi[:, M.pidx[b]]
    rs = phi[:, M.pidx['rs']] if M.rs_free else 0.0
    return np.exp(ln('apar') - rs), np.exp(ln('aper') - rs) * np.asarray(M.bao_dm_factor)
ALPH = {w: bao_alphas(w, CH[w]) for w in RUNS}
old = np.exp(CH['old'][:, M0.idx['bao']])
fig, axs = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
for j, z in enumerate(M0.z):
    sl = M0.bao_slice[j]; iso = M0.bao_iso[j]
    for comp, ax in ((0, axs[0]), (1, axs[1])):
        if iso and comp == 1: continue
        o = old[:, sl][:, 0] if iso else old[:, sl][:, comp]
        ax.errorbar(z - 0.02, o.mean(), o.std(), fmt='o', color=C_OLD, label='baseline: free BAO α (BAO data only)' if j == 0 else None)
        for i, w in enumerate(RUNS):
            ap, ae = ALPH[w]; v = ap[:, j]**(1/3) * ae[:, j]**(2/3) if iso else (ap[:, j] if comp == 0 else ae[:, j])
            ax.errorbar(z + 0.012 * i, v.mean(), v.std(), fmt='s', color=COL[w], label=(f'model {w}: ' + (r'α/α$_{r_s}$' if MW[w].rs_free else 'α') + ' (FS + BAO)') if j == 0 else None)
            log(f"z={z:.3f} {'iso' if iso else ['par', 'per'][comp]}: baseline {o.mean():.4f}+-{o.std():.4f}  {w} {v.mean():.4f}+-{v.std():.4f} (width ratio {v.std()/o.std():.2f})")
axs[0].set_ylabel(r'$\alpha_\parallel$ (BGS, QSO: $\alpha_{\rm iso}$)'); axs[1].set_ylabel(r'$\alpha_\perp$')
for ax in axs: ax.set_xlabel('z'); ax.axhline(1, color='0.8', lw=0.8)
axs[0].legend(fontsize=8); savefig('alphas'); plt.close('all')
for w in RUNS:
    M = MW[w]; ph = CH[w][:, M.n_eft:]
    if M.rs_free:
        lnrs = ph[:, M.pidx['rs'][0]]
        log(f"model {w}: ln alpha_rs = {lnrs.mean():+.4f} +- {lnrs.std():.4f}  (prior 0 +- {M.prior['s_lnrs']}): posterior/prior width {lnrs.std() / M.prior['s_lnrs']:.2f}; "
            f"corr with the mean ln alpha {np.corrcoef(lnrs, ph[:, np.r_[M.pidx['apar'], M.pidx['aper']]].mean(1))[0, 1]:+.3f}")
    if M.n_env:
        e = ph[:, M.pidx['env']]
        for j, (nm, sg) in enumerate(zip(M.env_names(), M.env_sigmas())):
            log(f"model {w}: {nm} = {e[:, j].mean():+.3f} +- {e[:, j].std():.3f} (prior 0 +- {sg})")

# %% [markdown]
# The broadband: $\ln a(k)$ on the knots, with its 68% band (dashed for A). In B and Be2 it is the spectrum in
# sound-horizon units relative to the fiducial. The direct ΛCDM posterior, mapped through B's least-squares
# projection, is shown for comparison.

# %%
fig, axs = plt.subplots(1, 2, figsize=(14, 4)); kh = MB.knots_h; ip = np.argmin(np.abs(kh - 0.05))
thd = np.load(S.files(s0)['chain_direct'])['x'].reshape(-1, M0.n_eft + 3)[:, M0.n_eft:]
fmap = MB.phi_ln_keys(COSMO_MODELS['lcdm3'])
lad = np.asarray(jax.vmap(lambda t: MB.lna_to_knots(fmap(t)[MB.pidx['lna']]))(jnp.array(thd[np.random.default_rng(1).choice(len(thd), 400, replace=False)])))
for ax, rel in zip(axs, (False, True)):
    for w in RUNS:
        M = MW[w]; sub = CH[w][np.random.default_rng(0).choice(len(CH[w]), 2000, replace=False), M.n_eft:]
        la = np.asarray(jax.vmap(M.lna_to_knots)(jnp.array(sub[:, M.pidx['lna']])))
        if rel: la = la - la[:, [ip]]
        q = np.percentile(la, [16, 50, 84], axis=0)
        ax.plot(kh, q[1], color=COL[w], label=f'model {w}')
        if w == 'B': ax.fill_between(kh, q[0], q[2], color=COL[w], alpha=0.3)
        else: ax.plot(kh, q[0], color=COL[w], lw=0.7, ls='--'); ax.plot(kh, q[2], color=COL[w], lw=0.7, ls='--')
    ld = lad - lad[:, [ip]] if rel else lad
    ax.plot(kh, np.percentile(ld, 16, 0), color=C_D, lw=1); ax.plot(kh, np.percentile(ld, 84, 0), color=C_D, lw=1, label='direct ΛCDM in B coordinates (68%)')
    ax.set_xscale('log'); ax.axvspan(0.02, 0.2, color='0.93', zorder=0); ax.set_xlim(1e-3, 0.7); ax.set_xlabel(r'$k\ [h_{\rm fid}/{\rm Mpc}]$')
axs[0].set_ylabel(r'$\ln a(k)$'); axs[1].set_ylabel(r'$\ln a(k)-\ln a(0.05\,h/{\rm Mpc})$  (shape)'); axs[0].legend(fontsize=9)
savefig('broadband'); plt.close('all')

# %% [markdown]
# ## 6. The recovery tests
#
# Each row is one model. The columns give the largest |shift| of any parameter of the MI → model projection, in units
# of the reference σ, against the **true direct likelihood**. That is the engine's spectrum at the emulator's knots
# with the corrected sound horizon (`wiggle_direct.py <model> exact`, `output/desi_fsbao/exact_*`), sampled with the
# same NUTS settings, priors and boxes as the existing direct chains. Section 3b showed why the existing chains are not
# a good enough reference. "repr." is the shift the representation alone causes: the reference chain reweighted by
# $L_{\rm wiggle}/L_{\rm true}$ on 2000 draws. The baseline 106-parameter MI model, with the same 64k samples and the
# same two density estimators (Gaussian, flow ensemble c5), is shown against the truth and against its own direct
# chain, as in the meeting. The pass criterion of the existing tests is |shift| < 0.3σ with widths within 20%.
#
# **The sound horizon.** `pybird.symbolic.rs_drag`, the $r_d$ of every CosmoPower direct model, has the $\omega_b$
# exponent with the wrong sign: $d\ln r_d/d\ln\omega_b=+0.09$ where a numerical integral gives $-0.17$. That is 0.65% per
# BBN σ of $\omega_b$, and it matters for lcdm5 and w0wa7, where $\omega_b$ is free. The true-likelihood references use
# the corrected $r_d$; the baseline columns "own" keep the original.

# %%
def baseline_samples(cv, est):
    if cv in ('baseline', 'ede7'):
        f = os.path.join(S.OUT_ROOT, 'meeting', f'B0_flow_study_{est}.npz')
        return np.load(f)[f"{'lcdm3' if cv == 'baseline' else 'ede7'}_samples"] if os.path.exists(f) else None
    f = os.path.join(S.OUT_ROOT, cv, f'chain_projected_{est}.npz')
    return np.load(f)['x'].reshape(-1, len(COSMO_MODELS[S.resolve(cv)['cosmo_model']])) if os.path.exists(f) else None

from wiggle_direct import rdfix_files
def direct_file(cv, fixed=True):
    """The reference chain, as wiggle_recovery.reference: the true-likelihood chain (wiggle_direct.py exact), else the
    corrected-r_d one, else the original; fixed=False: always the original direct chain."""
    for kind in (('exact', 'exactrw', 'rdfix') if fixed else ()):
        f = rdfix_files(cv, kind)[1]['chain_direct']
        if os.path.exists(f): return f
    return S.files(S.resolve(cv))['chain_direct']

def direct_samples(cv, fixed=True):
    x = np.load(direct_file(cv, fixed))['x']; return x.reshape(-1, x.shape[-1])[:, M0.n_eft:]

def shifts(th_d, th):
    return (th.mean(0) - th_d.mean(0)) / th_d.std(0), th.std(0) / th_d.std(0)

REC = {(w, est): dict(np.load(os.path.join(OUT[w], f'recovery_{est}.npz'))) for w in REC_RUNS for est in ('gauss', 'c5')
       if os.path.exists(os.path.join(OUT[w], f'recovery_{est}.npz'))}
for cv in W.COSMO_VARIANTS: log(f"reference for {cv}: {direct_file(cv)}")
cols = [('baseline', 'gauss', 'own'), ('baseline', 'gauss', 'truth'), ('baseline', 'c5', 'truth')] + \
       [(w, e, 'truth') for w in REC_RUNS for e in ('gauss', 'c5', 'repr.')]
print(f"{'model':9s} | " + " ".join(f"{(w + ' ' + e + ('' if r == 'truth' else ' ' + r)):>15s}" for w, e, r in cols)); print('-' * (12 + 16 * len(cols)))
for cv in W.COSMO_VARIANTS:
    th_d, row = direct_samples(cv), []
    for w, est, ref in cols:
        if w == 'baseline':
            b = baseline_samples(cv, est); ref_th = direct_samples(cv, fixed=(ref == 'truth'))
            row.append(np.abs(shifts(ref_th, b)[0]).max() if b is not None else np.nan)
        else:
            r = REC.get((w, 'gauss' if est == 'repr.' else est))
            row.append(np.nan if r is None or f'{cv}_shift' not in r else np.abs(r[f'{cv}_shift_rep' if est == 'repr.' else f'{cv}_shift']).max())
    print(f"{cv:9s} | " + " ".join(f"{x:15.2f}" for x in row))

# %%
for (w, est), r in REC.items():
    for cv in [c for c in W.COSMO_VARIANTS if f'{c}_names' in r]:
        log(f"[{w} {est}] {cv:9s} " + "  ".join(f"{n} {a:+.2f}σ ×{b:.2f}" for n, a, b in zip(list(r[f'{cv}_names']), r[f'{cv}_shift'], r[f'{cv}_ratio']))
            + "   | representation " + " ".join(f"{a:+.2f}" for a in r[f'{cv}_shift_rep']))

# %% [markdown]
# Triangles: the true-likelihood direct chain is filled. Four projections are drawn with the flow estimate (c5), the one
# to use (section 7):
# - the recommended model **Ag245F** (BAO-fit wiggle template);
# - Ae245F, the same model with the $e_0$, $e_1$ envelope;
# - the original proposal A;
# - the baseline 106-parameter model (dashed grey).
#
# The legend gives each curve's largest |shift| in units of the direct σ for the flow, with the Gaussian value in brackets.
# The table above has every variant.

# %%
SHOW = [('Ag245F', '#00838F', r'recommended: BAO template $(1+A)\,e^{-k^2\Delta\Sigma^2/2}\,O(\alpha_{r_s}k)$, 21 nodes'),
        ('Ae245F', '#6A1B9A', r'envelope $1+e_0+e_1\ln k$, otherwise the same'), ('A', '#E8710A', r'original proposal: $\alpha_{r_s}$, 16 nodes, no envelope')]
for cv in W.COSMO_VARIANTS:
    sv = S.resolve(cv); model = sv['cosmo_model']; keys = COSMO_MODELS[model]; names = ['logA' if k == 'lnAs' else k for k in keys]
    box = {n: (EDE_BOX.get(k, COSMO_BOX[k]) if model == 'ede7' else COSMO_BOX[k]) for n, k in zip(names, keys)}
    mk = lambda th, lab: MCSamples(samples=th, names=names, labels=[COSMO_TEX[k] for k in keys], ranges=box, label=lab)
    th_d = direct_samples(cv)
    roots, cols_, labs, filled, ls, lws = [mk(th_d, 'direct')], [C_D], [f'direct {model}, true likelihood'], [True], ['-'], [1.0]
    for w, c, desc in SHOW:
        rg, rc = REC.get((w, 'gauss')), REC.get((w, 'c5'))
        if rc is None or f'{cv}_samples' not in rc: continue
        mx = f"{np.abs(rc[f'{cv}_shift']).max():.2f}σ" + (f" (Gaussian {np.abs(rg[f'{cv}_shift']).max():.2f}σ)" if rg is not None and f'{cv}_shift' in rg else '')
        roots.append(mk(rc[f'{cv}_samples'], w)); cols_.append(c); labs.append(f'MI {w} ({desc}): {mx}'); filled.append(False); ls.append('-'); lws.append(1.6 if w == 'Ag245F' else 1.0)
    b, bc = baseline_samples(cv, 'gauss'), baseline_samples(cv, 'c5')
    if bc is not None:
        mx = f"{np.abs(shifts(th_d, bc)[0]).max():.2f}σ" + (f" (Gaussian {np.abs(shifts(th_d, b)[0]).max():.2f}σ)" if b is not None else '')
        roots.append(mk(bc, 'old')); cols_.append(C_OLD); filled.append(False); ls.append('--'); lws.append(1.0)
        labs.append(f'baseline 106-parameter MI' + (' (pybird $r_d$)' if cv in ('lcdm5', 'w0wa7') else '') + f': {mx}')
    g = plots.get_subplot_plotter(subplot_size=2.2 if len(keys) > 4 else 2.7); g.settings.num_plot_contours = 2; g.settings.alpha_filled_add = 0.3
    g.settings.legend_fontsize = 10 if len(keys) > 4 else 9
    g.triangle_plot(roots, filled=filled, contour_colors=cols_, contour_ls=ls, contour_lws=lws, legend_labels=labs)
    g.export(os.path.join(FIG, f'recovery_{cv}.png')); g.export(os.path.join(FIG, f'recovery_{cv}.pdf')); plt.show(); plt.close('all')

# %% [markdown]
# ## 7. Summary
#
# 1. **$\alpha_{r_s}$ is a near-gauge.** Scaling $\alpha_{r_s}$ and all twelve α's by 3% together, with the spline
#    refitted, costs $\Delta\chi^2\le0.6$ at 16, 8 and 6 nodes. $\alpha_{r_s}$ alone costs 66. Even 6 nodes hold 96% of a
#    broadband dilation, so fewer nodes do not pin it. Model B ($\alpha_{r_s}\equiv1$) is the same model without this
#    direction. Its two α per redshift *are* the BAO α's, its template is $P(k\,r_d)/r_d^3$, and it needs no wiggle split.
# 2. **A rigid wiggle template needs a free amplitude and damping.** Without them (A, B), real spectra map into the
#    model with posterior shifts of 0.6–2.2σ against the true likelihood: the BAO amplitude and damping follow
#    $\omega_b/\omega_m$, which $\alpha_{r_s}$ cannot do.
#    - The **BAO-fit form** $(1+A)\,e^{-k^2\Delta\Sigma^2/2}$ (Ag245F) and the log form $1+e_0+e_1\ln(k/k_*)$ (Ae245F) do
#      equally well: map-only shifts 0.04–0.15σ against 0.09–0.17σ.
#    - **The amplitude is needed.** Damping alone diverges. An amplitude alone ($e_0$) fails w0wa at 0.51σ with 21 nodes;
#      only about 60 nodes absorb the damping, and those can also slide the wiggles.
#    - The data barely constrain the two: $A=0.16\pm0.16$ and $\Delta\Sigma^2=7\pm21\,{\rm Mpc}^2$, against
#      $\Lambda$CDM's $-0.02\pm0.07$ and $-1.5\pm4.4\,{\rm Mpc}^2$.
# 3. **Frame, nodes and map** (section 3c).
#    - **Keep $\alpha_{r_s}$ explicit.** With $\alpha_{r_s}\equiv1$ the dimensionful EFT priors act in $r_d$ units, which
#      costs 0.4σ in $h$ and $w_0$ for w0wa. With $\alpha_{r_s}$ explicit they act in $h$/Mpc, as in a direct fit, and
#      the cost is 0.09σ.
#    - **Nodes and map.** 21 nodes (spacing 0.45, too coarse to slide a wiggle) with the data-weighted map
#      ($W=J^TC^{-1}J$ on the knots) bring every model to ≤0.15σ, map only.
# 4. **Found on the way (they affect the existing direct chains, not the MI chains):**
#    `pybird.symbolic.rs_drag` has the $\omega_b$ exponent with the wrong sign (0.65% in $r_d$ per BBN σ).
#    `symbolic.comoving_distance` starts at $z=10^{-3}$ (BAO $D_M$ low by 0.37% at $z=0.295$).
#    The 60-node composition of the existing direct likelihood shifts ω_cdm by 0.5–0.9σ against the true likelihood.
#    True-likelihood chains (`exact_*`) against the existing direct chains: ΛCDM ω_cdm 0.1235 → 0.1211 (−0.50σ), h unchanged;
#    lcdm5 h 0.6915 → 0.6850 ± 0.0074 (−1.0σ) and ω_b back on its BBN prior (0.02163 ± 0.00037 → 0.02206 ± 0.00054). The
#    meeting's open "ω_b offset" was the $r_d$ sign. w0wa7: ω_cdm −0.56σ, ω_b +0.96σ, w0 and wa < 0.2σ.
# 5. **The two shared α's per redshift are as tight as post-reconstruction BAO alone** (width ratios 0.93–1.02 against
#    the baseline's free BAO α's). Tying the full-shape AP to them gives one consistent set of distances at no cost.
# 6. **Recovery tests** (section 6; reference: the sampled true-likelihood chain of every model).
#    - **The recommended configuration, Ag245F,** passes with the flow estimator. In max |shift|: 0.18σ (ΛCDM),
#      0.26σ (ΛCDM + ω_b, n_s), 0.20σ (w0wa) and 0.11σ (EDE), widths within 5%. Its full specification is in
#      `WIGGLE_SETUP.md` (59 sampled parameters).
#    - **Use the flows.** The Gaussian estimate fails w0wa (0.66σ), because the $\alpha_{r_s}$ near-symmetry makes the MI
#      posterior non-Gaussian.
#    - Ae245F gives 0.21, 0.14, 0.12 and 0.21σ.
#    - The 106-parameter baseline, against the same references: 0.67 / 0.47σ (ΛCDM), 1.36 / 0.78σ (lcdm5), 1.52 / 0.88σ
#      (w0wa7) and 0.85 / 0.97σ (EDE), Gaussian / flow. Its old passes were agreement with its own 60-node composition.
#    - The original proposal (A: $\alpha_{r_s}$, no envelope) misses by 0.7–1.9σ.
#    - $A$ and $\Delta\Sigma^2$ are predicted by the cosmology in the projection, not sampled. Marginalizing them widens
#      ω_cdm and n_s by about 25% (lcdm5) and shifts EDE by about 0.4σ.
# 7. **Model A.** $\ln\alpha_{r_s}=+0.038\pm0.028$ against its 0.05 prior, and it moves with the mean ln α
#    (correlation 0.97): it follows the near-symmetry. A's α/α_rs and B's α agree within 0.003 (≤ 0.15σ) at every redshift.
#
# The per-model, per-estimator lines below are the record (reference file named on each line).

# %%
rows = []
for cv in W.COSMO_VARIANTS:
    kind = direct_file(cv).split('/')[-2]
    for w in REC_RUNS:
        for est in ('gauss', 'c5'):
            r = REC.get((w, est))
            if r is None or f'{cv}_shift' not in r: continue
            sh, ra, rep = r[f'{cv}_shift'], r[f'{cv}_ratio'], r[f'{cv}_shift_rep']
            ok = (np.abs(sh) < 0.3).all() and (np.abs(ra - 1) < 0.2).all()
            rows.append(f"{cv:9s} {w:4s} {est:6s} max|shift| {np.abs(sh).max():.2f}σ (representation {np.abs(rep).max():.2f}σ), "
                        f"widths [{ra.min():.2f}, {ra.max():.2f}]  {'PASS' if ok else 'FAIL'}   reference {kind}")
print('\n'.join(rows) if rows else 'no recovery results yet')
