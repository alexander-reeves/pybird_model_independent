"""AP / BAO parameters from P(k) vs P(k)+B(k), with and without a BAO phase prior.

Tables and figures for 09_bao_pk_bk.ipynb.  Everything is read from
../../output/fisher_pb/jacobians_pb.npz (the model Jacobians) through mi_prior_fisher.py, so this
script is seconds, not minutes.  Nothing here re-fits: a Fisher matrix is built for each
assumption and inverted.

THE THREE STAGES, in the order the notebook tells them.
  A  free template, no prior at all (80 knots): read from robust_pb_results.npz.  D_M/r_d and
     D_H/r_d are NOT measurable -- the template slides its own wiggles, so it carries its own
     ruler.  What survives is F_AP = D_M/D_H, the distance RATIOS between redshifts, and f.
  B  + the smoothness prior the HMC actually samples (60 ln-a nodes, lambda = 200).  The
     wiggles can still slide: this basis represents 96% of the slide pattern, and the prior
     alone allows a 3.2% slide at 1 sigma.  D_M/r_d lands at ~2%.
  C  + the BAO PHASE PRIOR, which is the assumption a standard BAO analysis makes.  See
     mi_prior_fisher.phase_modes: P_lin = a(k) T_nw(k) [1 + O(k)], sliding the wiggles by
     delta in ln k changes ln P by delta * v(k) with v = dln(1+O)/dlnk, so the prior is a
     Gaussian of width sigma_delta on the component of ln a along v.  One global slide is not
     enough (the phase can drift slowly across the band and take most of the freedom back), so
     the prior covers v times Legendre envelopes of degree 0..5 in ln k: the slide is pinned
     EVERYWHERE, not just on average.  Broadband free, wiggle amplitude (the damping nuisance
     of a BAO fit) free, wiggle PHASE tied down.  sigma_delta is exactly "how far the template
     may move the ruler", in per cent.

In every stage the exact symmetries g1 (ruler) and g2 (amplitude) are projected out of the
DATA, so no discretization artifact is read as information; the priors keep them.
alpha_perp = ln D_A - ln h_conv = D_M/r_d and alpha_par = -ln H - ln h_conv = D_H/r_d, both
invariant under g1 and therefore measurable without any prior on h.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mi_prior_fisher as M

OUT = M.OUT
FIG = os.path.join(OUT, 'meeting')
os.makedirs(FIG, exist_ok=True)
C_P, C_PB, C_INK, C_MUTED = '#E8710A', '#1E88E5', '#1A1A1A', '#6E6E6E'
zz, nz = M.zz, M.nz
IZ = 2                                        # z = 0.706, the bin the tables quote
KEYS = (('p', 'P(k)', C_P), ('pb', 'P(k)+B(k)', C_PB))
QTS = [('perp', r'$D_M/r_d$'), ('par', r'$D_H/r_d$'), ('F_AP', r'$F_{\rm AP}=D_M/D_H$'),
       ('f', r'$f(z)$')]
PHASE_REF = 0.002        # sigma_delta for the tables: the wiggles may slide by 0.2%
N_PHASE_SCAN = (1, 2, 3, 4, 6, 8)             # envelope modes, for the second scan panel


def alpha_c(sysd, which, iz):
    """alpha_perp = ln D_A - ln h_conv, alpha_par = -ln H - ln h_conv (same as bao_basis_scan).

    Distances in units of the template's own ruler: both are invariant under g1, the exact
    symmetry that rescales h_conv, every D_A and every 1/H together, so they are measurable
    without any prior on h -- unlike absolute D_A and H, which in this model are set by the
    h_conv and growth priors alone."""
    c = np.zeros(sysd['npar']); o = sysd['o_g']
    c[o + M.ihc] = -1.0
    if which == 'perp':
        c[o + M.iD[iz]] = 1.0
    else:
        c[o + M.iH[iz]] = -1.0
    return c


def sig_q(sy, q, iz):
    c = alpha_c(sy, q, iz) if q in ('perp', 'par') else M.growth_c(sy, q, iz)
    return M.sig(sy, c)


def systems(phase_sigma=None, spacing=M.NODE_SPACING, n_phase=M.N_PHASE):
    return {k: M.build(k, project_data=True, spacing=spacing, project=(0, 1),
                       phase_sigma=phase_sigma, n_phase=n_phase) for k, _, _ in KEYS}


# =======================================================================================
# 0.  what the phase prior is, drawn
# =======================================================================================
nl15, S15, D15, _ = M.node_basis(0.15)
nl60, S60, D60, _ = M.node_basis(0.60)
v15, v60 = M.phase_modes(S15, 1)[:, 0], M.phase_modes(S60, 1)[:, 0]
cap15 = np.linalg.norm(S15 @ v15) / np.linalg.norm(M.wiggle_v)
cap60 = np.linalg.norm(S60 @ v60) / np.linalg.norm(M.wiggle_v)
Pi15 = np.eye(len(nl15)) / M.PRIORS_HMC['s_lna']**2 + M.PRIORS_HMC['lam'] * (D15.T @ D15)
c15 = v15 / (v15 @ v15)
prior_slide = float(np.sqrt(c15 @ np.linalg.inv(Pi15) @ c15))
print(f"BAO period in ln k = 2pi/(r_d k): {2*np.pi/(105*0.05):.2f} at k=0.05, "
      f"{2*np.pi/(105*0.1):.2f} at 0.1, {2*np.pi/(105*0.2):.2f} at 0.2 h/Mpc")
print(f"slide pattern v = dln(1+O)/dlnk represented by the node basis: "
      f"{cap15:.1%} at spacing 0.15 (60 nodes), {cap60:.1%} at spacing 0.60 (16 nodes)")
print(f"smoothness prior alone allows a template slide of {100*prior_slide:.2f}% (1 sigma) "
      f"in the 60-node basis -> that is the ruler the current model really has\n")

sel = (M.knots_h > 0.02) & (M.knots_h < 0.32)
fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, layout='constrained')
axes[0].plot(M.knots_h[sel], 1 + M.wiggle_O[sel], color=C_INK, lw=2)
axes[0].axhline(1, color=C_MUTED, lw=0.8)
axes[0].set_ylabel(r'$1+O(k) = T/T_{\rm nw}$', fontsize=12)
axes[0].set_title('BAO wiggles of the template, and the pattern that slides them', fontsize=14)
axes[1].plot(M.knots_h[sel], M.wiggle_v[sel], color=C_INK, lw=2,
             label=r'$v = \mathrm{d}\ln(1+O)/\mathrm{d}\ln k$')
axes[1].plot(M.knots_h[sel], (S15 @ v15)[sel], color=C_PB, lw=1.8, ls='--',
             label=f'60 nodes, spacing 0.15: {cap15:.0%}')
axes[1].plot(M.knots_h[sel], (S60 @ v60)[sel], color=C_P, lw=1.8, ls='-.',
             label=f'16 nodes, spacing 0.60: {cap60:.0%}')
axes[1].axhline(0, color=C_MUTED, lw=0.8)
axes[1].set_xscale('log'); axes[1].set_xlabel(r'$k$  [$h$/Mpc]', fontsize=12)
axes[1].set_ylabel(r'$v(k)$', fontsize=12)
axes[1].legend(frameon=False, fontsize=9.5)
for a in axes:
    a.grid(alpha=0.2)
p = os.path.join(FIG, 'fig_phase_prior_explained.png')
fig.savefig(p, dpi=170); plt.close(fig); print(f"saved {p}")

# =======================================================================================
# 1.  the scan: how tight a phase prior buys how much BAO precision
# =======================================================================================
SCAN = [None, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
scan = {}
for ps in SCAN:
    sy = systems(phase_sigma=ps)
    for k, _, _ in KEYS:
        for q, _ in QTS:
            scan[(ps, k, q)] = sig_q(sy[k], q, IZ)
nscan = {}
for npz_ in N_PHASE_SCAN:
    sy = systems(phase_sigma=PHASE_REF, n_phase=npz_)
    for k, _, _ in KEYS:
        for q, _ in QTS:
            nscan[(npz_, k, q)] = sig_q(sy[k], q, IZ)
rigid = systems(spacing=0.60)                 # the coarse-node basis, for reference
_se = os.path.join(OUT, 'se07_forecast.npz')
S7 = np.load(_se) if os.path.exists(_se) else None

print(f"z = {zz[IZ]:.3f}: fractional 1 sigma vs the phase prior (sigma_delta = how far the "
      f"template may slide its own wiggles, {M.N_PHASE} envelope modes)")
print(f"{'sigma_delta':>12s}   {'D_M/r_d  P     P+B':>22s}   {'D_H/r_d  P     P+B':>22s}"
      f"   {'F_AP  P      P+B':>20s}   {'f  P      P+B':>18s}")
for ps in SCAN:
    lab = 'free' if ps is None else f'{100*ps:.2f}%'
    print(f"{lab:>12s}   " + "   ".join(
        f"{100*scan[(ps,'p',q)]:8.2f}% {100*scan[(ps,'pb',q)]:7.2f}%" for q, _ in QTS))
print(f"{'16 nodes':>12s}   " + "   ".join(
    f"{100*sig_q(rigid['p'], q, IZ):8.2f}% {100*sig_q(rigid['pb'], q, IZ):7.2f}%"
    for q, _ in QTS) + "   <- coarse-node basis instead of a prior")
if S7 is not None:
    print(f"{'BAO pre-rec':>12s}   {100*S7['pre_perp'][IZ]:8.2f}%"
          f"           {100*S7['pre_par'][IZ]:8.2f}%            <- standard BAO, same bins")
    print(f"{'BAO post-rec':>12s}   {100*S7['post_perp'][IZ]:8.2f}%"
          f"           {100*S7['post_par'][IZ]:8.2f}%")
print(f"\nhow completely the phase is pinned, at sigma_delta = {100*PHASE_REF:.1f}%:")
print(f"{'envelopes':>12s}   {'D_M/r_d  P     P+B':>22s}   {'D_H/r_d  P     P+B':>22s}")
for npz_ in N_PHASE_SCAN:
    print(f"{npz_:12d}   " + "   ".join(
        f"{100*nscan[(npz_,'p',q)]:8.2f}% {100*nscan[(npz_,'pb',q)]:7.2f}%"
        for q, _ in QTS[:2]))

fig, axes = plt.subplots(1, 3, figsize=(17, 4.9), layout='constrained')
for ax, (q, title) in zip(axes[:2], QTS[:2]):
    for k, lab, col in KEYS:
        ax.plot([100 * ps for ps in SCAN[1:]], [100 * scan[(ps, k, q)] for ps in SCAN[1:]],
                color=col, lw=2.2, marker='o', ms=5, label=lab)
        ax.axhline(100 * scan[(None, k, q)], color=col, lw=1.4, ls=':')
        ax.axhline(100 * sig_q(rigid[k], q, IZ), color=col, lw=1.4, ls='--')
    if S7 is not None:
        ax.axhline(100 * S7[f'pre_{q}'][IZ], color=C_INK, lw=1.4, ls='-')
        ax.axhline(100 * S7[f'post_{q}'][IZ], color=C_INK, lw=1.4, ls='-.')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.invert_xaxis()
    ax.set_xlabel(r'phase prior $\sigma_\delta$  [%]', fontsize=12)
    ax.set_title(title, fontsize=13); ax.grid(alpha=0.2, which='both')
axes[0].set_ylabel(r'fractional $1\sigma$ [%]', fontsize=12)
h = [plt.Line2D([], [], color=C_P, lw=2.2, marker='o'),
     plt.Line2D([], [], color=C_PB, lw=2.2, marker='o'),
     plt.Line2D([], [], color=C_MUTED, lw=1.4, ls=':'),
     plt.Line2D([], [], color=C_MUTED, lw=1.4, ls='--'),
     plt.Line2D([], [], color=C_INK, lw=1.4, ls='-'),
     plt.Line2D([], [], color=C_INK, lw=1.4, ls='-.')]
axes[0].legend(h, ['P(k)', 'P(k)+B(k)', 'no phase prior', 'coarse 16-node basis',
                   'standard BAO, pre-recon', 'standard BAO, post-recon'],
               frameon=False, fontsize=9, ncol=1, loc='upper right')
for k, lab, col in KEYS:
    for q, ls, mk in (('perp', '-', 'o'), ('par', '--', 's')):
        axes[2].plot(N_PHASE_SCAN, [100 * nscan[(n_, k, q)] for n_ in N_PHASE_SCAN],
                     color=col, ls=ls, lw=2.0, marker=mk, ms=5,
                     label=f'{lab}, {"$D_M/r_d$" if q == "perp" else "$D_H/r_d$"}')
        axes[2].axhline(100 * sig_q(rigid[k], q, IZ), color=col, lw=1.2, ls='--')
axes[2].set_xlabel('envelope modes', fontsize=12)
axes[2].set_title(rf'$\sigma_\delta = {100*PHASE_REF:.1f}\%$', fontsize=13)
axes[2].grid(alpha=0.2); axes[2].legend(frameon=False, fontsize=9)
fig.suptitle(f'BAO distance precision vs the phase prior, z = {zz[IZ]:.2f}', fontsize=14)
p = os.path.join(FIG, 'fig_phase_scan.png')
fig.savefig(p, dpi=170); plt.close(fig); print(f"\nsaved {p}")

# =======================================================================================
# 2.  per-redshift summary, no phase prior vs the standard-BAO phase prior
# =======================================================================================
free_sy, phase_sy = systems(), systems(phase_sigma=PHASE_REF)
summ = {}
for tag, sy in (('free', free_sy), ('phase', phase_sy)):
    for k, _, _ in KEYS:
        for q, _ in QTS:
            summ[(tag, k, q)] = np.array([sig_q(sy[k], q, iz) for iz in range(nz)])
np.savez(os.path.join(OUT, 'meeting_bao_pb.npz'), zeff=zz, phase_ref=PHASE_REF,
         scan_sigma=np.array([np.nan if s is None else s for s in SCAN]),
         cap15=cap15, cap60=cap60, prior_slide=prior_slide, n_phase=M.N_PHASE,
         n_phase_scan=np.array(N_PHASE_SCAN),
         **{f'nscan_{n_}_{k}_{q}': nscan[(n_, k, q)]
            for n_ in N_PHASE_SCAN for k, _, _ in KEYS for q, _ in QTS},
         **{f'{t}_{k}_{q}': v for (t, k, q), v in summ.items()},
         **{f'scan_{"free" if s is None else f"{s:g}"}_{k}_{q}': scan[(s, k, q)]
            for s in SCAN for k, _, _ in KEYS for q, _ in QTS})

print("\nPer-redshift fractional 1 sigma:")
for tag, lab in (('free', 'no phase prior (the model we sample)'),
                 ('phase', f'phase prior {100*PHASE_REF:.1f}% (standard BAO)')):
    print(f"  {lab}")
    print(f"{'':16s}" + "".join(f"{'z='+format(z,'.2f'):>16s}" for z in zz))
    for q, _ in QTS:
        for k, kl, _ in KEYS:
            print(f"    {q:5s} {kl:9s}" + "".join(
                f"{100*v:15.2f}%" for v in summ[(tag, k, q)]))

fig, axes = plt.subplots(2, 4, figsize=(19, 7.4), layout='constrained', sharex=True,
                         gridspec_kw={'height_ratios': [3, 1.25]})
for j, (q, title) in enumerate(QTS):
    ax, axr = axes[0, j], axes[1, j]
    for tag, ls, lab in (('free', '-', 'no phase prior'),
                         ('phase', '--', f'phase prior {100*PHASE_REF:.1f}%')):
        for k, kl, col in KEYS:
            ax.plot(zz, 100 * summ[(tag, k, q)], color=col, ls=ls, lw=2.0, marker='o', ms=4,
                    label=f'{kl}, {lab}')
        axr.plot(zz, summ[(tag, 'pb', q)] / summ[(tag, 'p', q)], color=C_INK, ls=ls, lw=1.8,
                 marker='o', ms=4, label=lab)
    if S7 is not None and q in ('perp', 'par'):
        ax.plot(zz, 100 * S7[f'pre_{q}'], color=C_MUTED, lw=1.6, ls='-', marker='^', ms=4,
                label='standard BAO, pre-recon')
        ax.plot(zz, 100 * S7[f'post_{q}'], color=C_MUTED, lw=1.6, ls='-.', marker='v', ms=4,
                label='standard BAO, post-recon')
    ax.set_yscale('log'); ax.set_title(title, fontsize=13.5); ax.grid(alpha=0.2, which='both')
    axr.axhline(1.0, color=C_MUTED, lw=1); axr.set_ylim(0.4, 1.1); axr.grid(alpha=0.2)
    axr.set_xlabel('$z$', fontsize=12)
axes[0, 0].set_ylabel(r'fractional $1\sigma$ [%]', fontsize=12)
axes[1, 0].set_ylabel(r'$\sigma_{P+B}/\sigma_P$', fontsize=12)
axes[0, 0].legend(frameon=False, fontsize=8.5, handlelength=3.0)
axes[1, 0].legend(frameon=False, fontsize=8.5, handlelength=3.0, loc='lower left')
fig.suptitle('BAO and growth per redshift', fontsize=14)
p = os.path.join(FIG, 'fig_bao_per_z.png')
fig.savefig(p, dpi=170); plt.close(fig); print(f"saved {p}")

# =======================================================================================
# 3.  the BAO corner at one redshift
# =======================================================================================
from getdist import plots
from getdist.gaussian_mixtures import GaussianND

names = ['aperp', 'apar', 'f']
labels = [r'\Delta (D_M/r_d)', r'\Delta (D_H/r_d)', r'\Delta f/f']
dists, cols, lss, fills, legs = [], [], [], [], []
for tag, lab, ls in (('free', 'no phase prior', '--'),
                     ('phase', f'phase prior {100*PHASE_REF:.1f}%', '-')):
    sy = free_sy if tag == 'free' else phase_sy
    for k, kl, col in KEYS:
        cs = np.array([alpha_c(sy[k], 'perp', IZ), alpha_c(sy[k], 'par', IZ),
                       M.growth_c(sy[k], 'f', IZ)])
        cov = cs @ sy[k]['C'] @ cs.T
        dists.append(GaussianND(np.zeros(3), cov, names=names, labels=labels,
                                label=f'{kl}, {lab}'))
        cols.append(col); lss.append(ls); fills.append(tag == 'phase' and k == 'pb')
        legs.append(f'{kl}, {lab}')
g = plots.get_subplot_plotter(subplot_size=2.8)
g.settings.num_plot_contours = 2
g.settings.alpha_filled_add = 0.30
g.settings.legend_fontsize = 13
g.triangle_plot(dists, params=names, filled=fills, contour_colors=cols, contour_ls=lss,
                line_args=[{'color': c, 'ls': l, 'lw': 2.0} for c, l in zip(cols, lss)],
                legend_labels=legs, markers={n: 0.0 for n in names})
g.fig.suptitle(f'BAO and growth at z = {zz[IZ]:.2f}', fontsize=15, y=1.02)
p = os.path.join(FIG, 'fig_bao_corner.png')
g.export(p); plt.close('all'); print(f"saved {p}")


# =======================================================================================
# 4.  the big triangle: both BAO parameters at every redshift
# =======================================================================================
names12, labels12, qty12 = [], [], []
for iz, z in enumerate(zz):
    names12 += [f'perp{iz}', f'par{iz}']
    labels12 += [rf'D_M/r_d\,({z:.2f})', rf'D_H/r_d\,({z:.2f})']
    qty12 += [('perp', iz), ('par', iz)]
ds12, cols12, leg12 = [], [], []
for k, kl, col in KEYS:                       # the BAO measurement: with the phase prior
    cs = np.array([alpha_c(phase_sy[k], q, iz) for q, iz in qty12])
    ds12.append(GaussianND(np.zeros(12), cs @ phase_sy[k]['C'] @ cs.T, names=names12,
                           labels=labels12, label=kl))
    cols12.append(col); leg12.append(kl)
g = plots.get_subplot_plotter(subplot_size=1.25)
g.settings.num_plot_contours = 2
g.settings.alpha_filled_add = 0.30
g.settings.legend_fontsize = 19
g.settings.axes_fontsize = 9
g.settings.axes_labelsize = 13
g.triangle_plot(ds12, params=names12, filled=[False, True], contour_colors=cols12,
                line_args=[{'color': c, 'lw': 2.0} for c in cols12],
                legend_labels=leg12, markers={n: 0.0 for n in names12})
g.fig.suptitle(f'BAO parameters at every redshift, phase prior {100*PHASE_REF:.1f}%',
               fontsize=22, y=1.045)
p = os.path.join(FIG, 'fig_bao_triangle_all_z.png')
g.export(p); plt.close('all'); print(f"saved {p}")
