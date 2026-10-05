"""P(k) recovery, the growth sector and the LCDM recovery: figures for 08_growth_pk_bk.ipynb.

All three come from the SAME Fisher system: the whitened model Jacobian of jacobian_pb.py in
the basis the HMC samples (60 ln-a nodes + the setups.COMMON priors), with the exact
symmetries projected out of the data. Nothing here re-fits; each figure is a different linear
functional of one covariance matrix.

  fig_bands_pb.png     the linear P(k) recovered with everything else marginalized, P vs P+B,
                       plus the per-node width and the prior it is competing with.
  fig_growth_corner    (f, H, D_A) at one redshift, P vs P+B.  The H and D_A widths here are
                       set by the h_conv/growth priors (the ruler direction g1 contains no
                       amplitudes, so no data can fix it) -- the ruler-free combinations are
                       the subject of 09_bao_pk_bk.ipynb.
  fig_growth_f_vs_z    sigma(f)/f at every redshift: where the bispectrum actually pays.
  fig_cosmo_pb.png     (omega_cdm, ln 10^10 A_s, h) for the DIRECT LCDM model = the MI model
                       composed with cosmo_map_pb.npz.  Built from the same whitened Jacobian,
                       with and without the data-side symmetry projection, so the printed
                       gate says how much of the LCDM constraint was discretization artifact.
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
KEYS = (('p', 'P(k)', C_P), ('pb', 'P(k)+B(k)', C_PB))
zz, nz = M.zz, M.nz
IZ = 2
KMIN_P, KMAX_P = 0.01, 0.2          # P(k) scale cut of the mock
KMIN_B, KMAX_B = 0.02, 0.10         # B(k) triangle range

sysd = {k: M.build(k, project_data=True, project=(0, 1)) for k, _, _ in KEYS}

# =======================================================================================
# 1.  the linear P(k) bands
# =======================================================================================
S = sysd['p']['S']
band, node_sig = {}, {}
for k, _, _ in KEYS:
    sy = sysd[k]
    Ca = sy['C'][sy['o_a']:sy['o_a'] + sy['n_amp'], sy['o_a']:sy['o_a'] + sy['n_amp']]
    band[k] = np.sqrt(np.clip(np.diag(S @ Ca @ S.T), 0, None))      # sigma(ln a) per knot
    node_sig[k] = np.sqrt(np.diag(Ca))
nl, Sb, Db, _ = M.node_basis()


def _prior_band(pr):
    Pi_a = np.eye(len(nl)) / pr['s_lna']**2 + pr['lam'] * (Db.T @ Db)
    return np.sqrt(np.diag(Sb @ np.linalg.inv(Pi_a) @ Sb.T))


prior_band = _prior_band(M.PRIORS_HMC)
# the same band with the wider priors recommended by prior_decomposition_pb.py
band_wide = {}
for _k, _, _ in (('p', 0, 0), ('pb', 0, 0)):
    _sy = M.build(_k, priors=M.PRIORS_WIDE, project_data=True, project=(0, 1))
    _Ca = _sy['C'][_sy['o_a']:_sy['o_a'] + _sy['n_amp'], _sy['o_a']:_sy['o_a'] + _sy['n_amp']]
    band_wide[_k] = np.sqrt(np.clip(np.diag(_sy['S'] @ _Ca @ _sy['S'].T), 0, None))

inb = (M.knots_h >= KMIN_P) & (M.knots_h <= KMAX_P)
print("sigma(ln P_lin) per knot, marginalized over everything else:")
print(f"  inside the P(k) window [{KMIN_P}, {KMAX_P}] h/Mpc:  P {band['p'][inb].mean():.4f}   "
      f"P+B {band['pb'][inb].mean():.4f}   ratio {band['pb'][inb].mean()/band['p'][inb].mean():.3f}"
      f"   (prior alone {prior_band[inb].mean():.4f})")
best = np.argmax(band['p'] / np.maximum(band['pb'], 1e-12))
print(f"  best single knot for B(k): k = {M.knots_h[best]:.3f} h/Mpc, "
      f"{band['p'][best]:.4f} -> {band['pb'][best]:.4f}")
print(f"  wider priors (s_lna {M.PRIORS_WIDE['s_lna']:g}, lambda {M.PRIORS_WIDE['lam']:g}): "
      f"P {band_wide['p'][inb].mean():.4f}   P+B {band_wide['pb'][inb].mean():.4f}   "
      f"ratio {band_wide['pb'][inb].mean()/band_wide['p'][inb].mean():.3f}")
_r = band_wide['pb'] / band_wide['p']
print(f"  best knot with the wider priors: k = {M.knots_h[np.argmin(_r)]:.3f} h/Mpc, "
      f"ratio {_r.min():.3f}")

sel = (M.knots_h > 2e-3) & (M.knots_h < 0.6)
fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.2), sharex=True, layout='constrained',
                         gridspec_kw={'height_ratios': [2.0, 1.3, 1.0]})
kk, T = M.knots_h[sel], M.T_[sel]
BW = band_wide
axes[0].plot(kk, T, color=C_INK, lw=1.8, ls='--', label='truth')
axes[0].fill_between(kk, T * np.exp(-BW['pb'][sel]), T * np.exp(BW['pb'][sel]),
                     color=C_PB, alpha=0.32, lw=0, label=r'P(k)+B(k)  $\pm1\sigma$')
for sgn in (+1, -1):
    axes[0].plot(kk, T * np.exp(sgn * BW['p'][sel]), color=C_P, lw=1.9,
                 label=r'P(k)  $\pm1\sigma$' if sgn > 0 else None)
axes[0].set_xscale('log'); axes[0].set_yscale('log')
axes[0].set_ylabel(r'$P_{\rm lin}(k)$  [Mpc$^3$]', fontsize=12)
axes[0].set_title('Linear power spectrum recovery', fontsize=14)

for k, lab, col in KEYS:
    axes[1].plot(kk, 100 * BW[k][sel], color=col, lw=2.3, label=lab)
    axes[1].plot(kk, 100 * band[k][sel], color=col, lw=1.4, ls=':')
axes[1].set_ylabel(r'$1\sigma$ on $P_{\rm lin}$  [%]', fontsize=12)
axes[1].set_ylim(0, 105 * BW['p'][sel].max())

axes[2].plot(kk, BW['pb'][sel] / BW['p'][sel], color=C_PB, lw=2.3)
axes[2].plot(kk, band['pb'][sel] / band['p'][sel], color=C_PB, lw=1.4, ls=':')
axes[2].axhline(1.0, color=C_MUTED, lw=1.2)
axes[2].set_ylabel(r'$\sigma_{P+B}\,/\,\sigma_{P}$', fontsize=12)
axes[2].set_xlabel(r'$k$  [$h$/Mpc]', fontsize=12)
axes[2].set_ylim(0.55, 1.05)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D
for a in axes:
    a.axvspan(KMIN_P, KMAX_P, color='0.55', alpha=0.13, lw=0)
    a.axvspan(KMIN_B, KMAX_B, color='0.55', alpha=0.13, lw=0)
    a.grid(alpha=0.2, which='both')
axes[0].legend(frameon=False, fontsize=11, loc='lower left')
axes[1].legend(axes[1].get_legend_handles_labels()[0] +
               [Line2D([], [], color=C_MUTED, lw=1.4, ls=':'),
                Patch(facecolor='0.55', alpha=0.13), Patch(facecolor='0.55', alpha=0.26)],
               axes[1].get_legend_handles_labels()[1] +
               ['tighter priors as sampled', 'P(k) data range', 'B(k) data range'],
               frameon=False, fontsize=9.5, ncol=2, loc='upper left')
p = os.path.join(FIG, 'fig_bands_pb.png')
fig.savefig(p, dpi=170); plt.close(fig); print(f"saved {p}")

# =======================================================================================
# 2.  the growth sector
# =======================================================================================
from getdist import plots
from getdist.gaussian_mixtures import GaussianND

nm3 = ['f', 'lnH', 'lnDA']
lb3 = [r'\Delta f/f', r'\Delta H/H', r'\Delta D_A/D_A']
ds = [GaussianND(np.zeros(3), M.block_cov(sysd[k], nm3, IZ), names=nm3, labels=lb3, label=lab)
      for k, lab, _ in KEYS]
g = plots.get_subplot_plotter(subplot_size=2.9)
g.settings.num_plot_contours = 2; g.settings.alpha_filled_add = 0.32
g.settings.legend_fontsize = 14
g.triangle_plot(ds, params=nm3, filled=[False, True], contour_colors=[C_P, C_PB],
                line_args=[{'color': C_P, 'lw': 2.0}, {'color': C_PB, 'lw': 2.0}],
                legend_labels=[lab for _, lab, _ in KEYS], markers={n: 0.0 for n in nm3})
g.fig.suptitle(rf'Growth and geometry at $z = {zz[IZ]:.2f}$', fontsize=15, y=1.02)
p = os.path.join(FIG, 'fig_growth_corner.png')
g.export(p); plt.close('all'); print(f"saved {p}")

sig_f = {k: np.array([M.sig(sysd[k], M.growth_c(sysd[k], 'f', iz)) for iz in range(nz)])
         for k, _, _ in KEYS}
sig_fap = {k: np.array([M.sig(sysd[k], M.growth_c(sysd[k], 'F_AP', iz)) for iz in range(nz)])
           for k, _, _ in KEYS}
print("\nfractional sigma(f):   " + "  ".join(f"z={z:.2f}" for z in zz))
for k, lab, _ in KEYS:
    print(f"  {lab:10s} " + "  ".join(f"{100*v:6.1f}%" for v in sig_f[k]))
print("  ratio      " + "  ".join(f"{b/a:6.2f} " for a, b in zip(sig_f['p'], sig_f['pb'])))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), layout='constrained')
for ax, (d, title) in zip(axes, [(sig_f, r'growth rate $f(z)$'),
                                 (sig_fap, r'$F_{\rm AP} = D_M/D_H$')]):
    for k, lab, col in KEYS:
        ax.plot(zz, 100 * d[k], color=col, lw=2.2, marker='o', ms=6, label=lab)
    ax.set_yscale('log'); ax.set_xlabel('$z$', fontsize=12); ax.set_title(title, fontsize=13)
    ax.grid(alpha=0.2, which='both'); ax.legend(frameon=False, fontsize=11)
axes[0].set_ylabel(r'fractional $1\sigma$ [%]', fontsize=12)
fig.suptitle('Fractional precision vs redshift', fontsize=14)
p = os.path.join(FIG, 'fig_growth_vs_z.png')
fig.savefig(p, dpi=170); plt.close(fig); print(f"saved {p}")

# =======================================================================================
# 3.  the LCDM recovery: the same Jacobian composed with the cosmology map
# =======================================================================================
import cosmo_fisher_pb as CF
from cosmo_fisher_pb import cosmo_fisher
theta_fid = CF.theta_fid

print(f"\nGATE: the 60-node basis represents the LCDM template response to "
      f"{' '.join(f'{v:.1e}' for v in CF.REP)} for (omega_cdm, lnAs, h)")
if CF.GATE:
    print(f"GATE: ||J_direct - J_MI M|| / ||J_direct|| = {CF.GATE['p']:.2e} (P), "
          f"{CF.GATE['pb']:.2e} (P+B)")

cov, sig = {}, {}
ROUTES = (('direct', 'direct LCDM fit'),
          ('through_mi_flat', 'through the MI model'),
          ('through_mi', 'through the MI model + its prior'))
print("\nLCDM recovery, EFT marginalized:")
print(f"{'':40s}{'omega_cdm':>12s}{'ln(1e10 As)':>14s}{'h':>10s}")
for key, lab, _ in KEYS:
    for route, rlab in ROUTES:
        c = np.linalg.inv(cosmo_fisher(key, route=route))
        cov[(key, route)] = c; sig[(key, route)] = np.sqrt(np.diag(c))
        print(f"  {lab:10s} {rlab:28s}" + "".join(f"{v:12.4f}" for v in sig[(key, route)]))
    r = sig[(key, 'through_mi_flat')] / sig[(key, 'direct')]
    print(f"  {'':10s} {'MI / direct':28s}" + "".join(f"{v:12.3f}" for v in r))
sp = np.sqrt(np.diag(np.linalg.inv(cosmo_fisher('p', route='prior_only'))))
print(f"  {'':10s} {'MI prior alone, no data':28s}" + "".join(f"{v:12.4f}" for v in sp))
print("\n  Row 1 vs row 2 is the check: the same data through the model-independent"
      "\n  parametrization returns the same LCDM errors, so the extra freedom neither adds nor"
      "\n  removes information. Row 3 is a warning, not a result: the MI prior is NOT flat along"
      "\n  the LCDM directions (last row), so compressing an MI posterior onto LCDM without"
      "\n  dividing that prior out tightens omega_cdm and A_s artificially.")

names = ['omega_cdm', 'logA', 'h']
labels = [r'\omega_{\rm cdm}', r'\ln(10^{10}A_s)', r'h']
ds, cols, lss, fills, legs = [], [], [], [], []
for route, rlab, ls in (('direct', 'direct $\\Lambda$CDM fit', '--'),
                        ('through_mi_flat', 'through the MI model', '-')):
    for key, lab, col in KEYS:
        ds.append(GaussianND(theta_fid, cov[(key, route)], names=names, labels=labels,
                             label=f'{lab}, {rlab}'))
        cols.append(col); lss.append(ls); fills.append(route == 'through_mi' and key == 'pb')
        legs.append(f'{lab}, {rlab}')
g = plots.get_subplot_plotter(subplot_size=2.9)
g.settings.num_plot_contours = 2; g.settings.alpha_filled_add = 0.30
g.settings.legend_fontsize = 13
g.triangle_plot(ds, params=names, filled=fills, contour_colors=cols, contour_ls=lss,
                line_args=[{'color': c, 'ls': l, 'lw': 2.0} for c, l in zip(cols, lss)],
                legend_labels=legs, markers={n: v for n, v in zip(names, theta_fid)})
g.fig.suptitle('$\\Lambda$CDM recovery', fontsize=15, y=1.02)
p = os.path.join(FIG, 'fig_cosmo_pb.png')
g.export(p); plt.close('all'); print(f"saved {p}")

np.savez(os.path.join(OUT, 'meeting_growth_pb.npz'), knots_h=M.knots_h,
         template_mpc=M.T_, prior_band=prior_band, zeff=zz,
         **{f'band_{k}': band[k] for k, _, _ in KEYS},
         **{f'band_wide_{k}': band_wide[k] for k, _, _ in KEYS},
         **{f'sig_f_{k}': sig_f[k] for k, _, _ in KEYS},
         **{f'sig_FAP_{k}': sig_fap[k] for k, _, _ in KEYS},
         **{f'cosmo_cov_{k}_{r}': cov[(k, r)] for k, _, _ in KEYS
            for r, _ in ROUTES}, theta_fid=theta_fid)
