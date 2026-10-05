"""Where the cosmological information comes from, in a basis that makes it legible.

Run standalone against output/fisher_v3/fisher_v3_results.npz (seconds):

    sbatch information_story.sbatch

The point of the figure. In (omega_cdm, lnAs, h) the two MI sectors overlap confusingly:
the template constrains omega_cdm and the growth sector looks like it constrains "some
combination of omega_cdm and h". Trading h for Omega_m = (omega_cdm + omega_b)/h^2
DIAGONALIZES the split, because each sector measures exactly one of them:

  * the TEMPLATE is the linear P(k) shape in Mpc at fixed 1/Mpc wavenumbers -- the
    equality turnover k_eq ~ omega_m and the sound horizon r_d(omega_b, omega_m) in Mpc.
    Pure early-universe physics, and dimensionless in the sense that matters: it needs no
    knowledge of the expansion history or of the units the data are measured in. It gives
    omega_m and nothing else.
  * the GROWTH/AP sector is dimensionless late-universe geometry: the AP anisotropy
    H(z) D_A(z), the RSD amplitude f(z), and the relative evolution of D(z) between
    redshifts. In LCDM all three depend on Omega_m alone. It gives Omega_m and nothing else.

h is then not measured by either sector -- it is the RATIO h = sqrt(omega_m / Omega_m),
i.e. an early-universe quantity divided by a late-universe one. Combining the two sector
marginals as independent measurements reproduces the measured product-of-marginals
sigma(h) to four digits, which is the check that this reading is the right one.

The full fit beats that by ~10x on h, and the extra is the STANDARD RULER: the template
fixes a physical length in Mpc, the data locate the same feature in h/Mpc, and the ratio is
h directly. That comparison needs both sectors at once, which is why it lives in the
cross-correlation and in neither marginal. A_s is the same kind of quantity: the data see
only A_s D(z)^2 h^3 b^2, so the template's normalization is degenerate with the growth
amplitudes, and A_s is unconstrained by either sector alone AND by their product.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

OMEGA_B, OMEGA_CDM_FID, H_FID, LNAS_FID = 0.02235, 0.120, 0.675, 3.044
OM_FID = (OMEGA_CDM_FID + OMEGA_B) / H_FID**2
WM_FID = OMEGA_CDM_FID + OMEGA_B

C_DIRECT, C_TEMPLATE, C_GROWTH, C_PRODUCT = '#424242', '#D32F2F', '#7B1FA2', '#1565C0'


def fisher_to_cov(F, rtol=1e-10, big_var=1e6):
    F = 0.5 * (F + F.T)
    w, V = np.linalg.eigh(F)
    good = w > rtol * (float(np.abs(w).max()) or 1.0)
    return (V * np.where(good, 1.0 / np.where(good, w, 1.0), big_var)) @ V.T


def to_Om_basis(F):
    """(omega_cdm, lnAs, h) -> (omega_cdm, lnAs, Omega_m), h = sqrt(omega_m/Omega_m)."""
    M = np.eye(3)
    M[2, 0] = 1.0 / (2 * H_FID * OM_FID)    # dh/d omega_cdm at fixed Omega_m
    M[2, 2] = -H_FID / (2 * OM_FID)         # dh/d Omega_m at fixed omega_cdm
    return M.T @ F @ M


# 68% / 95% of a 2-D Gaussian
LEVELS = [2.2958, 5.9915]


def region(ax, cov2, centre, color, fill, ls='-', lw=2.0, alpha=(0.28, 0.14), n=420):
    """Draw the 68/95% region of a 2-D Gaussian by contouring its chi^2 on the axes grid.

    Contouring rather than drawing an Ellipse patch is what makes an UNCONSTRAINED
    direction render honestly: its curvature is ~0, so the contours come out as a straight
    band across the frame instead of a giant ellipse clipped into a pair of stray lines.
    """
    x = np.linspace(*ax.get_xlim(), n)
    y = np.linspace(*ax.get_ylim(), n)
    X, Y = np.meshgrid(x, y)
    P = np.linalg.inv(cov2)          # marginal precision; flat directions -> ~0 curvature
    dx, dy = X - centre[0], Y - centre[1]
    chi2 = P[0, 0] * dx**2 + 2 * P[0, 1] * dx * dy + P[1, 1] * dy**2
    if fill:
        ax.contourf(X, Y, chi2, levels=[0] + LEVELS, colors=[color, color],
                    alpha=alpha[0], zorder=2)
        ax.contourf(X, Y, chi2, levels=[LEVELS[0], LEVELS[1]], colors=[color],
                    alpha=alpha[1], zorder=2)
    ax.contour(X, Y, chi2, levels=LEVELS, colors=color, linewidths=lw,
               linestyles=ls, zorder=4)


def make_figure(R, outpath, log=print):
    F = {'direct': R['F_cosmo_direct'], 'template': R['F_pk_marg_3d'],
         'growth': R['F_growth_marg_3d']}
    F['product'] = F['template'] + F['growth']
    covH = {k: fisher_to_cov(v) for k, v in F.items()}                  # (wc, lnAs, h)
    covOm = {k: fisher_to_cov(to_Om_basis(v)) for k, v in F.items()}    # (wc, lnAs, Om)
    idx = np.ix_([0, 2], [0, 2])

    L_T = 'Template alone  (early universe)'
    L_G = 'Growth + AP alone  (late universe)'
    L_P = 'Both sectors combined'
    L_D = 'Full fit  (adds the standard ruler)'

    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2),
                             gridspec_kw={'width_ratios': [1.1, 1.1, 1.15], 'wspace': 0.30})

    # ---- (a) (omega_cdm, h): neither sector alone constrains h ----------------
    ax = axes[0]
    ax.set_xlim(0.104, 0.136); ax.set_ylim(0.60, 0.76)
    c = (OMEGA_CDM_FID, H_FID)
    region(ax, covH['template'][idx], c, C_TEMPLATE, True)
    region(ax, covH['growth'][idx], c, C_GROWTH, True)
    region(ax, covH['product'][idx], c, C_PRODUCT, False, ls='--', lw=1.8)
    region(ax, covH['direct'][idx], c, C_DIRECT, True, alpha=(0.55, 0.30), lw=1.4)
    bb = dict(boxstyle='round,pad=0.3', fc='white', ec='0.75', alpha=0.92)
    ax.annotate('no $h$ at all:\nthe band is vertical', xy=(0.1252, 0.647),
                xytext=(0.1288, 0.614), fontsize=9.5, color=C_TEMPLATE, ha='center',
                bbox=bb, arrowprops=dict(arrowstyle='->', color=C_TEMPLATE, lw=1.2))
    ax.annotate('no $h$ either: only\n$\\Omega_m$, i.e. $\\omega_m\\propto h^2$',
                xy=(0.1078, 0.676), xytext=(0.1108, 0.744), fontsize=9.5,
                color=C_GROWTH, ha='center', bbox=bb,
                arrowprops=dict(arrowstyle='->', color=C_GROWTH, lw=1.2))
    ax.set_xlabel(r'$\omega_{\rm cdm}$'); ax.set_ylabel(r'$h$')
    ax.set_title('(a)  neither sector alone measures $h$', fontsize=12)

    # ---- (b) (omega_cdm, Omega_m): the split is orthogonal here ---------------
    ax = axes[1]
    ax.set_xlim(0.104, 0.136); ax.set_ylim(0.22, 0.41)
    c = (OMEGA_CDM_FID, OM_FID)
    region(ax, covOm['template'][idx], c, C_TEMPLATE, True)
    region(ax, covOm['growth'][idx], c, C_GROWTH, True)
    region(ax, covOm['product'][idx], c, C_PRODUCT, False, ls='--', lw=1.8)
    region(ax, covOm['direct'][idx], c, C_DIRECT, True, alpha=(0.55, 0.30), lw=1.4)
    ax.set_xlabel(r'$\omega_{\rm cdm}$   —   $P(k)$ shape in Mpc ($k_{\rm eq}$, $r_d$)')
    ax.set_ylabel(r'$\Omega_m$   —   AP, RSD, $D(z)$ evolution')
    ax.set_title(r'(b)  each sector owns one axis;  $h=\sqrt{\omega_m/\Omega_m}$',
                 fontsize=12)

    # ---- (c) the sigma(h) ladder ---------------------------------------------
    ax = axes[2]
    s_t = np.sqrt(covH['template'][2, 2]); s_g = np.sqrt(covH['growth'][2, 2])
    s_p = np.sqrt(covH['product'][2, 2]); s_d = np.sqrt(covH['direct'][2, 2])
    rows = [('Template alone', s_t, C_TEMPLATE, True),
            ('Growth + AP alone', s_g, C_GROWTH, False),
            ('Both combined', s_p, C_PRODUCT, False),
            ('Full fit', s_d, C_DIRECT, False)]
    XLO = 2e-3
    ax.set_xscale('log'); ax.set_xlim(XLO, 6.0); ax.set_ylim(3.6, -0.6)
    for i, (nm, sg, col, unbounded) in enumerate(rows):
        if unbounded:
            ax.hlines(i, XLO, 0.9, color=col, lw=4, alpha=0.5)
            ax.annotate('', xy=(5.5, i), xytext=(0.9, i),
                        arrowprops=dict(arrowstyle='-|>', color=col, lw=2.5))
            ax.text(1.15, i - 0.17, 'unconstrained', color=col, fontsize=10.5)
        else:
            ax.hlines(i, XLO, sg, color=col, lw=4, alpha=0.5)
            ax.plot([sg], [i], 'o', color=col, ms=9)
            ax.text(sg * 1.35, i + 0.02, '%.3g' % sg, va='center', fontsize=11.5, color=col)
    ax.set_yticks(range(4))
    ax.set_yticklabels([r[0] for r in rows], fontsize=11)
    for tick, r in zip(ax.get_yticklabels(), rows):
        tick.set_color(r[2])
    ax.tick_params(axis='y', length=0)
    ax.text(1.05, 1.42, r'$\times%.0f$' % (s_g / s_p) + '\ncombining\nearly $+$ late',
            fontsize=10.5, color='0.25', ha='left', va='center')
    ax.text(0.105, 2.42, r'$\times%.1f$' % (s_p / s_d) + '\nthe standard\nruler',
            fontsize=10.5, color='0.25', ha='left', va='center')
    for y0, y1 in ((1, 2), (2, 3)):
        ax.annotate('', xy=(rows[y1][1], y1 - 0.22), xytext=(rows[y0][1], y0 + 0.22),
                    arrowprops=dict(arrowstyle='-|>', color='0.45', lw=1.5,
                                    connectionstyle='arc3,rad=0.30'))
    ax.set_xlabel(r'$\sigma(h)$'); ax.grid(axis='x', alpha=0.25)
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    ax.set_title(r'(c)  where the $h$ information comes from', fontsize=12)

    for a_ in axes[:2]:
        a_.tick_params(labelsize=9)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C_TEMPLATE, lw=3, label=L_T),
               Line2D([], [], color=C_GROWTH, lw=3, label=L_G),
               Line2D([], [], color=C_PRODUCT, lw=2, ls='--', label=L_P),
               Line2D([], [], color=C_DIRECT, lw=3, label=L_D)]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=11.5,
               frameon=False, bbox_to_anchor=(0.5, -0.055))
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    log(f"saved {outpath}")

    st = np.sqrt(np.diag(fisher_to_cov(to_Om_basis(F['template']))))
    sg2 = np.sqrt(np.diag(fisher_to_cov(to_Om_basis(F['growth']))))
    rel_wm, rel_Om = st[0] / WM_FID, sg2[2] / OM_FID
    pred = 0.5 * np.hypot(rel_wm, rel_Om) * H_FID
    sA_p = np.sqrt(covH['product'][1, 1]); sA_d = np.sqrt(covH['direct'][1, 1])
    log("--- the chain, in words and numbers -------------------------------------")
    log(f"   sigma(h), template alone          = {s_t:.3g}    (unconstrained)")
    log(f"   sigma(h), growth+AP alone         = {s_g:.4f}   (unconstrained: omega_m ~ h^2)")
    log(f"1. template (early) -> omega_m      : sigma/omega_m = {rel_wm:.4f}")
    log(f"2. growth+AP (late) -> Omega_m      : sigma/Omega_m = {rel_Om:.4f}")
    log(f"3. h = sqrt(omega_m/Omega_m)        : predicted sigma(h) = {pred:.4f}")
    log(f"   measured, sectors combined       :           sigma(h) = {s_p:.4f}"
        f"  <- agrees to {abs(pred/s_p-1)*100:.2f}%")
    log(f"4. full fit (adds the ruler)        : sigma(h) = {s_d:.5f}"
        f"  -> ruler gain {s_p/s_d:.1f}x")
    log(f"5. lnAs: sectors combined {sA_p:.2f} (nothing); full fit {sA_d:.4f}")
    return outpath


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    figdir = os.path.join(root, 'output', 'fisher_v3')
    make_figure(np.load(os.path.join(figdir, 'fisher_v3_results.npz')),
                os.path.join(figdir, 'information_story.png'))
