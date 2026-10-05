"""Figures for the P(k) vs P(k)+B(k) Fisher comparison (run_fisher_pb.py / 06_fisher_pk_bk).

Single source for every figure, so the notebook diagnostics and the saved figures cannot
drift apart. `run_fisher_pb.py` cell 23 calls `make_figures` with the dict it has just
computed; running this file standalone regenerates everything from the saved npz, which is
much faster than re-executing the notebook when only the plotting changes:

    srun -A a0158 -p debug --environment=~/.edf/pybird-jax.toml \
        bash -lc 'source ~/pybird/jax_env/bin/activate && python3 make_fisher_pb_fig.py'

Outputs (in output/fisher_pb/):
  corner_growth_z<z>.png   per redshift: the growth/AP triple (f, H/H0, D_A H0) in
                           FRACTIONAL units, P vs P+B, TEMPLATE HELD FIXED.
  corner_ap_all_z.png      all redshifts at once: (H/H0, D_A H0), P vs P+B, template fixed.
  corner_cosmo_pb.png      (omega_cdm, lnAs, h): Direct and Growth|P(k), P vs P+B.
  ap_summary.png           fractional sigma vs z for f, H, D_A and F_AP = H D_A, with the
                           P+B / P ratio underneath.
  growth_spectrum.png      the cap-free statement for the TEMPLATE-FREE block: the sorted
                           eigen-spectrum of the growth Fisher, P vs P+B.

Why the template-free case gets a spectrum and not a corner. Once the 80 template amplitudes
are marginalized, the growth block has directions that are flat for all practical purposes
(the a*D^2 invariant and the h_conv-dilation family). A covariance therefore only exists once
those directions are truncated at some sigma_max, and the truncation is not innocent: it is
performed along each data set's OWN eigenvectors, so it can and does destroy the ordering
that Gate 7 guarantees for the Fisher matrices, letting the P(k)+B(k) contour render WIDER
than the P(k) one. Measured for this run: no choice of cap in [0.5, 20] preserves the
ordering of the template-free (H, D_A) block, and the P(k)-alone widths drift by a factor of
6 across that range while the P(k)+B(k) ones move by 50%. (For the template-FIXED block a cap
of 5 truncates nothing and the ordering does hold, which is why those corners are drawn.) That drift IS the result -- P(k)
alone does not determine the AP sector with a free template -- and the eigen-spectrum states
it without a truncation, so that is what the figure shows.

Color encodes the DATA SET (P orange, P+B blue) and line style the CONDITIONING of the
template (solid = held at the fiducial, dashed = marginalized), so a reader can see at a
glance which of the two comparisons a difference belongs to. Exactly one curve per figure is
filled -- P(k)+B(k) with the template fixed, the headline result -- because two overlapping
semi-transparent fills blend into a third colour and destroy the identity encoding.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fisher_utils import fisher_to_cov, fisher_to_cov_capped, psd_clip

# --- palette: identity = data set; validated for CVD separation against a light surface ---
C_P, C_PB = '#E8710A', '#1E88E5'      # P(k) only, P(k)+B(k)
C_INK, C_MUTED = '#1A1A1A', '#6E6E6E'

# Cap on the width of an unconstrained direction. getdist builds its analytic grid from the
# distribution's own sigma, so a direction floored far above the plot window renders as
# nothing at all. The growth block is in FRACTIONAL units, so the cap is a percentage: a
# cap must be large enough that nothing real is truncated -- capping at 1.0 pulled sigma(f) at
# z=1.491 from its true 3.00 down to 0.71 for P(k) alone, inverting the comparison at the very
# redshift where the bispectrum helps most. COV_CAP and MEASURED are the same two values as in
# run_fisher_pb.py, so the figures and the printed tables cannot disagree. Panels are kept
# readable with explicit plot limits instead, which clip the view without touching the
# distribution.
COV_CAP = 5.0        # only so a covariance exists; nothing real is truncated at this value
MEASURED = 1.0       # a fractional width worse than this is not a measurement
FLAT_SIGMA = COV_CAP # backwards-compatible alias

COSMO_FID = {'omega_cdm': 0.120, 'ln10^{10}A_s': 3.044, 'h': 0.675}
COSMO_NAMES = ['omega_cdm', 'logA', 'h']
COSMO_LABELS = [r'\omega_{\rm cdm}', r'\ln(10^{10}A_s)', r'h']
LIM_TIGHT = {"omega_cdm": [0.105, 0.135], "logA": [2.8, 3.3], "h": [0.62, 0.73]}
# fractional-deviation windows for the growth corners: wide enough to show the P(k) curve
# where it is measured, and to show it running off the frame where it is not.
LIM_GROWTH = {'dlnf': [-1.5, 1.5], 'dlnH': [-0.35, 0.35], 'dlnDA': [-0.35, 0.35]}


def _sub_fisher(F_ln, idx):
    """Fisher of a subset of the fractional growth parameters, the rest MARGINALIZED.

    Marginalizing (not conditioning) is the only correct reduction here: the parameters left
    out are constrained by the same data and correlated with the ones kept."""
    idx = np.asarray(idx)
    cov = fisher_to_cov_capped(F_ln, COV_CAP)
    return cov[np.ix_(idx, idx)]


def _corner(dists, names, labels, path, markers=None, param_limits=None, subplot_size=2.6,
            legend_fontsize=14, title=None):
    """dists: (label, mean, cov, color, linestyle, fill) tuples.

    Exactly ONE curve is filled -- the headline P(k)+B(k) result. Two overlapping
    semi-transparent fills blend into a third colour and destroy the identity encoding, so
    everything else is drawn as an outline."""
    from getdist import plots
    from getdist.gaussian_mixtures import GaussianND
    gd = [GaussianND(mean, cov, names=names, labels=labels, label=nm)
          for nm, mean, cov, _, _, _ in dists]
    g = plots.get_subplot_plotter(subplot_size=subplot_size)
    g.settings.num_plot_contours = 2
    g.settings.lw_contour = 2.0
    g.settings.alpha_filled_add = 0.30
    g.settings.legend_fontsize = legend_fontsize
    g.settings.axes_fontsize = 11
    g.settings.axes_labelsize = 14
    filled = [f for _, _, _, _, _, f in dists]
    kwargs = dict(
        params=names, filled=filled,
        contour_colors=[c for _, _, _, c, _, _ in dists],
        contour_ls=[ls for _, _, _, _, ls, _ in dists],
        line_args=[{'color': c, 'ls': ls, 'lw': 2.4 if f else 2.0}
                   for _, _, _, c, ls, f in dists],
        legend_labels=[nm for nm, _, _, _, _, _ in dists], markers=markers)
    # getdist dereferences param_limits unconditionally, so it must be a dict, not None
    if param_limits is not None:
        kwargs['param_limits'] = param_limits
    g.triangle_plot(gd, **kwargs)
    if title:
        g.fig.suptitle(title, fontsize=15, y=1.02)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close('all')
    return path


def _growth_dists(R, idx, which_list=('template fixed', 'template free')):
    """(label, mean, cov, color, ls, fill) entries for the fractional growth parameters `idx`.
    The filled curve is P(k)+B(k) with the template fixed -- the headline result."""
    out = []
    for which in which_list:
        key = which.replace(' ', '_')
        for tag, key_tag, color in [('P(k)', 'p', C_P), ('P(k)+B(k)', 'pb', C_PB)]:
            cov = _sub_fisher(R[f'growth_ln_{key}_{key_tag}'], idx)
            ls = '-' if which == 'template fixed' else '--'
            fill = (tag == 'P(k)+B(k)' and which == 'template fixed')
            out.append((f'{tag}, {which}', np.zeros(len(idx)), cov, color, ls, fill))
    return out


def make_corner_growth_per_z(R, figdir, log=print):
    """One 3x3 corner per redshift: fractional (f, H/H0, D_A H0), template held fixed.

    Template-free contours are deliberately NOT drawn here -- see the module docstring: that
    block is degenerate, so its contours depend on the truncation rather than on the data.
    `make_growth_spectrum` carries the template-free statement instead."""
    zeff = np.atleast_1d(R['zeff_unique'])
    names = ['dlnf', 'dlnH', 'dlnDA']
    labels = [r'\Delta f/f', r'\Delta H/H', r'\Delta D_A/D_A']
    paths = []
    for iz, z in enumerate(zeff):
        idx = [int(R['idx_f'][iz]), int(R['idx_H'][iz]), int(R['idx_DA'][iz])]
        dists = _growth_dists(R, idx, which_list=('template fixed',))
        _check_ordering(R, idx, 'template fixed', z, log)
        p = os.path.join(figdir, f'corner_growth_z{z:.3f}.png'.replace('.', 'p', 1))
        _corner(dists, names, labels, p, markers={n: 0.0 for n in names}, subplot_size=3.0,
                param_limits=LIM_GROWTH,
                title=rf'Growth / AP sector at $z = {z:.3f}$, template held fixed '
                      r'(EFT marginalized)')
        paths.append(p)
        log(f"saved {p}")
    return paths


def _check_ordering(R, idx, which, z, log):
    """Gate 7 says F(P+B) - F(P) is PSD, so every P+B contour must lie inside its P
    counterpart. Truncating flat directions can break that, which would draw a misleading
    figure; this verifies it for the block actually plotted."""
    key = which.replace(' ', '_')
    idx = np.asarray(idx)
    a = fisher_to_cov_capped(R[f'growth_ln_{key}_p'], COV_CAP)[np.ix_(idx, idx)]
    b = fisher_to_cov_capped(R[f'growth_ln_{key}_pb'], COV_CAP)[np.ix_(idx, idx)]
    w = np.linalg.eigvalsh(0.5 * ((a - b) + (a - b).T))
    if w.min() < -1e-10 * max(np.abs(w).max(), 1.0):
        log(f"  WARNING z={z}: cov(P) - cov(P+B) has eigenvalue {w.min():.2e} < 0 -- the "
            f"truncation has broken the Gate 7 ordering, contours may mislead")


def make_growth_spectrum(R, figdir, log=print):
    """Sorted eigen-spectrum of the fractional growth Fisher: how many directions of the
    growth/AP block are measured, and how well, for each data set and each conditioning.

    This is the cap-free version of the template-free comparison: eigenvalues of the Fisher
    need no truncation, so nothing here depends on a plotting choice."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout='constrained')
    for ax, which in zip(axes, ['template fixed', 'template free']):
        key = which.replace(' ', '_')
        for tag, key_tag, color, marker in [('P(k)', 'p', C_P, 'o'),
                                            ('P(k)+B(k)', 'pb', C_PB, 's')]:
            w = np.linalg.eigvalsh(R[f'growth_ln_{key}_{key_tag}'])
            sig = 1.0 / np.sqrt(np.clip(w, 1e-300, None))
            sig = np.sort(sig)                       # best-measured direction first
            ax.plot(np.arange(1, len(sig) + 1), np.minimum(sig, 1e3), marker=marker, ms=5,
                    color=color, lw=2, label=tag)
        n_dir = len(sig)
        ax.axhline(MEASURED, color=C_MUTED, lw=1.5, ls=':')
        ax.text(n_dir * 0.02, MEASURED * 1.25, 'unconstrained above this line',
                color=C_MUTED, fontsize=10)
        ax.set_yscale('log')
        ax.set_xlabel('eigen-direction of the growth / AP block (best measured first)',
                      fontsize=12)
        ax.set_title(which, fontsize=14, color=C_INK)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel(r'fractional $\sigma$ of the direction', fontsize=13)
    axes[0].legend(frameon=False, fontsize=12)
    fig.suptitle('How many growth / AP directions the data actually measure', fontsize=15)
    p = os.path.join(figdir, 'growth_spectrum.png')
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.close('all')
    log(f"saved {p}")
    for which in ['template fixed', 'template free']:
        key = which.replace(' ', '_')
        counts = {}
        for tag, key_tag in [('P', 'p'), ('P+B', 'pb')]:
            w = np.linalg.eigvalsh(R[f'growth_ln_{key}_{key_tag}'])
            counts[tag] = int((w > 1.0 / MEASURED**2).sum())
        log(f"  {which}: directions measured to better than {MEASURED:g} -- "
            f"P {counts['P']}, P+B {counts['P+B']} (of {len(w)})")
    return p


def make_corner_ap_all_z(R, figdir, log=print):
    """All redshifts at once: fractional (H/H0, D_A H0), template held fixed."""
    zeff = np.atleast_1d(R['zeff_unique'])
    idx, names, labels = [], [], []
    for iz, z in enumerate(zeff):
        idx += [int(R['idx_H'][iz]), int(R['idx_DA'][iz])]
        names += [f'dlnH{iz}', f'dlnDA{iz}']
        labels += [rf'\Delta H/H({z:.2f})', rf'\Delta D_A/D_A({z:.2f})']
    dists = _growth_dists(R, idx, which_list=('template fixed',))
    _check_ordering(R, idx, 'template fixed', 'all', log)
    p = os.path.join(figdir, 'corner_ap_all_z.png')
    _corner(dists, names, labels, p, markers={n: 0.0 for n in names},
            subplot_size=1.5, legend_fontsize=16,
            title='AP parameters at every redshift, template held at the fiducial')
    log(f"saved {p}")
    return p


def make_corner_cosmo(R, figdir, log=print):
    """(omega_cdm, lnAs, h): the full direct fit and the growth-conditional, P vs P+B."""
    mean = [COSMO_FID['omega_cdm'], COSMO_FID['ln10^{10}A_s'], COSMO_FID['h']]
    dists = []
    for nm, key, ls in [('Direct $\\Lambda$CDM', 'Direct', '-'),
                        ('Growth+AP $|$ P(k)', 'GrowthAP_given_Pk', '--')]:
        for tag, key_tag, color in [('P(k)', 'p', C_P), ('P(k)+B(k)', 'pb', C_PB)]:
            F = R[f'cosmo_{key}_{key_tag}']
            dists.append((f'{tag}, {nm}', mean, fisher_to_cov(psd_clip(F), big_var=9.0),
                          color, ls, tag == 'P(k)+B(k)' and ls == '-'))
    p = os.path.join(figdir, 'corner_cosmo_pb.png')
    _corner(dists, COSMO_NAMES, COSMO_LABELS, p, param_limits=LIM_TIGHT, subplot_size=3.0,
            markers={'omega_cdm': COSMO_FID['omega_cdm'], 'logA': COSMO_FID['ln10^{10}A_s'],
                     'h': COSMO_FID['h']})
    log(f"saved {p}")
    return p


def make_ap_summary(R, figdir, log=print):
    """Fractional sigma vs z for f, H, D_A, F_AP, with the P+B / P ratio underneath."""
    zeff = np.atleast_1d(R['zeff_unique'])
    quantities = [('f', r'$f$'), ('H', r'$H/H_0$'), ('DA', r'$D_A H_0$'), ('FAP', r'$F_{\rm AP}=H D_A$')]
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), sharex=True,
                             gridspec_kw={'height_ratios': [2.4, 1]}, layout='constrained')
    for j, (q, qlabel) in enumerate(quantities):
        ax, axr = axes[0, j], axes[1, j]
        for which, ls, marker in [('template fixed', '-', 'o'), ('template free', '--', 's')]:
            key = which.replace(' ', '_')
            sp = np.atleast_1d(R[f'sig_{q}_{key}_p'])
            spb = np.atleast_1d(R[f'sig_{q}_{key}_pb'])
            ax.plot(zeff, sp, ls=ls, marker=marker, ms=6, color=C_P, lw=2,
                    label=f'P(k), {which}')
            ax.plot(zeff, spb, ls=ls, marker=marker, ms=6, color=C_PB, lw=2,
                    label=f'P(k)+B(k), {which}')
            axr.plot(zeff, spb / sp, ls=ls, marker=marker, ms=6, color=C_INK, lw=2)
        ax.set_yscale('log')
        ax.set_title(qlabel, fontsize=14, color=C_INK)
        ax.grid(alpha=0.2)
        axr.axhline(1.0, color=C_MUTED, lw=1)
        axr.grid(alpha=0.2)
        axr.set_xlabel(r'$z$', fontsize=13)
        axr.set_ylim(0, 1.15)
        if j == 0:
            ax.set_ylabel(r'fractional $\sigma$', fontsize=13)
            axr.set_ylabel(r'$\sigma_{P+B}/\sigma_{P}$', fontsize=13)
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle('Growth / AP precision, P(k) alone vs P(k) + tree-level B(k)', fontsize=15)
    p = os.path.join(figdir, 'ap_summary.png')
    plt.savefig(p, dpi=200, bbox_inches='tight')
    plt.close('all')
    log(f"saved {p}")
    return p


def make_figures(R, figdir, log=print):
    """Figures of the Hessian-route analysis that are still valid: only the direct-LCDM corner.

    The per-redshift growth/AP corners, the all-z AP corner, the AP summary and the growth
    eigen-spectrum that this module used to write are SUPERSEDED (2026-09-11): with the template
    fixed and h_conv free they showed the discretization breaking of the exact ruler symmetry g1
    (sigma ~0.05) as if it were a measurement of H and D_A. Their corrected versions are built by
    corner_audit_pb.py, hmc_prior_bands.py and bao_basis_scan.py (see 07_robust_pk_bk.ipynb and
    output/fisher_pb/superseded/README.md). The functions are kept for the record, not called."""
    out = [make_corner_cosmo(R, figdir, log=log)]
    log("per-z growth/AP corners, all-z AP corner, AP summary and growth spectrum are superseded: "
        "see 07_robust_pk_bk.ipynb")
    return out


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    figdir = os.path.join(root, 'output', 'fisher_pb')
    make_figures(np.load(os.path.join(figdir, 'fisher_pb_results.npz'), allow_pickle=True), figdir)
