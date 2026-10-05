"""Triangle plots for the Fisher information split (01_fisher_v3 / run_fisher_v3.py).

Single source for every version of the figure, so the paper figure and the notebook
diagnostics cannot drift apart. `run_fisher_v3.py` cell 23 calls `make_figures` with the
matrices it has just computed; running this file standalone regenerates all three from the
saved `output/fisher_v3/fisher_v3_results.npz`, which is much faster than re-executing the
notebook when only the plotting changes:

    srun -A a0158 -p debug --environment=~/.edf/pybird-jax.toml \
        bash -lc 'source ~/pybird/jax_env/bin/activate && python3 make_fisher_fig.py'

Outputs (in output/fisher_v3/, and paper/figs/ for the first):
  triangle_v3.png        the PAPER figure: four curves, matching the caption of
                         Fig.~\\ref{fig:fisher} -- Direct, Combined, template alone,
                         growth | template. No title, no diagnostics.
  triangle_v3_split.png  all six pieces (both sector marginals AND both chain
                         conditionals) at the scale of the direct fit.
  triangle_v3_wide.png   the same six on a window wide enough that the two sector
                         marginals are visible; at direct-fit scale both are wider than
                         the frame.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COSMO_FID = {'omega_cdm': 0.120, 'ln10^{10}A_s': 3.044, 'h': 0.675}
NAMES = ['omega_cdm', 'logA', 'h']
LABELS = [r'\omega_{\rm cdm}', r'\ln(10^{10}A_s)', r'h']

# Variance floor for the exactly-flat directions of the marginals. getdist builds its
# analytic density grid from the distribution's own sigma, so a floor orders of magnitude
# above the plot window renders as nothing at all; sigma = 3 is outside every window used
# here and still plots. The same floor is used for every curve so they stay comparable.
FLAT_SIGMA = 3.0

LIM_TIGHT = {"omega_cdm": [0.105, 0.135], "logA": [2.8, 3.3], "h": [0.62, 0.73]}
LIM_WIDE = {"omega_cdm": [0.06, 0.18], "logA": [2.3, 3.8], "h": [0.50, 0.85]}


def fisher_to_cov(Fm, rtol=1e-10, big_var=FLAT_SIGMA**2):
    """Fisher -> covariance, unconstrained directions floored to a LARGE variance."""
    Fm = 0.5 * (Fm + Fm.T)
    w, V = np.linalg.eigh(Fm)
    good = w > rtol * (float(np.abs(w).max()) or 1.0)
    return (V * np.where(good, 1.0 / np.where(good, w, 1.0), big_var)) @ V.T


def _triangle(pieces, param_limits, path, legend_fontsize=None, subplot_size=3.5):
    from getdist import plots
    from getdist.gaussian_mixtures import GaussianND
    means = [COSMO_FID['omega_cdm'], COSMO_FID['ln10^{10}A_s'], COSMO_FID['h']]
    dists = [GaussianND(means, fisher_to_cov(F), names=NAMES, labels=LABELS, label=nm)
             for nm, F, _, _, _ in pieces]
    g = plots.get_subplot_plotter(subplot_size=subplot_size)
    g.settings.num_plot_contours = 2
    g.settings.lw_contour = 2.0
    g.settings.alpha_filled_add = 0.2
    if legend_fontsize:
        g.settings.legend_fontsize = legend_fontsize
    g.triangle_plot(
        dists, params=NAMES, filled=[True] + [False] * (len(pieces) - 1),
        contour_colors=[c for _, _, c, _, _ in pieces],
        contour_ls=[l for _, _, _, l, _ in pieces],
        line_args=[{'color': c, 'lw': lw, 'ls': l} for _, _, c, l, lw in pieces],
        legend_labels=[nm for nm, _, _, _, _ in pieces],
        param_limits=param_limits,
        markers={'omega_cdm': COSMO_FID['omega_cdm'],
                 'logA': COSMO_FID['ln10^{10}A_s'], 'h': COSMO_FID['h']})
    # (title_limit is not supported for analytic GaussianND distributions)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close('all')
    return path


def make_figures(F, figdir, paper_figdir=None, log=print):
    """F: mapping with the six 3x3 cosmology Fishers (an npz or the notebook's locals)."""
    paper = [
        (r'Direct $\Lambda$CDM',      F['F_cosmo_direct'],   '#757575', '-', 2.0),
        ('MI, projected',             F['F_cosmo_combined'], '#1E88E5', '-', 2.5),
        ('Template alone',            F['F_pk_marg_3d'],     '#D32F2F', '-', 1.6),
        (r'Growth $|$ template',      F['F_growth_given_pk'], '#388E3C', '-', 1.6),
    ]
    six = paper[:2] + [
        ('P(k) marginal',       F['F_pk_marg_3d'],      '#D32F2F', '-',  1.5),
        ('Growth+AP marginal',  F['F_growth_marg_3d'],  '#7B1FA2', ':',  2.0),
        ('Growth+AP | P(k)',    F['F_growth_given_pk'], '#388E3C', '-',  1.5),
        ('P(k) | Growth+AP',    F['F_pk_given_growth'], '#F9A825', '--', 1.5),
    ]
    six[0] = ('Direct (Full)', six[0][1], six[0][2], six[0][3], six[0][4])
    six[1] = ('Combined (P(k) + Growth)', six[1][1], six[1][2], six[1][3], six[1][4])

    out = [_triangle(paper, LIM_TIGHT, os.path.join(figdir, 'triangle_v3.png'),
                     legend_fontsize=15),
           _triangle(six, LIM_TIGHT, os.path.join(figdir, 'triangle_v3_split.png')),
           _triangle(six, LIM_WIDE, os.path.join(figdir, 'triangle_v3_wide.png'))]
    for p in out:
        log(f"saved {p}")
    if paper_figdir:
        import shutil
        dst = os.path.join(paper_figdir, 'triangle_v3.png')
        shutil.copyfile(out[0], dst)
        log(f"copied the paper figure to {dst}")
    return out


if __name__ == '__main__':
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    figdir = os.path.join(root, 'output', 'fisher_v3')
    make_figures(np.load(os.path.join(figdir, 'fisher_v3_results.npz')), figdir,
                 paper_figdir=os.path.join(root, 'paper', 'figs'))
