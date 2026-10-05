"""fig_bao_triangle_pb_overlay.png: P(k)+B(k) BAO recovery with and without the BAO phase prior,
in ONE triangle (D_M/r_d and D_H/r_d at every z). Reuses meeting_bao_pb.py up to its scan section."""
import os
_here = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_here, 'meeting_bao_pb.py')).read()
_head = _src.split('# 1.  the scan')[0].rsplit('# ====', 1)[0]
__file__ = os.path.join(_here, 'meeting_bao_pb.py')
exec(compile(_head, 'meeting_bao_pb.py[head]', 'exec'))
from getdist import plots
from getdist.gaussian_mixtures import GaussianND
free_sy, phase_sy = systems(), systems(phase_sigma=PHASE_REF)
names, labels, qty = [], [], []
for iz, z in enumerate(zz):
    names += [f'perp{iz}', f'par{iz}']
    labels += [rf'D_M/r_d\,({z:.2f})', rf'D_H/r_d\,({z:.2f})']
    qty += [('perp', iz), ('par', iz)]
ds, cols = [], ['#c0392b', '#1f77d4']
for sy, lab in ((free_sy, 'P(k)+B(k), no phase prior (sampled model)'),
                (phase_sy, f'P(k)+B(k), phase prior {100*PHASE_REF:.1f}%')):
    cs = np.array([alpha_c(sy['pb'], q, iz) for q, iz in qty])
    ds.append(GaussianND(np.zeros(12), cs @ sy['pb']['C'] @ cs.T, names=names, labels=labels, label=lab))
g = plots.get_subplot_plotter(subplot_size=1.25)
g.settings.num_plot_contours = 2
g.settings.alpha_filled_add = 0.30
g.settings.legend_fontsize = 17
g.settings.axes_fontsize = 9
g.settings.axes_labelsize = 13
g.triangle_plot(ds, params=names, filled=[True, True], contour_colors=cols,
                line_args=[{'color': c, 'lw': 2.0} for c in cols],
                legend_labels=[d.label for d in ds], markers={n: 0.0 for n in names})
g.fig.suptitle('BAO recovery with P(k)+B(k): effect of the wiggle-phase prior', fontsize=22, y=1.045)
p = os.path.join(FIG, 'fig_bao_triangle_pb_overlay.png')
g.export(p); print(f"saved {p}")
for sy, t in ((free_sy, 'free'), (phase_sy, 'phase')):
    print(t, 'z=0.71  perp %.2f%%  par %.2f%%' % (100*sig_q(sy['pb'], 'perp', IZ), 100*sig_q(sy['pb'], 'par', IZ)))
