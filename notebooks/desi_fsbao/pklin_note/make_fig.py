"""Figure for pklin_model.tex: the fixed template, its split, and what alpha_rs and e0 do to the wiggles.
Reads the split saved by wiggle_gates.py (output/desi_fsbao/wiggle_gates/gates.npz) and the Ae245 node grid."""
import os, json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
O_ROOT = '/capstor/store/cscs/swissai/a0158/areeves/pybird_model_independent/output/desi_fsbao'
G = np.load(os.path.join(O_ROOT, 'wiggle_gates', 'gates.npz'))
k, O, T = G['split_k'], G['split_O'], G['split_P']; Tnw = T / (1 + O)
h = 0.6736
nodes = np.array(json.load(open(os.path.join(O_ROOT, 'wiggle_Be245', 'meta.json')))['nodes_h']) * h
Oat = lambda q: np.interp(np.log(q), np.log(k), O, left=0.0, right=0.0)
plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
m = (k > 1e-3) & (k < 0.5); data = (0.02 * h, 0.2 * h)
ax[0].loglog(k[m], T[m], color='k', lw=1.5, label=r'$T(k)$: fiducial $P_{\rm lin}$ at $z_N$')
ax[0].loglog(k[m], Tnw[m], color='0.55', lw=1.5, ls='--', label=r'$T_{\rm nw}(k)$: its smooth part')
for x in nodes: ax[0].axvline(x, color='#1E88E5', lw=0.6, alpha=0.6, zorder=0)
ax[0].plot([], [], color='#1E88E5', lw=0.6, label='the 21 nodes of $a(k)$')
ax[0].set_xlim(1e-3, 0.5); ax[0].set_xlabel(r'$k\ [{\rm Mpc}^{-1}]$'); ax[0].set_ylabel(r'$[{\rm Mpc}^3]$'); ax[0].legend(fontsize=9, loc='lower left')
ax[0].set_title('(a) fixed template (fiducial cosmology)')
mm = (k > 5e-3) & (k < 0.4)
for a_, ls in ((1.0, 0), (0.97, 1), (1.03, 2)):
    c, lab = [('k', r'$O(k)$, $\alpha_{r_s}=1$ (fiducial)'), ('#E8710A', r'$O(\alpha_{r_s}k)$, $\alpha_{r_s}=0.97$'), ('#1E88E5', r'$O(\alpha_{r_s}k)$, $\alpha_{r_s}=1.03$')][ls]
    ax[1].semilogx(k[mm], Oat(k[mm] * a_), color=c, lw=1.5 if ls == 0 else 1.1, label=lab)
ax[1].set_xlabel(r'$k\ [{\rm Mpc}^{-1}]$'); ax[1].set_ylabel(r'$O$'); ax[1].legend(fontsize=9); ax[1].set_title(r'(b) $\alpha_{r_s}$ slides the wiggles')
ax[2].semilogx(k[mm], O[mm], color='k', lw=1.5, label=r'$O(k)$: $A=0$, $\Delta\Sigma^2=0$ (fiducial)')
for A_, c in ((-0.3, '#E8710A'), (0.3, '#1E88E5')):
    ax[2].semilogx(k[mm], (1 + A_) * O[mm], color=c, lw=1.0, ls='--', label=rf'$A={A_:+.1f}$')
for S2, c in ((25.0, '#E8710A'), (-25.0, '#1E88E5')):
    ax[2].semilogx(k[mm], np.exp(-0.5 * k[mm]**2 * S2) * O[mm], color=c, lw=1.1, label=rf'$\Delta\Sigma^2={S2:+.0f}\,{{\rm Mpc}}^2$')
ax[2].set_xlabel(r'$k\ [{\rm Mpc}^{-1}]$'); ax[2].set_ylabel(r'$(1+A)\,e^{-k^2\Delta\Sigma^2/2}\,O(k)$'); ax[2].legend(fontsize=8.5)
ax[2].set_title(r'(c) amplitude $A$ (dashed), damping $\Delta\Sigma^2$ (solid)')
for a in ax: a.axvspan(*data, color='0.93', zorder=-1)
plt.tight_layout(); plt.savefig('pklin_template.pdf', bbox_inches='tight'); plt.savefig('pklin_template.png', dpi=130, bbox_inches='tight')
print('saved pklin_template.pdf; max|O| =', np.abs(O[mm]).max(), 'at k =', k[mm][np.argmax(np.abs(O[mm]))])
