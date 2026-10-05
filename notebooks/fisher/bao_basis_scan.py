"""Why the MI model is not (yet) a BAO method, and what makes it one.

A BAO fit measures D_M/r_d and D_H/r_d because the wiggle SHAPE is fixed: only the broadband
is free. The MI model frees ln a on nodes spaced 0.15 in ln k, while a BAO oscillation spans
~0.6 in ln k -- about four nodes per wiggle -- so the template can slide its own wiggles and
the ruler floats. This scans the node spacing and watches the BAO parameters sharpen.

alpha_perp = ln D_A - ln h_conv,  alpha_par = -ln H - ln h_conv: distances in units of the
template's ruler, invariant under g1 (the exact uniform-rescaling symmetry). g1 and g2 are
projected out of the data (exact symmetries; the g1 artifact is sigma~0.05 and would otherwise
masquerade as a ruler). g3 is left in: it is only a symmetry when the basis can represent a
dilation, which is exactly the effect being scanned, and its own artifact is weak (sigma~0.2).
"""
import numpy as np
import mi_prior_fisher as M

zz, nz = M.zz, M.nz
IZ = 2   # z = 0.706


def alpha_c(sysd, which, iz):
    c = np.zeros(sysd['npar']); o = sysd['o_g']
    c[o + M.ihc] = -1.0
    if which == 'perp':
        c[o + M.iD[iz]] = 1.0
    else:
        c[o + M.iH[iz]] = -1.0
    return c


print("BAO oscillation period in ln k at k=0.1 h/Mpc: %.2f  (2pi/(r_d k), r_d=105 Mpc/h)"
      % (2 * np.pi / (105 * 0.1)))
print(f"{'spacing':>8s} {'nodes':>6s} {'dilation in basis':>18s} {'prior sig(ruler)':>17s}   "
      f"{'sigma(alpha_perp)':>32s}   {'sigma(alpha_par)':>32s}   {'sigma(F_AP)':>17s}")
print(f"{'':8s} {'':6s} {'':18s} {'':17s}   {'P        P+B     ratio':>32s}   "
      f"{'P        P+B     ratio':>32s}   {'P       P+B':>17s}")
for spacing in (0.15, 0.25, 0.4, 0.6, 0.9, 1.4):
    out = {}
    for key in ('p', 'pb'):
        out[key] = M.build(key, project_data=True, spacing=spacing, project=(0, 1))
    s0 = out['p']
    # how much of a pure template dilation the basis can actually represent
    nl, Sb, Db, gnodes = M.node_basis(spacing)
    frac = np.linalg.norm(Sb @ gnodes) / np.linalg.norm(M.gamma)
    # the prior's own 1-sigma on a template dilation in this basis
    Pi_a = np.eye(len(nl)) / M.PRIORS_HMC['s_lna']**2 + M.PRIORS_HMC['lam'] * (Db.T @ Db)
    sig_ruler = 1.0 / np.sqrt(max(gnodes @ Pi_a @ gnodes, 1e-300))
    ap = [M.sig(out[k], alpha_c(out[k], 'perp', IZ)) for k in ('p', 'pb')]
    al = [M.sig(out[k], alpha_c(out[k], 'par', IZ)) for k in ('p', 'pb')]
    fa = [M.sig(out[k], M.growth_c(out[k], 'F_AP', IZ)) for k in ('p', 'pb')]
    print(f"{spacing:8.2f} {len(nl):6d} {frac:18.3f} {sig_ruler:17.4f}   "
          f"{ap[0]:8.4f} {ap[1]:8.4f} {ap[1]/ap[0]:6.3f}         "
          f"{al[0]:8.4f} {al[1]:8.4f} {al[1]/al[0]:6.3f}         {fa[0]:8.4f} {fa[1]:8.4f}")
print("\n(z = %.3f; 'dilation in basis' = |P_basis gamma| / |gamma|, the fraction of a pure "
      "template\n dilation the node basis can absorb; 1 = the template is its own ruler.)" % zz[IZ])


# ---------------------------------------------------------------------------------------
# Per-redshift BAO summary: the corrected version of ap_summary.png
# ---------------------------------------------------------------------------------------
# Two bases, both with the HMC priors and the data-side g1/g2 projection:
#   spacing 0.15 (60 nodes)  the model the HMC samples; the template can slide its wiggles
#   spacing 0.60 (16 nodes)  one node per BAO period: the wiggle shape is rigid (a BAO fit)
# Quantities are the ruler-free ones: alpha_perp = D_M/r_d, alpha_par = D_H/r_d (in units of the
# template's ruler, g1-invariant), F_AP, and f. Absolute H and D_A are not shown: g1 contains no
# amplitudes, so they are fixed by the h_conv / growth priors, never by the data.
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from make_fisher_pb_fig import C_P, C_PB, C_MUTED, C_INK
except Exception:                                                   # noqa: BLE001
    C_P, C_PB, C_MUTED, C_INK = '#E8710A', '#1A73E8', '#80868B', '#202124'

BASES = [(0.15, 'HMC model (60 nodes)', '-'), (0.60, 'BAO-rigid basis (16 nodes)', '--')]
QTS = [('perp', r'$D_M/r_d$  ($\alpha_\perp$)'), ('par', r'$D_H/r_d$  ($\alpha_\parallel$)'),
       ('F_AP', r'$F_{\rm AP} = D_M/D_H$'), ('f', r'$f(z)$')]
summ = {}
for spacing, _, _ in BASES:
    for key in ('p', 'pb'):
        sy = M.build(key, project_data=True, spacing=spacing, project=(0, 1))
        for q, _ in QTS:
            if q in ('perp', 'par'):
                v = [M.sig(sy, alpha_c(sy, q, iz)) for iz in range(nz)]
            else:
                v = [M.sig(sy, M.growth_c(sy, q, iz)) for iz in range(nz)]
            summ[(spacing, key, q)] = np.array(v)

print("\nPer-redshift fractional 1 sigma (HMC priors, data-side g1/g2 projected):")
print(f"{'':30s}" + "".join(f"{'z='+format(z,'.3f'):>10s}" for z in zz))
for spacing, lab, _ in BASES:
    for q, _ in QTS:
        for key, tag in (('p', 'P'), ('pb', 'P+B')):
            print(f"{lab[:14]:14s} {q:5s} {tag:4s}      " + "".join(f"{x:10.4f}" for x in summ[(spacing, key, q)]))

fig, axes = plt.subplots(2, 4, figsize=(20, 7.8), layout='constrained', sharex=True,
                         gridspec_kw={'height_ratios': [3, 1.3]})
COL = {'p': C_P, 'pb': C_PB}; LAB = {'p': 'P(k)', 'pb': 'P(k)+B(k)'}
for j, (q, title) in enumerate(QTS):
    ax, axr = axes[0, j], axes[1, j]
    for spacing, lab, ls in BASES:
        for key in ('p', 'pb'):
            ax.plot(zz, summ[(spacing, key, q)], color=COL[key], ls=ls, lw=2.0, marker='o', ms=4,
                    label=f'{LAB[key]}, {lab}')
        axr.plot(zz, summ[(spacing, 'pb', q)] / summ[(spacing, 'p', q)], color=C_INK, ls=ls,
                 lw=1.8, marker='o', ms=4, label=lab)
    ax.set_yscale('log'); ax.set_title(title, fontsize=14, color=C_INK)
    ax.grid(alpha=0.2, which='both')
    axr.axhline(1.0, color=C_MUTED, lw=1); axr.set_ylim(0.4, 1.1); axr.grid(alpha=0.2)
    axr.set_xlabel('z', fontsize=13)
axes[0, 0].set_ylabel(r'fractional $1\sigma$', fontsize=13)
axes[1, 0].set_ylabel(r'$\sigma_{P+B}/\sigma_P$', fontsize=13)
axes[0, 0].legend(frameon=False, fontsize=9, handlelength=3.2)
axes[1, 0].legend(frameon=False, fontsize=9, handlelength=3.2, loc='lower left')
fig.suptitle('Per-redshift BAO and growth, P(k) alone vs P(k)+B(k): model-independent template '
             'with the HMC priors, everything marginalized', fontsize=14)
out = os.path.join(M.OUT, 'bao_summary_pb.png')
fig.savefig(out, dpi=170); plt.close(fig)
np.savez(os.path.join(M.OUT, 'bao_summary_pb.npz'), zeff=zz,
         **{f"s{sp:.2f}_{k}_{q}": v for (sp, k, q), v in summ.items()})
print(f"saved {out}")
