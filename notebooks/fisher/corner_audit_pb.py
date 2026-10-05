"""Reproduce corner_ap_all_z.png in the model the HMC samples, and audit whether its priors
are fair.

CORNER. The original output/fisher_pb/corner_ap_all_z.png showed (ln H, ln D_A) at all six
redshifts with the TEMPLATE FIXED and no handling of the exact symmetries, so its individual
H and D_A were the numerical breaking of the ruler symmetry (sigma ~0.05), not measurements.
Here the same 12 parameters come from mi_prior_fisher.build: the MI model the HMC samples
(60 ln-a nodes + the four priors), with the data's response along the three exact symmetries
projected out, so nothing in the contour comes from discretization error.

AUDIT. Four questions about the priors:
  A. How many degrees of freedom in each block does the DATA actually constrain?
     n_eff = trace[(C F_data)] per block, against the block size (Spiegelhalter's effective
     number of parameters). n_eff ~ 0 means that block is prior.
  B. Ruler: sigma along g1 (h_conv + a uniform D_A, H rescaling) from the prior alone versus
     from the posterior. g1 does not touch the amplitudes, so NO template prior can constrain
     it: if the two agree, every absolute distance in this model is prior, not data.
  C. Is the prior wide enough for real cosmologies? For Planck-1sigma and 5sigma shifts of
     omega_cdm, omega_b, n_s and ln A_s, take ln a = ln[P_lin(cosmo)/P_lin(fid)] at z_ref on the
     knots, project onto the 60 nodes, and evaluate the prior chi^2 (split into the amplitude
     term and the smoothness term). chi^2 >> 1 would mean the prior excludes real cosmologies.
  D. What does the smoothness prior say about the BAO scale? The same chi^2 for a pure dilation
     of the template gives the prior's own sigma on the ruler -- the number to compare against
     any "model-independent BAO" claim.

Outputs: ../../output/fisher_pb/{corner_ap_all_z_mi.png, corner_growth_z*_mi.png, prior_audit.npz}
"""
import os
import numpy as np

import mi_prior_fisher as M

OUT = M.OUT
zz, nz, ng = M.zz, M.nz, M.ng
DATASETS = M.DATASETS
Z_REF = int(np.argmin(np.abs(zz - 0.93)))


def log(m=''):
    print(m, flush=True)


sysd = {key: M.build(key, project_data=True) for _, key in DATASETS}
for tag, key in DATASETS:
    s = sysd[key]
    log(f"{tag:4s}: {s['npar']} params ({s['n_e']}x{M.nsky} EFT + {M.n_nodes} nodes + {ng} growth); "
        f"min eig {s['min_eig']:.3e}; the data's own (spurious) sigma along g1/g2/g3 = "
        + ", ".join(f"{b:.3g}" for b in s['breaking']) + "  (projected out)")

# ---------------------------------------------------------------------------------------
log("\n" + "=" * 100)
log("CORNER quantities: fractional 1 sigma of ln H and ln D_A per redshift, MI model, "
    "everything marginalized.")
log(f"{'':12s}" + "".join(f"{'z='+format(z,'.3f'):>12s}" for z in zz))
for nm in ('lnDA', 'lnH', 'F_AP', 'f'):
    for tag, key in DATASETS:
        v = [M.sig(sysd[key], M.growth_c(sysd[key], nm, iz)) for iz in range(nz)]
        log(f"{nm:7s}{tag:5s}" + "".join(f"{x:12.4f}" for x in v))

# ---- A. how much does the data constrain, block by block --------------------------------
log("\n" + "=" * 100)
log("AUDIT A. Effective number of parameters the DATA constrains, n_eff = trace(C F_data), "
    "per block\n   (block size in brackets; n_eff ~ 0 means the block is prior-dominated).")
for tag, key in DATASETS:
    s = sysd[key]
    Pi = np.zeros((s['npar'], s['npar']))
    Pi[s['o_a']:, s['o_a']:] = M.prior_precision(M.n_nodes, **s['priors'])
    F_data = s['F'] - Pi                        # EFT priors stay with the data side here
    CF = s['C'] @ F_data
    o_a, o_g = s['o_a'], s['o_g']
    blocks = {'EFT': slice(0, o_a), 'ln a nodes': slice(o_a, o_g),
              'f (per z)': None, 'H, D_A (per z)': None, 'D-ratios': None, 'h_conv': None}
    idx = {'f (per z)': o_g + M.iF, 'H, D_A (per z)': np.concatenate([o_g + M.iH, o_g + M.iD]),
           'D-ratios': o_g + M.iDr, 'h_conv': np.array([o_g + M.ihc])}
    parts = []
    for nm, sl in blocks.items():
        ii = np.arange(sl.start, sl.stop) if sl is not None else idx[nm]
        parts.append(f"{nm} {np.trace(CF[np.ix_(ii, ii)]):.1f}/[{len(ii)}]")
    log(f"   {tag:4s}: " + "   ".join(parts))

# ---- B. the ruler direction -------------------------------------------------------------
log("\n" + "=" * 100)
log("AUDIT B. The ruler g1 (ln h_conv +1, ln D_A +1, ln H -1 at every z). It does not involve "
    "the\n   template, so no template prior can constrain it.")
for tag, key in DATASETS:
    s = sysd[key]
    g1 = M.gauge_vectors(s['npar'], s['o_a'], s['o_g'])[:, 0]
    g1n = g1 / np.linalg.norm(g1)
    Pi = np.zeros((s['npar'], s['npar']))
    Pi[s['o_a']:, s['o_a']:] = M.prior_precision(M.n_nodes, **s['priors'])
    sig_post = float(np.sqrt(g1n @ s['C'] @ g1n))
    sig_prior_only = 1.0 / np.sqrt(g1n @ Pi @ g1n)
    log(f"   {tag:4s}: sigma(g1) posterior {sig_post:.4f}   prior alone {sig_prior_only:.4f}   "
        f"-> data adds {100*(1 - sig_post/sig_prior_only):.1f}% ")
s = sysd['p']
for slnh in (0.05, 0.3, 3.0):
    st = M.build('p', priors=dict(M.PRIORS_HMC, s_lnh=slnh), project_data=True)
    log(f"   P with s_lnh={slnh:4.2f}: sigma(ln D_A) at z=0.71 = "
        f"{M.sig(st, M.growth_c(st, 'lnDA', 2)):.4f}, sigma(F_AP) = "
        f"{M.sig(st, M.growth_c(st, 'F_AP', 2)):.4f}")

# ---- C/D. is the prior wide enough for real cosmologies, and what does it say about r_d? --
log("\n" + "=" * 100)
log("AUDIT C/D. Prior chi^2 of ln a = ln[P_lin(cosmo)/P_lin(fid)] at z_ref, projected on the "
    "60 nodes.\n   Split: amplitude term sum (ln a / s_lna)^2 and smoothness term "
    "lambda |D2 ln a|^2. chi^2 >> 1 = excluded by the prior.")
Pi_a = np.eye(M.n_nodes) / M.PRIORS_HMC['s_lna']**2 + M.PRIORS_HMC['lam'] * (M.D2.T @ M.D2)
Pi_amp = np.eye(M.n_nodes) / M.PRIORS_HMC['s_lna']**2
Pi_sm = M.PRIORS_HMC['lam'] * (M.D2.T @ M.D2)

def prior_chi2(lna_nodes):
    """chi^2 of the prior for a deviation given AT THE NODES.

    The model's parameters ARE the node values, so ln a is evaluated at the nodes directly.
    (An earlier version least-squares-projected ln a from the 80 knots onto the 60 nodes; the
    spline evaluation matrix has a near-null space, so that let wiggly node patterns in and
    gave a nonzero smoothness penalty even for an exactly constant ln a -- see the sanity
    check below, which must return zero.)"""
    x = np.asarray(lna_nodes)
    return float(x @ Pi_a @ x), float(x @ Pi_amp @ x), float(x @ Pi_sm @ x), 0.0, x

# Two engines. CosmoPower is what the model uses, but its k-dependent error does not cancel
# between two nearby cosmologies: a pure ln A_s shift, whose ln a is EXACTLY constant, comes out
# with a smoothness chi^2 of ~4, which is the noise floor of the test. symbolic_pofk is an
# analytic fitting formula, so differences between cosmologies are smooth by construction and
# the smoothness penalty is meaningful. Evaluated at z=0 (the shape ratio is z-independent in
# LCDM; symbolic_pofk supports z<=3, the template's z_ref=5 does not matter for a ratio).
import sys
sys.path.insert(0, os.environ.get('PYBIRD_DEV',
                                  '/capstor/store/cscs/swissai/a0158/areeves/pybird-dev'))
FID = dict(omega_b=0.02235, omega_cdm=0.120, h=0.675, lnAs=3.044, n_s=0.965)
SIG1 = dict(omega_cdm=0.0012, omega_b=0.00015, n_s=0.0042, lnAs=0.014)
kk_mpc = M.nodes_h * M.h_fid          # NODE wavenumbers: the model's own parameters
engines = {}
try:
    import jax, jax.numpy as jnp
    jax.config.update('jax_enable_x64', True)
    from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ
    cpj = CPJ(probe='mpk_lin'); modes = np.array(cpj.modes)

    def pk_cpj(**kw):
        c = dict(FID, **kw)
        d = {'omega_b': jnp.atleast_1d(c['omega_b']), 'omega_cdm': jnp.atleast_1d(c['omega_cdm']),
             'h': jnp.atleast_1d(c['h']), 'ln10^{10}A_s': jnp.atleast_1d(c['lnAs']),
             'n_s': jnp.atleast_1d(c['n_s']), 'z': jnp.atleast_1d(5.0)}
        p = np.array(cpj.predict(d)).ravel()
        return np.exp(np.interp(np.log(kk_mpc), np.log(modes), np.log(p)))
    engines['CosmoPower (used by the model)'] = pk_cpj
except Exception as exc:                                            # noqa: BLE001
    log(f"   CosmoPower unavailable: {exc!r}")
try:
    from pybird.symbolic_pofk_linear import plin_emulated

    def pk_sym(**kw):
        c = dict(FID, **kw)
        A_s = 1e-10 * np.exp(c['lnAs'])
        Om = (c['omega_cdm'] + c['omega_b']) / c['h']**2
        Ob = c['omega_b'] / c['h']**2
        return np.array(plin_emulated(M.nodes_h, A_s, Om, Ob, c['h'], c['n_s'], 0.0, -1.0, 0.0,
                                      z=0.0, lcdm=True))
    engines['symbolic_pofk (analytic)'] = pk_sym
except Exception as exc:                                            # noqa: BLE001
    log(f"   symbolic_pofk unavailable: {exc!r}")

rows = {}
for eng_name, pkf in engines.items():
    log(f"\n   --- {eng_name} ---")
    P0 = pkf()
    log(f"   {'variation':>26s} {'chi2_total':>11s} {'amplitude':>10s} {'smoothness':>11s}")
    for nmv, s1 in SIG1.items():
        for mult in (1, 5):
            c2, ca, cs, res, _ = prior_chi2(np.log(pkf(**{nmv: FID[nmv] + mult * s1}) / P0))
            rows[(eng_name.split()[0], nmv, mult)] = (c2, ca, cs, res)
            log(f"   {nmv+f' +{mult}sigma_Planck':>26s} {c2:11.3f} {ca:10.3f} {cs:11.3f}"
                + ("   <- ln a exactly constant: smoothness MUST be 0"
                   if nmv == 'lnAs' else ("   <- ln a linear in ln k: smoothness MUST be 0"
                                          if nmv == 'n_s' else "")))
    for eps in (0.01, 0.03):
        Pd = np.exp(np.interp(np.log(kk_mpc / (1 + eps)), np.log(kk_mpc), np.log(P0)))
        c2, ca, cs, res, _ = prior_chi2(np.log((1 + eps)**-3 * Pd / P0))
        rows[(eng_name.split()[0], 'dilation', eps)] = (c2, ca, cs, res)
        log(f"   {f'template dilation {100*eps:.0f}%':>26s} {c2:11.3f} {ca:10.3f} {cs:11.3f}"
            f"   -> prior sigma on the ruler = {eps/np.sqrt(max(c2,1e-30)):.4f}")
np.savez(os.path.join(OUT, 'prior_audit.npz'),
         **{f"chi2_{a}_{b}_{c}": np.array(v) for (a, b, c), v in rows.items()})

log("\n" + "=" * 100)
log("AUDIT E. Where does B(k) actually add? Eigen-spectrum of the 12-dim (ln H, ln D_A) "
    "covariance.\n   The widest direction is the prior ruler (identical for both, by "
    "construction); the rest is data.")
qty12 = [(p_, i) for i in range(nz) for p_ in ('lnH', 'lnDA')]
cov12 = {}
for tag, key in DATASETS:
    Cs = np.array([M.growth_c(sysd[key], p_, i) for p_, i in qty12])
    cov12[key] = Cs @ sysd[key]['C'] @ Cs.T
ev_p = np.sqrt(np.linalg.eigvalsh(cov12['p']))[::-1]
ev_pb = np.sqrt(np.linalg.eigvalsh(cov12['pb']))[::-1]
log("   sigma of each eigen-direction, widest first:")
log("      P   " + " ".join(f"{v:.4f}" for v in ev_p))
log("      P+B " + " ".join(f"{v:.4f}" for v in ev_pb))
log("      P+B/P " + " ".join(f"{b/a:.3f}" for a, b in zip(ev_p, ev_pb)))
log(f"   volume ratio |C_PB|^(1/12) / |C_P|^(1/12) = "
    f"{np.exp((np.linalg.slogdet(cov12['pb'])[1] - np.linalg.slogdet(cov12['p'])[1]) / 24):.4f}")

# ---------------------------------------------------------------------------------------
# corner figures
# ---------------------------------------------------------------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from make_fisher_pb_fig import C_P, C_PB, C_INK
except Exception:                                                   # noqa: BLE001
    C_P, C_PB, C_INK = '#E8710A', '#1A73E8', '#202124'
from getdist import plots
from getdist.gaussian_mixtures import GaussianND

names = [f'{p}{i}' for i in range(nz) for p in ('lnH', 'lnDA')]
labels = [rf'\Delta {p}/{p}({zz[i]:.2f})'.replace('lnH', 'H').replace('lnDA', 'D_A')
          for i in range(nz) for p in ('H', 'D_A')]
qty = [(p, i) for i in range(nz) for p in ('lnH', 'lnDA')]
dists = []
for tag, key in [('P(k)', 'p'), ('P(k)+B(k)', 'pb')]:
    Cs = np.array([M.growth_c(sysd[key], p, i) for p, i in qty])
    cov = Cs @ sysd[key]['C'] @ Cs.T
    dists.append(GaussianND(np.zeros(len(names)), cov, names=names, labels=labels, label=tag))
g = plots.get_subplot_plotter(subplot_size=1.15)
g.settings.num_plot_contours = 2
g.settings.lw_contour = 1.8
g.settings.alpha_filled_add = 0.35
g.settings.legend_fontsize = 15
g.triangle_plot(dists, params=names, filled=[False, True],
                contour_colors=[C_P, C_PB], contour_ls=['-', '-'],
                line_args=[{'color': C_P, 'lw': 1.8}, {'color': C_PB, 'lw': 1.8}],
                legend_labels=['P(k)', 'P(k)+B(k)'], markers={n: 0.0 for n in names})
g.fig.suptitle('AP parameters at every redshift, model-independent template (60 nodes + the '
               'HMC priors), everything marginalized,\nexact symmetries projected out of the '
               'data', fontsize=17, y=1.02)
p1 = os.path.join(OUT, 'corner_ap_all_z_mi.png')
g.export(p1)
log(f"\nsaved {p1}")

# the same corner with the prior-dominated common direction (the ruler) removed: each
# parameter minus the mean of all twelve, which is exactly the direction the prior fixes.
# The prior-fixed direction is g1 restricted to these 12 parameters: +1 on every ln D_A and
# -1 on every ln H. Its MEAN over the 12 is zero, so subtracting the mean (an earlier attempt)
# removed nothing at all -- the two corners came out identical. Project out g1 itself.
u_g1 = np.array([-1.0 if p_ == 'lnH' else 1.0 for p_, _ in qty12])
u_g1 /= np.linalg.norm(u_g1)
Pm = np.eye(12) - np.outer(u_g1, u_g1)
dists2 = []
for tag, key in [('P(k)', 'p'), ('P(k)+B(k)', 'pb')]:
    dists2.append(GaussianND(np.zeros(12), Pm @ cov12[key] @ Pm.T, names=names, labels=labels,
                             label=tag))
g2 = plots.get_subplot_plotter(subplot_size=1.15)
g2.settings.num_plot_contours = 2; g2.settings.alpha_filled_add = 0.35
g2.triangle_plot(dists2, params=names, filled=[False, True], contour_colors=[C_P, C_PB],
                 line_args=[{'color': C_P, 'lw': 1.8}, {'color': C_PB, 'lw': 1.8}],
                 legend_labels=['P(k)', 'P(k)+B(k)'], markers={n: 0.0 for n in names})
g2.fig.suptitle('The same AP parameters with the prior-fixed ruler direction projected out '
                '(a common rescaling of every D_A up and every H down): what the DATA measure',
                fontsize=16, y=1.02)
p1b = os.path.join(OUT, 'corner_ap_all_z_mi_rulerfree.png')
g2.export(p1b)
log(f"saved {p1b}")

for iz, z in enumerate(zz):
    nm3 = ['f', 'lnH', 'lnDA']
    lb3 = [r'\Delta f/f', r'\Delta H/H', r'\Delta D_A/D_A']
    ds = []
    for tag, key in [('P(k)', 'p'), ('P(k)+B(k)', 'pb')]:
        cov = M.block_cov(sysd[key], nm3, iz)
        ds.append(GaussianND(np.zeros(3), cov, names=nm3, labels=lb3, label=tag))
    gg = plots.get_subplot_plotter(subplot_size=3.0)
    gg.settings.num_plot_contours = 2; gg.settings.alpha_filled_add = 0.35
    gg.triangle_plot(ds, params=nm3, filled=[False, True], contour_colors=[C_P, C_PB],
                     line_args=[{'color': C_P, 'lw': 2.0}, {'color': C_PB, 'lw': 2.0}],
                     legend_labels=['P(k)', 'P(k)+B(k)'], markers={n: 0.0 for n in nm3})
    gg.fig.suptitle(rf'Growth / AP at $z={z:.3f}$, model-independent template + HMC priors',
                    fontsize=14)
    pth = os.path.join(OUT, f'corner_growth_z{z:.3f}_mi.png'.replace('.', 'p', 1))
    gg.export(pth)
log(f"saved per-redshift corners corner_growth_z*_mi.png")
