"""Result figures of the recommended wiggle model Ae245F (MI chain wiggle_Ae245, data-weighted direct map), as PDFs in
output/desi_fsbao/wiggle_figs/Ae245F/:

  recovery_<model>.pdf     MI -> model (flow ensemble c5) against the true-likelihood direct chain (filled)
  mi_wiggles.pdf           MI posterior: ln alpha_rs, e0, e1, <ln a> (amplitude at z_N), <ln alpha>
  mi_bao_alphas.pdf        MI posterior: alpha_par / alpha_rs and alpha_per / alpha_rs at the six redshifts (= D/r_d / fid)
  mi_growth.pdf            MI posterior: f(z) and D(z)/D(z_N)
  pklin.pdf                reconstructed P_lin(k, z_N): absolute, ratio to the fiducial, shape; the MI prior's 68% (grey dashed)
In the MI triangles the true-likelihood LCDM chain is mapped into the same parameters (red).
Also: the Gaussian projection with e0, e1 marginalized ("sampled over") against the default (e0, e1 predicted by the
cosmology), for lcdm3 / lcdm5 / ede7 -> how much cosmological information the wiggle amplitude carries. Log: stdout.

    SCRIPT=wiggle_results_figs.py sbatch exec_wiggle.sbatch
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist import MCSamples, plots
import settings as S
import wiggle_settings as W
from model import COSMO_MODELS, COSMO_TEX, COSMO_BOX, EDE_BOX
from wiggle_direct import rdfix_files
from compress import gaussian_estimate, density_logpost, sample_logpost
from sampling import log

NAME = os.environ.get('WIGGLE', 'Ae245F')                      # the wiggle variant (its MI chain: wiggle_<NAME without F>)
OUT = os.path.join(S.OUT_ROOT, 'wiggle_figs', NAME + os.environ.get('MI_TAG', '')); os.makedirs(OUT, exist_ok=True)
C_D, C_MI = '#D84315', '#00838F'
SW = W.resolve(NAME); MI_NAME = os.path.basename(SW['mi_out'])[len('wiggle_'):]     # 'Ag245WFM' -> its MI chain 'Ag245W'
M, DATA = W.build(SW, verbose=False)
log(f"EFT priors (data cfg): {DATA['cfg']['eft_prior']}"); log(f"MI prior: {M.prior}; lambda_eff {M.lambda_eff:.2f}")
xm = np.load(os.path.join(SW['mi_out'], 'chain_mi.npz'))['x']; xm = xm.reshape(-1, xm.shape[-1]); phi = xm[:, M.n_eft:]
REC = {e: dict(np.load(os.path.join(SW['out'], f'recovery_{e}.npz'))) for e in ('gauss', 'c5')}
log(f"MI chain {phi.shape}; model {M.n_mi} parameters")

def direct(cv):
    x = np.load(rdfix_files(cv, 'exact')[1]['chain_direct'])['x']; return x.reshape(-1, x.shape[-1])[:, M.n_eft:]

def box_names(cv):
    model = S.resolve(cv)['cosmo_model']; keys = COSMO_MODELS[model]; names = ['logA' if k == 'lnAs' else k for k in keys]
    box = {n: (EDE_BOX.get(k, COSMO_BOX[k]) if model == 'ede7' else COSMO_BOX[k]) for n, k in zip(names, keys)}
    return model, keys, names, box

# ---- 1. recovery triangles ------------------------------------------------------------------------------
CMP = os.environ.get('COMPARE')
REC_CMP = dict(np.load(os.path.join(S.OUT_ROOT, f'wiggle_{CMP}', 'recovery_c5.npz'))) if CMP else None
for cv in W.COSMO_VARIANTS:
    model, keys, names, box = box_names(cv)
    mk = lambda th, lab: MCSamples(samples=th, names=names, labels=[COSMO_TEX[k] for k in keys], ranges=box, label=lab)
    lab = lambda e: (f"MI ({NAME}) → {model}, {'flow ensemble' if e == 'c5' else 'Gaussian'}: max |shift| "
                     f"{np.abs(REC[e][f'{cv}_shift']).max():.2f}σ, widths {REC[e][f'{cv}_ratio'].min():.2f}–{REC[e][f'{cv}_ratio'].max():.2f}")
    g = plots.get_subplot_plotter(subplot_size=3.4 if len(keys) <= 3 else 2.7)
    g.settings.num_plot_contours = 2; g.settings.alpha_filled_add = 0.35; g.settings.legend_fontsize = 13 if len(keys) <= 3 else 15
    g.settings.axes_labelsize = 15; g.settings.axes_fontsize = 11
    roots, cols, lss, lws, labs = [mk(direct(cv), 'd'), mk(REC['c5'][f'{cv}_samples'], 'f')], [C_D, C_MI], ['-', '-'], [1.0, 2.0], \
        [f'direct {model}, true likelihood', lab('c5')]
    if REC_CMP is not None:                  # COMPARE=<variant>: its flow projection, dashed (recovery_<model>_vs_<variant>.*)
        roots.append(mk(REC_CMP[f'{cv}_samples'], 'c')); cols.append('#6A1B9A'); lss.append('--'); lws.append(1.5)
        labs.append(f"MI ({CMP}) → {model}, flow ensemble: max |shift| {np.abs(REC_CMP[f'{cv}_shift']).max():.2f}σ, "
                    f"widths {REC_CMP[f'{cv}_ratio'].min():.2f}–{REC_CMP[f'{cv}_ratio'].max():.2f}")
    g.triangle_plot(roots, filled=[True] + [False] * (len(roots) - 1), contour_colors=cols, contour_ls=lss, contour_lws=lws, legend_labels=labs)
    fn = f'recovery_{model}' + (f'_vs_{CMP}' if CMP else '')
    for ext in ('pdf', 'png'): g.export(os.path.join(OUT, f'{fn}.{ext}'))
    plt.close('all'); log(f"{fn}.pdf")

# ---- 2. the MI posterior, with LCDM mapped in ----------------------------------------------------------
P = M.pidx; apar, aper, lna = P['apar'], P['aper'], P['lna']
def derived(ph):
    rs = ph[:, P['rs'][0]]
    return {'lnars': rs, 'e0': ph[:, P['env'][0]], 'e1': ph[:, P['env'][-1]], 'lna': ph[:, lna].mean(1),
            'lnal': ph[:, np.r_[apar, aper]].mean(1),
            **{f'bp{j}': np.exp(ph[:, apar[j]] - rs) for j in range(M.N)}, **{f'bq{j}': np.exp(ph[:, aper[j]] - rs) for j in range(M.N)},
            **{f'f{j}': np.exp(ph[:, P['f'][j]]) for j in range(M.N)}, **{f'D{j}': np.exp(ph[:, P['D'][j]]) for j in range(M.N - 1)}}
th_l = direct('baseline'); sel = np.random.default_rng(0).choice(len(th_l), 3000, replace=False)
fmap = M.phi_ln_keys(COSMO_MODELS['lcdm3'])
ph_l = np.concatenate([np.asarray(jax.jit(jax.vmap(fmap))(jnp.array(th_l[sel[i:i + 250]]))) for i in range(0, len(sel), 250)])
dm, dl = derived(phi), derived(ph_l)
zl = [f'{z:.2f}' for z in M.z]
GAUSS = M.env_type == 'gauss'
LAB = {'lnars': r'\ln\alpha_{r_s}', 'e0': 'A' if GAUSS else 'e_0', 'e1': r'\Delta\Sigma^2\,[{\rm Mpc}^2]' if GAUSS else 'e_1', 'lna': r'\langle\ln a\rangle', 'lnal': r'\langle\ln\alpha\rangle',
       **{f'bp{j}': rf'\alpha_\parallel/\alpha_{{r_s}}\,({zl[j]})' for j in range(M.N)}, **{f'bq{j}': rf'\alpha_\perp/\alpha_{{r_s}}\,({zl[j]})' for j in range(M.N)},
       **{f'f{j}': rf'f({zl[j]})' for j in range(M.N)}, **{f'D{j}': rf'D({zl[j]})/D({zl[-1]})' for j in range(M.N - 1)}}
for fname, keys, size in (('mi_wiggles', ['lnars', 'e0', 'e1', 'lna', 'lnal'], 3.0),
                          ('mi_bao_alphas', [f'bp{j}' for j in range(M.N)] + [f'bq{j}' for j in range(M.N)], 1.9),
                          ('mi_growth', [f'f{j}' for j in range(M.N)] + [f'D{j}' for j in range(M.N - 1)], 1.9)):
    a = MCSamples(samples=np.stack([dm[k] for k in keys], 1), names=keys, labels=[LAB[k] for k in keys], label='MI')
    b = MCSamples(samples=np.stack([dl[k] for k in keys], 1), names=keys, labels=[LAB[k] for k in keys], label='LCDM')
    g = plots.get_subplot_plotter(subplot_size=size); g.settings.num_plot_contours = 2; g.settings.alpha_filled_add = 0.4
    g.settings.legend_fontsize = 16; g.settings.axes_labelsize = 13
    g.triangle_plot([a, b], filled=[True, False], contour_colors=[C_MI, C_D], contour_lws=[1.0, 1.5],
                    legend_labels=[f"MI posterior ({MI_NAME}, {len(phi) // 1000}k draws)", r'$\Lambda$CDM true-likelihood posterior mapped into these parameters'])
    g.export(os.path.join(OUT, f'{fname}.pdf')); plt.close('all'); log(f"{fname}.pdf")
X = np.stack([dm[k] for k in ['e0', 'e1', 'lnars', 'lnal', 'lna'] + [f'bq{j}' for j in range(M.N)]], 1)
C = np.corrcoef(X, rowvar=False)
log("correlations of e0, e1 with [e0, e1, ln alpha_rs, <ln alpha>, <ln a>, alpha_per/alpha_rs(z_1..6)]:")
log(f"   e0: {np.round(C[0], 2)}"); log(f"   e1: {np.round(C[1], 2)}")
for k in ('lnars', 'e0', 'e1'):
    log(f"   {k}: MI {dm[k].mean():+.3f} +- {dm[k].std():.3f}; LCDM mapped {dl[k].mean():+.3f} +- {dl[k].std():.3f}")

# ---- 3. P_lin(k, z_N) ----------------------------------------------------------------------------------
sub = phi[np.random.default_rng(1).choice(len(phi), 3000, replace=False)]
spec = jax.jit(jax.vmap(lambda p: M.spectrum_knots(p[lna], p[P['rs'][0]], p[P['env']])))
Pm = np.concatenate([np.asarray(spec(jnp.array(sub[i:i + 500]))) for i in range(0, len(sub), 500)])
fx = dict(M.cosmo_fixed); keys3 = COSMO_MODELS['lcdm3']
def truth(t):
    c, pl = M.engine(M.theta_to_full(t, keys3, fx)); s = c['h'] / M.h_fid
    return pl(M.knots_mpc * s) * s**3
Pl = np.asarray(jax.jit(jax.vmap(truth))(jnp.array(th_l[sel[:800]])))
# the MI prior alone (Gaussian, M.prior_prec about M.prior_mean), through the same spectrum
php = np.random.default_rng(2).multivariate_normal(M.prior_mean, np.linalg.inv(M.prior_prec), 3000)
Pp = np.concatenate([np.asarray(spec(jnp.array(php[i:i + 500]))) for i in range(0, len(php), 500)])
log(f"prior draws with P_lin <= 0 on some knot: {np.mean((Pp <= 0).any(1)):.3%} (dropped from the prior band)"); Pp = Pp[(Pp > 0).all(1)]
k, T = np.asarray(M.knots_mpc), np.asarray(M.T_knots)
ip = np.argmin(np.abs(k - 0.05))
fig, ax = plt.subplots(3, 1, figsize=(8.5, 11), sharex=True, gridspec_kw={'height_ratios': [1.4, 1, 1]})
for i, (Y, Yl, Yp, ylab) in enumerate(((Pm, Pl, Pp, r'$P_{\rm lin}(k,z_N)\ [{\rm Mpc}^3]$'), (Pm / T, Pl / T, Pp / T, r'$P_{\rm lin}/T$'),
                                       (Pm / T / (Pm / T)[:, [ip]], Pl / T / (Pl / T)[:, [ip]], Pp / T / (Pp / T)[:, [ip]],
                                        r'$P_{\rm lin}/T$, normalized at $k=0.05\,{\rm Mpc}^{-1}$'))):
    q = np.percentile(Y, [2.5, 16, 50, 84, 97.5], axis=0); ql = np.percentile(Yl, [16, 84], axis=0); qp = np.percentile(Yp, [16, 84], axis=0)
    ax[i].plot(k, qp[0], color='0.45', lw=1.0, ls='--'); ax[i].plot(k, qp[1], color='0.45', lw=1.0, ls='--', label='MI prior 68%')
    ax[i].fill_between(k, q[0], q[4], color=C_MI, alpha=0.18, lw=0, label='MI 95%')
    ax[i].fill_between(k, q[1], q[3], color=C_MI, alpha=0.4, lw=0, label='MI 68%')
    ax[i].plot(k, q[2], color=C_MI, lw=1.2)
    ax[i].plot(k, ql[0], color=C_D, lw=1.2); ax[i].plot(k, ql[1], color=C_D, lw=1.2, label=r'$\Lambda$CDM (true likelihood) 68%')
    ax[i].plot(k, T if i == 0 else np.ones_like(k), color='k', lw=0.9, ls='--', label='fiducial')
    ax[i].axvspan(0.02 * M.h_fid, 0.2 * M.h_fid, color='0.93', zorder=-1); ax[i].set_ylabel(ylab)
ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlim(1e-3, 0.47); ax[0].legend(fontsize=9)
for a_ in ax[1:]: a_.set_yscale('log')                                        # the wide priors span more than a factor 10
ax[2].set_ylim(0.3, 3.5)                                                     # the shape: the prior band runs off it above the data range
from matplotlib.ticker import FuncFormatter
ax[2].set_yticks([0.3, 0.5, 0.7, 1, 1.5, 2, 3]); ax[2].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); ax[2].yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
ax[2].set_xlabel(r'$k\ [{\rm Mpc}^{-1}]$ (fiducial $h$)')
ax[0].set_title(rf"Reconstructed linear spectrum at $z_N=1.491$ ({MI_NAME}); grey: data range $0.02<k<0.2\,h/{{\rm Mpc}}$", fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'pklin.pdf'), bbox_inches='tight'); plt.savefig(os.path.join(OUT, 'pklin.png'), dpi=120, bbox_inches='tight')
plt.close('all'); log("pklin.pdf")

# ---- 4. e0, e1 predicted (default) vs marginalized ("sampled over") ------------------------------------
if os.environ.get('SKIP_MARG') != '1':                                      # SKIP_MARG=1: figures only (~2 min)
    fl_full = gaussian_estimate(M, xm)
    fl_marg = gaussian_estimate(M, xm, block=np.setdiff1d(np.arange(M.n_phys), P['env']))
    res = {}
    for cv in ('baseline', 'lcdm5', 'ede7'):
        sv = S.resolve(cv); model, keys, names, box = box_names(cv)
        M.set_cosmo_model(sv['cosmo_model'], gauss=sv['cosmo_gauss'])
        lo, hi = [M.cosmo_box[kk][0] for kk in keys], [M.cosmo_box[kk][1] for kk in keys]
        th_d = direct(cv); out = {}
        for tag, fl in (('predicted', fl_full), ('marginalized', fl_marg)):
            lp = density_logpost(M, fl, M.phi_ln_keys(keys), lo, hi, M.logprior_theta_keys(keys))
            out[tag] = sample_logpost(lp, lo, hi, np.median(th_d, 0), seed=7, n_warmup=500, n_samples=3000, n_chains=4).reshape(-1, len(keys))
        a, b = out['predicted'], out['marginalized']
        log(f"=== {model} (Gaussian estimate): e0, e1 predicted by the cosmology vs marginalized")
        for i, n in enumerate(names):
            log(f"   {n:11s} direct {th_d[:, i].mean():.4f}+-{th_d[:, i].std():.4f}   predicted {a[:, i].mean():.4f}+-{a[:, i].std():.4f}   "
                f"marginalized {b[:, i].mean():.4f}+-{b[:, i].std():.4f}   (marg/pred width {b[:, i].std() / a[:, i].std():.2f})")
        res[cv] = out
    np.savez(os.path.join(OUT, 'e_marginalized.npz'), **{f'{cv}_{t}': v for cv, o in res.items() for t, v in o.items()})
    M.set_cosmo_model('lcdm3')
log("done")
