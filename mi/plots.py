"""Figures and tables of the notebooks, from data/mi_chains/ only (numpy, matplotlib, getdist; no model code).

Colours: agnostic #00897B (the one filled band), bao_template #6A1B9A (outline), true likelihood / Planck grey (filled in
triangles), LCDM black, w0waCDM #C62828, Planck #8D6E63.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), 'data', 'mi_chains')
OPTS = ('agnostic', 'bao_template')
COL = {'agnostic': '#00897B', 'bao_template': '#6A1B9A', 'bao_template_no_rs': '#C2185B', 'direct': '#757575', 'planck': '#8D6E63', 'lcdm': '#212121', 'w0wa': '#C62828',
       'prior': '#9E9E9E', 'growth_from_h': '#1565C0', 'direct_rd': '#EF6C00', 'adapted': '#00897B'}
LAB = {'agnostic': 'agnostic (60 nodes)', 'bao_template': 'BAO template (21 nodes)', 'bao_template_no_rs': r'BAO template, $\alpha_{r_s}=1$',
       'growth_from_h': r'growth from $H(z)$'}
KEYS = {'lcdm': ['omega_cdm', 'omega_b', 'n_s', 'lnAs', 'h'], 'w0wa': ['omega_cdm', 'omega_b', 'n_s', 'lnAs', 'h', 'w0', 'wa'],
        'ede': ['omega_cdm', 'omega_b', 'lnAs', 'h', 'fEDE', 'log10z_c', 'thetai_scf'],
        'mnu': ['omega_cdm', 'omega_b', 'n_s', 'lnAs', 'h', 'm_ncdm']}
TEX = {'omega_cdm': r'\omega_{\rm cdm}', 'omega_b': r'\omega_b', 'n_s': 'n_s', 'lnAs': r'\ln(10^{10}A_s)', 'h': 'h', 'w0': 'w_0',
       'wa': 'w_a', 'fEDE': r'f_{\rm EDE}', 'log10z_c': r'\log_{10}z_c', 'thetai_scf': r'\theta_i', 'Om': r'\Omega_m', 's8': r'\sigma_8',
       'm_ncdm': r'm_\nu\,[{\rm eV}]'}
COSMO_NAME = {'lcdm': 'LCDM', 'w0wa': 'w0waCDM', 'ede': 'EDE', 'mnu': 'LCDM + m_nu'}
RECOVERY = ('lcdm', 'w0wa', 'ede')          # the models of run.py recovery and checks.py map
plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})


def load(name):
    return np.load(os.path.join(DATA, name), allow_pickle=False)


def use(tree=''):
    """Read from another product folder: '' the DESI DR1 data, 'mock_w0wa' a noiseless mock (data/mi_chains_mock_w0wa), 'y6_p', ..."""
    global DATA
    DATA = os.path.join(os.path.dirname(HERE), 'data', 'mi_chains' + ('_' + tree if tree else ''))


def flat(x):
    x = np.asarray(x, float); return x.reshape(-1, x.shape[-1])


def mcsamples(th, keys, label, weights=None):
    from getdist import MCSamples
    return MCSamples(samples=th, weights=weights, names=list(keys), labels=[TEX.get(k, k) for k in keys], label=label,
                     ranges={k: (0., None) for k in keys if k in ('m_ncdm', 'fEDE')},      # the hard lower edge of their flat priors
                     settings={'smooth_scale_2D': 0.5, 'smooth_scale_1D': 0.5})


def triangle(roots, filled, colors, legend, params=None, ls=None):
    from getdist import plots
    g = plots.get_subplot_plotter(subplot_size=2.2); g.settings.num_plot_contours = 2; g.settings.legend_fontsize = 11
    g.triangle_plot(roots, params=params, filled=filled, contour_colors=colors, legend_labels=legend,
                    contour_ls=ls or ['-'] * len(roots), contour_lws=[1.6] * len(roots))
    return g


# ------------------------------------------------------------------------------------------------- chains
def rhat_ess(x):
    """Split R-hat and a bulk ESS (Geyer initial positive sequence) per parameter; x (chains, draws, params)."""
    x = np.asarray(x, float); c, n, p = x.shape; h = n // 2
    s = np.concatenate([x[:, :h], x[:, h:2 * h]], 0); m = s.shape[0]
    W = s.var(1, ddof=1).mean(0); B = h * s.mean(1).var(0, ddof=1)
    rhat = np.sqrt(((h - 1) / h * W + B / h) / W)
    z = s - s.mean(1, keepdims=True); f = np.fft.rfft(z, 2 * h, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :h] / h; ac = ac.mean(0) / ac.mean(0)[0]
    ess = np.empty(p)
    for j in range(p):
        r = ac[:, j]; tau = 1.0
        for t in range(1, h - 1, 2):
            pair = r[t] + r[t + 1]
            if pair < 0: break
            tau += 2 * pair
        ess[j] = m * h / tau
    return rhat, ess


def chain_table():
    print(f"{'template':19s} {'params':>6s} {'chains x draws':>15s} {'max R-hat':>9s} {'min ESS':>8s} {'median ESS':>10s} {'divergent':>9s}  seeds")
    for o in OPTS:
        d = load(f'chain_{o}.npz'); r, e = rhat_ess(d['x'])
        print(f"{o:19s} {d['x'].shape[-1]:6d} {d['x'].shape[0]:>7d} x {d['x'].shape[1]:<5d} {r.max():9.3f} "
              f"{e.min():8.0f} {np.median(e):10.0f} {float(d['divergent']):9.4f}  {', '.join(str(v) for v in d['seeds'])}")


def bestfit_table():
    """checks.py bestfit: chi2 and ln posterior at the best fit the chains start from, and at the end points of L-BFGS-B restarted
    from it, from the chain mean, from the chain's best draw and from random draws: the highest posterior found, and all of them."""
    print(f"{'template':19s} {'params':>6s} {'chain start':>18s} {'highest found':>18s}   chi2 / ln posterior;  chi2 of every end point")
    for o in OPTS:
        d = load(f'bestfit_{o}.npz'); j = np.argmax(d['lnpost'])
        print(f"{o:19s} {d['x'].shape[1]:6d} {d['chi2'][0]:9.2f} /{d['lnpost'][0]:8.2f} {d['chi2'][j]:9.2f} /{d['lnpost'][j]:8.2f}   "
              + ' '.join(f"{c:.1f}" for c in d['chi2'][1:]))


def blocks(names):
    out = {}
    for i, n in enumerate(names):
        b = n.split('(')[0].replace('ln ', '').strip()
        b = {'a': 'ln a (nodes)', 'f': 'ln f', 'alpha_par': 'alphas', 'alpha_per': 'alphas', 'D': 'ln D ratios'}.get(b, b)
        out.setdefault(b, []).append(i)
    return out


def posterior_prior():
    """Posterior / prior width of every MI parameter (physical block)."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, o in zip(axs, OPTS):
        d = load(f'chain_{o}.npz'); ne = int(d['n_eft']); x = flat(d['x'])[:, ne:]; names = list(d['names'][ne:])
        r = x.std(0) / d['prior_sd']
        for b, idx in blocks(names).items():
            ax.plot(idx, r[idx], 'o', ms=4, label=b)
        ax.axhline(1, color='k', lw=0.8); ax.set_yscale('log'); ax.set_title(LAB[o], fontsize=10); ax.set_xlabel('MI parameter')
        print(f"{o:9s} posterior/prior width: " + ", ".join(f"{b} {np.median(r[i]):.2f}" for b, i in blocks(names).items()) + " (median per block)")
    axs[0].set_ylabel(r'$\sigma_{\rm post}/\sigma_{\rm prior}$'); axs[1].legend(fontsize=8, ncol=2)
    return fig


# ------------------------------------------------------------------------------------------------- representation
def implied(th, d):
    w = np.exp(d - d.max()); w /= w.sum()
    return (w @ th - th.mean(0)) / th.std(0), 1.0 / np.sum(w**2) / len(w)


def map_table():
    m = load('map_test.npz'); rows = [r for r in ['legacy', 'agnostic', 'bao_template', 'bao_template_no_rs', 'oracle'] if f'lcdm_{r}-truth' in m.files]
    print("max |implied shift| of the posterior mean, in direct sigma (map only, vs the true likelihood; 300 draws)")
    print(f"{'cosmology':12s}" + "".join(f"{r:>14s}" for r in rows))
    for cv in RECOVERY:
        th = m[f'{cv}_th']; vals = []
        for r in rows:
            k = f'{cv}_{r}-truth'
            vals.append(np.abs(implied(th, m[k])[0]).max() if k in m.files else np.nan)
        print(f"{COSMO_NAME[cv]:12s}" + "".join(f"{v:14.2f}" for v in vals))


# ------------------------------------------------------------------------------------------------- recovery
def recovery(cv, show=True, est='flow'):
    """Projection of each template's MI posterior onto cosmology cv vs the reference (direct fit, full likelihood), with the
    density estimate est: 'flow' (normalizing flows of the MI posterior, the default) or 'score' (score regression of the MI likelihood, the back-up)."""
    keys = KEYS[cv]; dd = load(f'reference_{cv}.npz'); ne = int(dd['n_eft']); th_d = flat(dd['x'])[:, ne:]
    roots, out = [mcsamples(th_d, keys, 'reference')], {}
    for o in OPTS:
        r = load(f'recovery_{o}.npz'); k = f'{cv}_x' if est == 'flow' else f'{cv}_{est}_x'
        if k not in r.files: print(f"{o}: no {est} recovery exported"); continue
        th = flat(r[k]); roots.append(mcsamples(th, keys, LAB[o]))
        out[o] = ((th.mean(0) - th_d.mean(0)) / th_d.std(0), th.std(0) / th_d.std(0))
    if not show: return out
    print(f"{COSMO_NAME[cv]} ({est}): shift [reference sigma] / width ratio")
    for i, k in enumerate(keys):
        print(f"   {k:11s} reference {th_d[:, i].mean():.4f} +- {th_d[:, i].std():.4f}   " +
              "   ".join(f"{o} {out[o][0][i]:+.2f} / {out[o][1][i]:.2f}" for o in out))
    return triangle(roots, [True] + [False] * len(out), [COL['direct']] + [COL[o] for o in out],
                    ['reference (direct fit, full likelihood)'] + [LAB[o] for o in out])


def recovery_table(ests=('flow', 'score')):
    """max |shift| of the projected posterior mean vs the reference, and the width ratios, per template and density estimate."""
    cols = [(o, e) for o in OPTS for e in ests]
    print(f"{'cosmology':12s}" + "".join(f"{o + ' (' + e + ')':>34s}" for o, e in cols))
    for cv in RECOVERY:
        res = {e: recovery(cv, show=False, est=e) for e in ests}
        print(f"{COSMO_NAME[cv]:12s}" + "".join(f"{np.abs(res[e][o][0]).max():20.2f} [{res[e][o][1].min():.2f}, {res[e][o][1].max():.2f}]"
                                               if o in res[e] else f"{'-':>34s}" for o, e in cols))


def direct_vs_mi(cv, theta_s=False, rd=False, opts=('bao_template_no_rs',), show=True):
    """The MI posterior projected onto cosmology cv (projections_<template>.npz, score density) against the direct fit with the
    full likelihood (reference_<cv>.npz; rd: reference_<cv>_rd.npz, the fit with the EFT priors in sound-horizon units).
    theta_s: the projection with the theta_s prior against the direct draws weighted by that prior (their effective number is
    printed). On a mock (DATA = a data/mi_chains_mock_<name> folder) the truth is shown too. Shifts in direct-fit sigma."""
    keys = KEYS[cv]; name = f"reference_{cv}{'_rd' if rd else ''}"; sampled = theta_s and os.path.exists(os.path.join(DATA, name + '_theta_s.npz'))
    dd = load(name + ('_theta_s' if sampled else '') + '.npz'); th_d = flat(dd['x'])[:, int(dd['n_eft']):]; w = np.ones(len(th_d))
    if theta_s and not sampled:                               # no direct chain with theta_s: weight the DESI-alone one
        c, sg = dd['theta_s_prior']; lw = -0.5 * ((np.asarray(dd['theta_s'], float).reshape(-1) - c) / sg)**2; w = np.exp(lw - lw.max())
    w = w / w.sum(); mu = w @ th_d; sd = np.sqrt(w @ (th_d - mu)**2); out = {}
    truth = load('truth.npz')['x'][-len(keys):] if os.path.exists(os.path.join(DATA, 'truth.npz')) else None
    for o in opts:
        pr = load(f'projections_{o}.npz'); th = flat(pr[cv + ('_theta_s' if theta_s else '') + '_x']); out[o] = ((th.mean(0) - mu) / sd, th.std(0) / sd, th)
    if not show: return out
    lab = 'direct fit' + (', EFT priors in sound-horizon units' if rd else '') + (r' $\times\ \theta_s$ prior' if theta_s else '')
    print(f"{COSMO_NAME[cv]}{' + theta_s' if theta_s else ''}: {lab.replace('$', '')}"
          + (f" ({1 / np.sum(w**2):.0f} effective of {len(w)} draws)" if theta_s and not sampled else ' (sampled with it)' if sampled else '')
          + "; MI: shift [direct sigma] / width ratio" + ("; truth: (direct mean - truth) / sigma" if truth is not None else ''))
    for i, k in enumerate(keys):
        print(f"   {k:11s} direct {mu[i]:.4f} +- {sd[i]:.4f}" + (f"   truth {truth[i]:.4f} ({(mu[i] - truth[i]) / sd[i]:+.2f})" if truth is not None else '')
              + "".join(f"   {o} {v[2][:, i].mean():.4f} +- {v[2][:, i].std():.4f} ({v[0][i]:+.2f} / {v[1][i]:.2f})" for o, v in out.items()))
    g = triangle([mcsamples(th_d, keys, 'direct', weights=w)] + [mcsamples(v[2], keys, o) for o, v in out.items()], [True] + [False] * len(out),
                 [COL['direct']] + [COL[o] for o in out], [lab] + [LAB[o] + (r' + $\theta_s$' if theta_s else '') for o in out])
    return truth_lines(g, truth)


def truth_lines(g, truth):
    """Dotted lines at the truth of a mock on a triangle plot of len(truth) parameters (None: nothing)."""
    for i in range(0 if truth is None else len(truth)):
        for j in range(i + 1):
            ax = g.subplots[i, j]; ax.axvline(truth[j], color='k', lw=0.6, ls=':')
            if i != j: ax.axhline(truth[i], color='k', lw=0.6, ls=':')
    return g


def prior_units(cv='w0wa', opts=('bao_template', 'bao_template_no_rs'), params=('omega_cdm', 'lnAs', 'h', 'w0', 'wa')):
    """The same data fitted directly with the EFT priors in h/Mpc (reference_<cv>) and in units of the sound horizon
    (reference_<cv>_rd), and through each template (projections). Printed: mean +- sd, and the shift against the h/Mpc fit in
    its sigma. On a mock the truth is shown (dotted)."""
    keys = KEYS[cv]; ip = [keys.index(p) for p in params]; ref = lambda n: (lambda d: flat(d['x'])[:, int(d['n_eft']):])(load(f'reference_{cv}{n}.npz'))
    R = [('direct fit, EFT priors in $h$/Mpc', ref(''), COL['direct']), ('direct fit, EFT priors in sound-horizon units', ref('_rd'), COL['direct_rd'])]
    R += [(LAB[o] + ', projected', flat(load(f'projections_{o}.npz')[cv + '_x']), COL[o]) for o in opts]
    truth = load('truth.npz')['x'][-len(keys):][ip] if os.path.exists(os.path.join(DATA, 'truth.npz')) else None
    mu, sd = R[0][1].mean(0), R[0][1].std(0)
    print(f"{COSMO_NAME[cv]}: mean +- sd (shift against the h/Mpc direct fit, in its sigma)" + ('' if truth is None else '; truth: ' + ', '.join(f"{p} {t:.4f}" for p, t in zip(params, truth))))
    for lab, th, _ in R:
        print(f"   {lab.replace('$', ''):48s}" + ''.join(f"  {keys[i]} {th[:, i].mean():.4f} +- {th[:, i].std():.4f} ({(th[:, i].mean() - mu[i]) / sd[i]:+.2f})" for i in ip))
    g = triangle([mcsamples(th[:, ip], list(params), lab) for lab, th, _ in R], [True] + [False] * (len(R) - 1), [c for _, _, c in R], [lab for lab, _, _ in R])
    return truth_lines(g, truth)


def mi_truth(o='bao_template_no_rs'):
    """A mock's MI posterior against the MI parameters of its truth (the map of the true cosmology): (mean - truth) / sd of
    every MI parameter, and the ratio of the posterior width to the prior's (1 = the prior alone holds it at the fiducial)."""
    c = load(f'chain_{o}.npz'); ne = int(c['n_eft']); x = flat(c['x'])[:, ne:]; names = [str(n) for n in c['names'][ne:]]; tr = np.asarray(c['truth'], float)
    pull, wr = (x.mean(0) - tr) / x.std(0), x.std(0) / c['prior_sd']
    print(f"{LAB[o]}: (posterior mean - truth) / posterior sd per block; in brackets the parameters the data measure (width < 0.5 x prior)")
    B = {lab: np.array(idx) for lab, idx in blocks(names).items()}
    for lab, idx in B.items():
        m = wr[idx] < 0.5
        print(f"   {lab:22s} max |pull| {np.abs(pull[idx]).max():.2f}, rms {np.sqrt(np.mean(pull[idx]**2)):.2f}"
              + (f"   [measured: {m.sum()} of {len(idx)}, max |pull| {np.abs(pull[idx][m]).max():.2f}]" if m.any() else "   [none measured]"))
    if 'ln f' in B:      # b1^2 a and f / b1 are measured: a, b1 and f move together along a^-1/2. The measured growth combination:
        ia = [i for i, n in enumerate(names) if n.startswith('ln a(') and 0.02 <= float(n[5:-1]) <= 0.2]
        comb = lambda v: (v[..., B['ln f']] + 0.5 * v[..., ia].mean(-1, keepdims=True)
                          + np.pad(v[..., B['ln D ratios']], [(0, 0)] * (v.ndim - 1) + [(0, 1)]))        # D(z_N) / D(z_N) = 1
        c, ct = comb(x), comb(tr)
        print(f"   f D a^1/2 per redshift (a: mean of the nodes at 0.02 < k < 0.2 h/Mpc): pull {np.round((c.mean(0) - ct) / c.std(0), 2)}, "
              f"width / width of ln f {np.round(c.std(0) / x[:, B['ln f']].std(0), 2)}")
    fig, ax = plt.subplots(figsize=(13, 3.2)); ax.axhspan(-1, 1, color=COL['prior'], alpha=0.15)
    ax.scatter(np.arange(len(pull)), pull, s=14, c=np.where(wr < 0.5, COL[o], COL['prior'])); top = ax.get_ylim()[1]
    for j, (lab, idx) in enumerate(B.items()):               # one-parameter blocks: labels on alternating lines
        ax.axvline(idx[0] - 0.5, color='k', lw=0.4); ax.text(idx[0], top, ' ' + '\n' * (len(idx) == 1 and j % 2) + lab, va='top', fontsize=8)
    ax.set(xlabel='MI parameter (grey: prior-dominated)', ylabel='(mean $-$ truth) / sd'); fig.tight_layout(); return fig


# ------------------------------------------------------------------------------------------------- Y6 mock: P alone against P + B (Fisher)
def y6_fisher(o='bao_template_no_rs'):
    """DESI-Y6 mock, Gauss-Newton Fisher at the two MI best fits (run.py fisher, MI prior included): sd(P + B) / sd(P) of every MI
    parameter and of LCDM / w0waCDM through the reconstruction (DESI-alone priors)."""
    F = {}
    for t in ('y6_p', 'y6_pb'): use(t); F[t] = load('fisher.npz')
    a, b = F['y6_p'], F['y6_pb']; names = [str(n) for n in a[f'{o}_names']]; r = b[f'{o}_sd_post'] / a[f'{o}_sd_post']; meas = a[f'{o}_sd_post'] < 0.5 * a[f'{o}_sd_prior']
    print(f"{'block':22s}{'sd(P+B)/sd(P): median':>24s}{'best':>8s}{'measured by P (sd < 0.5 prior)':>34s}")
    B = {lab: np.array(i) for lab, i in blocks(names).items()}
    for lab, i in B.items(): print(f"{lab:22s}{np.median(r[i]):24.2f}{r[i].min():8.2f}{f'{meas[i].sum()} of {len(i)}':>34s}")
    for cv in ('lcdm', 'w0wa'):
        sa, sb = np.sqrt(np.diag(a[f'{o}_{cv}_C'])), np.sqrt(np.diag(b[f'{o}_{cv}_C']))
        print(f"{COSMO_NAME[cv]:8s} sd P alone: " + ", ".join(f"{k} {v:.4f}" for k, v in zip(KEYS[cv], sa)) + "\n         sd(P+B)/sd(P): "
              + ", ".join(f"{k} {v:.2f}" for k, v in zip(KEYS[cv], sb / sa)))
    fig, ax = plt.subplots(figsize=(13, 3.2)); ax.axhline(1, color='k', lw=0.6)
    ax.scatter(np.arange(len(r)), r, s=14, c=np.where(meas, '#1E88E5', COL['prior']))
    for lab, i in B.items(): ax.axvline(i[0] - 0.5, color='k', lw=0.4); ax.text(i[0], ax.get_ylim()[1], ' ' + lab, va='top', fontsize=8)
    ax.set(xlabel='MI parameter (grey: prior-dominated with P alone)', ylabel=r'Fisher $\sigma(P+B)\,/\,\sigma(P)$'); fig.tight_layout(); return fig


# ------------------------------------------------------------------------------------------------- growth from the expansion
def growth_from_h(a='bao_template_no_rs', b='growth_from_h'):
    """f, the two alphas and the growth ratios at the data redshifts: free per redshift (template a) against derived from the
    H(z) spline and the growth equation (template b). Filled: mean and 68% of each chain; open: the maxima of the posteriors
    (checks.py bestfit); grey: the fiducial LCDM. Printed: sd of b over sd of a, and the matter density of b at its best fit and in its chain."""
    ca, cb = load(f'chain_{a}.npz'), load(f'chain_{b}.npz'); z = np.asarray(cb['z'], float); n = len(z)
    na = [str(v) for v in ca['names']]; xa = flat(ca['x']); ia = lambda pre: [i for i, v in enumerate(na) if v.startswith(pre)]
    pre = {'f': 'ln f(', 'apar': 'ln alpha_par(', 'aper': 'ln alpha_per(', 'D': 'ln D('}
    cut = lambda d: {'f': d[..., :n], 'apar': d[..., n:2 * n], 'aper': d[..., 2 * n:3 * n], 'D': d[..., 3 * n:]}
    ba, bb = load(f'bestfit_{a}.npz'), load(f'bestfit_{b}.npz'); ja, jb = np.argmax(ba['lnpost']), np.argmax(bb['lnpost'])     # the maxima
    B, Bb = cut(np.asarray(cb['derived'], float)), cut(np.asarray(bb['derived'][jb], float))
    labs = {'f': r'$f(z)$', 'apar': r'$\alpha_\parallel(z)$', 'aper': r'$\alpha_\perp(z)$', 'D': r'$D(z)/D(z_N)$'}
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.8))
    for axi, k in zip(ax, labs):
        A = np.exp(xa[:, ia(pre[k])]); zz = z[:A.shape[1]]
        axi.plot(zz, np.exp(np.asarray(ca['prior_mean'])[np.array(ia(pre[k])) - int(ca['n_eft'])]), color='0.6', lw=1, label=r'fiducial $\Lambda$CDM')
        for v, bf, col, off, lab in ((A, np.exp(ba['x'][ja][ia(pre[k])]), COL[a], -0.012, LAB[a]), (B[k], Bb[k], COL[b], 0.012, LAB[b])):
            axi.errorbar(zz + off, v.mean(0), v.std(0), fmt='o', ms=4, color=col, capsize=2, label=lab)
            axi.plot(zz + off, bf, 'o', ms=6, mfc='none', color=col)
        axi.set(xlabel='z', ylabel=labs[k])
        print(f"{k:5s} sd(derived) / sd(free) at z = {np.round(zz, 2)}: {np.round(B[k].std(0) / A.std(0), 2)};  (derived - free) / sd(free): "
              f"{np.round((B[k].mean(0) - A.mean(0)) / A.std(0), 2)}")
    i = [str(v) for v in cb['names']].index('ln Omega_m (H0 r_d)^2'); m = flat(cb['x'])[:, i]
    print(f"ln Omega_m (H0 r_d)^2 / fiducial: best fit {bb['x'][jb][i]:+.2f}, chain {m.mean():+.2f} +- {m.std():.2f}")
    ax[0].legend(fontsize=8); fig.tight_layout(); return fig


def growth_validity():
    """checks.py growth: what the H(z) spline + growth equation give at the map of each direct-fit draw against the cosmology's
    own f, alphas and D ratios (max |Delta ln| over 300 draws and all redshifts), and the shift of the posterior mean implied
    by the likelihood difference, in direct sigma."""
    d = load('growth_from_h.npz'); print(f"{'cosmology':10s}{'f':>10s}{'alpha_par':>11s}{'alpha_per':>11s}{'D ratio':>10s}{'Delta chi2 sd':>15s}{'max |implied shift|':>21s}")
    for cv in ('lcdm', 'w0wa', 'ede'):
        e = np.abs(d[f'{cv}_dln']); n = (e.shape[1] + 1) // 4; sh = implied(d[f'{cv}_th'], d[f'{cv}_dlnL'])[0]
        print(f"{COSMO_NAME[cv]:10s}{e[:, :n].max():10.1e}{e[:, n:2 * n].max():11.1e}{e[:, 2 * n:3 * n].max():11.1e}{e[:, 3 * n:].max():10.1e}"
              f"{2 * d[f'{cv}_dlnL'].std():15.2f}{np.abs(sh).max():21.2f}")


# ------------------------------------------------------------------------------------------------- alpha_rs = 1: the frame
def frame_table(model='w0wa'):
    """checks.py exact_repr: the 2000 reference draws importance-weighted to each likelihood; shift of the posterior mean against
    the true likelihood with the same loops, in reference sigma. oracle: the exact spectrum in the alpha_rs = 1 frame (no template).
    _adapted (checks.py adapted): with the EFT priors rescaled per draw by its alpha_rs^-n, i.e. the h/Mpc priors in that frame."""
    keys = KEYS[model]; sel = [keys.index(k) for k in ('h', 'w0', 'wa') if k in keys]
    fmt = lambda v: ' / '.join(f'{v[i]:+.2f}' for i in sel) + f'  (max {np.abs(v).max():.2f})'
    print(f"{COSMO_NAME[model]}: shift of the mean of " + ' / '.join(keys[i] for i in sel) + " in reference sigma (max over all parameters)")
    print(f"{'likelihood':24s}{'EFT priors':>12s}{'loop emulator':>34s}{'exact loops':>34s}")
    for f, lab in (('exact_repr.npz', 'analysis'), ('exact_repr_wide.npz', 'x 1000')):
        d = load(f)
        for t in ('bao_template', 'bao_template_no_rs', 'oracle', 'no_rs_vs_oracle', 'oracle_adapted', 'bao_template_no_rs_adapted'):
            ks = [f'{model}_{t}_{e}_shift' for e in ('emu', 'exact')]
            if ks[0] in d.files: print(f"{t:28s}{'per draw' if 'adapted' in t else lab:>8s}" + "".join(f"{fmt(d[k]) if k in d.files else '':>34s}" for k in ks))


def prior_adapted(model='w0wa', params=('omega_cdm', 'lnAs', 'h', 'w0', 'wa')):
    """alpha_rs = 1 with the EFT priors as they are, and adapted per cosmology to the h/Mpc priors (checks.py adapted), against
    the direct fit: the 2000 direct-fit draws of checks.py exact_repr, weighted by L / L_true of bao_template_no_rs with either
    prior. Printed: mean +- sd, the shift against the direct fit in its sigma, and the effective fraction of the weighting."""
    d = load('exact_repr.npz'); keys = [str(k) for k in d[f'{model}_keys']]; th = d[f'{model}_th'][:, [keys.index(p) for p in params]]
    R = [('direct fit', None, 'direct'), (LAB['bao_template_no_rs'] + ', EFT priors as they are', 'bao_template_no_rs', 'bao_template_no_rs'),
         (LAB['bao_template_no_rs'] + ', EFT priors adapted per cosmology', 'bao_template_no_rs_adapted', 'adapted')]
    W = []; print(f"{COSMO_NAME[model]}: mean +- sd (shift against the direct fit, in its sigma)")
    for lab, t, _ in R:
        lw = np.zeros(len(th)) if t is None else d[f'{model}_L_{t}_emu'] - d[f'{model}_L_true_emu']; w = np.exp(lw - lw.max()); w /= w.sum(); W.append(w)
        mu = w @ th; sd = np.sqrt(w @ (th - mu)**2)
        print(f"   {lab.replace('$', ''):62s}" + ''.join(f"  {p} {m:.4f} +- {v:.4f} ({(m - th.mean(0)[i]) / th.std(0)[i]:+.2f})" for i, (p, m, v) in enumerate(zip(params, mu, sd)))
              + f"   [effective {1 / np.sum(w**2) / len(w):.2f}]")
    return triangle([mcsamples(th, list(params), lab, weights=w) for (lab, _, _), w in zip(R, W)], [True, False, False], [COL[c] for _, _, c in R], [lab for lab, _, _ in R])


def frame_figure(model='w0wa'):
    """Delta chi2 of each reference draw between the alpha_rs = 1 frame (oracle) and the true likelihood against ln alpha_rs of the
    draw; what the EFT priors alone predict for it; and what is left for the template (bao_template_no_rs - oracle)."""
    d, wd = load('exact_repr.npz'), load('exact_repr_wide.npz'); x = d[f'{model}_lnars']; pred = d[f'{model}_prior_pred']
    dc = lambda z, a, b, e: -2 * (z[f'{model}_L_{a}_{e}'] - z[f'{model}_L_{b}_{e}'])
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3)); c = COL['bao_template_no_rs']
    ax[0].scatter(x, dc(d, 'oracle', 'true', 'emu'), s=3, color=c, label='analysis EFT priors')
    if f'{model}_L_oracle_emu' in wd.files: ax[0].scatter(x, dc(wd, 'oracle', 'true', 'emu'), s=3, color=COL['prior'], label=r'EFT priors $\times 1000$')
    ax[0].set(xlabel=r'$\ln\alpha_{r_s}=\ln[h r_d/(h r_d)_{\rm fid}]$ of the draw', ylabel=r'$\Delta\chi^2$: $\alpha_{r_s}=1$ frame $-$ truth'); ax[0].legend(markerscale=4)
    for e, col, lab in (('emu', c, 'loop emulator'), ('exact', COL['lcdm'], 'exact loops')):
        m = dc(d, 'oracle', 'true', e); ax[1].scatter(pred, m, s=3, color=col, label=f'{lab}: corr {np.corrcoef(pred, m)[0, 1]:.2f}')
        print(f"{lab:14s}: slope of Delta chi2 against ln alpha_rs: measured {np.polyfit(x, m, 1)[0]:+.1f}, predicted from the EFT priors "
              f"{np.polyfit(x, pred, 1)[0]:+.1f}, correlation {np.corrcoef(pred, m)[0, 1]:.2f}; template alone (no_rs - oracle) "
              f"{np.polyfit(x, dc(d, 'bao_template_no_rs', 'oracle', e), 1)[0]:+.1f}, sd {dc(d, 'bao_template_no_rs', 'oracle', e).std():.2f}")
    lim = [min(pred.min(), -1), pred.max()]; ax[1].plot(lim, lim, 'k:', lw=0.8)
    ax[1].set(xlabel=r'predicted: $\sum[(\alpha_{r_s}^{-n}c-\mu)^2-(c-\mu)^2]/\sigma^2$ over the 42 profiled EFT parameters', ylabel=r'measured $\Delta\chi^2$'); ax[1].legend(markerscale=4)
    ax[2].scatter(x, dc(d, 'bao_template_no_rs', 'oracle', 'emu'), s=3, color=c)
    ax[2].set(xlabel=r'$\ln\alpha_{r_s}$ of the draw', ylabel=r'$\Delta\chi^2$: bao_template_no_rs $-$ $\alpha_{r_s}=1$ frame', ylim=ax[0].get_ylim())
    fig.tight_layout(); return fig


def projections(model):
    """LCDM / w0waCDM from DESI FS+BAO (BBN, n_s prior), without and with theta_s, both options."""
    keys = None; roots, cols, ls, leg = [], [], [], []
    for o in OPTS:
        p = load(f'projections_{o}.npz')
        for prior, lab in (('', 'DESI'), ('_theta_s', r'DESI + $\theta_s$')):
            n = f'{model}{prior}'
            if f'{n}_x' not in p.files: continue
            keys = [str(k) for k in p[f'{n}_keys']]; th = flat(p[f'{n}_x'])
            Om = (th[:, keys.index('omega_cdm')] + th[:, keys.index('omega_b')]) / th[:, keys.index('h')]**2
            roots.append(mcsamples(np.c_[th, Om, p[f'{n}_sigma8']], keys + ['Om', 's8'], f'{o}{prior}'))
            cols.append(COL[o]); ls.append('-' if prior else '--'); leg.append(f'{LAB[o]}: {lab}')
            print(f"{o:9s} {n:18s} " + ", ".join(f"{k} {th[:, i].mean():.4f}+-{th[:, i].std():.4f}" for i, k in enumerate(keys))
                  + f", Omega_m {Om.mean():.4f}+-{Om.std():.4f}, sigma8 {p[f'{n}_sigma8'].mean():.4f}+-{p[f'{n}_sigma8'].std():.4f}")
    params = ['Om', 'h', 's8'] + [k for k in ('w0', 'wa') if k in keys]
    return triangle(roots, [False] * len(roots), cols, leg, params=params, ls=ls)


# ------------------------------------------------------------------------------------------------- sectors
SECTORS = {'shape_vs_evolution': ('shape', 'evolution'), 'background_vs_perturbations': ('background', 'perturbations')}
SEC_COL = {'shape': '#2E7D32', 'evolution': '#E65100', 'background': '#E65100', 'perturbations': '#2E7D32'}


# what each sector holds (bao_template adds the BAO-template parameters alpha_rs, A, dSigma^2)
SEC_TEXT = {'shape': 'shape: P(k) bands (relative to the template)',
            'evolution': r'evolution: $f(z)$, $D(z)$, $\alpha_\parallel(z)$, $\alpha_\perp(z)$',
            'background': r'background: BAO/AP dilations $\alpha_\parallel(z)$, $\alpha_\perp(z)$',
            'perturbations': r'perturbations: P(k) bands, $f(z)$, $D(z)$'}
SEC_EXTRA = {'shape': ' + BAO amplitude, damping', 'evolution': r' + $\alpha_{r_s}$', 'background': r' + $\alpha_{r_s}$',
             'perturbations': ' + BAO amplitude, damping'}


def derived(th, s8):
    k = KEYS['lcdm']; oc = th[:, k.index('omega_cdm')]
    return np.c_[oc, (oc + th[:, k.index('omega_b')]) / th[:, k.index('h')]**2, th[:, k.index('h')], s8]


def sectors(split, o):
    """LCDM (DESI alone, no theta_s) from each sector of one split, against the fit to all MI parameters."""
    s = load(f'sectors_{o}.npz'); pj = load(f'projections_{o}.npz'); names = ['omega_cdm', 'Om', 'h', 's8']
    roots, cols, leg, ls = [], [], [], []
    for sec in SECTORS[split]:
        roots.append(mcsamples(derived(flat(s[f'{sec}_x']), s[f'{sec}_sigma8']), names, sec))
        extra = SEC_EXTRA[sec] if o == 'bao_template' or (o == 'bao_template_no_rs' and sec in ('shape', 'perturbations')) else ''
        cols.append(SEC_COL[sec]); ls.append('-'); leg.append(SEC_TEXT[sec] + extra)
    roots.append(mcsamples(derived(flat(pj['lcdm_x']), pj['lcdm_sigma8']), names, 'combined'))
    cols.append('#212121'); leg.append('all MI parameters together'); ls.append('--')
    print(f"{LAB[o]}, {split}:")
    for r_, sec in zip(roots, list(SECTORS[split]) + ['all']):
        v = r_.samples
        print(f"   {sec:14s}" + "   ".join(f"{n} {v[:, i].mean():.4f}+-{v[:, i].std():.4f}" for i, n in enumerate(('omega_cdm', 'Omega_m', 'h', 'sigma8'))))
    g = triangle(roots, [False] * len(roots), cols, leg, ls=ls)
    g.fig.suptitle(f"{LAB[o]}: {' | '.join(SECTORS[split])}", y=1.02, fontsize=11)
    return g


def sectors_vs_planck(split, o='agnostic', theta_s=False):
    """LCDM from each sector of one split and from all MI parameters, against Planck PR4 (filled). theta_s=True: the theta_s
    prior is applied to each sector and to the full fit separately (run.py sectors --theta-s yes, project)."""
    sfx = '_theta_s' if theta_s else ''; names = ['omega_cdm', 'Om', 'h', 's8']
    s = load(f'sectors_{o}.npz'); pj = load(f'projections_{o}.npz'); pl = load('planck_early_lcdm_samples.npz')
    th_pl = np.c_[pl['omega_cdm'], pl['omega_b'], pl['n_s'], pl['ln10As'], pl['h']]
    roots, cols, leg, ls, filled, tag = [mcsamples(derived(th_pl, load('planck_early_lcdm_sigma8.npz')['sigma8']), names, 'planck')], \
        [COL['planck']], [r'Planck PR4 $\Lambda$CDM'], ['-'], [True], ['planck']
    for sec in SECTORS[split]:
        roots.append(mcsamples(derived(flat(s[f'{sec}{sfx}_x']), s[f'{sec}{sfx}_sigma8']), names, sec)); tag.append(sec)
        cols.append(SEC_COL[sec]); ls.append('-'); filled.append(False)
        leg.append(SEC_TEXT[sec] + (SEC_EXTRA[sec] if o == 'bao_template' else '') + (r' + $\theta_s$' if theta_s else ''))
    roots.append(mcsamples(derived(flat(pj[f'lcdm{sfx}_x']), pj[f'lcdm{sfx}_sigma8']), names, 'combined')); tag.append('all')
    cols.append('#212121'); ls.append('--'); filled.append(False); leg.append('all MI parameters together' + (r' + $\theta_s$' if theta_s else ''))
    print(f"{LAB[o]}, {split}, " + (r'with theta_s' if theta_s else 'DESI alone') + ":")
    for r_, sec in zip(roots, tag):
        v = r_.samples
        print(f"   {sec:14s}" + "   ".join(f"{n} {v[:, i].mean():.4f}+-{v[:, i].std():.4f}" for i, n in enumerate(('omega_cdm', 'Omega_m', 'h', 'sigma8'))))
    g = triangle(roots, filled, cols, leg, ls=ls)
    g.fig.suptitle(f"{LAB[o]}: {' | '.join(SECTORS[split])}" + (r' with $\theta_s$' if theta_s else ', DESI alone'), y=1.02, fontsize=11)
    return g


def sector_theta_s(sector, o='agnostic'):
    """One sector projected onto lcdm, DESI alone (dashed) and with the theta_s prior (solid), against the theta_s prior alone
    (grey: the DESI-alone priors x theta_s, no data) and Planck PR4: where theta_s adds information to the sector."""
    names = ['omega_cdm', 'Om', 'h', 's8']; s = load(f'sectors_{o}.npz'); pl = load('planck_early_lcdm_samples.npz'); ts = load('theta_s_alone_lcdm.npz')
    th_pl = np.c_[pl['omega_cdm'], pl['omega_b'], pl['n_s'], pl['ln10As'], pl['h']]
    roots = [mcsamples(derived(th_pl, load('planck_early_lcdm_sigma8.npz')['sigma8']), names, 'planck'),
             mcsamples(derived(flat(ts['x']), ts['sigma8']), names, 'theta_s_alone'),
             mcsamples(derived(flat(s[f'{sector}_x']), s[f'{sector}_sigma8']), names, sector),
             mcsamples(derived(flat(s[f'{sector}_theta_s_x']), s[f'{sector}_theta_s_sigma8']), names, sector + '_theta_s')]
    text = SEC_TEXT[sector] + (SEC_EXTRA[sector] if o == 'bao_template' else '')
    leg = [r'Planck PR4 $\Lambda$CDM', r'$\theta_s$ prior alone (no DESI data)', text + ', DESI alone', text + r' + $\theta_s$']
    print(f"{LAB[o]}, {sector}:")
    for r_, lab in zip(roots, ('Planck PR4', 'theta_s alone', 'DESI alone', '+ theta_s')):
        v = r_.samples
        print(f"   {lab:14s}" + "   ".join(f"{n} {v[:, i].mean():.4f}+-{v[:, i].std():.4f}" for i, n in enumerate(('omega_cdm', 'Omega_m', 'h', 'sigma8'))))
    g = triangle(roots, [True, True, False, False], [COL['planck'], '#B0BEC5', SEC_COL[sector], SEC_COL[sector]], leg, ls=['-', '-', '--', '-'])
    g.fig.suptitle(f"{LAB[o]}: {sector} sector, where " + r'$\theta_s$' + " adds information", y=1.02, fontsize=11)
    return g


# ------------------------------------------------------------------------------------------------- master figure
def _band(ax, x, q, color, filled, lab=None, lw=1.2, ls='-'):
    if filled: ax.fill_between(x, q[1], q[3], color=color, alpha=0.35, lw=0, label=lab)
    else:
        ax.plot(x, q[1], color=color, lw=lw, ls=ls, label=lab); ax.plot(x, q[3], color=color, lw=lw, ls=ls)


def pk_compare():
    """The reconstructed linear spectrum at z_N of both options, absolute, with their MI priors."""
    b = load('master_bands.npz'); kh, T = b['kh'], b['T_fid']
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 7.5), sharex=True, gridspec_kw={'height_ratios': [2, 1.2]})
    for ax, norm in ((a1, 1.0), (a2, T)):
        for o in OPTS:
            _band(ax, kh, b[f'{o}_post_Pk'] / norm, COL[o], o == OPTS[0], f'{LAB[o]} (68%)')
            _band(ax, kh, b[f'{o}_prior_Pk'] / norm, COL[o], False, f'{LAB[o]}: MI prior (68%)', lw=0.9, ls=':')
            ax.plot(kh, b[f'{o}_post_Pk'][2] / norm, color=COL[o], lw=1.6, ls='-' if o == OPTS[0] else '--', label=f'{LAB[o]}: median')
        ax.axvspan(*b['data_range'], color='0.94', zorder=-1)
    a1.plot(kh, T, color='k', lw=0.8, label='fiducial (template)')
    a1.set_xscale('log'); a1.set_yscale('log'); a1.set_ylim(1e2, 1e5)
    a1.set_ylabel(r'$\tilde P_{\rm lin}(\tilde k, z_N)\ [({\rm Mpc}/h_{\rm fid})^3]$'); a1.legend(fontsize=8, loc='lower left')
    a2.axhline(1, color='k', lw=0.8); a2.set_yscale('log'); a2.set_ylim(0.08, 8); a2.set_ylabel(r'$\tilde P_{\rm lin} / P_{\rm fid}$')
    a2.set_xlabel(r'$\tilde k = k\, r_d/r_d^{\rm fid}\ [h_{\rm fid}/{\rm Mpc}]$')
    a1.set_title(f"linear power spectrum at z_N = {float(b['zN']):.3f}, sound-horizon units (grey: data range)", fontsize=10)
    m = (kh > 0.02) & (kh < 0.2)
    for o in OPTS:
        w = 0.5 * np.log(b[f'{o}_post_Pk'][3] / b[f'{o}_post_Pk'][1]); wp = 0.5 * np.log(b[f'{o}_prior_Pk'][3] / b[f'{o}_prior_Pk'][1])
        print(f"{o:9s} 68% half-width of ln P in 0.02-0.2 h/Mpc: median {np.median(w[m]):.3f} (prior {np.median(wp[m]):.3f})")
    o0, o1 = OPTS[0], OPTS[1]; da = np.log(b[f'{o0}_post_Pk'][2] / b[f'{o1}_post_Pk'][2]); sa = 0.5 * np.log(b[f'{o0}_post_Pk'][3] / b[f'{o0}_post_Pk'][1])
    print(f"{o0} - {o1} medians in 0.02-0.2 h/Mpc: max |Delta ln P| / sigma_{o0} = {np.abs(da[m] / sa[m]).max():.2f}")
    fig.tight_layout(); return fig


def master(theta_s=False):
    """MI measurements (both options) vs LCDM and w0waCDM from DESI FS+BAO [+ theta_s] and Planck PR4 LCDM."""
    b = load('master_bands.npz'); kh, zg, zi, zN = b['kh'], b['zg'], b['zi'], float(b['zN'])
    sfx = '_theta_s' if theta_s else ''
    th = [('lcdm' + sfx, COL['lcdm'], r'$\Lambda$CDM', True), ('w0wa' + sfx, COL['w0wa'], r'$w_0w_a$CDM', False)]
    fig = plt.figure(figsize=(17, 9.5)); gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.27)
    axA, axB, axC = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    axV, axM, axH = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]), fig.add_subplot(gs[1, 2])
    # A: the spectrum
    for o in OPTS:
        _band(axA, kh, b[f'{o}_post_Pk'], COL[o], o == OPTS[0], LAB[o])
        _band(axA, kh, b[f'{o}_prior_Pk'], COL[o], False, None, lw=0.9, ls=':')
    for m, c, lab, _ in th: axA.plot(kh, b[f'{m}_Pk'][2], color=c, lw=1.0, label=lab)
    axA.plot(kh, b['planck_Pk'][2], color=COL['planck'], lw=1.0, ls='--', label=r'Planck PR4 $\Lambda$CDM')
    axA.axvspan(*b['data_range'], color='0.94', zorder=-1); axA.set_xscale('log'); axA.set_yscale('log')
    axA.set_xlim(2e-3, 0.5); axA.set_ylim(3e2, 3e4); axA.legend(fontsize=7.5, loc='lower left')
    axA.set_xlabel(r'$\tilde k\ [h_{\rm fid}/{\rm Mpc}]$'); axA.set_ylabel(r'$\tilde P_{\rm lin}(\tilde k, z_N)\ [({\rm Mpc}/h_{\rm fid})^3]$')
    axA.set_title('A  linear power spectrum (dotted: MI priors)', fontsize=10)
    # B, C: growth
    for ax, q, ylab, title in ((axB, 'fsig', r'$f\sigma_{s8}(z)$', r'B  growth: $f\sigma_{s8}$'),
                               (axC, 'G', r'$[D(a)/a]\,/\,[D(a_N)/a_N]$', r'C  growth: $D(a)/a$')):
        for m, c, lab, fill in th: _band(ax, zg, b[f'{m}_{q}'], c, fill, lab)
        _band(ax, zg, b[f'planck_{q}'], COL['planck'], False, r'Planck PR4 $\Lambda$CDM', ls='--')
        n = len(zi) if q == 'fsig' else len(zi) - 1
        for o, dz, mk in zip(OPTS, (-0.02, 0.02), ('o', 's')):
            v = b[f'{o}_post_{q}'][:, :n]; p = b[f'{o}_prior_{q}'][:, :n]
            ax.errorbar(zi[:n] + dz, v[2], yerr=[v[2] - v[1], v[3] - v[2]], fmt=mk, color=COL[o], ms=6, capsize=2, label=LAB[o])
            ax.errorbar(zi[:n] + 2 * dz, p[2], yerr=[p[2] - p[1], p[3] - p[2]], fmt='none', ecolor=COL['prior'], elinewidth=0.8, capsize=4)
        ax.set_xlabel('$z$'); ax.set_ylabel(ylab); ax.set_title(title + ' (grey bars: MI priors)', fontsize=10); ax.set_xlim(0.1, 1.7)
    axC.plot([zN], [1.0], 's', color='k', mfc='white', ms=6, label=r'reference $a_N$')
    axB.set_ylim(0.15, 0.75); axC.set_ylim(0.3, 1.9); axB.legend(fontsize=7.5); axC.legend(fontsize=7.5)
    # D: BAO, relative to the LCDM best fit of this version
    for q, ax, lab in (('V', axV, 'D_V'), ('M', axM, 'D_M'), ('H', axH, 'D_H')):
        ref = b[f'lcdm{sfx}_best_{q}']
        for m, c, l, fill in th: _band(ax, zg, b[f'{m}_{q}'] / ref, c, fill, l)
        _band(ax, zg, b[f'planck_{q}'] / ref, COL['planck'], False, r'Planck PR4 $\Lambda$CDM', ls='--')
        for o, dz, mk in zip(OPTS, (-0.012, 0.012), ('o', 's')):
            z = b[f'{o}_post_{q}_z']; v = b[f'{o}_post_{q}'] / np.interp(z, zg, ref)[:, None]
            ax.errorbar(z + dz, v[:, 2], yerr=[v[:, 2] - v[:, 1], v[:, 3] - v[:, 2]], fmt=mk, color=COL[o], ms=6, capsize=2, label=LAB[o])
        ax.axhline(1, color='k', lw=0.6); ax.set_xlim(0.1, 1.7); ax.set_xlabel('$z$')
        ax.set_ylabel(rf'$({lab}/r_d)\,/\,({lab}/r_d)_{{\Lambda{{\rm CDM\ bf}}}}$'); ax.set_title(f'D  BAO: ${lab}/r_d$', fontsize=10)
    axV.legend(fontsize=7.5)
    fig.suptitle('DESI DR1 FS+BAO, model-independent measurements (both templates) vs fits from DESI FS+BAO'
                 + (r' + $\theta_s$' if theta_s else ' alone') + ' and Planck PR4 (68%)', fontsize=12)
    return fig


# ------------------------------------------------------------------------------------------------- Fisher
def fisher_vs_chain():
    f = load('fisher.npz')
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
    for ax, o in zip(axs, OPTS):
        d = load(f'chain_{o}.npz'); ne = int(d['n_eft']); sd = flat(d['x'])[:, ne:].std(0); names = list(d['names'][ne:])
        r = f[f'{o}_sd_post'] / sd
        for blk, idx in blocks(names).items(): ax.plot(idx, r[idx], 'o', ms=4, label=blk)
        ax.axhline(1, color='k', lw=0.8); ax.set_title(LAB[o], fontsize=10); ax.set_xlabel('MI parameter'); ax.set_yscale('log')
        print(f"{o:9s} Fisher / chain sigma: " + ", ".join(f"{blk} {np.median(r[i]):.2f}" for blk, i in blocks(names).items()) + " (median)")
    axs[0].set_ylabel(r'$\sigma_{\rm Fisher}/\sigma_{\rm chain}$'); axs[1].legend(fontsize=8, ncol=2)
    return fig


def fisher_pk():
    f, b = load('fisher.npz'), load('master_bands.npz'); kh = f['kh']
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for o in OPTS:
        ax.plot(kh, f[f'{o}_lnP_sd_post'], color=COL[o], lw=1.6, label=f'{LAB[o]}: Fisher')
        ax.plot(kh, f[f'{o}_lnP_sd_prior'], color=COL[o], lw=1.0, ls=':', label=f'{LAB[o]}: prior')
        ax.plot(kh, 0.5 * np.log(b[f'{o}_post_Pk'][3] / b[f'{o}_post_Pk'][1]), color=COL[o], lw=1.0, ls='--', label=f'{LAB[o]}: chain (68%/2)')
    ax.axvspan(0.02, 0.2, color='0.94', zorder=-1); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(2e-3, 0.5)
    ax.set_xlabel(r'$\tilde k\ [h_{\rm fid}/{\rm Mpc}]$'); ax.set_ylabel(r'$\sigma(\ln\tilde P_{\rm lin})$'); ax.legend(fontsize=7.5)
    return fig


def fisher_growth_bao():
    f = load('fisher.npz'); zi = f['zi']
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.8))
    for o, dz, mk in zip(OPTS, (-0.015, 0.015), ('o', 's')):
        a1.plot(zi + dz, f[f'{o}_lnfsig_sd_post'], mk, color=COL[o], label=f'{LAB[o]}: Fisher')
        a1.plot(zi + dz, f[f'{o}_lnfsig_sd_prior'], mk, color=COL[o], mfc='none', label=f'{LAB[o]}: prior')
        a2.plot(np.arange(len(f[f'{o}_lnbao_sd_post'])) + dz * 10, f[f'{o}_lnbao_sd_post'], mk, color=COL[o], label=LAB[o])
    a1.set_yscale('log'); a1.set_xlabel('$z$'); a1.set_ylabel(r'$\sigma(\ln f\sigma_{s8})$'); a1.legend(fontsize=8)
    a2.set_xlabel('BAO point (BGS, LRG1 par/per, ..., QSO)'); a2.set_ylabel(r'$\sigma(\ln\alpha_{\rm BAO})$'); a2.legend(fontsize=8)
    for o in OPTS:
        print(f"{o:9s} sigma(ln f sigma_s8): {np.round(f[f'{o}_lnfsig_sd_post'], 3)}; sigma(ln alpha_BAO): {np.round(f[f'{o}_lnbao_sd_post'], 4)}")
    fig.tight_layout(); return fig


def fisher_cosmo(model):
    """Fisher ellipses of both options against their projected chains (BBN, n_s prior, no theta_s)."""
    from getdist.gaussian_mixtures import GaussianND
    f = load('fisher.npz'); roots, cols, ls, leg = [], [], [], []
    keys = [str(k) for k in f[f'agnostic_{model}_keys']]
    for o in OPTS:
        C = f[f'{o}_{model}_C']; mu = f[f'{o}_{model}_theta']
        roots.append(GaussianND(mu, C, names=keys, labels=[TEX[k] for k in keys], label=f'{o} Fisher'))
        cols.append(COL[o]); ls.append('--'); leg.append(f'{LAB[o]}: Fisher')
        p = load(f'projections_{o}.npz'); n = model
        if f'{n}_x' in p.files:
            th = flat(p[f'{n}_x']); roots.append(mcsamples(th, keys, f'{o} chain')); cols.append(COL[o]); ls.append('-'); leg.append(f'{LAB[o]}: projected chain')
            print(f"{o:9s} {model}: sigma Fisher / chain " + ", ".join(f"{k} {np.sqrt(C[i, i]):.4f}/{th[:, i].std():.4f}" for i, k in enumerate(keys)))
    return triangle(roots, [False] * len(roots), cols, leg, ls=ls)


# ------------------------------------------------------------------------------------------------- the alpha_rs gauge (checks.py degeneracy)
DEG = {'mock': 'noiseless mock', 'data': 'DESI DR1 FS+BAO'}
MOVE_COL = {'A': '#C62828', 'B': '#5D4037', 'Bc': '#2E7D32', 'T': '#212121'}
MOVE_LAB = {'A': r'A: $\alpha_{r_s}\to\lambda\,\alpha_{r_s}$', 'B': r'B: every $\alpha\to\alpha/\lambda$',
            'Bc': r'Bc: B + broadband dilated back', 'T': r'$T_\lambda$: $\alpha_{r_s}$, every $\alpha$ $\times\lambda$ + broadband'}
LADDER = [('exact_noresum_wide', 'exact 1-loop, no IR resummation, wide EFT priors'), ('exact_wide', '+ IR resummation'),
          ('wide', 'loop emulator instead of exact loops'), ('analysis', '+ EFT priors = the analysis likelihood'),
          ('nodes', '+ broadband on the 21 nodes')]


def _deg(case):
    return load(f'degeneracy_{case}.npz')


def deg_moves(case):
    d = _deg(case); lam, lp = d['lams'], d['lams_profile']
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for m in ('A', 'B', 'Bc', 'T'):
        ax[0].plot(lam, d[f'fixed_{m}_nodes'], color=MOVE_COL[m], lw=1.6, label=MOVE_LAB[m])
        ax[0].plot(lam, d[f'fixed_{m}_exact'], color=MOVE_COL[m], lw=1.0, ls=':')
        ax[1].plot(lam, d[f'bao_{m}'], color=MOVE_COL[m], lw={'A': 3.0, 'B': 1.8, 'Bc': 1.0, 'T': 1.6}[m], ls={'B': '--', 'Bc': ':'}.get(m, '-'), label=MOVE_LAB[m])
    for m in ('A', 'B', 'T'):
        ax[2].plot(lp, d[f'prof_{m}'], 'o-', color=MOVE_COL[m], ms=4, lw=1.2, label=MOVE_LAB[m])
    ax[0].set_yscale('symlog', linthresh=1.0)
    for a, t in zip(ax, ['fixed broadband + EFT (solid: 21 nodes, dotted: exact spectrum)', 'BAO α alone', 'profiled over ln a, A, ΔΣ², b1, c2']):
        a.axhline(0, color='k', lw=0.6); a.set_xlabel(r'$\lambda$'); a.set_title(t, fontsize=10)
    ax[0].set_ylabel(r'$\Delta\chi^2$'); ax[0].legend(fontsize=8); ax[1].legend(fontsize=8); fig.suptitle(DEG[case]); fig.tight_layout()
    print(f"{DEG[case]}: Delta chi2 from the MI best fit (chi2 = {float(d['chi2_base']):.3f})")
    print(f"{'lam':>6s} {'A':>9s} {'B':>9s} {'Bc exact':>9s} {'Bc nodes':>9s} {'T exact':>9s} {'T nodes':>9s} | {'BAO: A':>8s} {'B':>8s} {'Bc':>8s} {'T':>8s}")
    for i, l in enumerate(lam):
        print(f"{l:6.3f} " + " ".join(f"{d[k][i]:9.3f}" for k in ('fixed_A_nodes', 'fixed_B_nodes', 'fixed_Bc_exact', 'fixed_Bc_nodes', 'fixed_T_exact', 'fixed_T_nodes'))
              + " | " + " ".join(f"{d[f'bao_{m}'][i]:8.3f}" for m in ('A', 'B', 'Bc', 'T')))
    print(f"\nprofiled (Delta from the profile at lam = 1, chi2 = {float(d['prof_base']):.3f}):")
    print(f"{'lam':>6s} {'A':>9s} {'B':>9s} {'A - B':>9s} {'T':>9s}")
    for i, l in enumerate(lp):
        print(f"{l:6.3f} {d['prof_A'][i]:9.3f} {d['prof_B'][i]:9.3f} {d['prof_A'][i] - d['prof_B'][i]:9.3f} {d['prof_T'][i]:9.3f}")
    return fig


def deg_ladder(case):
    d = _deg(case); lam, lp = d['lams_ladder'], d['lams_profile']; cols = plt.cm.viridis(np.linspace(0, 0.9, len(LADDER) + 1))
    nsum = float(np.sum(d['eft_n_pred'])) * len(d['skies'])
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
    for a in ax:
        for (k, lab), c in zip(LADDER, cols):
            a.plot(lam, d['ladder_nodes'] if k == 'nodes' else d[f'ladder_{k}'], 'o-', color=c, ms=3.5, lw=1.4, label=lab)
        a.plot(lp, d['prof_T'], 's-', color=cols[-1], ms=3.5, lw=1.4, label='+ re-fitted (ln a, A, ΔΣ², b1, c2)')
        a.axhline(0, color='k', lw=0.6); a.set_xlabel(r'$\lambda$')
    ll = np.linspace(lam[0], lam[-1], 100)
    ax[1].plot(lam, d['ladder_marg_wide'], 'D-', color='#8D6E63', ms=3.5, lw=1.2, label='EFT marginalized with log det F₂ (wide priors)')
    ax[1].plot(ll, -2 * nsum * np.log(ll), '--', color='#8D6E63', lw=1, label=rf'$-2\,\Sigma n\,\ln\lambda$ = ${-2 * nsum:.0f}\ln\lambda$')
    ax[1].plot(d['lams'], d['fixed_A_nodes'], color=MOVE_COL['A'], lw=1.6, label=r'A: $\alpha_{r_s}$ alone (scale)')
    ax[1].set_yscale('symlog', linthresh=0.1)
    ax[0].set_ylabel(r'$\Delta\chi^2$ along $T_\lambda$'); ax[0].set_title('one ingredient added at a time', fontsize=10)
    ax[1].set_title('same, symlog, with α_rs alone and the marginalized likelihood', fontsize=10)
    ax[0].legend(fontsize=8); ax[1].legend(fontsize=7, loc='lower right'); fig.suptitle(DEG[case]); fig.tight_layout()
    keys = [k for k, _ in LADDER]
    print(f"{DEG[case]}: Delta chi2 along T_lam (exact transformed spectrum unless 'nodes')")
    print(f"{'lam':>6s} " + " ".join(f"{k[:14]:>15s}" for k in keys) + f" {'marg_wide':>10s} {'-2 Sn lnlam':>12s}")
    for i, l in enumerate(lam):
        print(f"{l:6.3f} " + " ".join(f"{(d['ladder_nodes'] if k == 'nodes' else d[f'ladder_{k}'])[i]:15.4f}" for k in keys)
              + f" {d['ladder_marg_wide'][i]:10.3f} {-2 * nsum * np.log(l):12.3f}")
    print("chi2 at lam = 1 per variant: " + ", ".join(f"{k} {float(d[f'ladder_base_{k}']):.2f}" for k in d['variants']))
    return fig


def deg_eft_powers(case):
    d = _deg(case); n, sp, names, pred = d['eft_n'], d['eft_n_spread'], list(d['eft_names']), d['eft_n_pred']
    print(f"{DEG[case]}: EFT templates at T_1.02 vs the best fit, T' = lam^-n T; median n over the k-bins (5-95% spread)")
    print(f"{'sky':6s} " + " ".join(f"{p:>14s}" for p in names)); print(f"{'pred.':6s} " + " ".join(f"{v:14d}" for v in pred))
    for i, s in enumerate(d['skies']):
        print(f"{s:6s} " + " ".join(f"{n[i, k]:+7.3f} ({sp[i, k]:.0e})" for k in range(len(names))))
    if 'exact_eft_n' in d.files:
        print("exact one-loop (no emulator):")
        for i, s in enumerate(d['skies']):
            print(f"{s:6s} " + " ".join(f"{d['exact_eft_n'][i, k]:+7.3f} ({d['exact_eft_n_spread'][i, k]:.0e})" for k in range(len(names))))


def deg_fisher(case, exact=False):
    d0 = _deg(case); p = 'exact_' if exact else ''; d = {k[len(p):]: d0[k] for k in d0.files if k.startswith(p)}
    names = list(d0['names']); kap = {m: float(d[f'kappa_{m}']) for m in ('A', 'B', 'Bc', 'T')}; kmin = float(d['kappa_min_rs'])
    v, dopt = d['gauge'], d['d_opt']; what = 'exact one-loop, wide EFT priors' if exact else 'analysis likelihood (loop emulator)'
    ia = [i for i, s in enumerate(names) if s.startswith('ln alpha_par')], [i for i, s in enumerate(names) if s.startswith('ln alpha_per')]
    il = [i for i, s in enumerate(names) if s.startswith('ln a(')]; kn = np.array([float(names[i][5:-1]) for i in il])
    z = np.array([float(names[i].split('(')[1][:-1]) for i in ia[0]])
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5)); ax = ax.ravel()
    labs = ['A: α_rs alone', 'B: α alone', 'Bc', 'T: gauge v', 'α_rs + the data\'s best\ncompensation']; vals = [kap['A'], kap['B'], kap['Bc'], kap['T'], kmin]
    ax[0].bar(range(5), vals, color=[MOVE_COL['A'], MOVE_COL['B'], MOVE_COL['Bc'], MOVE_COL['T'], '#9E9E9E']); ax[0].set_yscale('log')
    for i, val in enumerate(vals): ax[0].text(i, val * 1.3, rf'$\sigma$={1 / np.sqrt(val):.3g}', ha='center', fontsize=8)
    ax[0].set_xticks(range(5)); ax[0].set_xticklabels(labs, fontsize=8); ax[0].set_ylabel(r'$\kappa=d^\top F d$ per unit $\ln\lambda$')
    ax[0].set_title(r'curvature: $\Delta\chi^2\simeq\kappa\,(\ln\lambda)^2$', fontsize=10)
    mu = d['mu']; j = np.arange(len(mu))
    ax[1].semilogy(j, mu, 'k.-', lw=0.8, label='eigenvalues of the prior-whitened Fisher'); ax[1].set_xlabel('eigenvector (stiff → soft)')
    ax[1].set_ylabel('data / prior information'); b = ax[1].twinx()
    b.bar(j - 0.2, d['overlap_T'], 0.4, color=MOVE_COL['T'], label=r'gauge $v$: $|\langle \hat v, u_j\rangle|^2$')
    b.bar(j + 0.2, d['overlap_A'], 0.4, color=MOVE_COL['A'], alpha=0.6, label=r'$\alpha_{r_s}$ alone')
    b.set_ylabel('overlap'); b.set_ylim(0, 1); b.legend(fontsize=8, loc='upper center'); ax[1].legend(fontsize=8, loc='lower left')
    ax[1].set_title('where the gauge direction sits in the Fisher spectrum', fontsize=10)
    for k, mk, lab in ((0, 'o', r'$\alpha_\parallel$'), (1, 's', r'$\alpha_\perp$')):
        ax[2].plot(z, v[ia[k]], mk + '-', color=MOVE_COL['T'], ms=6, mfc='none', label=f'gauge v: {lab}')
        ax[2].plot(z, dopt[ia[k]], mk, color='#9E9E9E', ms=4, label=f'data best compensation: {lab}')
    ax[2].set_xlabel('z'); ax[2].set_ylabel(r'$d\ln\alpha$ per unit $d\ln\alpha_{r_s}$'); ax[2].legend(fontsize=8)
    ax[2].set_title(r'how the $\alpha$ move when $\alpha_{r_s}$ moves', fontsize=10)
    ax[3].semilogx(kn, v[il], 'o-', color=MOVE_COL['T'], mfc='none', label='gauge v (dilation of a(k) on the nodes)')
    ax[3].semilogx(kn, dopt[il], 's', color='#9E9E9E', ms=4, label='data best compensation')
    ax[3].axvspan(0.02, 0.2, color='0.93', zorder=-1); ax[3].set_xlabel('node k [h/Mpc]'); ax[3].set_ylabel(r'$d\ln a$ per unit $d\ln\alpha_{r_s}$')
    ax[3].legend(fontsize=8); ax[3].set_title('… and the broadband (grey: data k-range)', fontsize=10)
    fig.suptitle(f'{DEG[case]}: {what}'); fig.tight_layout()
    ie = [names.index('BAO amplitude A'), names.index('wiggle damping dSigma^2 [Mpc^2]')]
    small = d['overlap_T'][mu < 1].sum()
    print(f"{DEG[case]}, {what}: Gauss-Newton Fisher check, -2 ln L rebuilt from residuals {d['fisher_chi2_check'][0]:.4f} vs likelihood {d['fisher_chi2_check'][1]:.4f}")
    print("curvature kappa per unit ln lam and sigma = kappa^-1/2:  " + ", ".join(f"{l.split(':')[0]} {k:.4g} ({1 / np.sqrt(k):.3g})"
                                                                       for l, k in zip(labs, vals)))
    print(f"kappa(T) / kappa(A) = {kap['T'] / kap['A']:.2e};  correlation of A and Bc: rho = {float(d['rho_A_Bc']):.8f} (1 - rho = {1 - float(d['rho_A_Bc']):.2e})")
    print(f"cos(gauge, softest eigenvector) = {float(d['cos_soft_T']):.4f};  fraction of the gauge in directions the data inform less than the prior "
          f"(eigenvalue < 1): {small:.4f};  data/prior information along the gauge {float(d['rayleigh_T']):.3g} vs along alpha_rs alone {float(d['rayleigh_A']):.3g}")
    print(f"A, dSigma^2 (barely constrained by the data, so its best compensation is arbitrary): gauge v {v[ie[0]]:+.3f}, {v[ie[1]]:+.3f};  data best compensation {dopt[ie[0]]:+.3f}, {dopt[ie[1]]:+.3f}")
    return fig


def deg_map():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for a, case in zip(ax, ('mock', 'data')):
        d = _deg(case)
        a.plot(d['map_draw_lnrs'], d['map_exact'], '.', color=MOVE_COL['T'], ms=3, label='analysis likelihood, exact transformed spectrum')
        a.plot(d['map_draw_lnrs'], d['map_nodes'], '.', color=COL['bao_template'], ms=3, alpha=0.6, label='broadband on the 21 nodes')
        if 'map_sub' in d.files:
            x = d['map_draw_lnrs'][d['map_sub']]
            a.plot(x, d['map_wide'], 'o', mfc='none', color='#2E7D32', ms=5, label='exact spectrum, wide EFT priors (80 draws)')
            a.plot(x, d['map_exact_wide'], 's', color='#C62828', ms=4, label='+ exact loops instead of the emulator')
            print(f"{DEG[case]}, 80 of the draws, |Delta chi2| median / 95%: analysis {np.median(np.abs(d['map_exact'][d['map_sub']])):.3f} / "
                  f"{np.percentile(np.abs(d['map_exact'][d['map_sub']]), 95):.3f}, wide EFT priors {np.median(np.abs(d['map_wide'])):.3f} / "
                  f"{np.percentile(np.abs(d['map_wide']), 95):.3f}, exact loops {np.median(np.abs(d['map_exact_wide'])):.3f} / "
                  f"{np.percentile(np.abs(d['map_exact_wide']), 95):.3f}")
        a.axhline(0, color='k', lw=0.6); a.set_xlabel(r'$\ln\alpha_{r_s}$ of the draw'); a.set_title(DEG[case], fontsize=10)
        print(f"{DEG[case]}: {len(d['map_exact'])} draws, Delta chi2 after T_(1/alpha_rs): exact median {np.median(d['map_exact']):+.4f}, "
              f"5-95% [{np.percentile(d['map_exact'], 5):+.4f}, {np.percentile(d['map_exact'], 95):+.4f}]; nodes median {np.median(d['map_nodes']):+.4f}, "
              f"5-95% [{np.percentile(d['map_nodes'], 5):+.4f}, {np.percentile(d['map_nodes'], 95):+.4f}]")
    ax[0].set_ylabel(r'$\Delta\chi^2$: $\alpha_{r_s}\to1$, $\alpha\to\alpha/\alpha_{r_s}$, broadband dilated'); ax[0].legend(fontsize=8); fig.tight_layout()
    return fig


def deg_ridge():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2)); x = np.linspace(-0.2, 0.2, 400)
    gauss = lambda x, m, s: np.exp(-0.5 * ((x - m) / s)**2) / (np.sqrt(2 * np.pi) * s)
    print(f"{'':16s} {'chain mean':>10s} {'chain sd':>9s} {'best fit':>9s} {'truth':>8s} {'sd prior|v':>10s} {'sd lkl|v':>9s} {'sd both|v':>9s} {'sd lkl|v exact loops':>20s}")
    for a, case in zip(ax, ('mock', 'data')):
        d = _deg(case); s = d['chain_lnrs']; tr = float(d['truth_lnrs']) if 'truth_lnrs' in d.files else np.nan
        a.hist(s, bins=60, density=True, color=COL['bao_template'], alpha=0.45, label='chain')
        a.plot(x, gauss(x, float(d['map_lnrs']), float(d['sigma_prior_T'])), 'k--', lw=1.2, label=rf"MI prior along $v$ at the best fit ($\sigma$={float(d['sigma_prior_T']):.3f})")
        a.plot(x, gauss(x, 0., float(d['sigma_prior_A'])), color='#9E9E9E', lw=1, label=rf"prior on $\ln\alpha_{{r_s}}$ alone ($\sigma$={float(d['sigma_prior_A']):.3f})")
        a.axvline(float(d['map_lnrs']), color='k', lw=0.8, label='best fit')
        if np.isfinite(tr): a.axvline(tr, color=MOVE_COL['A'], lw=1.2, label='truth')
        a.set_xlabel(r'$\ln\alpha_{r_s}$'); a.set_title(DEG[case], fontsize=10); a.legend(fontsize=7)
        print(f"{DEG[case]:16s} {s.mean():+10.4f} {s.std():9.4f} {float(d['map_lnrs']):+9.4f} {tr:+8.4f} {float(d['sigma_prior_T']):10.4f} "
              f"{float(d['sigma_like_T']):9.3g} {float(d['sigma_post_T']):9.4f} {1 / np.sqrt(float(d['exact_kappa_T'])):20.3g}")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------- alpha_rs free vs alpha_rs = 1
def compare_mi(a='bao_template', b='bao_template_no_rs'):
    """The MI posteriors of two templates on what both define: the BAO alphas (alpha / alpha_rs where alpha_rs is free, and
    alpha_iso = alpha_par^1/3 alpha_per^2/3), ln f, ln D ratios, the BAO amplitude A and damping dSigma^2. Shift of b's mean
    in a's sigma, and b's width over a's."""
    def q(o):
        d = load(f'chain_{o}.npz'); ne = int(d['n_eft']); x = flat(d['x'])[:, ne:]; names = list(d['names'][ne:])
        lrs = x[:, names.index('ln alpha_rs')] if 'ln alpha_rs' in names else 0.
        zs = [n.split('(')[1][:-1] for n in names if n.startswith('ln alpha_par')]; out = {}
        for kind in ('par', 'per'):
            for z in zs: out[rf'$\alpha_{{\rm {kind}}}^{{\rm BAO}}({z})$'] = np.exp(x[:, names.index(f'ln alpha_{kind}({z})')] - lrs)
        for z in zs: out[rf'$\alpha_{{\rm iso}}^{{\rm BAO}}({z})$'] = (out[rf'$\alpha_{{\rm par}}^{{\rm BAO}}({z})$']**(1 / 3)
                                                                       * out[rf'$\alpha_{{\rm per}}^{{\rm BAO}}({z})$']**(2 / 3))
        for i, n in enumerate(names):
            if n.startswith('ln f') or n.startswith('ln D'): out[n] = x[:, i]
        out['A'], out[r'$\Delta\Sigma^2$'] = x[:, names.index('BAO amplitude A')], x[:, names.index('wiggle damping dSigma^2 [Mpc^2]')]
        return out
    A, B = q(a), q(b); keys = [k for k in A if k in B]
    sh = np.array([(B[k].mean() - A[k].mean()) / A[k].std() for k in keys]); wr = np.array([B[k].std() / A[k].std() for k in keys])
    fig, ax = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True); i = np.arange(len(keys))
    ax[0].bar(i, sh, color=COL[b]); ax[0].axhline(0, color='k', lw=0.6); ax[0].set_ylabel(f'shift of the mean\n[{a} $\\sigma$]')
    ax[1].bar(i, wr - 1, bottom=1, color=COL[b]); ax[1].axhline(1, color='k', lw=0.6); ax[1].set_ylabel('width ratio')
    ax[1].set_xticks(i); ax[1].set_xticklabels([k.replace('ln ', r'ln ') for k in keys], rotation=90, fontsize=8)
    ax[0].set_title(f'{LAB[b]} vs {LAB[a]}: MI posterior on the quantities both define', fontsize=10); fig.tight_layout()
    print(f"{'':28s} {a:>22s} {b:>22s} {'shift':>7s} {'width':>6s}")
    for k, s, w in zip(keys, sh, wr):
        print(f"{k.replace('$', ''):28s} {A[k].mean():12.4f} +- {A[k].std():.4f} {B[k].mean():12.4f} +- {B[k].std():.4f} {s:+7.2f} {w:6.2f}")
    return fig


def _mi_all(o):
    """Every MI parameter of template o that has a frame-independent meaning: sampled EFT, BAO alphas (alpha / alpha_rs where
    alpha_rs is free) and alpha_iso, ln f, ln D ratios, A, dSigma^2, ln alpha_rs (if free). {name: (samples, LaTeX label)}."""
    d = load(f'chain_{o}.npz'); ne = int(d['n_eft']); x = flat(d['x']); names = list(d['names']); ph, pn = x[:, ne:], names[ne:]
    out = {}
    for i, n in enumerate(names[:ne]):
        p, s = n.split('_'); out[n] = (x[:, i], rf'$b_1^{{\rm {s}}}$' if p == 'b1' else rf'$c_2^{{\rm {s}}}$')
    lrs = ph[:, pn.index('ln alpha_rs')] if 'ln alpha_rs' in pn else 0.
    zs = [n.split('(')[1][:-1] for n in pn if n.startswith('ln alpha_par')]
    for z in zs:
        a = {k: np.exp(ph[:, pn.index(f'ln alpha_{k}({z})')] - lrs) for k in ('par', 'per')}
        out[f'apar{z}'] = (a['par'], rf'$\alpha_\parallel^{{\rm BAO}}({z})$'); out[f'aper{z}'] = (a['per'], rf'$\alpha_\perp^{{\rm BAO}}({z})$')
        out[f'aiso{z}'] = (a['par']**(1 / 3) * a['per']**(2 / 3), rf'$\alpha_{{\rm iso}}^{{\rm BAO}}({z})$')
    for i, n in enumerate(pn):
        if n.startswith('ln f'): out[n] = (ph[:, i], rf"$\ln f({n.split('(')[1][:-1]})$")
        if n.startswith('ln D'): out[n] = (ph[:, i], rf"$\ln D({n.split('(')[1].split(')')[0]})/D_N$")
    out['A'] = (ph[:, pn.index('BAO amplitude A')], r'BAO amplitude $A$')
    out['dS2'] = (ph[:, pn.index('wiggle damping dSigma^2 [Mpc^2]')], r'$\Delta\Sigma^2\ [{\rm Mpc}^2]$')
    if 'ln alpha_rs' in pn: out['lnrs'] = (lrs, r'$\ln\alpha_{r_s}$')
    return out


def compare_mi_1d(a='bao_template', b='bao_template_no_rs'):
    """1D posterior of every MI parameter, both templates (filled: a). ln a(k) depends on the frame: see pk_compare."""
    A, B = _mi_all(a), _mi_all(b); keys = list(A) + [k for k in B if k not in A]; nc = 8; nr = int(np.ceil(len(keys) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(2.2 * nc, 1.75 * nr)); axs = axs.ravel()
    for ax, k in zip(axs, keys):
        ref = A.get(k, B.get(k))[0]; lo, hi = np.percentile(ref, [0.2, 99.8]); w = hi - lo; bins = np.linspace(lo - 0.15 * w, hi + 0.15 * w, 50)
        if k in A: ax.hist(A[k][0], bins, density=True, color=COL[a], alpha=0.35, lw=0)
        if k in B: ax.hist(B[k][0], bins, density=True, histtype='step', color=COL[b], lw=1.4)
        ax.set_title(A.get(k, B.get(k))[1], fontsize=8); ax.set_yticks([]); ax.tick_params(labelsize=7)
    for ax in axs[len(keys):]: ax.axis('off')
    axs[0].legend([plt.Rectangle((0, 0), 1, 1, color=COL[a], alpha=0.35), plt.Line2D([], [], color=COL[b])], [LAB[a], LAB[b]], fontsize=7, loc='upper left')
    fig.suptitle(f'MI posterior: {LAB[a]} (filled) vs {LAB[b]} (line)', fontsize=11); fig.tight_layout()
    return fig


def compare_bao_2d(a='bao_template', b='bao_template_no_rs'):
    """alpha_par vs alpha_perp (BAO) at each redshift, both templates."""
    from getdist import plots
    roots = []
    for o in (a, b):
        q = _mi_all(o); ks = [k for k in q if k.startswith('apar') or k.startswith('aper')]
        from getdist import MCSamples
        roots.append(MCSamples(samples=np.column_stack([q[k][0] for k in ks]), names=ks, labels=[q[k][1].strip('$') for k in ks], label=LAB[o],
                               settings={'smooth_scale_2D': 0.5, 'smooth_scale_1D': 0.5}))
    zs = [k[4:] for k in ks if k.startswith('apar')]
    g = plots.get_subplot_plotter(subplot_size=3.0); g.settings.num_plot_contours = 2
    g.plots_2d(roots, param_pairs=[[f'apar{z}', f'aper{z}'] for z in zs], nx=3, filled=[True, False], colors=[COL[a], COL[b]])
    g.add_legend([LAB[a], LAB[b]], fontsize=9)
    return g


def compare_cosmo(cv, theta_s=False, a='bao_template', b='bao_template_no_rs'):
    """Every parameter of cosmology cv (+ Omega_m, sigma_8 where all have it) projected from both templates' MI posteriors
    (score density), DESI alone with the reference (direct fit, true likelihood) or + theta_s. EDE: the recovery chains."""
    keys = KEYS[cv]; roots, cols, leg, filled, have_s8 = [], [], [], [], True
    if not theta_s:
        dd = load(f'reference_{cv}.npz'); th = flat(dd['x'])[:, int(dd['n_eft']):]; s8 = None; have_s8 = False
        roots.append((th, s8, 'reference')); cols.append(COL['direct']); leg.append('reference (direct fit, full likelihood)'); filled.append(True)
    for o in (a, b):
        if cv == 'ede':
            r = load(f'recovery_{o}.npz'); th, s8 = flat(r['ede_score_x']), None; have_s8 = False
        else:
            p = load(f'projections_{o}.npz'); n = cv + ('_theta_s' if theta_s else ''); th, s8 = flat(p[f'{n}_x']), p[f'{n}_sigma8']
        roots.append((th, s8, o)); cols.append(COL[o]); leg.append(LAB[o] + (r' + $\theta_s$' if theta_s else '')); filled.append(False)
    names = keys + ['Om'] + (['s8'] if have_s8 else []); ms = []
    for th, s8, lab in roots:
        Om = (th[:, keys.index('omega_cdm')] + th[:, keys.index('omega_b')]) / th[:, keys.index('h')]**2
        ms.append(mcsamples(np.column_stack([th, Om] + ([s8] if have_s8 else [])), names, lab))
        print(f"{lab:20s} " + ", ".join(f"{k} {v.mean():.4f}+-{v.std():.4f}" for k, v in zip(names, ms[-1].samples.T)))
    g = triangle(ms, filled, cols, leg)
    g.fig.suptitle(f"{COSMO_NAME[cv]}: DESI DR1 FS+BAO{' + θ_s' if theta_s else ' alone'}", y=1.01, fontsize=12)
    return g
