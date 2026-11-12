"""P(k) vs P(k)+B(k): Fisher forecast of the legacy model-independent (MI) model, and every step
of ../../notebooks/5_fisher_pk_bk.ipynb. Reads the caches in output/fisher_pb/ (made by rebuild.py); figures go
to output/fisher_pb/meeting/.

THE MODEL (legacy mi_model.py, what the HMC samples). ln a on 60 nodes log-spaced over
[1e-4, 0.7] h/Mpc (spacing 0.15 in ln k), cubic onto the 80 emulator knots, with priors
    s_lna 0.5 per node,  lambda 200 on second differences of ln a,
    s_lng 0.3 on ln of every (f, H, D_A, D-ratio),  s_lnh 0.05 on ln h_conv.
The Fisher is the whitened model Jacobian (jacobians_pb.npz) in that basis: nothing is re-fitted,
every number is a linear functional of one covariance matrix.

EXACT SYMMETRIES of the continuum model:
    g1 ruler      ln h_conv +1, ln D_A(z) +1, ln H(z) -1 at every z   (no amplitudes)
    g2 amplitude  ln a +1 at every node, ln D(z) -1/2
    g3 dilation   ln h_conv +1, ln a -(3 + dlnT/dlnk)
The discretization breaks them (g1 at sigma ~0.05, g3 at ~0.3) and a Fisher would read that as
information. The data response along g1, g2 is projected out of the DATA rows only,
W_data -> (I - Q Q^T) W_data with Q an orthonormal basis of {W_data g}; the priors may still
constrain those directions. g1 involves no amplitude, so absolute distances come from the h_conv /
growth priors only; the smoothness prior breaks g3, which is why D_M/r_d (the template's own
wiggles as the ruler) is measurable at all.

LCDM. The direct LCDM model is the MI model composed with the cosmology map (cosmo_map_pb.npz,
J_direct = J_MI M to 1.4e-6), so the LCDM Fisher is a projection of the MI one; the routes differ
only in which priors they carry.
"""
import os
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'output', 'fisher_pb'))
FIG = os.path.join(OUT, 'meeting')
R = np.load(os.path.join(OUT, 'jacobians_pb.npz'), allow_pickle=True)

knots_h = R['knots_h']; nk = int(R['n_knots']); ng = int(R['n_growth'])
nz = int(R['n_z_unique']); nsky = int(R['num_skies']); zz = R['zeff_unique']
gfid = R['growth_fid']; h_fid = float(R['h_true'])
iF, iH, iD = R['idx_f'], R['idx_H'], R['idx_DA']
iDr = 3 * nz + np.arange(nz); ihc = ng - 1
gamma = 3.0 + R['dlnT_dlnk']
T_ = R['template_mpc']
Tnw = R['template_nw_mpc']

PRIORS_HMC = dict(s_lna=0.5, s_lng=0.3, s_lnh=0.05, lam=200.0)
# the wider priors prior_audit() recommends: lambda is the block that injects information into LCDM and
# s_lna sets most of the P(k) band; these leave a roughness of 0.135 per node (from 0.066)
PRIORS_WIDE = dict(s_lna=2.0, s_lng=0.3, s_lnh=0.05, lam=50.0)
NODE_RANGE, NODE_SPACING = (1e-4, 0.7), 0.15
C_P, C_PB, C_INK, C_MUTED = '#E8710A', '#1E88E5', '#1A1A1A', '#6E6E6E'
C_N60, C_N16 = '#8E24AA', '#43A047'           # node bases (orange/blue are reserved for P, P+B)
KEYS = (('p', 'P(k)', C_P), ('pb', 'P(k)+B(k)', C_PB))
IZ = 2                                        # z = 0.706, the bin the tables quote
KMIN_P, KMAX_P = 0.01, 0.2                    # P(k) scale cut of the mock
KMIN_B, KMAX_B = 0.02, 0.10                   # B(k) triangle range

lnk_knots = np.log(knots_h * h_fid)
_lo, _hi = np.log(NODE_RANGE[0] * h_fid), np.log(NODE_RANGE[1] * h_fid)


def node_basis(spacing=NODE_SPACING):
    """(nodes_lnk, S, D2, gamma_nodes) for a node spacing in ln k: S the cubic spline nodes ->
    knots, D2 the second differences, gamma_nodes the dilation pattern on the nodes (least
    squares through the spline). The BAO period is ~0.6 in ln k: a finer spacing lets the nodes
    slide the wiggles (the template becomes its own ruler), a coarser one cannot."""
    nn = int(np.round((_hi - _lo) / spacing)) + 1
    nl = np.linspace(_lo, _hi, nn)
    qq = np.clip(lnk_knots, nl[0], nl[-1])
    Sm = np.zeros((nk, nn))
    for i in range(nn):
        e = np.zeros(nn); e[i] = 1.0
        Sm[:, i] = CubicSpline(nl, e, bc_type='not-a-knot')(qq)
    Dm = np.zeros((max(nn - 2, 0), nn))
    for i in range(nn - 2):
        Dm[i, i:i + 3] = (1.0, -2.0, 1.0)
    return nl, Sm, Dm, np.linalg.lstsq(Sm, gamma, rcond=None)[0]


# ---- the BAO phase prior ------------------------------------------------------------------
# P_lin = a(k) T_nw(k) [1 + O(k)]. Sliding the wiggles by delta in ln k, O(k) -> O(k e^delta),
# changes ln P by delta * v(k), v = dln(1 + O)/dln k: if ln a can contain v the template moves its
# OWN BAO feature and the ruler floats. A Gaussian prior sigma_delta on that slide is the
# assumption of a standard BAO fit (broadband and wiggle amplitude free, phase fixed);
# sigma_delta -> 0 is rigid wiggles, -> inf the free template.
wiggle_O = T_ / Tnw - 1.0
wiggle_v = np.gradient(np.log1p(wiggle_O), np.log(knots_h))
PHASE_RANGE = (0.01, 0.4)       # h/Mpc: where the wiggles carry weight, for the envelopes
N_PHASE = 6                     # envelope modes (Legendre degree 0..5) across that range


def phase_modes(Sb, n_phase=N_PHASE):
    """The wiggle-slide patterns, on the ln-a nodes of basis `Sb`: columns v * P_j(ln k).

    A single column (n_phase = 1) pins one GLOBAL slide of the wiggle pattern. That is not
    enough: the template can still drift the phase slowly across k and recover most of the
    freedom (measured: sigma(D_M/r_d) 2.13% -> 1.37% only). The envelopes P_j are Legendre
    polynomials of degree j in ln k rescaled to [-1, 1] over PHASE_RANGE, so column j is a
    slide whose size varies across the band; pinning j = 0..5 leaves the phase fixed
    everywhere and takes sigma(D_M/r_d) to 0.88%, close to the 0.71% of a basis too coarse to
    slide anything. The wiggle AMPLITUDE is never touched -- it is the damping nuisance of a
    BAO fit and stays free."""
    from numpy.polynomial.legendre import legval
    lo, hi = np.log(PHASE_RANGE[0]), np.log(PHASE_RANGE[1])
    x = np.clip((np.log(knots_h) - lo) / (hi - lo) * 2 - 1, -1, 1)
    cols = []
    for j in range(n_phase):
        c = np.zeros(j + 1); c[j] = 1.0
        cols.append(np.linalg.lstsq(Sb, wiggle_v * legval(x, c), rcond=None)[0])
    return np.array(cols).T


def phase_precision(Sb, sigma, n_phase=N_PHASE):
    """Precision on the ln-a nodes for a prior `sigma` on every slide coefficient.

    ln a is read as delta(k) * v(k) + (the rest), with delta(k) = sum_j c_j P_j(ln k) the local
    slide; c = (V^T V)^+ V^T ln a, and the prior sum_j (c_j/sigma)^2 is the rank-n_phase
    precision W W^T / sigma^2 with W = V (V^T V)^+."""
    V = phase_modes(Sb, n_phase)
    W = V @ np.linalg.pinv(V.T @ V)
    return (W @ W.T) / sigma**2, V


# ---- the Fisher system ---------------------------------------------------------------------
def prior_precision(n_amp, s_lna, s_lng, s_lnh, lam, Db):
    """Prior precision on [ln a nodes, ln growth], exactly mi_model.prior_prec_mat."""
    Pi = np.zeros((n_amp + ng, n_amp + ng))
    Pi[:n_amp, :n_amp] = np.eye(n_amp) / s_lna**2 + lam * (Db.T @ Db)
    for j in range(4 * nz):
        Pi[n_amp + j, n_amp + j] = 1.0 / s_lng**2
    Pi[n_amp + ihc, n_amp + ihc] = 1.0 / s_lnh**2
    return Pi


def whiten(key):
    """Whitened model Jacobian in KNOT coordinates for data set `key` ('p' or 'pb').

    Rows are whitened data points (L^T J, P = L L^T the data precision); columns are [EFT per
    sky, 80 ln-a knots, 25 ln-growth], each dimensionless: EFT by its prior width (or its own size
    where it has none), template and growth by their fiducial values (ln-parameters)."""
    J, P, eftpr, x0 = R[f'J_{key}'], R[f'P_{key}'], R[f'prior_{key}'], R[f'x0_{key}']
    n_e = int(R[f'n_e_{key}']); o_a = n_e * nsky; o_gk = o_a + nk
    Lc = np.linalg.cholesky(0.5 * (P + P.T))
    s = np.ones(len(x0)); e = np.arange(o_a); has = eftpr[e] > 0
    s[e] = np.where(has, 1 / np.sqrt(np.where(has, eftpr[e], 1.0)), np.maximum(np.abs(x0[e]), 1.0))
    s[o_a:o_gk] = x0[o_a:o_gk]; s[o_gk:] = gfid
    return dict(Wd=(Lc.T @ J) * s, has=has, o_a=o_a, o_gk=o_gk, n_e=n_e)


def build(key, priors=None, spacing=NODE_SPACING, phase_sigma=None, n_phase=N_PHASE):
    """Covariance of the node-basis system for one data set, g1 and g2 projected out of the data.

    phase_sigma: add the BAO phase prior, the template's wiggles may slide by this fraction at
    1 sigma anywhere in k (phase_modes). None leaves them free, the model the HMC samples."""
    pr = dict(PRIORS_HMC if priors is None else priors)
    nl, Sb, Db, gnodes = node_basis(spacing)
    n_amp = len(nl)
    wk = whiten(key)
    Wd_k, has, o_a, o_gk, n_e = wk['Wd'], wk['has'], wk['o_a'], wk['o_gk'], wk['n_e']
    Wd = np.hstack([Wd_k[:, :o_a], Wd_k[:, o_a:o_gk] @ Sb, Wd_k[:, o_gk:]])  # knots -> nodes
    o_g = o_a + n_amp
    npar = o_g + ng
    G = np.zeros((npar, 3))                                     # g1, g2, g3 (columns)
    G[o_g + ihc, 0] = 1.0; G[o_g + iD, 0] = 1.0; G[o_g + iH, 0] = -1.0
    G[o_a:o_a + n_amp, 1] = 1.0; G[o_g + iDr, 1] = -0.5
    G[o_g + ihc, 2] = 1.0; G[o_a:o_a + n_amp, 2] = -gnodes
    G = G[:, [0, 1]]                                            # g3 is left to the prior
    Q, Rq = np.linalg.qr(Wd @ G)
    keep = np.abs(np.diag(Rq)) > 1e-14 * np.linalg.norm(Wd)
    Q = Q[:, keep]
    Wd = Wd - Q @ (Q.T @ Wd)
    idx = np.where(has)[0]
    Wp = np.zeros((len(idx), npar)); Wp[np.arange(len(idx)), idx] = 1.0     # EFT priors
    Pi = np.zeros((npar, npar))
    Pi[o_a:, o_a:] = prior_precision(n_amp, pr['s_lna'], pr['s_lng'], pr['s_lnh'], pr['lam'], Db)
    if phase_sigma is not None:
        Pi[o_a:o_a + n_amp, o_a:o_a + n_amp] += phase_precision(Sb, phase_sigma, n_phase)[0]
    F = Wd.T @ Wd + Wp.T @ Wp + Pi
    F = 0.5 * (F + F.T)
    return dict(C=np.linalg.inv(F), o_a=o_a, o_g=o_g, npar=npar, S=Sb, n_amp=n_amp)


def systems(phase_sigma=None, spacing=NODE_SPACING, n_phase=N_PHASE):
    return {k: build(k, spacing=spacing, phase_sigma=phase_sigma, n_phase=n_phase)
            for k, _, _ in KEYS}


def sig(sysd, c):
    return float(np.sqrt(c @ sysd['C'] @ c))


def band(sy):
    """sigma(ln P_lin) per knot, everything else marginalized."""
    Ca = sy['C'][sy['o_a']:sy['o_a'] + sy['n_amp'], sy['o_a']:sy['o_a'] + sy['n_amp']]
    return np.sqrt(np.clip(np.diag(sy['S'] @ Ca @ sy['S'].T), 0, None))


def growth_c(sysd, name, iz):
    """Unit functional for a growth quantity, in fractional (ln) units."""
    c = np.zeros(sysd['npar']); o = sysd['o_g']
    if name == 'f':
        c[o + iF[iz]] = 1.0
    elif name == 'lnH':
        c[o + iH[iz]] = 1.0
    elif name == 'lnDA':
        c[o + iD[iz]] = 1.0
    elif name == 'F_AP':
        c[o + iH[iz]] = 1.0; c[o + iD[iz]] = 1.0
    return c


def alpha_c(sysd, which, iz):
    """alpha_perp = ln D_A - ln h_conv = D_M/r_d, alpha_par = -ln H - ln h_conv = D_H/r_d.

    Distances in units of the template's own ruler: invariant under g1, so measurable without
    any prior on h -- unlike absolute D_A and H, set by the h_conv and growth priors alone."""
    c = np.zeros(sysd['npar']); o = sysd['o_g']
    c[o + ihc] = -1.0
    if which == 'perp':
        c[o + iD[iz]] = 1.0
    else:
        c[o + iH[iz]] = -1.0
    return c


def sig_q(sy, q, iz):
    c = alpha_c(sy, q, iz) if q in ('perp', 'par') else growth_c(sy, q, iz)
    return sig(sy, c)


# ---- LCDM as a projection of the MI Fisher -------------------------------------------------
_LCDM = {}


def lcdm():
    """The cosmology map d(MI parameters)/d(omega_cdm, lnAs, h) (cosmo_map_pb.npz), also on the
    60 nodes (least squares through the spline), loaded once."""
    if not _LCDM:
        CM = np.load(os.path.join(OUT, 'cosmo_map_pb.npz'))
        Mmap = CM['M']
        nl, Sb, Db, _ = node_basis()
        M_amp_nodes = np.linalg.lstsq(Sb, Mmap[:nk], rcond=None)[0]
        _LCDM.update(M=Mmap, theta_fid=CM['theta_fid'], Sb=Sb, Db=Db, n_amp=len(nl),
                     Mnode=np.vstack([M_amp_nodes, Mmap[nk:]]),
                     gate={k: float(CM[f'gate_{k}']) for k in ('p', 'pb')},
                     rep=[np.linalg.norm(Sb @ M_amp_nodes[:, j] - Mmap[:nk, j])
                          / np.linalg.norm(Mmap[:nk, j]) for j in range(3)])
    return _LCDM


def cosmo_fisher(key, route='direct'):
    """3x3 LCDM Fisher (omega_cdm, ln 10^10 A_s, h), EFT marginalized, g1-g3 projected out of
    the data in knot coordinates.

    'direct'           LCDM fitted at the 80 knots, no MI priors: an ordinary full-shape fit.
    'through_mi_flat'  the same data through the MI parametrization (60 nodes), no MI priors;
                       must reproduce 'direct' (no information inserted).
    'through_mi'       ... plus the MI priors evaluated on the LCDM submanifold."""
    L = lcdm()
    wk = whiten(key)
    Wd, o_a, o_gk = wk['Wd'], wk['o_a'], wk['o_gk']
    G = np.zeros((Wd.shape[1], 3))
    G[o_gk + ihc, 0] = 1.0; G[o_gk + iD, 0] = 1.0; G[o_gk + iH, 0] = -1.0
    G[o_a:o_gk, 1] = 1.0; G[o_gk + iDr, 1] = -0.5
    G[o_gk + ihc, 2] = 1.0; G[o_a:o_gk, 2] = -gamma
    Q, Rq = np.linalg.qr(Wd @ G)
    Q = Q[:, np.abs(np.diag(Rq)) > 1e-14 * np.linalg.norm(Wd)]
    Wd = Wd - Q @ (Q.T @ Wd)
    idx = np.where(wk['has'])[0]
    if route == 'direct':
        W = np.hstack([Wd[:, :o_a], Wd[:, o_a:] @ L['M']])
        Pi = np.zeros((3, 3))
    else:
        Wn = np.hstack([Wd[:, :o_a], Wd[:, o_a:o_a + nk] @ L['Sb'], Wd[:, o_a + nk:]])
        W = np.hstack([Wn[:, :o_a], Wn[:, o_a:] @ L['Mnode']])
        if route == 'through_mi_flat':
            Pi = np.zeros((3, 3))
        else:
            pr = PRIORS_HMC
            Pi = L['Mnode'].T @ prior_precision(L['n_amp'], pr['s_lna'], pr['s_lng'], pr['s_lnh'],
                                                pr['lam'], L['Db']) @ L['Mnode']
    Wp = np.zeros((len(idx), o_a + 3)); Wp[np.arange(len(idx)), idx] = 1.0
    F = W.T @ W + Wp.T @ Wp
    F = 0.5 * (F + F.T)
    Fe, Fx, Fxx = F[:o_a, :o_a], F[:o_a, o_a:], F[o_a:, o_a:]
    return Fxx + Pi - Fx.T @ np.linalg.solve(Fe, Fx)


# ---- notebook steps -----------------------------------------------------------------------
def show(name, width=1000):
    from IPython.display import Image, display
    display(Image(filename=os.path.join(FIG, name), width=width))


def _save(name, fig=None, g=None, pre=''):
    """Write a matplotlib figure (dpi 170) or a getdist plotter `g` to FIG and close it."""
    os.makedirs(FIG, exist_ok=True)
    p = os.path.join(FIG, name)
    if g is not None:
        g.export(p); plt.close('all')
    else:
        fig.savefig(p, dpi=170); plt.close(fig)
    print(f"{pre}saved {p}")


def data_summary():
    print(f"redshift bins: {np.round(zz, 3).tolist()}   ({nsky} skies, "
          f"sky -> z index {R['sky_to_z_idx'].tolist()})")
    print(f"free parameters per data set: {R['J_p'].shape[1]} (P)   {R['J_pb'].shape[1]} (P+B)")
    print(f"   = per-sky EFT ({len(R['eft_names_p'])} P / {len(R['eft_names_pb'])} P+B names x "
          f"{nsky} skies) + {nk} template amplitudes + {ng} growth/geometry")
    print(f"data vector: {R['J_p'].shape[0]} bins (P0,P2,P4)   {R['J_pb'].shape[0]} bins (+B0: "
          f"{int(R['isB_pb'].sum())} triangles)")
    eps, c_h, c_g1, c_g2, c_g3, _ = R['sym_tests_pb'][1]          # rebuild.py jacobians
    print(f"delta chi^2 for a {100 * eps:g}% move (P+B): h_conv alone {c_h:.3g};  g1 {c_g1:.2g}, "
          f"g2 {c_g2:.1g}, g3 {c_g3:.2g}  (0 for an exact symmetry)")


def likelihood_summary():
    for key, lab in (('p', 'P(k)'), ('pb', 'P(k)+B(k)')):
        names = [str(n) for n in R[f'eft_names_{key}']]
        prec = R[f'prior_{key}'][:len(names)]                       # sky 1
        free = ' '.join(n for n, p in zip(names, prec) if p == 0)
        gauss = ', '.join(f'{n} {1 / np.sqrt(p):.3g}' for n, p in zip(names, prec) if p > 0)
        print(f"{lab:10s} free: {free}\n{'':10s} Gaussian prior (sigma): {gauss}")


def growth():
    """The linear P(k) band, (f, H, D_A) at z = 0.71, sigma(f) and sigma(F_AP) vs z, and LCDM
    direct vs through the MI model. Writes meeting_growth_pb.npz."""
    sysd = {k: build(k) for k, _, _ in KEYS}
    bnd = {k: band(sysd[k]) for k, _, _ in KEYS}
    bw = {k: band(build(k, priors=PRIORS_WIDE)) for k, _, _ in KEYS}
    nl, Sb, Db, _ = node_basis()
    Pi_a = np.eye(len(nl)) / PRIORS_HMC['s_lna']**2 + PRIORS_HMC['lam'] * (Db.T @ Db)
    prior_band = np.sqrt(np.diag(Sb @ np.linalg.inv(Pi_a) @ Sb.T))

    inb = (knots_h >= KMIN_P) & (knots_h <= KMAX_P)
    print("sigma(ln P_lin) per knot, marginalized over everything else:")
    print(f"  inside the P(k) window [{KMIN_P}, {KMAX_P}] h/Mpc:  P {bnd['p'][inb].mean():.4f}   "
          f"P+B {bnd['pb'][inb].mean():.4f}   ratio {bnd['pb'][inb].mean()/bnd['p'][inb].mean():.3f}"
          f"   (prior alone {prior_band[inb].mean():.4f})")
    best = np.argmax(bnd['p'] / np.maximum(bnd['pb'], 1e-12))
    print(f"  best single knot for B(k): k = {knots_h[best]:.3f} h/Mpc, "
          f"{bnd['p'][best]:.4f} -> {bnd['pb'][best]:.4f}")
    print(f"  wider priors (s_lna {PRIORS_WIDE['s_lna']:g}, lambda {PRIORS_WIDE['lam']:g}): "
          f"P {bw['p'][inb].mean():.4f}   P+B {bw['pb'][inb].mean():.4f}   "
          f"ratio {bw['pb'][inb].mean()/bw['p'][inb].mean():.3f}")
    r = bw['pb'] / bw['p']
    print(f"  best knot with the wider priors: k = {knots_h[np.argmin(r)]:.3f} h/Mpc, "
          f"ratio {r.min():.3f}")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    sel = (knots_h > 2e-3) & (knots_h < 0.6)
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.2), sharex=True, layout='constrained',
                             gridspec_kw={'height_ratios': [2.0, 1.3, 1.0]})
    kk, T = knots_h[sel], T_[sel]
    axes[0].plot(kk, T, color=C_INK, lw=1.8, ls='--', label='truth')
    axes[0].fill_between(kk, T * np.exp(-bw['pb'][sel]), T * np.exp(bw['pb'][sel]),
                         color=C_PB, alpha=0.32, lw=0, label=r'P(k)+B(k)  $\pm1\sigma$')
    for sgn in (+1, -1):
        axes[0].plot(kk, T * np.exp(sgn * bw['p'][sel]), color=C_P, lw=1.9,
                     label=r'P(k)  $\pm1\sigma$' if sgn > 0 else None)
    axes[0].set_xscale('log'); axes[0].set_yscale('log')
    axes[0].set_ylabel(r'$P_{\rm lin}(k)$  [Mpc$^3$]', fontsize=12)
    axes[0].set_title('Linear power spectrum recovery', fontsize=14)
    for k, lab, col in KEYS:
        axes[1].plot(kk, 100 * bw[k][sel], color=col, lw=2.3, label=lab)
        axes[1].plot(kk, 100 * bnd[k][sel], color=col, lw=1.4, ls=':')
    axes[1].set_ylabel(r'$1\sigma$ on $P_{\rm lin}$  [%]', fontsize=12)
    axes[1].set_ylim(0, 105 * bw['p'][sel].max())
    axes[2].plot(kk, bw['pb'][sel] / bw['p'][sel], color=C_PB, lw=2.3)
    axes[2].plot(kk, bnd['pb'][sel] / bnd['p'][sel], color=C_PB, lw=1.4, ls=':')
    axes[2].axhline(1.0, color=C_MUTED, lw=1.2)
    axes[2].set_ylabel(r'$\sigma_{P+B}\,/\,\sigma_{P}$', fontsize=12)
    axes[2].set_xlabel(r'$k$  [$h$/Mpc]', fontsize=12)
    axes[2].set_ylim(0.55, 1.05)
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
    _save('fig_bands_pb.png', fig)

    from getdist import plots
    from getdist.gaussian_mixtures import GaussianND
    nm3 = ['f', 'lnH', 'lnDA']
    lb3 = [r'\Delta f/f', r'\Delta H/H', r'\Delta D_A/D_A']
    ds = []
    for k, lab, _ in KEYS:
        Cs = np.array([growth_c(sysd[k], n, IZ) for n in nm3])
        ds.append(GaussianND(np.zeros(3), Cs @ sysd[k]['C'] @ Cs.T, names=nm3, labels=lb3,
                             label=lab))
    g = plots.get_subplot_plotter(subplot_size=2.9)
    g.settings.num_plot_contours = 2; g.settings.alpha_filled_add = 0.32
    g.settings.legend_fontsize = 14
    g.triangle_plot(ds, params=nm3, filled=[False, True], contour_colors=[C_P, C_PB],
                    line_args=[{'color': C_P, 'lw': 2.0}, {'color': C_PB, 'lw': 2.0}],
                    legend_labels=[lab for _, lab, _ in KEYS], markers={n: 0.0 for n in nm3})
    g.fig.suptitle(rf'Growth and geometry at $z = {zz[IZ]:.2f}$', fontsize=15, y=1.02)
    _save('fig_growth_corner.png', g=g)

    sig_f = {k: np.array([sig(sysd[k], growth_c(sysd[k], 'f', iz)) for iz in range(nz)])
             for k, _, _ in KEYS}
    sig_fap = {k: np.array([sig(sysd[k], growth_c(sysd[k], 'F_AP', iz)) for iz in range(nz)])
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
    _save('fig_growth_vs_z.png', fig)

    L = lcdm()
    print(f"\nGATE: the 60-node basis represents the LCDM template response to "
          f"{' '.join(f'{v:.1e}' for v in L['rep'])} for (omega_cdm, lnAs, h)")
    print(f"GATE: ||J_direct - J_MI M|| / ||J_direct|| = {L['gate']['p']:.2e} (P), "
          f"{L['gate']['pb']:.2e} (P+B)")
    cov, sg = {}, {}
    ROUTES = (('direct', 'direct LCDM fit'),
              ('through_mi_flat', 'through the MI model'),
              ('through_mi', 'through the MI model + its prior'))
    print("\nLCDM recovery, EFT marginalized:")
    print(f"{'':40s}{'omega_cdm':>12s}{'ln(1e10 As)':>14s}{'h':>10s}")
    for key, lab, _ in KEYS:
        for route, rlab in ROUTES:
            c = np.linalg.inv(cosmo_fisher(key, route=route))
            cov[(key, route)] = c; sg[(key, route)] = np.sqrt(np.diag(c))
            print(f"  {lab:10s} {rlab:28s}" + "".join(f"{v:12.4f}" for v in sg[(key, route)]))
        r = sg[(key, 'through_mi_flat')] / sg[(key, 'direct')]
        print(f"  {'':10s} {'MI / direct':28s}" + "".join(f"{v:12.3f}" for v in r))
    pr = PRIORS_HMC                                   # the MI prior on the LCDM submanifold, no data
    Pi = L['Mnode'].T @ prior_precision(L['n_amp'], pr['s_lna'], pr['s_lng'], pr['s_lnh'], pr['lam'], L['Db']) @ L['Mnode']
    print(f"  {'':10s} {'MI prior alone, no data':28s}" + "".join(f"{v:12.4f}" for v in np.sqrt(np.diag(np.linalg.inv(Pi)))))

    names = ['omega_cdm', 'logA', 'h']
    labels = [r'\omega_{\rm cdm}', r'\ln(10^{10}A_s)', r'h']
    ds, cols, lss, fills, legs = [], [], [], [], []
    for route, rlab, ls in (('direct', 'direct $\\Lambda$CDM fit', '--'),
                            ('through_mi_flat', 'through the MI model', '-')):
        for key, lab, col in KEYS:
            ds.append(GaussianND(L['theta_fid'], cov[(key, route)], names=names, labels=labels,
                                 label=f'{lab}, {rlab}'))
            cols.append(col); lss.append(ls)
            fills.append(route == 'through_mi_flat' and key == 'pb')
            legs.append(f'{lab}, {rlab}')
    g = plots.get_subplot_plotter(subplot_size=2.9)
    g.settings.num_plot_contours = 2; g.settings.alpha_filled_add = 0.30
    g.settings.legend_fontsize = 13
    g.triangle_plot(ds, params=names, filled=fills, contour_colors=cols, contour_ls=lss,
                    line_args=[{'color': c, 'ls': l, 'lw': 2.0} for c, l in zip(cols, lss)],
                    legend_labels=legs, markers={n: v for n, v in zip(names, L['theta_fid'])})
    g.fig.suptitle('$\\Lambda$CDM recovery', fontsize=15, y=1.02)
    _save('fig_cosmo_pb.png', g=g)

    np.savez(os.path.join(OUT, 'meeting_growth_pb.npz'), knots_h=knots_h,
             template_mpc=T_, prior_band=prior_band, zeff=zz,
             **{f'band_{k}': bnd[k] for k, _, _ in KEYS},
             **{f'band_wide_{k}': bw[k] for k, _, _ in KEYS},
             **{f'sig_f_{k}': sig_f[k] for k, _, _ in KEYS},
             **{f'sig_FAP_{k}': sig_fap[k] for k, _, _ in KEYS},
             **{f'cosmo_cov_{k}_{r}': cov[(k, r)] for k, _, _ in KEYS
                for r, _ in ROUTES}, theta_fid=L['theta_fid'])


def prior_audit():
    """Which MI prior block does the work: the induced LCDM prior block by block, each block on
    alone and off alone (LCDM and the MI numbers), then a scan over s_lna and lambda with the
    roughness each setting still allows. Writes prior_decomposition_pb.npz."""
    L = lcdm()
    n_amp, Db, Mnode = L['n_amp'], L['Db'], L['Mnode']
    PAR = ['omega_cdm', 'ln(1e10 As)', 'h']
    BIG = 1e6                     # "prior off": a width so wide it cannot constrain anything
    inb = (knots_h >= 0.01) & (knots_h <= 0.2)

    def sig3(F):
        return np.sqrt(np.diag(np.linalg.inv(F)))

    def row(label, s):
        return f"  {label:34s}" + "".join(f"{v:13.4f}" for v in s)

    def mi_numbers(pr, phase_sigma=None):
        """P(k) band, sigma(f), sigma(D_M/r_d) at z = 0.71 for P+B under priors `pr`."""
        sy = build('pb', priors=pr, phase_sigma=phase_sigma)
        b = band(sy)[inb].mean()
        c = np.zeros(sy['npar']); c[sy['o_g'] + ihc] = -1.0; c[sy['o_g'] + iD[IZ]] = 1.0
        return b, sig(sy, growth_c(sy, 'f', IZ)), sig(sy, c)

    def rough(Pa, D):
        return np.sqrt(np.median(np.diag(D @ np.linalg.inv(Pa) @ D.T)))

    n = n_amp + ng
    B = {name: np.zeros((n, n)) for name in ('s_lna', 'lambda', 's_lng', 's_lnh')}
    B['s_lna'][:n_amp, :n_amp] = np.eye(n_amp) / 0.5**2
    B['lambda'][:n_amp, :n_amp] = 200.0 * (Db.T @ Db)
    for j in range(4 * nz):
        B['s_lng'][n_amp + j, n_amp + j] = 1.0 / 0.3**2
    B['s_lnh'][n_amp + ihc, n_amp + ihc] = 1.0 / 0.05**2
    tot = sum(B.values())

    print("=" * 86)
    print("1. THE INDUCED PRIOR ON LCDM, NO DATA AT ALL")
    print(f"{'':36s}" + "".join(f"{p:>13s}" for p in PAR))
    for name, P in list(B.items()) + [('all four together', tot)]:
        Pi = Mnode.T @ P @ Mnode
        ev = np.linalg.eigvalsh(Pi)
        s = sig3(Pi) if ev.min() > 1e-12 * max(ev.max(), 1) else np.full(3, np.inf)
        print(row(name, s))
    F_data = {k: cosmo_fisher(k, route='through_mi_flat') for k in ('p', 'pb')}
    print("\n  for scale, the DATA alone:")
    for k, lab in (('p', 'P(k)'), ('pb', 'P(k)+B(k)')):
        print(row(lab, sig3(F_data[k])))

    print("\n" + "=" * 86)
    print("2. LCDM WITH EACH BLOCK ON ALONE, AND WITH EACH BLOCK OFF ALONE   [P(k)+B(k)]")
    print(f"{'':36s}" + "".join(f"{p:>13s}" for p in PAR))
    base = sig3(F_data['pb'])
    print(row('data only (no MI priors)', base))
    for name in B:
        print(row(f'+ {name} alone', sig3(F_data['pb'] + Mnode.T @ B[name] @ Mnode)))
    print(row('+ all four (the model as sampled)', sig3(F_data['pb'] + Mnode.T @ tot @ Mnode)))
    for name in B:
        off = sum(P for n_, P in B.items() if n_ != name)
        print(row(f'all but {name}', sig3(F_data['pb'] + Mnode.T @ off @ Mnode)))
    print("\n  ratio to data-only, all four on: " +
          "  ".join(f"{p}={v:.2f}" for p, v in
                    zip(PAR, sig3(F_data['pb'] + Mnode.T @ tot @ Mnode) / base)))

    print("\n" + "=" * 86)
    print("3. THE SAME BLOCKS, ON THE QUANTITIES THE MI ANALYSIS REPORTS   [P(k)+B(k)]")
    print(f"{'':36s}{'band sigma(lnP)':>16s}{'sigma(f)':>11s}{'D_M/r_d':>10s}"
          f"{'D_M/r_d + phase':>17s}")
    SETS = [('as sampled: 0.5 / 200', dict(s_lna=0.5, lam=200.0, s_lng=0.3, s_lnh=0.05)),
            ('s_lna off (1e6)', dict(s_lna=BIG, lam=200.0, s_lng=0.3, s_lnh=0.05)),
            ('lambda off (0)', dict(s_lna=0.5, lam=0.0, s_lng=0.3, s_lnh=0.05)),
            ('s_lng off (1e6)', dict(s_lna=0.5, lam=200.0, s_lng=BIG, s_lnh=0.05)),
            ('template priors off', dict(s_lna=BIG, lam=0.0, s_lng=0.3, s_lnh=0.05))]
    for lab, pr in SETS:
        b, f_, d = mi_numbers(pr)
        _, _, dp = mi_numbers(pr, phase_sigma=0.002)
        print(f"  {lab:34s}{b:16.4f}{100*f_:10.1f}%{100*d:9.2f}%{100*dp:16.2f}%")

    print("\n" + "=" * 86)
    print("4. SCAN OVER s_lna AND lambda   [P(k)+B(k)]")
    S_LNA = [0.5, 1.0, 2.0, 5.0, 10.0]
    LAM = [200.0, 50.0, 20.0, 5.0, 0.0]
    print(f"{'s_lna':>7s}{'lam':>6s} | {'LCDM sigma ratio to data-only':>31s} | {'rough':>7s} | "
          f"{'band':>7s}{'sig(f)':>9s}{'D_M/r_d':>9s}{'+phase':>9s}")
    print(f"{'':7s}{'':6s} | {'omega_cdm     lnAs        h':>31s} | {'':7s} | {'':7s}{'':9s}"
          f"{'free':>9s}{'0.2%':>9s}")
    scan = {}
    for s_lna in S_LNA:
        for lam in LAM:
            pr = dict(s_lna=s_lna, lam=lam, s_lng=0.3, s_lnh=0.05)
            Pi_full = prior_precision(n_amp, s_lna, 0.3, 0.05, lam, Db)
            ratio = sig3(F_data['pb'] + Mnode.T @ Pi_full @ Mnode) / base
            rg = rough(np.eye(n_amp) / s_lna**2 + lam * (Db.T @ Db), Db)
            b, f_, d = mi_numbers(pr)
            _, _, dp = mi_numbers(pr, phase_sigma=0.002)
            scan[(s_lna, lam)] = (ratio, rg, b, f_, d, dp)
            print(f"{s_lna:7.1f}{lam:6.0f} | " + "".join(f"{v:10.2f}" for v in ratio) +
                  f" | {rg:7.3f} | {b:7.3f}{100*f_:8.1f}%{100*d:8.2f}%{100*dp:8.2f}%")
    print("\n  ratio: sigma(LCDM) with the MI prior / data only;  rough: 1-sigma second difference of"
          "\n  ln a the prior allows;  band: mean sigma(ln P_lin) over knots in 0.01 < k < 0.2 h/Mpc")
    np.savez(os.path.join(OUT, 'prior_decomposition_pb.npz'),
             s_lna=np.array(S_LNA), lam=np.array(LAM), par=np.array(PAR),
             data_sigma_p=sig3(F_data['p']), data_sigma_pb=base,
             **{f'scan_{s:g}_{l:g}': np.array([*scan[(s, l)][0], *scan[(s, l)][1:]])
                for s in S_LNA for l in LAM})

    print("\n" + "=" * 86)
    print("5. lambda = 200 VS NODE SPACING (prior alone)")
    print(f"{'spacing':>9s}{'nodes':>7s}{'roughness':>12s}{'equivalent lambda at 0.15':>27s}")
    for spacing in (0.15, 0.25, 0.40, 0.60):
        nl_s, _, D_s, _ = node_basis(spacing)
        rg = rough(np.eye(len(nl_s)) / 0.5**2 + 200.0 * (D_s.T @ D_s), D_s)
        print(f"{spacing:9.2f}{len(nl_s):7d}{rg:12.3f}{200 * (spacing / 0.15)**4:27.0f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), layout='constrained')
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(LAM)))
    for i, lam in enumerate(LAM):
        axes[0].plot(S_LNA, [scan[(s, lam)][0][0] for s in S_LNA], color=cols[i], lw=2.2,
                     marker='o', ms=5, label=rf'$\lambda$ = {lam:g}')
        axes[1].plot(S_LNA, [scan[(s, lam)][1] for s in S_LNA], color=cols[i], lw=2.2,
                     marker='o', ms=5)
        axes[2].plot(S_LNA, [100 * scan[(s, lam)][2] for s in S_LNA], color=cols[i], lw=2.2,
                     marker='o', ms=5)
    axes[0].axhline(1, color=C_INK, lw=1.4, ls='--')
    axes[0].set_ylabel(r'$\sigma(\omega_{\rm cdm})$ with the prior / data alone', fontsize=12)
    axes[1].set_ylabel('node-to-node roughness allowed', fontsize=12)
    axes[2].set_ylabel(r'$1\sigma$ on $P_{\rm lin}$ in the data window [%]', fontsize=12)
    axes[0].set_ylim(0.35, 1.08)
    for ax in axes:
        ax.set_xscale('log'); ax.grid(alpha=0.2, which='both')
    for ax in axes[1:]:
        ax.set_yscale('log')
        ax.set_xlabel(r'$\sigma_{\ln a}$ per node', fontsize=12)
    axes[0].legend(frameon=False, fontsize=10)
    fig.suptitle('How much the template prior constrains, and what widening it costs', fontsize=14)
    _save('fig_prior_decomposition.png', fig, pre='\n')


def stage_a():
    """BAO stage A, the free 80-knot template without any prior (robust_pb_results.npz)."""
    S = np.load(os.path.join(OUT, 'robust_pb_results.npz'))
    fmt = lambda v: 'not measured' if v > 1 else f'{100 * v:.2f}%'
    print(f"free 80-knot template, no prior, everything marginalized, z = {S['zeff'][IZ]:.3f}")
    print(f"{'':32s}{'P(k)':>14s}{'P(k)+B(k)':>14s}")
    for lab, key in [('D_M/r_d', 'alpha_perp'), ('F_AP = D_M/D_H', 'F_AP'),
                     ('isotropic distance / z=0.93', 'alpha_iso_rel'), ('f', 'f')]:
        a, b = (S[f'geo_free_free_{k}_{key}'][IZ] for k in ('p', 'pb'))
        print(f"  {lab:30s}{fmt(a):>14s}{fmt(b):>14s}")
    print(f"  {'BAO wiggle S/N':30s}{S['snr_free_p_marg'][0]:14.1f}{S['snr_free_pb_marg'][0]:14.1f}")


def bao():
    """BAO stages B (smoothness prior) and C (+ phase prior): D_M/r_d, D_H/r_d, F_AP, f vs the
    phase prior, per redshift, and the corners. Writes meeting_bao_pb.npz."""
    QTS = [('perp', r'$D_M/r_d$'), ('par', r'$D_H/r_d$'), ('F_AP', r'$F_{\rm AP}=D_M/D_H$'),
           ('f', r'$f(z)$')]
    PHASE_REF = 0.002        # sigma_delta for the tables: the wiggles may slide by 0.2%
    N_PHASE_SCAN = (1, 2, 3, 4, 6, 8)

    # ---- what the phase prior is, drawn
    nl15, S15, D15, _ = node_basis(0.15)
    nl60, S60, D60, _ = node_basis(0.60)
    v15, v60 = phase_modes(S15, 1)[:, 0], phase_modes(S60, 1)[:, 0]
    cap15 = np.linalg.norm(S15 @ v15) / np.linalg.norm(wiggle_v)
    cap60 = np.linalg.norm(S60 @ v60) / np.linalg.norm(wiggle_v)
    Pi15 = np.eye(len(nl15)) / PRIORS_HMC['s_lna']**2 + PRIORS_HMC['lam'] * (D15.T @ D15)
    c15 = v15 / (v15 @ v15)
    prior_slide = float(np.sqrt(c15 @ np.linalg.inv(Pi15) @ c15))
    print(f"BAO period in ln k = 2pi/(r_d k): {2*np.pi/(105*0.05):.2f} at k=0.05, "
          f"{2*np.pi/(105*0.1):.2f} at 0.1, {2*np.pi/(105*0.2):.2f} at 0.2 h/Mpc")
    print(f"slide pattern v = dln(1+O)/dlnk represented by the node basis: "
          f"{cap15:.1%} at spacing 0.15 (60 nodes), {cap60:.1%} at spacing 0.60 (16 nodes)")
    print(f"smoothness prior alone allows a template slide of {100*prior_slide:.2f}% (1 sigma) "
          f"in the 60-node basis\n")
    sel = (knots_h > 0.02) & (knots_h < 0.32)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True, layout='constrained')
    axes[0].plot(knots_h[sel], 1 + wiggle_O[sel], color=C_INK, lw=2)
    axes[0].axhline(1, color=C_MUTED, lw=0.8)
    axes[0].set_ylabel(r'$1+O(k) = T/T_{\rm nw}$', fontsize=12)
    axes[0].set_title('BAO wiggles of the template, and the pattern that slides them', fontsize=14)
    axes[1].plot(knots_h[sel], wiggle_v[sel], color=C_INK, lw=2,
                 label=r'$v = \mathrm{d}\ln(1+O)/\mathrm{d}\ln k$')
    axes[1].plot(knots_h[sel], (S15 @ v15)[sel], color=C_N60, lw=1.8, ls='--',
                 label=f'60 nodes, spacing 0.15: {cap15:.0%}')
    axes[1].plot(knots_h[sel], (S60 @ v60)[sel], color=C_N16, lw=1.8, ls='-.',
                 label=f'16 nodes, spacing 0.60: {cap60:.0%}')
    axes[1].axhline(0, color=C_MUTED, lw=0.8)
    axes[1].set_xscale('log'); axes[1].set_xlabel(r'$k$  [$h$/Mpc]', fontsize=12)
    axes[1].set_ylabel(r'$v(k)$', fontsize=12)
    axes[1].legend(frameon=False, fontsize=9.5)
    for a in axes:
        a.grid(alpha=0.2)
    _save('fig_phase_prior_explained.png', fig)

    # ---- the scan: how tight a phase prior buys how much BAO precision
    SCAN = [None, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    scan, nscan = {}, {}
    for ps in SCAN:
        sy = systems(phase_sigma=ps)
        for k, _, _ in KEYS:
            for q, _ in QTS:
                scan[(ps, k, q)] = sig_q(sy[k], q, IZ)
    for n_ in N_PHASE_SCAN:
        sy = systems(phase_sigma=PHASE_REF, n_phase=n_)
        for k, _, _ in KEYS:
            for q, _ in QTS:
                nscan[(n_, k, q)] = sig_q(sy[k], q, IZ)
    rigid = systems(spacing=0.60)                 # the coarse-node basis, for reference
    S7 = np.load(os.path.join(OUT, 'se07_forecast.npz'))

    print(f"z = {zz[IZ]:.3f}: fractional 1 sigma vs the phase prior (sigma_delta = how far the "
          f"template may slide its own wiggles, {N_PHASE} envelope modes)")
    print(f"{'sigma_delta':>12s}   {'D_M/r_d  P     P+B':>22s}   {'D_H/r_d  P     P+B':>22s}"
          f"   {'F_AP  P      P+B':>20s}   {'f  P      P+B':>18s}")
    for ps in SCAN:
        lab = 'free' if ps is None else f'{100*ps:.2f}%'
        print(f"{lab:>12s}   " + "   ".join(
            f"{100*scan[(ps,'p',q)]:8.2f}% {100*scan[(ps,'pb',q)]:7.2f}%" for q, _ in QTS))
    print(f"{'16 nodes':>12s}   " + "   ".join(
        f"{100*sig_q(rigid['p'], q, IZ):8.2f}% {100*sig_q(rigid['pb'], q, IZ):7.2f}%"
        for q, _ in QTS) + "   <- coarse-node basis instead of a prior")
    print(f"{'BAO pre-rec':>12s}   {100*S7['pre_perp'][IZ]:8.2f}%"
          f"           {100*S7['pre_par'][IZ]:8.2f}%            <- standard BAO, same bins")
    print(f"{'BAO post-rec':>12s}   {100*S7['post_perp'][IZ]:8.2f}%"
          f"           {100*S7['post_par'][IZ]:8.2f}%")
    print(f"\nhow completely the phase is pinned, at sigma_delta = {100*PHASE_REF:.1f}%:")
    print(f"{'envelopes':>12s}   {'D_M/r_d  P     P+B':>22s}   {'D_H/r_d  P     P+B':>22s}")
    for n_ in N_PHASE_SCAN:
        print(f"{n_:12d}   " + "   ".join(
            f"{100*nscan[(n_,'p',q)]:8.2f}% {100*nscan[(n_,'pb',q)]:7.2f}%"
            for q, _ in QTS[:2]))

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.9), layout='constrained')
    for ax, (q, title) in zip(axes[:2], QTS[:2]):
        for k, lab, col in KEYS:
            ax.plot([100 * ps for ps in SCAN[1:]], [100 * scan[(ps, k, q)] for ps in SCAN[1:]],
                    color=col, lw=2.2, marker='o', ms=5, label=lab)
            ax.axhline(100 * scan[(None, k, q)], color=col, lw=1.4, ls=':')
            ax.axhline(100 * sig_q(rigid[k], q, IZ), color=col, lw=1.4, ls='--')
        ax.axhline(100 * S7[f'pre_{q}'][IZ], color=C_INK, lw=1.4, ls='-')
        ax.axhline(100 * S7[f'post_{q}'][IZ], color=C_INK, lw=1.4, ls='-.')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.invert_xaxis()
        ax.set_xlabel(r'phase prior $\sigma_\delta$  [%]', fontsize=12)
        ax.set_title(title, fontsize=13); ax.grid(alpha=0.2, which='both')
    axes[0].set_ylabel(r'fractional $1\sigma$ [%]', fontsize=12)
    h = [plt.Line2D([], [], color=C_P, lw=2.2, marker='o'),
         plt.Line2D([], [], color=C_PB, lw=2.2, marker='o'),
         plt.Line2D([], [], color=C_MUTED, lw=1.4, ls=':'),
         plt.Line2D([], [], color=C_MUTED, lw=1.4, ls='--'),
         plt.Line2D([], [], color=C_INK, lw=1.4, ls='-'),
         plt.Line2D([], [], color=C_INK, lw=1.4, ls='-.')]
    axes[0].legend(h, ['P(k)', 'P(k)+B(k)', 'no phase prior', 'coarse 16-node basis',
                       'standard BAO, pre-recon', 'standard BAO, post-recon'],
                   frameon=False, fontsize=9, ncol=1, loc='upper right')
    for k, lab, col in KEYS:
        for q, ls, mk in (('perp', '-', 'o'), ('par', '--', 's')):
            axes[2].plot(N_PHASE_SCAN, [100 * nscan[(n_, k, q)] for n_ in N_PHASE_SCAN],
                         color=col, ls=ls, lw=2.0, marker=mk, ms=5,
                         label=f'{lab}, {"$D_M/r_d$" if q == "perp" else "$D_H/r_d$"}')
            axes[2].axhline(100 * sig_q(rigid[k], q, IZ), color=col, lw=1.2, ls='--')
    axes[2].set_xlabel('envelope modes', fontsize=12)
    axes[2].set_title(rf'$\sigma_\delta = {100*PHASE_REF:.1f}\%$', fontsize=13)
    axes[2].grid(alpha=0.2); axes[2].legend(frameon=False, fontsize=9)
    fig.suptitle(f'BAO distance precision vs the phase prior, z = {zz[IZ]:.2f}', fontsize=14)
    _save('fig_phase_scan.png', fig, pre='\n')

    # ---- per redshift, no phase prior vs the standard-BAO phase prior
    free_sy, phase_sy = systems(), systems(phase_sigma=PHASE_REF)
    summ = {}
    for tag, sy in (('free', free_sy), ('phase', phase_sy)):
        for k, _, _ in KEYS:
            for q, _ in QTS:
                summ[(tag, k, q)] = np.array([sig_q(sy[k], q, iz) for iz in range(nz)])
    np.savez(os.path.join(OUT, 'meeting_bao_pb.npz'), zeff=zz, phase_ref=PHASE_REF,
             scan_sigma=np.array([np.nan if s is None else s for s in SCAN]),
             cap15=cap15, cap60=cap60, prior_slide=prior_slide, n_phase=N_PHASE,
             n_phase_scan=np.array(N_PHASE_SCAN),
             **{f'nscan_{n_}_{k}_{q}': nscan[(n_, k, q)]
                for n_ in N_PHASE_SCAN for k, _, _ in KEYS for q, _ in QTS},
             **{f'{t}_{k}_{q}': v for (t, k, q), v in summ.items()},
             **{f'scan_{"free" if s is None else f"{s:g}"}_{k}_{q}': scan[(s, k, q)]
                for s in SCAN for k, _, _ in KEYS for q, _ in QTS})
    print("\nPer-redshift fractional 1 sigma:")
    for tag, lab in (('free', 'no phase prior (the model we sample)'),
                     ('phase', f'phase prior {100*PHASE_REF:.1f}% (standard BAO)')):
        print(f"  {lab}")
        print(f"{'':16s}" + "".join(f"{'z='+format(z,'.2f'):>16s}" for z in zz))
        for q, _ in QTS:
            for k, kl, _ in KEYS:
                print(f"    {q:5s} {kl:9s}" + "".join(
                    f"{100*v:15.2f}%" for v in summ[(tag, k, q)]))

    fig, axes = plt.subplots(2, 4, figsize=(19, 7.4), layout='constrained', sharex=True,
                             gridspec_kw={'height_ratios': [3, 1.25]})
    for j, (q, title) in enumerate(QTS):
        ax, axr = axes[0, j], axes[1, j]
        for tag, ls, lab in (('free', '-', 'no phase prior'),
                             ('phase', '--', f'phase prior {100*PHASE_REF:.1f}%')):
            for k, kl, col in KEYS:
                ax.plot(zz, 100 * summ[(tag, k, q)], color=col, ls=ls, lw=2.0, marker='o',
                        ms=4, label=f'{kl}, {lab}')
            axr.plot(zz, summ[(tag, 'pb', q)] / summ[(tag, 'p', q)], color=C_INK, ls=ls,
                     lw=1.8, marker='o', ms=4, label=lab)
        if q in ('perp', 'par'):
            ax.plot(zz, 100 * S7[f'pre_{q}'], color=C_MUTED, lw=1.6, ls='-', marker='^', ms=4,
                    label='standard BAO, pre-recon')
            ax.plot(zz, 100 * S7[f'post_{q}'], color=C_MUTED, lw=1.6, ls='-.', marker='v', ms=4,
                    label='standard BAO, post-recon')
        ax.set_yscale('log'); ax.set_title(title, fontsize=13.5); ax.grid(alpha=0.2, which='both')
        axr.axhline(1.0, color=C_MUTED, lw=1); axr.set_ylim(0.4, 1.1); axr.grid(alpha=0.2)
        axr.set_xlabel('$z$', fontsize=12)
    axes[0, 0].set_ylabel(r'fractional $1\sigma$ [%]', fontsize=12)
    axes[1, 0].set_ylabel(r'$\sigma_{P+B}/\sigma_P$', fontsize=12)
    axes[0, 0].legend(frameon=False, fontsize=8.5, handlelength=3.0)
    axes[1, 0].legend(frameon=False, fontsize=8.5, handlelength=3.0, loc='lower left')
    fig.suptitle('BAO and growth per redshift', fontsize=14)
    _save('fig_bao_per_z.png', fig)

    # ---- the BAO corner at z = 0.71, and both BAO parameters at every redshift
    from getdist import plots
    from getdist.gaussian_mixtures import GaussianND
    names = ['aperp', 'apar', 'f']
    labels = [r'\Delta (D_M/r_d)', r'\Delta (D_H/r_d)', r'\Delta f/f']
    dists, cols, lss, fills, legs = [], [], [], [], []
    for tag, lab, ls in (('free', 'no phase prior', '--'),
                         ('phase', f'phase prior {100*PHASE_REF:.1f}%', '-')):
        sy = free_sy if tag == 'free' else phase_sy
        for k, kl, col in KEYS:
            cs = np.array([alpha_c(sy[k], 'perp', IZ), alpha_c(sy[k], 'par', IZ),
                           growth_c(sy[k], 'f', IZ)])
            dists.append(GaussianND(np.zeros(3), cs @ sy[k]['C'] @ cs.T, names=names,
                                    labels=labels, label=f'{kl}, {lab}'))
            cols.append(col); lss.append(ls); fills.append(tag == 'phase' and k == 'pb')
            legs.append(f'{kl}, {lab}')
    g = plots.get_subplot_plotter(subplot_size=2.8)
    g.settings.num_plot_contours = 2
    g.settings.alpha_filled_add = 0.30
    g.settings.legend_fontsize = 13
    g.triangle_plot(dists, params=names, filled=fills, contour_colors=cols, contour_ls=lss,
                    line_args=[{'color': c, 'ls': l, 'lw': 2.0} for c, l in zip(cols, lss)],
                    legend_labels=legs, markers={n: 0.0 for n in names})
    g.fig.suptitle(f'BAO and growth at z = {zz[IZ]:.2f}', fontsize=15, y=1.02)
    _save('fig_bao_corner.png', g=g)

    names12, labels12, qty12 = [], [], []
    for iz, z in enumerate(zz):
        names12 += [f'perp{iz}', f'par{iz}']
        labels12 += [rf'D_M/r_d\,({z:.2f})', rf'D_H/r_d\,({z:.2f})']
        qty12 += [('perp', iz), ('par', iz)]
    ds12, cols12, leg12 = [], [], []
    for k, kl, col in KEYS:                       # the BAO measurement: with the phase prior
        cs = np.array([alpha_c(phase_sy[k], q, iz) for q, iz in qty12])
        ds12.append(GaussianND(np.zeros(12), cs @ phase_sy[k]['C'] @ cs.T, names=names12,
                               labels=labels12, label=kl))
        cols12.append(col); leg12.append(kl)
    g = plots.get_subplot_plotter(subplot_size=1.25)
    g.settings.num_plot_contours = 2
    g.settings.alpha_filled_add = 0.30
    g.settings.legend_fontsize = 19
    g.settings.axes_fontsize = 9
    g.settings.axes_labelsize = 13
    g.triangle_plot(ds12, params=names12, filled=[False, True], contour_colors=cols12,
                    line_args=[{'color': c, 'lw': 2.0} for c in cols12],
                    legend_labels=leg12, markers={n: 0.0 for n in names12})
    g.fig.suptitle(f'BAO parameters at every redshift, phase prior {100*PHASE_REF:.1f}%',
                   fontsize=22, y=1.045)
    _save('fig_bao_triangle_all_z.png', g=g)
