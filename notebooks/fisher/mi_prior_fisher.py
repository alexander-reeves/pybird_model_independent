"""The MI model the HMC samples, as a Fisher system: shared machinery.

Used by hmc_prior_bands.py (validation against the chain) and corner_audit_pb.py (corners +
prior audit). One copy of the model so the two cannot drift.

THE MODEL (mi_model.py + setups.COMMON): 60 ln-a nodes log-spaced over [1e-4, 0.7] h/Mpc at
spacing 0.15 in ln k, cubic-interpolated onto the 80 emulator knots, with priors
    s_lna = 0.5 per node,  smooth_lambda = 200 on second differences of ln a,
    s_lng = 0.3 on ln of every (f, H, D_A, D-ratio),  s_lnh = 0.05 on ln h_conv.

DATA-SIDE SYMMETRY PROJECTION. The continuum model has three exact symmetries (jacobian_pb.py):
    g1 ruler      ln h_conv +1, ln D_A(z) +1, ln H(z) -1 at every z   (no amplitudes!)
    g2 amplitude  ln a +1 at every node, ln D(z) -1/2
    g3 dilation   ln h_conv +1, ln a -(3 + dlnT/dlnk)
The discretization breaks them (g1 at sigma ~0.05, g3 at ~0.3), and a Fisher would read that
breaking as data information. The PRIORS may legitimately constrain these directions -- that is
what a prior is for -- but the DATA must not. So the projection is applied to the data rows
only: W_data -> (I - Q Q^T) W_data with Q an orthonormal basis of {W_data g}, priors untouched.

This matters for the ruler: g1 does not involve the amplitudes, so no template prior can break
it. Absolute distances in this model come from the h_conv / growth priors, never from the
smoothness prior. What the smoothness prior does break is g3, which is why the ratio
D_A/r_d (ruler = the template's own wiggles) becomes measurable -- see robust_marg_pb.py.
"""
import os
import numpy as np
from scipy.interpolate import CubicSpline

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'output', 'fisher_pb')
R = np.load(os.path.join(OUT, 'jacobians_pb.npz'), allow_pickle=True)

knots_h = R['knots_h']; nk = int(R['n_knots']); ng = int(R['n_growth'])
nz = int(R['n_z_unique']); nsky = int(R['num_skies']); zz = R['zeff_unique']
gfid = R['growth_fid']; h_fid = float(R['h_true'])
iF, iH, iD = R['idx_f'], R['idx_H'], R['idx_DA']
iDr = 3 * nz + np.arange(nz); ihc = ng - 1
gamma = 3.0 + R['dlnT_dlnk']
T_ = R['template_mpc']
Tnw = R['template_nw_mpc'] if 'template_nw_mpc' in R.files else None

PRIORS_HMC = dict(s_lna=0.5, s_lng=0.3, s_lnh=0.05, lam=200.0)
# Recommended replacement (prior_decomposition_pb.py, 2026-09-29): lambda is the block that
# injects information into LCDM, and s_lna sets most of the reported P(k) band. Widening to
# these leaves the roughness the emulator sees at 0.135 per node (from 0.066) and moves the MI
# results by a few per cent, while the band widens to what the DATA actually constrain.
PRIORS_WIDE = dict(s_lna=2.0, s_lng=0.3, s_lnh=0.05, lam=50.0)
NODE_RANGE, NODE_SPACING = (1e-4, 0.7), 0.15
DATASETS = [('P', 'p'), ('P+B', 'pb')]

# ---- node basis (mi_model.lna_to_knots) ------------------------------------------------
lnk_knots = np.log(knots_h * h_fid)
_lo, _hi = np.log(NODE_RANGE[0] * h_fid), np.log(NODE_RANGE[1] * h_fid)
n_nodes = int(np.round((_hi - _lo) / NODE_SPACING)) + 1
nodes_lnk = np.linspace(_lo, _hi, n_nodes)
nodes_h = np.exp(nodes_lnk) / h_fid
_q = np.clip(lnk_knots, nodes_lnk[0], nodes_lnk[-1])
S = np.zeros((nk, n_nodes))
for _i in range(n_nodes):
    _e = np.zeros(n_nodes); _e[_i] = 1.0
    S[:, _i] = CubicSpline(nodes_lnk, _e, bc_type='not-a-knot')(_q)
D2 = np.zeros((n_nodes - 2, n_nodes))
for _i in range(n_nodes - 2):
    D2[_i, _i:_i + 3] = (1.0, -2.0, 1.0)
# the template-dilation pattern expressed on the nodes (least squares through the spline)
gamma_nodes = np.linalg.lstsq(S, gamma, rcond=None)[0]


# ---- the BAO phase direction ------------------------------------------------------------
# P_lin = a(k) T(k) with T = T_nw (1 + O): T_nw the smooth part, O the BAO wiggles. Sliding
# the wiggle pattern by delta in ln k, O(k) -> O(k e^delta), changes ln P by
#     dln P = delta * v(k),   v = dln(1 + O)/dln k,
# so if ln a can contain the pattern v the template moves its OWN BAO feature and the ruler
# floats: that is the whole reason absolute D/r_d is weak in this model. A Gaussian prior of
# width sigma_delta on the component of ln a along v is exactly the assumption a standard BAO
# fit makes -- broadband free, wiggle phase fixed -- with sigma_delta the fractional slide the
# template is still allowed. sigma_delta -> 0 is rigid wiggles, sigma_delta -> inf the free
# template. Nothing else about the template is touched: the broadband and the wiggle AMPLITUDE
# (the damping nuisance of a BAO fit) stay free.
wiggle_O = (T_ / Tnw - 1.0) if Tnw is not None else None
wiggle_v = np.gradient(np.log1p(wiggle_O), np.log(knots_h)) if Tnw is not None else None
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


def prior_precision(n_amp=None, s_lna=0.5, s_lng=0.3, s_lnh=0.05, lam=200.0, Db=None):
    """Prior precision on [ln a nodes, ln growth], exactly mi_model.prior_prec_mat."""
    if n_amp is None:
        n_amp = n_nodes
    Dm = D2 if Db is None else Db
    Pi = np.zeros((n_amp + ng, n_amp + ng))
    Pi[:n_amp, :n_amp] = np.eye(n_amp) / s_lna**2 + lam * (Dm.T @ Dm)
    g0 = n_amp
    for j in range(4 * nz):
        Pi[g0 + j, g0 + j] = 1.0 / s_lng**2
    Pi[g0 + ihc, g0 + ihc] = 1.0 / s_lnh**2
    return Pi


def gauge_vectors(npar, o_a, o_g, n_amp=None, gnodes=None):
    """g1, g2, g3 in the node parametrization (columns)."""
    if n_amp is None:
        n_amp = n_nodes
    gn = gamma_nodes if gnodes is None else gnodes
    G = np.zeros((npar, 3))
    G[o_g + ihc, 0] = 1.0; G[o_g + iD, 0] = 1.0; G[o_g + iH, 0] = -1.0
    G[o_a:o_a + n_amp, 1] = 1.0; G[o_g + iDr, 1] = -0.5
    G[o_g + ihc, 2] = 1.0; G[o_a:o_a + n_amp, 2] = -gn
    return G


def node_basis(spacing=NODE_SPACING):
    """(nodes_lnk, S, D2, gamma_nodes) for a given node spacing in ln k.

    The BAO oscillation spans ~0.6 in ln k, so a spacing well below that lets the nodes move
    the wiggles (the template can act as its own ruler); a spacing above it cannot represent a
    wiggle shift at all, which is what a BAO fit assumes."""
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


def whiten(key):
    """The whitened model Jacobian in KNOT coordinates, for one data set.

    Rows are whitened data points (L^T J with P = L L^T the data precision); columns are
    [EFT coefficients per sky, 80 ln-a knots, 25 ln-growth parameters], each scaled so the
    parameter is dimensionless: EFT by its prior width (or its own size where it has none),
    the template and growth by their fiducial values, which makes them ln-parameters.
    build() maps the knot block onto the ln-a nodes; a direct LCDM fit composes it with the
    cosmology mapping instead (meeting_growth_pb.py)."""
    J, P, eftpr, x0 = R[f'J_{key}'], R[f'P_{key}'], R[f'prior_{key}'], R[f'x0_{key}']
    n_e = int(R[f'n_e_{key}']); o_a = n_e * nsky; o_gk = o_a + nk
    Lc = np.linalg.cholesky(0.5 * (P + P.T))
    s = np.ones(len(x0)); e = np.arange(o_a); has = eftpr[e] > 0
    s[e] = np.where(has, 1 / np.sqrt(np.where(has, eftpr[e], 1.0)), np.maximum(np.abs(x0[e]), 1.0))
    s[o_a:o_gk] = x0[o_a:o_gk]; s[o_gk:] = gfid
    return dict(Wd=(Lc.T @ J) * s, has=has, o_a=o_a, o_gk=o_gk, n_e=n_e, x0=x0, scale=s)


def build(key, priors=None, project_data=True, drop_hconv=False, spacing=NODE_SPACING,
          project=(0, 1, 2), phase_sigma=None, n_phase=N_PHASE):
    """Whitened, node-basis system for one data set. Returns the covariance and the offsets.

    project_data: remove the data's response along the exact symmetries (the numerical
    breaking), leaving the priors to constrain those directions if they wish.
    phase_sigma: if given, add the BAO phase prior -- the template's own wiggles may slide by
    this fraction at 1 sigma, anywhere in k (see phase_modes / phase_precision). None leaves
    the wiggles free to move, which is the model the HMC samples today.
    """
    pr = dict(PRIORS_HMC if priors is None else priors)
    nl, Sb, Db, gnodes = node_basis(spacing)
    n_amp = len(nl)
    wk = whiten(key)
    Wd_k, has, o_a, o_gk, n_e = wk['Wd'], wk['has'], wk['o_a'], wk['o_gk'], wk['n_e']
    Wd = np.hstack([Wd_k[:, :o_a], Wd_k[:, o_a:o_gk] @ Sb, Wd_k[:, o_gk:]])  # knots -> nodes
    o_g = o_a + n_amp
    npar = o_g + ng
    G = gauge_vectors(npar, o_a, o_g, n_amp=n_amp, gnodes=gnodes)[:, list(project)]
    breaking = [1.0 / max(np.linalg.norm(Wd @ (G[:, j] / np.linalg.norm(G[:, j]))), 1e-300)
                for j in range(G.shape[1])]
    if project_data:
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
    if drop_hconv:
        k = np.ones(npar, bool); k[o_g + ihc] = False
        F = F[np.ix_(k, k)]
    C = np.linalg.inv(F)
    return dict(C=C, F=F, o_a=o_a, o_g=o_g, npar=npar, n_e=n_e, S=Sb, n_amp=n_amp,
                nodes_lnk=nl, D2=Db, phase_sigma=phase_sigma, n_phase=n_phase,
                min_eig=float(np.linalg.eigvalsh(F).min()), breaking=breaking, priors=pr)


def sig(sysd, c):
    return float(np.sqrt(c @ sysd['C'] @ c))


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
    elif name == 'h_conv':
        c[o + ihc] = 1.0
    return c


def block_cov(sysd, names, iz):
    """Covariance of a set of growth quantities at one redshift."""
    Cs = np.array([growth_c(sysd, n, iz) for n in names])
    return Cs @ sysd['C'] @ Cs.T
