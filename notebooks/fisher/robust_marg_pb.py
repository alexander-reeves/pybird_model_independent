"""Square-root (SVD) Fisher analysis of P(k) vs P(k)+B(k): properly marginalized geometry,
growth and template-band constraints, with the model's exact symmetries handled explicitly.

Reads  ../../output/fisher_pb/jacobians_pb.npz  (written by jacobian_pb.py)
Writes ../../output/fisher_pb/{robust_pb_results.npz, bands_pb.png, geometry_robust_pb.png}
and prints every table the notebook quotes.

METHOD
  1. Whitening. With P = L L^T the data precision, each data set is the linear model
     W theta = L^T J theta, stacked with sqrt(prior) rows for the EFT coefficients (and, for the
     'smooth' template, the project's smoothness prior on ln a). Columns are scaled to
     fractional units (d ln theta) for the template and growth block and to the prior width for
     the EFT coefficients: every reported quantity is invariant under this, only the
     conditioning improves.
  2. Symmetries. The continuum model has three EXACT symmetries (tested in jacobian_pb.py):
        g1 ruler      ln h_conv +1, ln D_A(z) +1, ln H(z) -1 at every z
        g2 amplitude  ln a_j +1 at every knot, ln D(z) -1/2 at every z
        g3 dilation   ln h_conv +1, ln a_j -(3 + dlnT/dlnk)_j
     The discretized model breaks them slightly, and a Fisher matrix reads that breaking as
     information. Each case projects out exactly the symmetries that survive in its reduced
     parameter space, W -> (I - Q Q^T) W with Q an orthonormal basis of {W g}. That is the Schur
     complement for marginalizing a free nuisance that moves along g: it can only remove
     information, never invent it.
  3. Marginals from the SVD of W: var(c) = sum_i (v_i . c)^2 / s_i^2 over singular values above a
     tolerance. The SVD works with the square root of the condition number of J^T P J, so it
     resolves directions that the explicit normal matrix (Gate 9) could not. A quantity is
     reported only if it is gauge invariant (orthogonal to the null space) AND its variance is
     stable across the tolerance sweep.

CASES
  template:  'fixed'  amplitudes held at the fiducial (the conventional template fit)
             'free'   80 independent knot amplitudes, no prior (the v3 model-independent block)
             'smooth' free amplitudes + lambda * |D2 ln a|^2 + |ln a|^2 / sigma_lna^2, the prior
                      form of mi_model.py. NB: it penalizes a wiggly ln a, and dilating the
                      template changes ln a by -(3 + dlnT/dlnk), which oscillates at the BAO: the
                      smooth prior therefore LOCKS the wiggles to the fiducial template (g3 is
                      broken by the prior, not projected).
  ruler:     free   h_conv free (the ruler length in Mpc/h unknown -- the MI model's default)
             known  h_conv held fixed (equivalent to a known r_d h, i.e. calibrated BAO)
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', 'output', 'fisher_pb')
R = np.load(os.path.join(OUT, 'jacobians_pb.npz'), allow_pickle=True)

zz = R['zeff_unique']; nz = int(R['n_z_unique']); nk = int(R['n_knots']); ng = int(R['n_growth'])
nsky = int(R['num_skies']); gfid = R['growth_fid']; knots_h = R['knots_h']
iF, iH, iD = R['idx_f'], R['idx_H'], R['idx_DA']
iDr = 3 * nz + np.arange(nz)
ihc = ng - 1
gamma = 3.0 + R['dlnT_dlnk']                    # d ln a_j / d ln(dilation) is -gamma_j
Z_REF = int(np.argmin(np.abs(zz - 0.93)))       # reference bin for relative distances

# ---- the template's own BAO ruler ---------------------------------------------------------
# A BAO fit measures D_M/r_d and D_H/r_d, the distances in units of the ruler the data carry;
# r_d itself is never needed. With a free template the ruler is the template's own wiggle scale.
# Its (log) change is read off the template by a linear functional rho = w . d ln a with
#   w . gamma = 1  (a template dilation by lambda moves the ruler by -ln lambda, g3)
#   w . 1     = 0  (an overall amplitude change does not move it, g2)
# and w built from the wiggle derivative dO/dlnk, O = P/P_nw - 1, with polynomials in ln k up to
# RULER_NPOLY projected out over the BAO range: so rho responds to where the wiggles are and is
# blind to smooth broadband distortions. Then
#   ln(D_M/r_d)(z) = ln D_A(z) - ln h_conv - rho,   ln(D_H/r_d)(z) = -ln H(z) - ln h_conv - rho
# are invariant under g1, g2 and g3 (verified numerically below). With the template fixed rho = 0
# and they are the standard BAO alpha_perp, alpha_par (h never enters).
T_ = R['template_mpc']
if 'template_nw_mpc' in R.files:
    Tnw = R['template_nw_mpc']
else:
    _lk = np.log(knots_h); _lkf = np.linspace(_lk.min(), _lk.max(), 4000)
    _lT = np.interp(_lkf, _lk, np.log(T_)); _w = np.exp(-0.5 * ((_lkf[:, None] - _lkf[None, :]) / 0.25)**2)
    Tnw = np.exp(np.interp(_lk, _lkf, (_w @ _lT) / _w.sum(1)))
RULER_NPOLY = int(os.environ.get('RULER_NPOLY', '3'))
RULER_KMIN, RULER_KMAX = 0.02, 0.30

def make_ruler(npoly=RULER_NPOLY, kmin=RULER_KMIN, kmax=RULER_KMAX):
    lk = np.log(knots_h)
    O = T_ / Tnw - 1.0
    dO = np.gradient(O, lk)
    sel = (knots_h >= kmin) & (knots_h <= kmax)
    Xp = np.vander(lk[sel] - lk[sel].mean(), npoly + 1)
    w0 = dO[sel] - Xp @ np.linalg.lstsq(Xp, dO[sel], rcond=None)[0]
    w = np.zeros(nk); w[sel] = w0
    return w / (w @ gamma)

W_RULER = make_ruler()
LAMBDA_SMOOTH = float(os.environ.get('LAMBDA_SMOOTH', '200'))
SIGMA_LNA = float(os.environ.get('SIGMA_LNA', '0.5'))
TOLS = [1e-8, 1e-10, 1e-12, 1e-14]              # relative singular-value cut-offs swept
TOL_REPORT = 1e-12
NULL_TOL = 1e-13
DRIFT_MAX = 0.02                                 # >2% change across the sweep = not converged
DATASETS = [('P', 'p'), ('P+B', 'pb')]


def log(msg=''):
    print(msg, flush=True)


# ---------------------------------------------------------------------------------------
# whitened, scaled linear system for one data set
# ---------------------------------------------------------------------------------------
def build(key):
    J, P = R[f'J_{key}'], R[f'P_{key}']
    prior, x0 = R[f'prior_{key}'], R[f'x0_{key}']
    n_e = int(R[f'n_e_{key}'])
    o_a = n_e * nsky; o_g = o_a + nk; npar = o_g + ng
    Lc = np.linalg.cholesky(0.5 * (P + P.T))
    s = np.ones(npar)
    e = np.arange(o_a)
    has = prior[e] > 0
    s[e] = np.where(has, 1.0 / np.sqrt(np.where(has, prior[e], 1.0)), np.maximum(np.abs(x0[e]), 1.0))
    s[o_a:o_g] = x0[o_a:o_g]
    s[o_g:] = gfid
    Wd = (Lc.T @ J) * s                           # data rows, d ln theta columns
    pr = np.where(has)[0]
    Wp = np.zeros((len(pr), npar)); Wp[np.arange(len(pr)), pr] = 1.0   # sqrt(prior)*s = 1
    G = np.zeros((npar, 3))
    G[o_g + ihc, 0] = 1.0; G[o_g + iD, 0] = 1.0; G[o_g + iH, 0] = -1.0     # g1 ruler
    G[o_a:o_g, 1] = 1.0; G[o_g + iDr, 1] = -0.5                            # g2 amplitude
    G[o_g + ihc, 2] = 1.0; G[o_a:o_g, 2] = -gamma                          # g3 dilation
    return dict(W=np.vstack([Wd, Wp]), Wd=Wd, n_e=n_e, o_a=o_a, o_g=o_g, npar=npar, G=G)


def smooth_rows(B):
    D2 = np.zeros((nk - 2, nk))
    for i in range(nk - 2):
        D2[i, i:i + 3] = (1.0, -2.0, 1.0)
    Ra = np.vstack([np.sqrt(LAMBDA_SMOOTH) * D2, np.eye(nk) / SIGMA_LNA])
    Rm = np.zeros((Ra.shape[0], B['npar'])); Rm[:, B['o_a']:B['o_a'] + nk] = Ra
    return Rm


def case_system(B, template, ruler_known, growth_known=False, project=True):
    """Reduced, symmetry-projected whitened system for one case."""
    npar, o_a, o_g = B['npar'], B['o_a'], B['o_g']
    keep = np.ones(npar, bool)
    if template == 'fixed':
        keep[o_a:o_g] = False
    if ruler_known:
        keep[o_g + ihc] = False
    if growth_known:
        keep[o_g:] = False
    W = B['W'] if template != 'smooth' else np.vstack([B['W'], smooth_rows(B)])
    allowed = [0, 1, 2] if template != 'smooth' else [0, 1]
    Gs = B['G'][:, allowed]
    drop = ~keep
    if drop.any():
        _, sv, vt = np.linalg.svd(Gs[drop], full_matrices=True)
        rank = int((sv > 1e-10 * max(sv.max(), 1.0)).sum()) if sv.size else 0
        Bc = vt[rank:].T
    else:
        Bc = np.eye(Gs.shape[1])
    Gk = (Gs @ Bc)[keep] if Bc.shape[1] else np.zeros((int(keep.sum()), 0))
    Wk = W[:, keep]
    breaking = []
    if Gk.shape[1]:
        for j in range(Gk.shape[1]):
            gn = Gk[:, j] / np.linalg.norm(Gk[:, j])
            breaking.append(1.0 / max(np.linalg.norm(Wk @ gn), 1e-300))   # sigma it would imply
    if project and Gk.shape[1]:
        Q, Rq = np.linalg.qr(Wk @ Gk)
        ok = np.abs(np.diag(Rq)) > 1e-14 * np.linalg.norm(Wk)
        Q = Q[:, ok]
        Wk = Wk - Q @ (Q.T @ Wk)
    _, S, Vt = np.linalg.svd(Wk, full_matrices=False)
    return dict(keep=keep, S=S, Vt=Vt, n_sym=Gk.shape[1], breaking=breaking)


def marginal(sys_, c_full):
    """sigma of the functional c (full scaled space) at every tolerance, and its gauge
    non-invariance (component along the null space)."""
    c = c_full[sys_['keep']]
    S, Vt = sys_['S'], sys_['Vt']
    nrm = max(np.linalg.norm(c), 1e-300)
    null = S <= NULL_TOL * S[0]
    ninv = float(np.linalg.norm(Vt[null] @ c) / nrm) if null.any() else 0.0
    a = Vt @ c
    sig = []
    for t in TOLS:
        g = S > t * S[0]
        sig.append(float(np.sqrt(np.sum((a[g] / S[g])**2))))
    return np.array(sig), ninv


def verdict(sig, ninv):
    """(value, status): status 'ok' / 'gauge' (not invariant) / 'drift' (not converged)."""
    if ninv > 1e-7:
        return np.nan, 'gauge'
    rep = sig[TOLS.index(TOL_REPORT)]
    lo, hi = sig[1:].min(), sig[1:].max()        # tolerances 1e-10 .. 1e-14
    if hi > (1 + DRIFT_MAX) * lo:
        return rep, 'drift'
    return rep, 'ok'


# ---------------------------------------------------------------------------------------
# functionals
# ---------------------------------------------------------------------------------------
RULER_W_OVERRIDE = None


def geo_c(B, name, iz):
    c = np.zeros(B['npar']); o = B['o_g']
    if name == 'f':
        c[o + iF[iz]] = 1.0
    elif name == 'F_AP':                      # ln(H D_A)
        c[o + iH[iz]] = 1.0; c[o + iD[iz]] = 1.0
    elif name == 'alpha_iso':                 # ln D_V up to a constant: (2 ln D_A - ln H)/3
        c[o + iD[iz]] = 2 / 3; c[o + iH[iz]] = -1 / 3
    elif name == 'alpha_iso_rel':             # relative to the reference bin
        c[o + iD[iz]] += 2 / 3; c[o + iH[iz]] += -1 / 3
        c[o + iD[Z_REF]] -= 2 / 3; c[o + iH[Z_REF]] -= -1 / 3
    elif name in ('alpha_perp', 'alpha_par'):  # D_M/r_d, D_H/r_d with the template's own ruler
        wr = RULER_W_OVERRIDE if RULER_W_OVERRIDE is not None else W_RULER
        if name == 'alpha_perp':
            c[o + iD[iz]] = 1.0
        else:
            c[o + iH[iz]] = -1.0
        c[o + ihc] = -1.0
        c[B['o_a']:B['o_a'] + nk] = -wr
    elif name == 'lnDA_rel':                  # D_A(z) / D_A(z_ref)
        c[o + iD[iz]] += 1.0; c[o + iD[Z_REF]] -= 1.0
    elif name == 'lnH_rel':                   # H(z) / H(z_ref)
        c[o + iH[iz]] += 1.0; c[o + iH[Z_REF]] -= 1.0
    elif name == 'lnDA':
        c[o + iD[iz]] = 1.0
    elif name == 'lnH':
        c[o + iH[iz]] = 1.0
    return c


def band_c(B, j, conditional, zstar=Z_REF):
    """Band power at knot j. Conditional (growth, geometry, h_conv known): ln a_j itself.
    Otherwise the gauge-invariant version: the linear power at redshift z*, in the observed
    coordinates of that bin,
        ln a_j + 2 ln D(z*) + gamma_j [ln h_conv - alpha_iso(z*)],
    which is invariant under g1, g2 and g3 (checked numerically below)."""
    c = np.zeros(B['npar']); o_a, o_g = B['o_a'], B['o_g']
    c[o_a + j] = 1.0
    if not conditional:
        g = gamma[j]
        c[o_g + iDr[zstar]] += 2.0
        c[o_g + ihc] += g
        c[o_g + iD[zstar]] += -g * 2 / 3
        c[o_g + iH[zstar]] += g / 3
    return c


# ---------------------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------------------
BUILT = {key: build(key) for _, key in DATASETS}
res = {}

log("=" * 100)
log("0. Numerical symmetry breaking: the sigma a Fisher analysis would ASSIGN to each exact "
    "symmetry\n   (1/|W g|, unit g in ln units); the continuum answer is infinity.")
fd = {key: R[f'sym_tests_{key}'] for _, key in DATASETS if f'sym_tests_{key}' in R.files}
for tag, key in DATASETS:
    B = BUILT[key]
    names = ['g1 ruler', 'g2 amplitude', 'g3 dilation']
    Wd = B['Wd']
    line = []
    for j, nm in enumerate(names):
        g = B['G'][:, j] / np.linalg.norm(B['G'][:, j])
        line.append(f"{nm} {1/max(np.linalg.norm(Wd @ g),1e-300):.3g}")
    log(f"   {tag:4s} from the Jacobian:          " + " | ".join(line))
    if key in fd:
        eps, c_h, c_g1, c_g2, c_g3, c_da = fd[key][0]
        g1n = np.linalg.norm(B['G'][:, 0]); g3n = np.linalg.norm(B['G'][:, 2])
        log(f"   {tag:4s} from finite differences:   g1 ruler {eps*g1n/np.sqrt(max(c_g1,1e-300)):.3g}"
            f" | g2 amplitude {eps*np.sqrt(nk + 0.25*nz)/np.sqrt(max(c_g2,1e-300)):.3g}"
            f" | g3 dilation {eps*g3n/np.sqrt(max(c_g3,1e-300)):.3g}"
            f"   (h_conv alone: {eps/np.sqrt(c_h):.2g})")

CASES = [('fixed', False), ('fixed', True), ('free', False), ('free', True),
         ('smooth', False), ('smooth', True)]
QTY = ['alpha_perp', 'alpha_par', 'F_AP', 'alpha_iso_rel', 'lnDA_rel', 'lnH_rel', 'f', 'alpha_iso', 'lnDA', 'lnH']

log("\n" + "=" * 100)
log("1. Geometry and growth, EVERYTHING else marginalized (EFT, template, f, D, h_conv), "
    "exact symmetries projected.\n   Fractional 1-sigma. 'gauge' = not measurable in this "
    "case (moves along an exact symmetry); '~' = not converged.")
for template, ruler_known in CASES:
    label = f"{template} template, ruler {'known' if ruler_known else 'free'}"
    for tag, key in DATASETS:
        B = BUILT[key]
        sy = case_system(B, template, ruler_known)
        tab = {}
        for q in QTY:
            vals = []
            for iz in range(nz):
                if q.endswith('_rel') and iz == Z_REF:
                    vals.append((0.0, 'ref')); continue
                sig, ninv = marginal(sy, geo_c(B, q, iz))
                vals.append(verdict(sig, ninv))
            tab[q] = vals
        res[(template, ruler_known, key)] = tab
    log(f"\n--- {label}  (symmetries projected: "
        f"{case_system(BUILT['p'], template, ruler_known)['n_sym']}) ---")
    log(f"{'':14s}" + "".join(f"{'z='+format(z,'.3f'):>17s}" for z in zz))
    for q in QTY:
        for tag, key in DATASETS:
            cells = []
            for v, st in res[(template, ruler_known, key)][q]:
                if st == 'gauge':
                    cells.append(f"{'gauge':>17s}")
                elif st == 'ref':
                    cells.append(f"{'(ref)':>17s}")
                else:
                    cells.append(f"{('~' if st == 'drift' else '') + format(v, '.4f'):>17s}")
            log(f"{q[:9]:>9s} {tag:4s}" + "".join(cells))

log("\n" + "=" * 100)
log("2. What the projection removes: fixed template, ruler free, WITHOUT projecting g1 "
    "(i.e. trusting the numerical breaking).")
for tag, key in DATASETS:
    B = BUILT[key]
    sy = case_system(B, 'fixed', False, project=False)
    vals = [marginal(sy, geo_c(B, 'lnDA', iz))[0][TOLS.index(TOL_REPORT)] for iz in range(nz)]
    valh = [marginal(sy, geo_c(B, 'lnH', iz))[0][TOLS.index(TOL_REPORT)] for iz in range(nz)]
    log(f"   {tag:4s} sigma(ln D_A) per z: " + " ".join(f"{v:.4f}" for v in vals))
    log(f"   {tag:4s} sigma(ln H)   per z: " + " ".join(f"{v:.4f}" for v in valh))
log("   -> these are the numbers the earlier Hessian/Schur analysis reported as 'template fixed'."
    " With g1 projected they are gauge quantities (unmeasurable) and only the ratios above "
    "survive.")

log("\n" + "=" * 100)
log("3. Convergence: sigma(F_AP) at every tolerance, free template, ruler free (the hardest "
    "case).")
for tag, key in DATASETS:
    B = BUILT[key]
    sy = case_system(B, 'free', False)
    for iz in range(nz):
        sig, ninv = marginal(sy, geo_c(B, 'F_AP', iz))
        log(f"   {tag:4s} z={zz[iz]:.3f}  " + "  ".join(f"tol {t:.0e}: {s:.5f}" for t, s in zip(TOLS, sig))
            + f"   non-invariance {ninv:.1e}")

# ---------------------------------------------------------------------------------------
# template bands
# ---------------------------------------------------------------------------------------
log("\n" + "=" * 100)
log("3b. Per-redshift BAO with the template FREE: sensitivity to how the ruler is read off the "
    "template\n    (polynomial order projected out, k-range of the wiggle functional). sigma of "
    "ln(D_M/r_d) | ln(D_H/r_d), z=0.51, 0.93, 1.32.")
for tag, key in DATASETS:
    B = BUILT[key]
    sy = case_system(B, 'free', False)
    for npoly, kmin, kmax in [(2, .02, .30), (3, .02, .30), (4, .02, .30), (5, .02, .30), (3, .03, .25), (3, .015, .35)]:
        RULER_W_OVERRIDE = make_ruler(npoly, kmin, kmax)
        cells = []
        for iz in (1, 3, 4):
            sp_, ni_p = marginal(sy, geo_c(B, 'alpha_perp', iz))
            sa_, ni_a = marginal(sy, geo_c(B, 'alpha_par', iz))
            cells.append(f"{sp_[TOLS.index(TOL_REPORT)]:.4f}|{sa_[TOLS.index(TOL_REPORT)]:.4f}"
                         + ("" if max(ni_p, ni_a) < 1e-7 else "(NOT INV)"))
        log(f"   {tag:4s} npoly={npoly} k=[{kmin},{kmax}]:  " + "   ".join(cells))
    RULER_W_OVERRIDE = None

log("\n" + "=" * 100)
log(f"4. Template band recovery (80 knots). Marginal = everything marginalized, reported as "
    f"the gauge-invariant\n   linear power at z={zz[Z_REF]:.3f} in observed coordinates; "
    f"conditional = growth, geometry and h_conv known.")
bands = {}
kin = (knots_h >= 0.01) & (knots_h <= 0.30)
for template in ('free', 'smooth'):
    for tag, key in DATASETS:
        B = BUILT[key]
        for cond in (True, False):
            sy = case_system(B, template, ruler_known=False, growth_known=cond)
            sig, ninv, drift = np.zeros(nk), np.zeros(nk), np.zeros(nk, bool)
            for j in range(nk):
                s_, n_ = marginal(sy, band_c(B, j, cond))
                sig[j] = s_[TOLS.index(TOL_REPORT)]
                ninv[j] = n_
                drift[j] = s_[1:].max() > (1 + DRIFT_MAX) * s_[1:].min()
            bands[(template, key, cond)] = (sig, ninv, drift)
            log(f"   {template:6s} {tag:4s} {'conditional' if cond else 'marginal   '}: "
                f"median sigma over 0.01<k<0.3 = {np.median(sig[kin]):.4f}, "
                f"max non-invariance {ninv.max():.1e}, knots not converged {int(drift.sum())}")

# binned bands (3 consecutive knots) for the BAO view
bin_edges = [j for j in range(nk) if 0.02 <= knots_h[j] <= 0.30]
bins = [bin_edges[i:i + 3] for i in range(0, len(bin_edges) - 2, 3)]
binned = {}
for template in ('free',):
    for tag, key in DATASETS:
        B = BUILT[key]
        sy = case_system(B, template, ruler_known=False)
        vals = []
        for bj in bins:
            c = np.mean([band_c(B, j, False) for j in bj], axis=0)
            s_, n_ = marginal(sy, c)
            vals.append(s_[TOLS.index(TOL_REPORT)] if n_ < 1e-7 else np.nan)
        binned[(template, key)] = np.array(vals)
kc = np.array([np.exp(np.mean(np.log(knots_h[bj]))) for bj in bins])
log(f"   binned (3 knots) free-template marginal sigma, P  : " +
    " ".join(f"{v:.3f}" for v in binned[('free', 'p')]))
log(f"   binned (3 knots) free-template marginal sigma, P+B: " +
    " ".join(f"{v:.3f}" for v in binned[('free', 'pb')]))
log(f"   bin centres k [h/Mpc]: " + " ".join(f"{k:.3f}" for k in kc))

# fiducial wiggles
T = R['template_mpc']
if 'template_nw_mpc' in R.files:
    Tnw = R['template_nw_mpc']
else:
    lk = np.log(knots_h); lkf = np.linspace(lk.min(), lk.max(), 4000)
    lT = np.interp(lkf, lk, np.log(T)); w = np.exp(-0.5 * ((lkf[:, None] - lkf[None, :]) / 0.25)**2)
    Tnw = np.exp(np.interp(lk, lkf, (w @ lT) / w.sum(1)))
wig = T / Tnw - 1.0
wig_bin = np.array([np.mean(wig[bj]) for bj in bins])
log(f"   fiducial wiggle amplitude per bin |P/P_nw - 1|: " + " ".join(f"{abs(v):.3f}" for v in wig_bin))

# BAO detection significance in the reconstructed template: S/N = sqrt(w^T C^+ w), with C the
# marginal covariance of the gauge-invariant band powers over the knots the data reach and w
# the fiducial wiggle pattern P/P_nw - 1 there. It asks whether the reconstruction can tell the
# fiducial template from its own no-wiggle version, using every band and their correlations.
sel = np.where((knots_h >= 0.02) & (knots_h <= 0.30))[0]
wsel = wig[sel]
snr = {}
for template in ('free', 'smooth'):
    for tag, key in DATASETS:
        B = BUILT[key]
        for cond in (True, False):
            sy = case_system(B, template, ruler_known=False, growth_known=cond)
            Phi = np.array([band_c(B, j, cond)[sy['keep']] for j in sel])
            g = sy['S'] > TOL_REPORT * sy['S'][0]
            M = (Phi @ sy['Vt'][g].T) / sy['S'][g]
            Um, sm, _ = np.linalg.svd(M, full_matrices=False)
            vals = []
            for t in (1e-6, 1e-8, 1e-10):
                gg = sm > t * sm[0]
                vals.append(np.sqrt(np.sum(((Um[:, gg].T @ wsel) / sm[gg])**2)))
            snr[(template, key, cond)] = vals
            log(f"   BAO wiggle S/N in the reconstruction, {template:6s} {tag:4s} "
                f"{'conditional' if cond else 'marginal   '}: {vals[1]:.2f}"
                f"   (cut 1e-6/1e-8/1e-10: " + "/".join(f"{v:.2f}" for v in vals) + ")")

np.savez(os.path.join(OUT, 'robust_pb_results.npz'),
         zeff=zz, z_ref=zz[Z_REF], knots_h=knots_h, wiggle=wig, kc=kc, wig_bin=wig_bin,
         **{f"geo_{t}_{'known' if rk else 'free'}_{k}_{q}": np.array([v for v, _ in res[(t, rk, k)][q]])
            for (t, rk, k) in res for q in QTY},
         **{f"band_{t}_{k}_{'cond' if c else 'marg'}": bands[(t, k, c)][0] for (t, k, c) in bands},
         **{f"binned_{t}_{k}": binned[(t, k)] for (t, k) in binned},
         **{f"snr_{t}_{k}_{'cond' if c else 'marg'}": np.array(v) for (t, k, c), v in snr.items()})
log(f"\nsaved {os.path.join(OUT, 'robust_pb_results.npz')}")

# ---------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    log("matplotlib unavailable: tables only")
    raise SystemExit(0)
try:
    from make_fisher_pb_fig import C_P, C_PB, C_MUTED, C_INK
except Exception:                                               # noqa: BLE001
    C_P, C_PB, C_MUTED, C_INK = '#E8710A', '#1A73E8', '#80868B', '#202124'
COL = {'p': C_P, 'pb': C_PB}
LAB = {'p': 'P(k)', 'pb': 'P(k)+B(k)'}

# ---- bands -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), layout='constrained')
for ax, template, title in [(axes[0], 'free', '80 free knots (no prior)'),
                            (axes[1], 'smooth', f'smooth prior ($\\lambda$={LAMBDA_SMOOTH:g})')]:
    ax.axvspan(0.01, 0.20, color=C_MUTED, alpha=0.10, lw=0)
    ax.axvspan(0.02, 0.10, color=C_MUTED, alpha=0.12, lw=0)
    for key in ('p', 'pb'):
        sm, _, dm = bands[(template, key, False)]
        sc, _, _ = bands[(template, key, True)]
        ax.plot(knots_h, np.clip(sm, 1e-4, 1e3), color=COL[key], lw=2.2,
                label=f'{LAB[key]}, everything marginalized')
        ax.plot(knots_h, np.clip(sc, 1e-4, 1e3), color=COL[key], lw=1.4, ls=':',
                label=f'{LAB[key]}, growth & geometry known')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(1e-3, 0.7); ax.set_ylim(1e-3, 50)
    ax.axhline(1.0, color=C_MUTED, lw=1, ls='--')
    ax.set_xlabel(r'$k\ [h\,{\rm Mpc}^{-1}]$', fontsize=13)
    ax.set_title(title, fontsize=13, color=C_INK)
    ax.text(0.0105, 30, 'P(k) data', color=C_MUTED, fontsize=9)
    ax.text(0.021, 14, 'B(k) data', color=C_MUTED, fontsize=9)
    ax.grid(alpha=0.2, which='both')
axes[0].set_ylabel(r'fractional $\sigma$ of the band power at each knot', fontsize=12)
axes[0].legend(frameon=False, fontsize=9, loc='lower left')

ax = axes[2]
ax.plot(kc, np.abs(wig_bin), color=C_INK, lw=1.8, marker='s', ms=4,
        label=r'fiducial BAO wiggle amplitude $|P/P_{\rm nw}-1|$')
for key in ('p', 'pb'):
    snr_m = snr[('free', key, False)][1]
    ax.plot(kc, binned[('free', key)], color=COL[key], lw=2.2, marker='o', ms=4,
            label=f'{LAB[key]}: $1\\sigma$ per 3-knot band  (wiggle S/N = {snr_m:.2f})')
ax.set_yscale('log'); ax.set_xlim(0.015, 0.31); ax.set_ylim(2e-3, 30)
ax.set_xlabel(r'$k\ [h\,{\rm Mpc}^{-1}]$', fontsize=13)
ax.set_ylabel(r'fractional amplitude / $1\sigma$', fontsize=12)
ax.set_title(f'BAO in the free-template reconstruction (z = {zz[Z_REF]:.2f})', fontsize=13, color=C_INK)
ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.grid(alpha=0.2, which='both')
fig.suptitle('Linear power spectrum recovery, P(k) alone vs P(k)+B(k)', fontsize=15)
p1 = os.path.join(OUT, 'bands_pb.png')
fig.savefig(p1, dpi=170); plt.close(fig)
log(f"saved {p1}")

# ---- geometry -----------------------------------------------------------------------------
LS = {'fixed': '-', 'smooth': ':', 'free': '--'}
fig, axes = plt.subplots(1, 4, figsize=(20, 5), layout='constrained', sharex=True)
panels = [('F_AP', False, r'$F_{\rm AP}=D_A H$  (ruler free)'),
          ('alpha_iso_rel', False, rf'$\alpha_{{\rm iso}}(z)/\alpha_{{\rm iso}}({zz[Z_REF]:.2f})$  (ruler free)'),
          ('f', False, r'$f(z)$  (ruler free)'),
          ('alpha_iso', True, r'absolute $\alpha_{\rm iso}$  (ruler KNOWN)')]
for ax, (q, rk, title) in zip(axes, panels):
    for template in ('fixed', 'smooth', 'free'):
        for key in ('p', 'pb'):
            v = np.array([x for x, _ in res[(template, rk, key)][q]], float)
            st = [s for _, s in res[(template, rk, key)][q]]
            v = np.where([s in ('gauge', 'ref') for s in st], np.nan, v)
            ax.plot(zz, v, color=COL[key], ls=LS[template], marker='o', ms=4, lw=1.8,
                    label=f'{LAB[key]}, {template} template')
        if all(s == 'gauge' for _, s in res[(template, rk, 'p')][q]):
            ax.text(0.03, 0.04 + 0.07 * ['fixed', 'smooth', 'free'].index(template),
                    f'{template} template: not measurable', transform=ax.transAxes,
                    fontsize=9, color=C_MUTED)
    ax.set_yscale('log'); ax.set_title(title, fontsize=12, color=C_INK)
    ax.set_xlabel('z', fontsize=12); ax.grid(alpha=0.2, which='both')
axes[0].set_ylabel(r'fractional $1\sigma$', fontsize=12)
axes[0].legend(frameon=False, fontsize=8.5, handlelength=3.5)
fig.suptitle('Geometry and growth with everything else marginalized and the exact symmetries '
             'projected', fontsize=14)
p2 = os.path.join(OUT, 'geometry_robust_pb.png')
fig.savefig(p2, dpi=170); plt.close(fig)
log(f"saved {p2}")


# ---- band recovery with 1-sigma bands ------------------------------------------------------
# Template FREE, EVERYTHING marginalized. For P(k) alone the overall amplitude of the linear power
# is degenerate with b1 (P ~ b1^2 D^2 a), so every band floats together at O(100%); showing that
# as a band hides the shape information. So: (i) the SHAPE, each 3-knot band relative to the mean
# over all bands, with the overall amplitude marginalized; (ii) the amplitude as its own number.
Dz = gfid[iDr[Z_REF]]
Pz = T_ * R['h_true']**3 * Dz**2
sel_all = [j for j in range(nk) if 0.01 <= knots_h[j] <= 0.30]
bins_all = [sel_all[i:i + 3] for i in range(0, len(sel_all) - 2, 3)]
union = [j for bj in bins_all for j in bj]
shape_sig, amp_sig = {}, {}
for key in ('p', 'pb'):
    B = BUILT[key]
    sy = case_system(B, 'free', ruler_known=False)
    cj = {j: band_c(B, j, False) for j in union}
    cmean = np.mean([cj[j] for j in union], axis=0)
    amp_sig[key] = marginal(sy, cmean)[0][TOLS.index(TOL_REPORT)]
    shape_sig[key] = np.array([marginal(sy, np.mean([cj[j] for j in bj], axis=0) - cmean)[0][TOLS.index(TOL_REPORT)]
                               for bj in bins_all])
log(f"   band SHAPE sigma per 3-knot band (amplitude marginalized), P  : " + " ".join(f"{v:.3f}" for v in shape_sig['p']))
log(f"   band SHAPE sigma per 3-knot band (amplitude marginalized), P+B: " + " ".join(f"{v:.3f}" for v in shape_sig['pb']))
log(f"   overall amplitude of P_lin(z*): sigma = {amp_sig['p']:.3f} (P)  {amp_sig['pb']:.3f} (P+B)")

fig, axes = plt.subplots(1, 2, figsize=(16, 5.8), layout='constrained')
for ax, ratio in ((axes[0], False), (axes[1], True)):
    base = Pz / (T_ / Tnw * 0 + 1) if not ratio else T_ / Tnw
    yfid = knots_h * Pz if not ratio else T_ / Tnw
    ax.plot(knots_h, yfid, color=C_INK, lw=1.8, label='true' + (r' $P/P_{\rm nw}$' if ratio else ' linear P(k)'), zorder=5)
    for n_, bj in enumerate(bins_all):
        k0, k1 = knots_h[bj[0]], knots_h[bj[-1]]
        yc = np.exp(np.mean(np.log(yfid[bj])))
        for key, fill in (('pb', True), ('p', False)):
            sg = min(shape_sig[key][n_], 5.0)
            lo, hi = yc * np.exp(-sg), yc * np.exp(sg)
            if fill:
                ax.fill_between([k0, k1], [lo, lo], [hi, hi], color=C_PB, alpha=0.35, lw=0,
                                label=r'P(k)+B(k)  $\pm1\sigma$' if n_ == 0 else None)
            else:
                ax.plot([k0, k1, k1, k0, k0], [lo, lo, hi, hi, lo], color=C_P, lw=1.5,
                        label=r'P(k)  $\pm1\sigma$' if n_ == 0 else None)
    ax.set_xlabel(r'$k\ [h\,{\rm Mpc}^{-1}]$', fontsize=13); ax.grid(alpha=0.2, which='both')
    if not ratio:
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(0.009, 0.33)
        ax.set_ylim(yfid[union].min() / 4, yfid[union].max() * 4)
        ax.set_ylabel(rf'$k\,P_{{\rm lin}}(k,\,z={zz[Z_REF]:.2f})\ [({{\rm Mpc}}/h)^2]$', fontsize=13)
        ax.set_title('Band shape, 3-knot bands (overall amplitude marginalized)', fontsize=13, color=C_INK)
        ax.axvspan(0.01, 0.20, color=C_MUTED, alpha=0.08, lw=0); ax.axvspan(0.02, 0.10, color=C_MUTED, alpha=0.10, lw=0)
    else:
        ax.set_xlim(0.009, 0.31); ax.set_ylim(0.5, 1.6); ax.axhline(1.0, color=C_MUTED, lw=0.8)
        ax.set_ylabel(r'$P_{\rm lin}/P_{\rm nw}$', fontsize=13)
        ax.set_title(f"Same bands against the BAO   (wiggle-pattern S/N: P {snr[('free','p',False)][1]:.1f}, "
                     f"P+B {snr[('free','pb',False)][1]:.1f})", fontsize=12, color=C_INK)
    ax.legend(frameon=False, fontsize=10, loc='lower left' if not ratio else 'upper right')
fig.suptitle(f"Linear P(k) recovered with the template FREE, everything marginalized.   Overall amplitude: "
             f"$\\sigma$ = {amp_sig['p']:.2f} (P)  vs  {amp_sig['pb']:.2f} (P+B)", fontsize=14)
p3 = os.path.join(OUT, 'band_recovery_pb_priorfree.png')
fig.savefig(p3, dpi=170); plt.close(fig)
log(f"saved {p3}")

# ---- AP summary, properly marginalized ------------------------------------------------------
cols = [('alpha_perp', r'$D_M/r_d$  ($\alpha_\perp$)'), ('alpha_par', r'$D_H/r_d$  ($\alpha_\parallel$)'),
        ('F_AP', r'$F_{\rm AP}=D_M/D_H$'), ('f', r'$f(z)$')]
fig, axes = plt.subplots(2, 4, figsize=(20, 7.5), layout='constrained', sharex=True,
                         gridspec_kw={'height_ratios': [3, 1.3]})
def _vals(template, key, q):
    v = np.array([x for x, _ in res[(template, False, key)][q]], float)
    st = [t for _, t in res[(template, False, key)][q]]
    return np.where([t in ('gauge', 'ref') for t in st], np.nan, v)
for j, (q, title) in enumerate(cols):
    ax, axr = axes[0, j], axes[1, j]
    for template, ls, lw, alpha, lab in (('free', '-', 2.2, 1.0, 'template free'),
                                         ('fixed', '--', 1.2, 0.55, 'true template inserted')):
        for key in ('p', 'pb'):
            ax.plot(zz, _vals(template, key, q), color=COL[key], ls=ls, lw=lw, alpha=alpha,
                    marker='o' if template == 'free' else None, ms=4,
                    label=f'{LAB[key]}, {lab}')
        axr.plot(zz, _vals(template, 'pb', q) / _vals(template, 'p', q), color=C_INK, ls=ls,
                 lw=lw, alpha=alpha, marker='o' if template == 'free' else None, ms=4,
                 label=lab)
    ax.set_yscale('log'); ax.set_title(title, fontsize=14, color=C_INK)
    ax.grid(alpha=0.2, which='both')
    axr.axhline(1.0, color=C_MUTED, lw=1); axr.set_ylim(0, 1.15); axr.grid(alpha=0.2)
    axr.set_xlabel('z', fontsize=13)
    if q.endswith('_rel'):
        ax.axvline(zz[Z_REF], color=C_MUTED, lw=1, ls=':')
        ax.text(zz[Z_REF], ax.get_ylim()[0] if False else 0.02, ' reference', color=C_MUTED,
                fontsize=9, transform=ax.get_xaxis_transform())
axes[0, 0].set_ylabel(r'fractional $1\sigma$', fontsize=13)
axes[1, 0].set_ylabel(r'$\sigma_{P+B}/\sigma_P$', fontsize=13)
axes[0, 0].legend(frameon=False, fontsize=9, handlelength=3.2)
axes[1, 0].legend(frameon=False, fontsize=9, handlelength=3.2, loc='lower left')
fig.suptitle('Per-redshift BAO and growth, P(k) alone vs P(k)+B(k): template free (ruler = the template\'s '
             'own wiggles), everything marginalized, exact symmetries projected', fontsize=13)
p4 = os.path.join(OUT, 'ap_summary_marg_pb_priorfree.png')
fig.savefig(p4, dpi=170); plt.close(fig)
log(f"saved {p4}")


# =============================================================================================
# 6. THE HMC MODEL: 60 smooth nodes + the mi_model priors (setups.COMMON), validated on the chain
# =============================================================================================
# Every case above leaves the template (and the growth block) WITHOUT priors: 80 independent
# knots, so the knots outside the data -- limited only through the loops -- are almost free and
# their freedom leaks into every marginal. The sampled analysis (hmc_v3_p5, mock_pk_reconstruction)
# never used that model: ln a lives on 60 log-spaced nodes (1e-4..0.7 h/Mpc, spacing 0.15 in ln k,
# cubic onto the 80 knots), with sigma_lna = 0.5 per node, lambda = 200 on second differences,
# sigma_lng = 0.3 on f, H, D_A, D per z and sigma_lnh = 0.05 on h_conv. Those priors also make every
# symmetry proper, so nothing is projected here. Validated two ways against the MI chain itself:
# the v3 Fisher (same data, same EFT as the chain) with these priors, and this pipeline's P-only.
from scipy.interpolate import CubicSpline
HMC = dict(s_lna=0.5, s_lng=0.3, s_lnh=0.05, lam=200.0)
h_fid = float(R['h_true'])
n_lo, n_hi = np.log(1e-4 * h_fid), np.log(0.7 * h_fid)
N_NODES = int(np.round((n_hi - n_lo) / 0.15)) + 1
nodes_lnk = np.linspace(n_lo, n_hi, N_NODES)
k_star = knots_h * h_fid                                   # 1/Mpc, as in the HMC figure
Smat = np.column_stack([CubicSpline(nodes_lnk, e)(np.clip(np.log(k_star), n_lo, n_hi))
                        for e in np.eye(N_NODES)])          # d ln a_knot / d ln a_node (80 x 60)


def hmc_prior_rows(n_eft):
    n2 = n_eft + N_NODES + ng
    D2 = np.zeros((N_NODES - 2, N_NODES))
    for i in range(N_NODES - 2):
        D2[i, i:i + 3] = (1.0, -2.0, 1.0)
    Rn = np.zeros((N_NODES, n2)); Rn[:, n_eft:n_eft + N_NODES] = np.eye(N_NODES) / HMC['s_lna']
    Rs = np.zeros((N_NODES - 2, n2)); Rs[:, n_eft:n_eft + N_NODES] = np.sqrt(HMC['lam']) * D2
    Rg = np.zeros((ng, n2)); og = n_eft + N_NODES
    for i in range(ng - 1):
        Rg[i, og + i] = 1.0 / HMC['s_lng']
    Rg[ng - 1, og + ng - 1] = 1.0 / HMC['s_lnh']
    return np.vstack([Rn, Rs, Rg])


def to_nodes(c, o_a, o_g):
    return np.concatenate([c[:o_a], Smat.T @ c[o_a:o_g], c[o_g:]])


def hmc_sys(B, with_data=True):
    o_a, o_g = B['o_a'], B['o_g']
    Wd = B['Wd']
    Wpe = B['W'][Wd.shape[0]:, :o_a]
    Wpe = np.hstack([Wpe, np.zeros((Wpe.shape[0], N_NODES + ng))])
    rows = [Wpe, hmc_prior_rows(o_a)]
    if with_data:
        rows.insert(0, np.hstack([Wd[:, :o_a], Wd[:, o_a:o_g] @ Smat, Wd[:, o_g:]]))
    W = np.vstack(rows)
    _, S, Vt = np.linalg.svd(W, full_matrices=False)
    return dict(keep=np.ones(W.shape[1], bool), S=S, Vt=Vt)


def sig_of(sy, c):
    sg, ninv = marginal(sy, c)
    return sg[TOLS.index(TOL_REPORT)] if ninv < 1e-7 else np.nan


log("\n" + "=" * 100)
log(f"6. The HMC model: {N_NODES} nodes, sigma_lna={HMC['s_lna']}, lambda={HMC['lam']:g}, "
    f"sigma_lng={HMC['s_lng']}, sigma_lnh={HMC['s_lnh']}")

# --- (a) the chain ---
ch = np.load(os.path.join(OUT, '..', 'hmc_v3_p5', 'chain_mi.npz'))['x']
ne3 = ch.shape[-1] - N_NODES - ng
xs = ch.reshape(-1, ch.shape[-1])[:, ne3:ne3 + N_NODES]
lnak_ch = xs @ Smat.T
sig_ch = lnak_ch.std(0)
a16_ch, a84_ch = np.percentile(np.exp(lnak_ch), [16, 84], axis=0)
log(f"   chain {ch.shape}: {ne3} EFT + {N_NODES} nodes + {ng} growth")

# --- (b) v3 Fisher (same data and EFT as the chain) + the HMC priors ---
v3 = np.load(os.path.join(OUT, '..', 'fisher_v3', 'fisher_v3_results.npz'))
F3 = 0.5 * (v3['F_full'] + v3['F_full'].T)
pf3 = v3['params_fid']; nE3 = len(pf3) - nk - ng
s3 = np.ones(len(pf3)); s3[nE3:nE3 + nk] = pf3[nE3:nE3 + nk]; s3[nE3 + nk:] = v3['growth_fid']
F3s = (F3 * s3).T * s3
w_, V_ = np.linalg.eigh(F3s); F3s = (V_ * np.clip(w_, 0, None)) @ V_.T
T3 = np.zeros((len(pf3), nE3 + N_NODES + ng))
T3[:nE3, :nE3] = np.eye(nE3); T3[nE3:nE3 + nk, nE3:nE3 + N_NODES] = Smat
T3[nE3 + nk:, nE3 + N_NODES:] = np.eye(ng)
Rp3 = hmc_prior_rows(nE3)
C3 = np.linalg.inv(T3.T @ F3s @ T3 + Rp3.T @ Rp3)
Cn3 = C3[nE3:nE3 + N_NODES, nE3:nE3 + N_NODES]
sig_v3 = np.sqrt(np.einsum('jk,kl,jl->j', Smat, Cn3, Smat))

# --- (c) this pipeline, P and P+B, with the same priors; and the priors alone ---
HS = {key: hmc_sys(BUILT[key]) for _, key in DATASETS}
HS0 = hmc_sys(BUILT['p'], with_data=False)
sig_h = {}
for key in ('p', 'pb'):
    B = BUILT[key]
    sig_h[key] = np.array([sig_of(HS[key], to_nodes(np.eye(B['npar'])[B['o_a'] + j], B['o_a'], B['o_g']))
                           for j in range(nk)])
B0 = BUILT['p']
sig_prior = np.array([sig_of(HS0, to_nodes(np.eye(B0['npar'])[B0['o_a'] + j], B0['o_a'], B0['o_g']))
                      for j in range(nk)])
dr = (knots_h >= 0.01) & (knots_h <= 0.20)
log(f"   sigma(ln a) per knot, median over the data range (0.01<k<0.2 h/Mpc) | outside:")
for lab_, v_ in [('HMC chain (truth)', sig_ch), ('v3 Fisher + HMC priors', sig_v3),
                 ('this pipeline, P  + HMC priors', sig_h['p']), ('this pipeline, P+B + HMC priors', sig_h['pb']),
                 ('HMC priors alone', sig_prior)]:
    log(f"      {lab_:34s} {np.nanmedian(v_[dr]):.3f} | {np.nanmedian(v_[~dr]):.3f}")
log(f"   Fisher / chain per knot in the data range: v3 {np.nanmedian(sig_v3[dr]/sig_ch[dr]):.2f}, "
    f"this pipeline P {np.nanmedian(sig_h['p'][dr]/sig_ch[dr]):.2f}")

# --- (d) per-redshift BAO, AP and growth under the HMC model ---
QH = ['alpha_perp', 'alpha_par', 'F_AP', 'f']
hgeo = {}
for key in ('p', 'pb'):
    B = BUILT[key]
    for q in QH:
        hgeo[(key, q)] = np.array([sig_of(HS[key], to_nodes(geo_c(B, q, iz), B['o_a'], B['o_g'])) for iz in range(nz)])
hgeo_prior = {q: np.array([sig_of(HS0, to_nodes(geo_c(B0, q, iz), B0['o_a'], B0['o_g'])) for iz in range(nz)]) for q in QH}
log(f"   per-z fractional 1-sigma under the HMC model (z = {', '.join(f'{z:.2f}' for z in zz)}):")
for q in QH:
    for key in ('p', 'pb'):
        log(f"      {q:10s} {LAB[key] if 'LAB' in dir() else key:10s} " + " ".join(f"{v:.4f}" for v in hgeo[(key, q)]))
    log(f"      {q:10s} {'prior only':10s} " + " ".join(f"{v:.4f}" for v in hgeo_prior[q]))
log("   ruler-definition sensitivity (sigma D_M/r_d | D_H/r_d at z=0.93, HMC model):")
for key in ('p', 'pb'):
    B = BUILT[key]; cells = []
    for npoly, kmin, kmax in [(2, .02, .30), (3, .02, .30), (5, .02, .30), (3, .03, .25), (3, .015, .35)]:
        RULER_W_OVERRIDE = make_ruler(npoly, kmin, kmax)
        cells.append(f"{sig_of(HS[key], to_nodes(geo_c(B, 'alpha_perp', Z_REF), B['o_a'], B['o_g'])):.4f}|"
                     f"{sig_of(HS[key], to_nodes(geo_c(B, 'alpha_par', Z_REF), B['o_a'], B['o_g'])):.4f}")
    RULER_W_OVERRIDE = None
    log(f"      {key:3s}: " + "  ".join(cells))

# --- figures -------------------------------------------------------------------------------
Tz5 = R['template_mpc']
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9.5), sharex=True, layout='constrained',
                               gridspec_kw={'height_ratios': [1.25, 1]})
for ax in (ax1, ax2):
    ax.axvspan(0.01 * h_fid, 0.20 * h_fid, color=C_MUTED, alpha=0.13, lw=0)
    ax.axvspan(0.02 * h_fid, 0.10 * h_fid, color=C_MUTED, alpha=0.10, lw=0)
ax1.fill_between(k_star, Tz5 * np.exp(-sig_h['pb']), Tz5 * np.exp(sig_h['pb']), color=C_PB, alpha=0.35, lw=0,
                 label=r'P(k)+B(k)  $\pm1\sigma$')
for sg_ in (-1, 1):
    ax1.plot(k_star, Tz5 * np.exp(sg_ * sig_h['p']), color=C_P, lw=1.8, label=r'P(k)  $\pm1\sigma$' if sg_ > 0 else None)
ax1.plot(k_star, Tz5, color=C_INK, lw=1.6, ls='--', label=r'truth ($\Lambda$CDM, $z_{\rm ref}=5$)')
ax1.set_xscale('log'); ax1.set_yscale('log')
ax1.set_ylabel(r'$P_{\rm lin}(k^*, z_{\rm ref}=5)\ [{\rm Mpc}^3]$', fontsize=13)
ax1.legend(frameon=False, fontsize=10, loc='lower center'); ax1.grid(alpha=0.2, which='both')
ax2.fill_between(k_star, np.exp(-sig_h['pb']), np.exp(sig_h['pb']), color=C_PB, alpha=0.35, lw=0,
                 label=r'P(k)+B(k)  $\pm1\sigma$ (Fisher)')
for sg_ in (-1, 1):
    ax2.plot(k_star, np.exp(sg_ * sig_h['p']), color=C_P, lw=1.8, label=r'P(k)  $\pm1\sigma$ (Fisher)' if sg_ > 0 else None)
    ax2.plot(k_star, [a16_ch, a84_ch][sg_ > 0], color=C_INK, lw=1.0, ls=(0, (1, 1.5)),
             label='P(k) HMC chain 68% (mock_pk_reconstruction)' if sg_ > 0 else None)
ax2.axhline(np.exp(HMC['s_lna']), color='#C5221F', lw=1, ls=':', label=r'prior $\pm1\sigma$')
ax2.axhline(np.exp(-HMC['s_lna']), color='#C5221F', lw=1, ls=':')
ax2.axhline(1.0, color=C_INK, lw=0.8, ls='--')
ax2.set_ylim(0.5, 2.1); ax2.set_ylabel(r'$a_i = P/P_{\rm template}$', fontsize=13)
ax2.set_xlabel(r'$k^*\ [1/{\rm Mpc}]$', fontsize=13)
ax2.legend(frameon=False, fontsize=9.5, loc='upper left', ncol=2); ax2.grid(alpha=0.2, which='both')
ax1.set_title(f"Linear P(k) recovery, P(k) alone vs P(k)+B(k): the HMC model ({N_NODES} nodes + priors), "
              f"everything marginalized", fontsize=12, color=C_INK)
p5 = os.path.join(OUT, 'band_recovery_pb.png')
fig.savefig(p5, dpi=170); plt.close(fig)
log(f"saved {p5}")

cols = [('alpha_perp', r'$D_M/r_d$  ($\alpha_\perp$)'), ('alpha_par', r'$D_H/r_d$  ($\alpha_\parallel$)'),
        ('F_AP', r'$F_{\rm AP}=D_M/D_H$'), ('f', r'$f(z)$')]
fig, axes = plt.subplots(2, 4, figsize=(20, 7.5), layout='constrained', sharex=True,
                         gridspec_kw={'height_ratios': [3, 1.3]})
for j, (q, title) in enumerate(cols):
    ax, axr = axes[0, j], axes[1, j]
    fx = {key: np.array([x if t not in ('gauge', 'ref') else np.nan for x, t in res[('fixed', False, key)][q]], float)
          for key in ('p', 'pb')}
    for key in ('p', 'pb'):
        ax.plot(zz, hgeo[(key, q)], color=COL[key], lw=2.2, marker='o', ms=4, label=f'{LAB[key]}, HMC model (template free)')
        ax.plot(zz, fx[key], color=COL[key], lw=1.2, ls='--', alpha=0.6, label=f'{LAB[key]}, true template inserted')
    ax.plot(zz, hgeo_prior[q], color=C_MUTED, lw=1.2, ls=':', label='priors alone')
    axr.plot(zz, hgeo[('pb', q)] / hgeo[('p', q)], color=C_INK, lw=2.2, marker='o', ms=4, label='HMC model')
    axr.plot(zz, fx['pb'] / fx['p'], color=C_INK, lw=1.2, ls='--', alpha=0.6, label='true template')
    ax.set_yscale('log'); ax.set_title(title, fontsize=14, color=C_INK); ax.grid(alpha=0.2, which='both')
    axr.axhline(1.0, color=C_MUTED, lw=1); axr.set_ylim(0, 1.15); axr.grid(alpha=0.2); axr.set_xlabel('z', fontsize=13)
axes[0, 0].set_ylabel(r'fractional $1\sigma$', fontsize=13)
axes[1, 0].set_ylabel(r'$\sigma_{P+B}/\sigma_P$', fontsize=13)
axes[0, 0].legend(frameon=False, fontsize=8.5, handlelength=3.2)
axes[1, 0].legend(frameon=False, fontsize=8.5, handlelength=3.2, loc='lower left')
fig.suptitle('Per-redshift BAO, AP and growth, P(k) alone vs P(k)+B(k): the HMC model (template free, 60 nodes + '
             'priors), everything marginalized', fontsize=13)
p6 = os.path.join(OUT, 'ap_summary_marg_pb.png')
fig.savefig(p6, dpi=170); plt.close(fig)
log(f"saved {p6}")
