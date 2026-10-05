"""Fisher-matrix linear algebra shared by the Fisher notebooks.

These are the v3.1 routines of `run_fisher_v3.py` (cell 17), lifted verbatim so that the
P(k)-only and the P(k)+B(k) notebooks marginalize and project with exactly one code path.
The reasoning behind each choice is documented at length in `run_fisher_v3.py`; in short:

  * an autodiff Hessian is PSD only to ~1e-7 of its largest eigenvalue, and every sector
    marginal is a Schur complement, i.e. DIVIDES by the block being marginalized, so every
    Fisher is PSD-clipped before it is marginalized or plotted and the pseudo-inverse
    threshold `RTOL_MARG` sits above the measured noise;
  * `pos_pinv` (zero weight for null directions) is used ONLY to invert nuisance blocks
    inside Schur complements;
  * `fisher_to_cov` floors unconstrained directions to a LARGE variance, never zero.
"""
import numpy as np

RTOL_MARG = 1e-8


def sym(M):
    """Symmetric part (an autodiff Hessian is symmetric only up to roundoff)."""
    return 0.5 * (M + M.T)


def noise_level(Fm):
    """Relative size of the numerical noise in a Hessian: the most negative eigenvalue and
    the antisymmetric part are two independent estimates of the same quantity."""
    Fs = sym(Fm)
    w = np.linalg.eigvalsh(Fs)
    return max(max(-w.min(), 0.0) / np.abs(w).max(),
               np.linalg.norm(Fm - Fs) / np.linalg.norm(Fs))


def psd_clip(Fm, rtol=0.0):
    """Symmetrize and clip eigenvalues below rtol*|max| to zero."""
    w, V = np.linalg.eigh(sym(Fm))
    w = np.where(w > rtol * np.abs(w).max(), w, 0.0)
    return (V * w) @ V.T


def pos_pinv(Fm, rtol=RTOL_MARG):
    """Pseudo-inverse keeping only the positive eigenvalues above rtol*|max|."""
    w, V = np.linalg.eigh(sym(Fm))
    good = w > rtol * (float(np.abs(w).max()) or 1.0)
    return (V * np.where(good, 1.0 / np.where(good, w, 1.0), 0.0)) @ V.T


def fisher_to_cov(Fm, rtol=1e-10, big_var=1e10):
    """Fisher -> covariance; tiny/negative eigenvalues are floored to a LARGE variance."""
    w, V = np.linalg.eigh(sym(Fm))
    good = w > rtol * (float(np.abs(w).max()) or 1.0)
    var = np.where(good, 1.0 / np.where(good, w, 1.0), big_var)
    return (V * var) @ V.T


def schur_marg(Fm, idx_keep, idx_marg, rtol=RTOL_MARG):
    """Marginalize idx_marg out of Fm by Schur complement, PSD-clipped in and out."""
    Fm = psd_clip(Fm)
    Fkk = Fm[np.ix_(idx_keep, idx_keep)]
    Fmm = Fm[np.ix_(idx_marg, idx_marg)]
    Fkm = Fm[np.ix_(idx_keep, idx_marg)]
    return psd_clip(Fkk - Fkm @ pos_pinv(Fmm, rtol) @ Fkm.T)


def sigmas(Fm, big_var=1.0):
    """Marginal 1-sigma widths; a value of O(sqrt(big_var)) means UNCONSTRAINED."""
    return np.sqrt(np.diag(fisher_to_cov(Fm, big_var=big_var)))


def sector_split(F_full, idx_eft, idx_pk, idx_growth, rtol=RTOL_MARG):
    """EFT-marginalized physical block of a full MI Fisher and its sector pieces.

    Returns a dict with F_phys_marg (ordered [amps, growth]), the three blocks A (template),
    G (growth/AP), C (cross), the two sector marginals A_marg (growth marginalized) and
    G_marg (template marginalized), and the pseudo-inverses used for the conditionals.
    """
    n_pk = len(idx_pk)
    idx_phys = np.concatenate([idx_pk, idx_growth])
    F_phys = schur_marg(F_full, idx_phys, idx_eft, rtol)
    A = F_phys[:n_pk, :n_pk]
    G = F_phys[n_pk:, n_pk:]
    C = F_phys[:n_pk, n_pk:]
    A_pinv, G_pinv = pos_pinv(A, rtol), pos_pinv(G, rtol)
    return dict(F_phys_marg=F_phys, A_blk=A, G_blk=G, C_blk=C, A_pinv=A_pinv, G_pinv=G_pinv,
                A_marg=psd_clip(A - C @ G_pinv @ C.T), G_marg=psd_clip(G - C.T @ A_pinv @ C))


def cosmo_pieces(split, J_pk, J_growth, F_direct_3):
    """The six 3x3 cosmology-space objects of the v3 information split, from one sector
    split and the Jacobians of the two mappings (J_pk: [n_pk x 3], J_growth: [n_growth x 3])."""
    A, G, C = split['A_blk'], split['G_blk'], split['C_blk']
    J_full = np.concatenate([J_pk, J_growth])
    Jg_corr = J_growth + split['G_pinv'] @ C.T @ J_pk
    Ja_corr = J_pk + split['A_pinv'] @ C @ J_growth
    return {
        'Direct': F_direct_3,
        'Combined': J_full.T @ split['F_phys_marg'] @ J_full,
        'P(k) marginal': psd_clip(J_pk.T @ split['A_marg'] @ J_pk),
        'Growth+AP marginal': psd_clip(J_growth.T @ split['G_marg'] @ J_growth),
        'Growth+AP | P(k)': psd_clip(Jg_corr.T @ G @ Jg_corr),
        'P(k) | Growth+AP': psd_clip(Ja_corr.T @ A @ Ja_corr),
    }


def fisher_to_cov_capped(Fm, sigma_max):
    """Fisher -> covariance with every direction's sigma capped at `sigma_max`.

    `fisher_to_cov` decides what counts as unconstrained by a RELATIVE eigenvalue threshold,
    which is the right criterion for an exactly-flat direction but not for a nearly-flat one:
    a growth block whose template has been marginalized has eigenvalues that are tiny in
    absolute terms and yet far above rtol * lambda_max, so the reported sigmas come out at
    1e4-1e5 instead of reading as "unconstrained". Here the criterion is absolute, which is
    meaningful because the block is parametrized in FRACTIONAL units: a direction measured
    worse than sigma_max is reported AT sigma_max, so a value at the cap means "no
    information", and correlations with genuinely measured directions are preserved.
    """
    w, V = np.linalg.eigh(sym(Fm))
    var = np.where(w > 1.0 / sigma_max**2, 1.0 / np.where(w > 0, w, 1.0), sigma_max**2)
    return (V * var) @ V.T


def report_combos(Fm, names, sigma_max, log=print, top=None):
    """Print the eigen-structure of a Fisher: which parameter combinations are measured, and
    how well. Directions worse than sigma_max are reported as flat."""
    w, V = np.linalg.eigh(sym(Fm))
    order = np.argsort(w)[::-1]
    shown = 0
    for i in order:
        combo = " ".join(f"{V[j, i]:+.2f}*{names[j]}" for j in range(len(names))
                         if abs(V[j, i]) > 0.15)
        if w[i] > 1.0 / sigma_max**2:
            log(f"   sigma({combo}) = {1/np.sqrt(w[i]):.3e}")
            shown += 1
            if top is not None and shown >= top:
                log(f"   ... ({len(w) - shown} further directions)")
                return
        else:
            log(f"   FLAT (sigma > {sigma_max:g}): {combo}")
            return
