#!/usr/bin/env python
"""
Physical constraints test suite for the model-independent Fisher decomposition.

Tests are organized in dependency order:
  Test 0 (foundational): loglkl at truth = 0 — data-model consistency
  Group 1: PSD of F_full and derived blocks
  Group 2: Information monotonicity — combined >= each component  [catches illogical behavior]
  Group 3: No h information in P(k) alone
  Group 4: Algebraic closure — combined matches direct
  Group 5: Marginalization reduces information

Usage:
    python scripts/test_fisher_physical_constraints.py
    python scripts/test_fisher_physical_constraints.py --npz output/fisher_decomposition_results.npz

Required NPZ keys (the notebook save cell must output all of these):
    F_full               (n_params, n_params) full Hessian
    F_phys_marg          (n_phys, n_phys) EFT-marginalized block
    F_pk_block           (n_knots, n_knots)
    F_pk_marginal        (n_knots, n_knots) Schur over growth
    F_growth_block       (n_growth, n_growth)
    F_growth_marginal    (n_growth, n_growth) Schur over P(k)
    F_cosmo_combined     (3,3)  projected combined [omega_cdm, ln10As, h]
    F_cosmo_direct_3d    (3,3)  direct from likelihood Hessian
    F_cosmo_from_pk_3d   (3,3)  J_pk.T @ F_pk_marginal @ J_pk
    F_cosmo_from_growth_only_2d  (2,2)  projected to [omega_cdm, h]
    Q_pp                 (3,3)  P(k) sector contribution
    Q_gg                 (3,3)  growth sector contribution
    F_cosmo_cross        (3,3)  cross term  (Q_pp + F_cosmo_cross + Q_gg = F_combined)
    J_pk                 (n_knots, 3)  d(pk_amps)/d(omega_cdm, ln10As, h)  [h-column must be zero]
    loglkl_at_truth      scalar  must be ~0
"""

import argparse
import sys
import numpy as np

PARAM_NAMES = ['omega_cdm', 'ln10As', 'h']

# ============================================================================
# Notebook fix: pos_pinv
# ============================================================================
# The bug in the notebook is using np.linalg.pinv(F_growth_block) and
# np.linalg.pinv(F_pk_block) for the Schur complement.
#
# pinv includes negative eigenvalues (from numerical noise in the Fisher blocks),
# so F_cross @ pinv(F_growth) @ F_cross.T can be NEGATIVE in some directions.
# The Schur complement then ADDS information instead of subtracting it, giving
# F_pk_marginal > F_pk_block — the root cause of the illogical P(k) > combined result.
#
# Fix in the notebook: replace np.linalg.pinv with pos_pinv defined as:
#
#   def pos_pinv(F, rtol=1e-10):
#       F = 0.5 * (F + F.T)
#       w, V = np.linalg.eigh(F)
#       scale = float(np.abs(w).max()) or 1.0
#       w_inv = np.where(w > rtol * scale, 1.0 / w, 0.0)
#       return (V * w_inv) @ V.T
#
# Then:
#   F_pk_marginal   = F_pk_block   - F_cross_block   @ pos_pinv(F_growth_block) @ F_cross_block.T
#   F_growth_marginal = F_growth_block - F_cross_block.T @ pos_pinv(F_pk_block)   @ F_cross_block

REQUIRED_KEYS = [
    'F_full', 'F_phys_marg',
    'F_pk_block', 'F_pk_marginal',
    'F_growth_block', 'F_growth_marginal',
    'F_cosmo_combined', 'F_cosmo_direct_3d', 'F_cosmo_from_pk_3d',
    'F_cosmo_from_growth_only_2d',
    'Q_pp', 'Q_gg', 'F_cosmo_cross',
    'J_pk',
    'loglkl_at_truth',
]

# ============================================================================
# Helpers
# ============================================================================

def sym(F):
    F = np.asarray(F, dtype=np.float64)
    return 0.5 * (F + F.T)


def is_psd(F, atol=1e-6):
    """(passed, min_eig, max_eig). Passed iff min_eig >= -atol * max|eig|.
    atol=1e-6 tolerates floating-point noise (~1e-14 of max eigenvalue) while
    catching genuine negative eigenvalues (e.g. pos_pinv bug gives ~-1e4)."""
    F = sym(F)
    w = np.linalg.eigvalsh(F)
    threshold = -atol * max(float(np.abs(w).max()), 1.0)
    return bool(np.all(w >= threshold)), float(w.min()), float(w.max())


def loewner_ge(A, B, atol=1e-6):
    """(passed, min_eig, max_eig). Passed iff A - B is PSD up to atol."""
    return is_psd(np.asarray(A, np.float64) - np.asarray(B, np.float64), atol=atol)


def sigma_from_fisher(F, idx=None):
    """
    Marginal sigma from Fisher via eigendecomposition (positive eigenspace only).
    Negative eigenvalues are excluded (physically: they are numerical noise).
    Near-zero / negative eigenvalues → unconstrained direction → sigma = inf.
    """
    F = sym(F)
    w, V = np.linalg.eigh(F)
    scale = max(float(np.abs(w).max()), 1.0)
    # Only invert strictly positive eigenvalues; zero/negative → variance = inf
    w_inv = np.where(w > 1e-14 * scale, 1.0 / w, 0.0)
    # C[i,i] = sum_k V[i,k]^2 / w_k  (only over positive eigenvalues)
    diag_C = (V ** 2) @ w_inv
    sigmas = np.sqrt(np.maximum(diag_C, 0.0))
    if idx is not None:
        return float(sigmas[idx])
    return sigmas

# ============================================================================
# Load NPZ
# ============================================================================

def load_data(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    missing = [k for k in REQUIRED_KEYS if k not in z.files]
    if missing:
        print('ERROR: NPZ is missing required keys:')
        for k in missing:
            print(f'  {k}')
        print('Update the notebook save cell to include all keys listed in this script\'s docstring.')
        sys.exit(2)
    return z

# ============================================================================
# Test runner
# ============================================================================

_results = []


def run_test(name, passed, details=''):
    status = 'PASS' if passed else 'FAIL'
    _results.append((name, passed))
    print(f'  [{status}] {name}')
    if details:
        for line in details.splitlines():
            print(f'         {line}')
    return passed

# ============================================================================
# Test 0: Foundational — data-model consistency
# ============================================================================

def test_loglkl_at_truth(z):
    print('\n=== TEST 0: Data-model consistency (loglkl at truth) ===')
    print('  Physical meaning: fake data must be generated with the exact same P(k)')
    print('  as the likelihood evaluates at the truth. Without this, F_full is not')
    print('  the Fisher information matrix and is not guaranteed to be PSD.')

    lkl = float(z['loglkl_at_truth'])
    tol = 0.5
    passed = abs(lkl) < tol
    details = f'loglkl(truth) = {lkl:.4e}  (tolerance |lkl| < {tol})'
    if not passed:
        details += (
            '\n=> Fix: regenerate fake data using exactly the same P(k) emulator,'
            '\n   k-grid, unit convention, and normalization as model_independent_loglkl.'
        )
    run_test('loglkl(truth) ~ 0', passed, details)

# ============================================================================
# Group 1: PSD
# ============================================================================

def test_group1_psd(z):
    print('\n=== GROUP 1: Positive semi-definiteness ===')
    print('  All Fisher matrices must have non-negative eigenvalues.')
    print('  F_full is PSD by construction iff Test 0 passes.')

    for name in ('F_full', 'F_phys_marg', 'F_pk_block', 'F_growth_block'):
        F = sym(z[name])
        ok, emin, emax = is_psd(F)
        run_test(
            f'{name} is PSD',
            ok,
            f'min_eig = {emin:.3e},  max_eig = {emax:.3e}',
        )

# ============================================================================
# Group 2: Information monotonicity
# ============================================================================

def test_group2_monotonicity(z):
    print('\n=== GROUP 2: Information monotonicity ===')
    print('  Adding data cannot decrease Fisher information (Loewner ordering).')
    print('  This directly tests the reported illogical result where P(k) alone')
    print('  appears to give tighter constraints than combined P(k)+growth.')

    F_comb = sym(z['F_cosmo_combined'])
    F_pk3d = sym(z['F_cosmo_from_pk_3d'])
    F_gr2d = sym(z['F_cosmo_from_growth_only_2d'])

    # 2a: combined >= P(k)-only in full 3D cosmo space
    ok2a, emin2a, _ = loewner_ge(F_comb, F_pk3d)
    run_test(
        'F_cosmo_combined >= F_cosmo_from_pk_3d  [all of omega_cdm, ln10As, h]',
        ok2a,
        f'min_eig(F_combined - F_pk_3d) = {emin2a:.3e}  (must be >= 0)'
        + ('\n=> P(k) alone is more constraining than combined — Jacobian or projection bug'
           if not ok2a else ''),
    )

    # 2b: combined (omega_cdm, h) subblock >= growth-only 2D
    # Indices [0, 2] correspond to omega_cdm and h in the 3D [omega_cdm, ln10As, h] space
    idx = np.ix_([0, 2], [0, 2])
    F_comb_sub = F_comb[idx]
    ok2b, emin2b, _ = loewner_ge(F_comb_sub, F_gr2d)
    run_test(
        'F_cosmo_combined[omega_cdm,h] >= F_cosmo_from_growth_only_2d',
        ok2b,
        f'min_eig(F_combined_sub - F_growth_2d) = {emin2b:.3e}  (must be >= 0)'
        + ('\n=> Growth alone outperforms combined in (omega_cdm, h) — cross-covariance sign error?'
           if not ok2b else ''),
    )

# ============================================================================
# Group 3: h information: growth dominates over P(k) alone
# ============================================================================

def test_group3_h_info(z):
    print('\n=== GROUP 3: h-independence of P(k) ===')
    print('  cosmo_to_pk evaluates at fixed k[1/Mpc] with fiducial h.')
    print('  J_pk[:, 2] must be zero (or machine-epsilon) by construction.')
    print('  All h information enters through the growth sector.')

    J_pk = np.asarray(z['J_pk'], dtype=np.float64)   # (n_knots, 3)

    # 3a: J_pk h-column should be zero (not merely small)
    h_col_rms  = float(np.sqrt(np.mean(J_pk[:, 2] ** 2)))
    other_rms  = float(np.sqrt(np.mean(J_pk[:, :2] ** 2)))
    ratio_j    = h_col_rms / (other_rms + 1e-30)
    run_test(
        'J_pk h-column is zero  [P(k) h-independent by construction, ratio < 1e-6]',
        ratio_j < 1e-6,
        f'rms|J[pk,h]| = {h_col_rms:.3e},  rms|J[pk,other]| = {other_rms:.3e},  ratio = {ratio_j:.3e}',
    )

    # 3b: sigma(h) from combined must be strictly tighter than from P(k) alone
    F_pk3d = sym(z['F_cosmo_from_pk_3d'])
    F_comb = sym(z['F_cosmo_combined'])
    F_dir  = sym(z['F_cosmo_direct_3d'])
    sigma_h_pk   = sigma_from_fisher(F_pk3d, idx=2)
    sigma_h_comb = sigma_from_fisher(F_comb, idx=2)
    sigma_h_dir  = sigma_from_fisher(F_dir,  idx=2)
    run_test(
        'sigma(h): combined << P(k) alone  [growth provides all h info]',
        sigma_h_comb < sigma_h_pk * 0.1,   # at least 10x tighter
        f'sigma_h: combined={sigma_h_comb:.4f}, pk_only={sigma_h_pk:.4f}, direct={sigma_h_dir:.4f}',
    )

# ============================================================================
# Group 4: Algebraic closure
# ============================================================================

def test_group4_closure(z):
    print('\n=== GROUP 4: Algebraic closure ===')

    F_comb = sym(z['F_cosmo_combined'])
    F_dir  = sym(z['F_cosmo_direct_3d'])
    Q_pp   = sym(z['Q_pp'])
    Q_gg   = sym(z['Q_gg'])
    Q_x    = sym(z['F_cosmo_cross'])

    # 4a: combined matches direct (end-to-end consistency)
    norm_dir = np.linalg.norm(F_dir, 'fro') + 1e-30
    rel_cd   = float(np.linalg.norm(F_comb - F_dir, 'fro') / norm_dir)
    run_test(
        '||F_combined - F_direct||_F / ||F_direct||_F < 0.05',
        rel_cd < 0.05,
        f'relative error = {rel_cd:.3e}',
    )

    # 4b: decomposition identity Q_pp + Q_cross + Q_gg = F_combined
    norm_comb = np.linalg.norm(F_comb, 'fro') + 1e-30
    S         = Q_pp + Q_x + Q_gg
    rel_cl    = float(np.linalg.norm(F_comb - S, 'fro') / norm_comb)
    run_test(
        '||Q_pp + Q_cross + Q_gg - F_combined||_F / ||F_combined||_F < 1e-6',
        rel_cl < 1e-6,
        f'relative error = {rel_cl:.3e}',
    )

# ============================================================================
# Group 5: Marginalization reduces information
# ============================================================================

def test_group5_marginalization(z):
    print('\n=== GROUP 5: Marginalization reduces information ===')
    print('  Schur complement: marginalizing over a block can only reduce information.')

    F_pk_blk  = sym(z['F_pk_block'])
    F_pk_marg = sym(z['F_pk_marginal'])
    ok5a, emin5a, _ = loewner_ge(F_pk_blk, F_pk_marg)
    run_test(
        'F_pk_block >= F_pk_marginal  (marginalizing over growth reduces P(k) info)',
        ok5a,
        f'min_eig(F_pk_block - F_pk_marginal) = {emin5a:.3e}',
    )

    F_gr_blk  = sym(z['F_growth_block'])
    F_gr_marg = sym(z['F_growth_marginal'])
    ok5b, emin5b, _ = loewner_ge(F_gr_blk, F_gr_marg)
    run_test(
        'F_growth_block >= F_growth_marginal  (marginalizing over P(k) reduces growth info)',
        ok5b,
        f'min_eig(F_growth_block - F_growth_marginal) = {emin5b:.3e}',
    )

# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        '--npz',
        default='output/fisher_decomposition_results.npz',
        help='Path to NPZ checkpoint from the Fisher decomposition notebook',
    )
    args = ap.parse_args()

    print(f'Loading {args.npz} ...')
    z = load_data(args.npz)

    test_loglkl_at_truth(z)
    test_group1_psd(z)
    test_group2_monotonicity(z)
    test_group3_h_info(z)
    test_group4_closure(z)
    test_group5_marginalization(z)

    n_total  = len(_results)
    n_passed = sum(1 for _, ok in _results if ok)
    n_failed = n_total - n_passed

    print(f'\n{"=" * 60}')
    print(f'SUMMARY: {n_passed}/{n_total} tests passed')
    if n_failed:
        print('FAILED:')
        for name, ok in _results:
            if not ok:
                print(f'  - {name}')

    test0_passed = _results[0][1]
    if not test0_passed:
        print('\nNOTE: Test 0 (data-model consistency) failed.')
        print('Fix fake data generation before interpreting other failures.')

    print('=' * 60)
    sys.exit(0 if n_failed == 0 else 1)


if __name__ == '__main__':
    main()
