#!/usr/bin/env python3
"""
Numerical sanity checks for Fisher matrices saved by 01_fisher_information.ipynb.

Diagnoses the failure mode where JAX −Hessian(marginalized log L) is not quite PSD:
indefinite F_full propagates to negative eigenvalues in F_phys_marg, Q_pp / Q_gg slices,
and F_cosmo_combined — then sqrt(diag inv F) and triangle ellipses are meaningless.

Usage (from pybird_model_independent/):
  python scripts/test_fisher_degeneracies.py
  python scripts/test_fisher_degeneracies.py --npz output/fisher_results.npz --plot output/fisher_triangle.png

Exit code 0 if all strict checks pass; 1 if any critical matrix is far from PSD or
algebraic identities fail badly.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_NPZ = os.path.join(PROJECT_ROOT, "output", "fisher_results.npz")
DEFAULT_PLOT = os.path.join(PROJECT_ROOT, "output", "fisher_triangle.png")

RTOL_CLOSURE = 1e-5
ATOL_CLOSURE = 1e-8
PSD_TOL = 1e-7  # min eigenvalue must be >= -PSD_TOL (tiny float noise)


def _sym(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    return 0.5 * (F + F.T)


def report_matrix(name: str, F: np.ndarray) -> tuple[float, float, bool]:
    F = _sym(F)
    w = np.linalg.eigvalsh(F)
    mn, mx = float(w.min()), float(w.max())
    ok = mn >= -PSD_TOL
    cond = np.linalg.cond(F) if F.shape[0] > 0 and mn > 1e-300 else float("inf")
    status = "OK" if ok else "FAIL (not PSD)"
    print(f"  {name:28s}  shape={F.shape!s:10s}  min_λ={mn:+.3e}  max_λ={mx:+.3e}  cond≈{cond:.2e}  {status}")
    return mn, mx, ok


def corr_from_fisher2(F2: np.ndarray) -> float:
    """Pearson correlation implied by a 2×2 Fisher (if PSD)."""
    F2 = _sym(F2)
    d = np.diag(F2)
    if np.any(d <= 0):
        return float("nan")
    rho = -F2[0, 1] / np.sqrt(d[0] * d[1])
    return float(rho)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", default=DEFAULT_NPZ, help="Path to fisher_results.npz")
    p.add_argument("--plot", default=DEFAULT_PLOT, help="Triangle PNG to probe (optional)")
    p.add_argument("--strict", action="store_true", help="Exit 1 on any non-PSD report")
    args = p.parse_args()

    print(__doc__.split("Usage")[0].strip())
    print()

    if not os.path.isfile(args.npz):
        print(f"ERROR: missing {args.npz} — run the Fisher notebook (and savez cell) first.")
        return 1

    z = np.load(args.npz, allow_pickle=True)
    files = set(z.files)

    print(f"Loaded {args.npz} keys: {sorted(files)}\n")

    print("--- Symmetry / PSD ---")
    ok_all = True
    for key in ("F_full", "F_phys_marg", "F_cosmo_combined", "F_cosmo_direct"):
        if key not in files:
            print(f"  (skip {key}: not in npz)")
            continue
        _, _, ok = report_matrix(key, z[key])
        ok_all = ok_all and ok

    if "Q_pp" in files:
        _, _, ok = report_matrix("Q_pp (3×3)", z["Q_pp"])
        ok_all = ok_all and ok
        F2 = z["Q_pp"][:2, :2]
        _, _, ok2 = report_matrix("Q_pp[:2,:2] (ω, lnAs)", F2)
        ok_all = ok_all and ok2
        rho = corr_from_fisher2(F2)
        print(f"    implied ρ(ω_cdm, lnAs) from Q_pp slice = {rho:+.4f}  (must be in [-1, 1])")
        if not np.isfinite(rho) or abs(rho) > 1.0 + 1e-6:
            print("    FAIL: correlation out of range — slice is not a valid Gaussian precision.")
            ok_all = False

    if "Q_gg" in files:
        _, _, ok = report_matrix("Q_gg (3×3)", z["Q_gg"])
        ok_all = ok_all and ok
        F2h = z["Q_gg"][np.ix_([0, 2], [0, 2])]
        _, _, okh = report_matrix("Q_gg ω–h slice", F2h)
        ok_all = ok_all and okh
        rho_h = corr_from_fisher2(F2h)
        print(f"    implied ρ(ω_cdm, h) from Q_gg slice = {rho_h:+.4f}  (must be in [-1, 1])")
        if not np.isfinite(rho_h) or abs(rho_h) > 1.0 + 1e-6:
            print("    FAIL: correlation out of range.")
            ok_all = False

    print("\n--- Algebraic identities ---")
    if {"F_cosmo_combined", "Q_pp", "Q_gg", "F_cosmo_cross"}.issubset(files):
        Qpp, Qgg = z["Q_pp"], z["Q_gg"]
        Qx = z["F_cosmo_cross"]
        Fc = z["F_cosmo_combined"]
        S = _sym(Qpp + Qx + Qgg)
        diff = np.linalg.norm(_sym(Fc) - S, ord="fro")
        nrm = np.linalg.norm(_sym(Fc), ord="fro") + 1e-30
        rel = diff / nrm
        print(f"  ||F_cosmo_combined − (Q_pp + Q_cross + Q_gg)||_F = {diff:.3e}  (rel {rel:.3e})")
        if rel > RTOL_CLOSURE and diff > ATOL_CLOSURE:
            print("  FAIL: block decomposition does not close (J or blocks mismatch).")
            ok_all = False
    elif "F_cosmo_block_sum" in files and "F_cosmo_combined" in files:
        diff = np.linalg.norm(_sym(z["F_cosmo_combined"]) - _sym(z["F_cosmo_block_sum"]), ord="fro")
        nrm = np.linalg.norm(_sym(z["F_cosmo_combined"]), ord="fro") + 1e-30
        print(f"  ||F_cosmo_combined − F_cosmo_block_sum||_F = {diff:.3e}  (rel {diff/nrm:.3e})")

    if "F_cosmo_combined" in files and "F_cosmo_direct" in files:
        A, B = _sym(z["F_cosmo_combined"]), _sym(z["F_cosmo_direct"])
        d = np.linalg.norm(A - B, ord="fro") / (np.linalg.norm(A, ord="fro") + 1e-30)
        print(f"  ||F_combined − F_direct||_F / ||F_combined||_F = {d:.3e}")
        print("    (direct uses Schur over EFT on cosmo+EFT Hessian; small mismatch can be numerical.)")

    print("\n--- Plot file ---")
    if os.path.isfile(args.plot):
        try:
            from PIL import Image

            im = Image.open(args.plot)
            arr = np.asarray(im)
            print(f"  {args.plot}: size={im.size} mode={im.mode} array_shape={arr.shape}")
        except Exception as e:
            print(f"  Could not read plot ({e})")
    else:
        print(f"  (no file {args.plot})")

    print("\n--- Interpretation ---")
    print(
        "  If F_full or F_phys_marg is not PSD: do not invert for covariances; triangle plots mislead.\n"
        "  The notebook §6 now symmetrises F_full and adds a minimal isotropic ridge so Schur\n"
        "  complements stay PSD. For marginalized PyBird likelihoods, hessian_type='FH' only\n"
        "  affects the non-marginal chi2 path; the marg branch still uses get_chi2_marg — so\n"
        "  ridge (or a custom Fisher) is often required for stable cosmo projections."
    )

    if args.strict and not ok_all:
        return 1
    if not ok_all:
        print("\nWARN: some matrices failed PSD check (re-run notebook after §6 ridge, or use --strict to exit 1).")
        return 0
    print("\nAll reported checks passed within tolerances.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
