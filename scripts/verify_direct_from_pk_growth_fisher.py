#!/usr/bin/env python3
"""
Verify that the *direct* 3×3 cosmo Fisher (Hessian path + Schur over EFT) matches the
cosmo Fisher obtained by correctly combining P(k) and *growth* pieces of the *physical*
Fisher F_phys_marg (EFT already marginalised).

Algebra (same parameterisation: ω_cdm, ln10As, h):

  F_direct   ≈  F_combined  :=  Jᵀ F_phys_marg J
                            =  Q_pp + Q_cross + Q_gg

where J = ∂(pk_amps, growth_obs) / ∂(cosmo) at the fiducial,
      F_phys_marg = [[F_pp, F_pg], [F_gp, F_gg]]  in (P(k) sector, growth sector),

      Q_pp    = J_pkᵀ F_pp J_pk,
      Q_gg    = J_grᵀ F_gg J_gr,
      Q_cross = J_pkᵀ F_pg J_gr + J_grᵀ F_gp J_pk.

So **direct is not** J_pkᵀ F_pk_marginal J_pk + J_grᵀ F_growth_marginal J_gr (Schur *within*
phys between pk and growth), and **not** Q_pp + Q_gg (dropping cross). This script prints
all of those mismatches so you can see which “combination” actually closes.

**Marginalised-sector plots (what you asked):**  
In *data/phys* space, “P(k) with growth marginalised” is the Gaussian Schur
F_pk_marg = F_pp − F_pg F_gg⁺ F_gp; “growth with P(k) marginalised” is
F_gr_marg = F_gg − F_gp F_pp⁺ F_pg.  
Projecting with J gives 3×3 matrices Q_pk_marg := J_pkᵀ F_pk_marg J_pk and
Q_gr_marg := J_grᵀ F_gr_marg J_gr. Those are **legitimate** summaries of
“how tight is cosmo along each sector if the *other* sector is integrated out
in the quadratic approx”.  
**But** Q_pk_marg + Q_gr_marg ≠ Jᵀ F J in general: adding them **double-counts**
the coupling encoded in F_pg, F_gp. They **only** sum to the full cosmo Fisher
when F_pg ≈ 0 (sectors decouple in the Fisher). The **correct** additive split
that always recovers F_direct is Q_pp + Q_cross + Q_gg (full blocks, not Schur
within phys between pk and growth).

Inputs: output/fisher_results.npz from 01_fisher_information.ipynb (needs F_phys_marg, J,
F_cosmo_direct, F_pk_block for n_knots; optional F_cosmo_combined, Q_pp, … for cross-checks).

Usage (from pybird_model_independent/):
  python scripts/verify_direct_from_pk_growth_fisher.py
  python scripts/verify_direct_from_pk_growth_fisher.py --tol 1e-3
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_NPZ = os.path.join(PROJECT_ROOT, "output", "fisher_results.npz")


def sym(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    return 0.5 * (F + F.T)


def rel_fro(A: np.ndarray, B: np.ndarray) -> float:
    A, B = sym(A), sym(B)
    d = np.linalg.norm(A - B, ord="fro")
    n = np.linalg.norm(A, ord="fro") + 1e-30
    return float(d / n)


def block_reconstruct_cosmo(
    F_phys_marg: np.ndarray, J: np.ndarray, n_knots: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Q_pp, Q_cross, Q_gg, F_sum, F_J."""
    F = sym(F_phys_marg)
    J = np.asarray(J, dtype=float)
    n = int(n_knots)
    if F.shape[0] != J.shape[0] or J.shape[1] != 3:
        raise ValueError(f"shape mismatch F_phys_marg {F.shape}, J {J.shape}")
    J_pk = J[:n, :]
    J_gr = J[n:, :]
    F_pp = F[:n, :n]
    F_pg = F[:n, n:]
    F_gp = F[n:, :n]
    F_gg = F[n:, n:]
    Q_pp = J_pk.T @ F_pp @ J_pk
    Q_pg = J_pk.T @ F_pg @ J_gr
    Q_gp = J_gr.T @ F_gp @ J_pk
    Q_cross = Q_pg + Q_gp
    Q_gg = J_gr.T @ F_gg @ J_gr
    F_sum = sym(Q_pp + Q_cross + Q_gg)
    F_J = sym(J.T @ F @ J)
    return sym(Q_pp), sym(Q_cross), sym(Q_gg), F_sum, F_J


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=DEFAULT_NPZ, help="Path to fisher_results.npz")
    ap.add_argument("--tol", type=float, default=1e-3, help="Max allowed rel Fro error direct vs block sum")
    args = ap.parse_args()

    if not os.path.isfile(args.npz):
        print(f"ERROR: {args.npz} not found. Run the Fisher notebook save cell first.")
        return 1

    z = np.load(args.npz, allow_pickle=True)
    need = ("F_phys_marg", "J", "F_cosmo_direct", "F_pk_block")
    missing = [k for k in need if k not in z.files]
    if missing:
        print(f"ERROR: npz missing keys {missing}")
        return 1

    F_phys = z["F_phys_marg"]
    J = z["J"]
    F_dir = sym(z["F_cosmo_direct"])
    n_knots = int(z["F_pk_block"].shape[0])

    Q_pp, Q_cross, Q_gg, F_sum, F_J = block_reconstruct_cosmo(F_phys, J, n_knots)

    print("=== Reconstruct cosmo Fisher from P(k) + growth *blocks* of F_phys_marg ===\n")
    print(f"  n_knots = {n_knots},  n_growth_obs = {J.shape[0] - n_knots},  J shape = {J.shape}")

    r_J = rel_fro(F_J, F_sum)
    print(f"\n  (internal) || JᵀF_phys J − (Q_pp + Q_cross + Q_gg) ||_F / ||·||_F = {r_J:.3e}")
    if r_J > 1e-12:
        print("  WARN: block sum does not match JᵀF J (bug in script or asymmetric F).")

    r_direct_sum = rel_fro(F_dir, F_sum)
    r_direct_J = rel_fro(F_dir, F_J)
    print(f"\n  ** Key: || F_direct − (Q_pp + Q_cross + Q_gg) ||_F / ||F_direct||_F = {r_direct_sum:.3e}")
    print(f"           || F_direct − JᵀF_phys J           ||_F / ||F_direct||_F = {r_direct_J:.3e}")

    F_wrong = sym(Q_pp + Q_gg)
    r_drop_cross = rel_fro(F_dir, F_wrong)
    print(f"\n  (wrong)  || F_direct − (Q_pp + Q_gg only) ||_F / ||F_direct||_F = {r_drop_cross:.3e}")
    print("           → dropping Q_cross is *not* the direct Fisher unless Q_cross ≈ 0.")

    n = n_knots
    J_pk = J[:n, :]
    J_gr = J[n:, :]
    F = sym(F_phys)
    F_pp = F[:n, :n]
    F_pg = F[:n, n:]
    F_gp = F[n:, :n]
    F_gg = F[n:, n:]
    nf = np.linalg.norm(F, ord="fro") + 1e-30
    npk = np.linalg.norm(F_pp, ord="fro") + 1e-30
    ngr = np.linalg.norm(F_gg, ord="fro") + 1e-30
    ncpl = np.linalg.norm(F_pg, ord="fro")
    print("\n=== Coupling in F_phys_marg (pk–growth) ===")
    print(f"  ||F_pg||_F / ||F||_F = {ncpl / nf:.4f}  (→ 0 means sectors nearly decoupled in Fisher)")
    print(f"  ||F_pg||_F / ||F_pp||_F = {ncpl / npk:.4f},  ||F_pg||_F / ||F_gg||_F = {ncpl / ngr:.4f}")

    if "F_pk_marginal" in z.files and "F_growth_marginal" in z.files:
        F_pkm = sym(z["F_pk_marginal"])
        F_grm = sym(z["F_growth_marginal"])
        Q_pk_marg = sym(J_pk.T @ F_pkm @ J_pk)
        Q_gr_marg = sym(J_gr.T @ F_grm @ J_gr)
        sum_marg = sym(Q_pk_marg + Q_gr_marg)
        r_naive = rel_fro(F_dir, sum_marg)
        print("\n=== Schur-marginal sector plots (growth out for pk; pk out for growth) ===")
        print("  Q_pk_marg = J_pkᵀ F_pk_marg J_pk   [P(k) sector, growth marginalised in phys]")
        print("  Q_gr_marg = J_grᵀ F_gr_marg J_gr   [growth sector, P(k) marginalised in phys]")
        print(f"\n  || F_direct − (Q_pk_marg + Q_gr_marg) ||_F / ||F_direct||_F = {r_naive:.3e}")
        if r_naive <= args.tol:
            print("  → Sectors are effectively independent for Fisher purposes (F_pg small or special structure).")
        else:
            print("  → **Do not** add the two marginalised-sector Fishers to recover F_direct; coupling matters.")
        print(f"\n  Compare: ||Q_cross||_F / ||F_direct||_F = {np.linalg.norm(Q_cross, ord='fro') / (np.linalg.norm(F_dir, ord='fro')+1e-30):.4f}")

    if "F_cosmo_combined" in z.files:
        F_c = sym(z["F_cosmo_combined"])
        print(f"\n  (cross-check) || F_cosmo_combined − F_direct ||_F / ||F_direct||_F = {rel_fro(F_c, F_dir):.3e}")
        print(f"                 || F_cosmo_combined − block sum   ||_F / ||F_direct||_F = {rel_fro(F_c, F_sum):.3e}")

    nc = np.linalg.norm(F_dir, ord="fro") + 1e-30
    print(f"\n  ||Q_cross||_F / ||F_direct||_F = {np.linalg.norm(Q_cross, ord='fro') / nc:.4f}")

    w = np.linalg.eigvalsh(F_dir)
    print(f"\n  F_direct eigenvalues: [{w.min():.3e}, {w.max():.3e}]")

    ok = r_direct_sum <= args.tol and r_direct_J <= args.tol
    if ok:
        print(f"\nPASS: direct Fisher matches pk+growth block combination (with cross) within tol={args.tol:g}.")
        return 0
    print(f"\nFAIL: mismatch exceeds tol={args.tol:g}. Check F_phys_marg PSD, J consistency, or EFT Schur in direct path.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
