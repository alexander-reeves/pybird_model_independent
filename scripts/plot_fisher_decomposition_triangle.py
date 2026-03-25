#!/usr/bin/env python3
"""
Triangle plot: cosmo Fisher decomposition for *interpretation*, not additivity.

Curves (all 3D cosmo, ω_cdm, ln10As, h):
  • Full / direct     — experiment (includes cross)
  • Q_pp only         — Jacobian projection of P(k) *diagonal* block F_pp
  • Q_gg only         — Jacobian projection of growth *diagonal* block F_gg
  • Q_pp + Q_gg       — diagonal blocks only in cosmo space (**cross Q_cross omitted**)

When the “no cross” contour differs from the full one, the **gap** is the effect of
Q_cross = J_pkᵀ F_pg J_gr + J_grᵀ F_gp J_pk (not drawable as its own Gaussian —
Q_cross is symmetric but usually indefinite). This is the visual you wanted.

Uses fisher_results.npz (numpy + matplotlib + getdist). No JAX required if Q_* saved.

Usage (from pybird_model_independent/):
  python scripts/plot_fisher_decomposition_triangle.py
  python scripts/plot_fisher_decomposition_triangle.py -o output/my_triangle.png
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_NPZ = os.path.join(PROJECT_ROOT, "output", "fisher_results.npz")
DEFAULT_OUT = os.path.join(PROJECT_ROOT, "output", "fisher_triangle_decomposition.png")


def sym(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    return 0.5 * (F + F.T)


def cov_from_fisher(F: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Inverse Fisher with minimal ridge if not PSD."""
    F = sym(F)
    w = np.linalg.eigvalsh(F)
    if w.min() <= 0:
        F = F + (abs(float(w.min())) + eps) * np.eye(F.shape[0])
    return np.linalg.inv(F)


def load_Q_blocks(z: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Q_pp, Q_gg, Q_cross, F_combined, cosmo_fid (3,)."""
    if "cosmo_fid_vec" not in z.files:
        raise KeyError("npz must contain cosmo_fid_vec")
    fid = np.asarray(z["cosmo_fid_vec"], dtype=float).ravel()
    if fid.size != 3:
        raise ValueError(f"cosmo_fid_vec must be length 3, got {fid.size}")

    if all(k in z.files for k in ("Q_pp", "Q_gg", "F_cosmo_cross", "F_cosmo_combined")):
        Q_pp = sym(z["Q_pp"])
        Q_gg = sym(z["Q_gg"])
        Qx = sym(z["F_cosmo_cross"])
        Fc = sym(z["F_cosmo_combined"])
        return Q_pp, Q_gg, Qx, Fc, fid

    if not all(k in z.files for k in ("F_phys_marg", "J", "F_pk_block")):
        raise KeyError("npz needs either Q_pp/Q_gg/F_cosmo_cross/F_cosmo_combined or F_phys_marg+J+F_pk_block")

    F = sym(z["F_phys_marg"])
    J = np.asarray(z["J"], dtype=float)
    n = int(z["F_pk_block"].shape[0])
    J_pk, J_gr = J[:n, :], J[n:, :]
    F_pp, F_pg = F[:n, :n], F[:n, n:]
    F_gp, F_gg = F[n:, :n], F[n:, n:]
    Q_pp = sym(J_pk.T @ F_pp @ J_pk)
    Q_gg = sym(J_gr.T @ F_gg @ J_gr)
    Qx = sym(J_pk.T @ F_pg @ J_gr + J_gr.T @ F_gp @ J_pk)
    Fc = sym(J.T @ F @ J)
    return Q_pp, Q_gg, Qx, Fc, fid


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("-o", "--out", default=DEFAULT_OUT)
    ap.add_argument("--n-samp", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    if not os.path.isfile(args.npz):
        print(f"ERROR: {args.npz} not found.")
        return 1

    z = np.load(args.npz, allow_pickle=True)
    Q_pp, Q_gg, Q_cross, F_comb, fid = load_Q_blocks(z)

    F_dir = sym(z["F_cosmo_direct"]) if "F_cosmo_direct" in z.files else F_comb
    F_no_cross = sym(Q_pp + Q_gg)

    # --- Numerical sanity (printed; cross matrix is not a precision for a Gaussian) ---
    w_x = np.linalg.eigvalsh(Q_cross)
    diff_alg = np.linalg.norm(F_comb - sym(Q_pp + Q_cross + Q_gg), ord="fro")
    nrm = np.linalg.norm(F_comb, ord="fro") + 1e-30
    diff_phys = np.linalg.norm(F_comb - F_no_cross, ord="fro")
    print("=== Fisher decomposition sanity ===")
    print(f"  ||F_comb - (Q_pp + Q_cross + Q_gg)||_F / ||F_comb||_F = {diff_alg / nrm:.3e}")
    print(f"  ||F_comb - (Q_pp + Q_gg)||_F / ||F_comb||_F          = {diff_phys / nrm:.3e}  (visual 'cross' gap)")
    print(f"  Q_cross eigenvalues (sym): [{w_x.min():+.3e}, {w_x.max():+.3e}]  (not a standalone Gaussian)")
    if "F_cosmo_direct" in z.files:
        print(f"  ||F_comb - F_direct||_F / ||F_comb||_F              = {np.linalg.norm(F_comb-F_dir,ord='fro')/nrm:.3e}")

    names_3d = ["omega_cdm", "ln10As", "h"]
    labels_3d = [r"\omega_{\rm cdm}", r"\ln(10^{10}A_s)", r"h"]

    # Covariances for sampling (same scale for all contours)
    C_full = cov_from_fisher(F_comb)
    C_dir = cov_from_fisher(F_dir)
    C_pp = cov_from_fisher(Q_pp)
    C_gg = cov_from_fisher(Q_gg)
    C_nox = cov_from_fisher(F_no_cross)

    rng = np.random.default_rng(args.seed)
    n = args.n_samp

    try:
        from getdist import MCSamples, plots
    except ImportError:
        print("ERROR: getdist required. pip install getdist")
        return 1

    def make_gd(label: str, C: np.ndarray):
        samps = rng.multivariate_normal(fid, C, n)
        return MCSamples(samples=samps, names=names_3d, labels=labels_3d, label=label)

    # Order: full first (filled), optional direct, sector blocks, no-cross (gap vs full = cross)
    curves = [
        (r"(c) Full $J^\top F J = Q_{pp}+Q_{\rm cross}+Q_{gg}$", C_full, True),
        (r"(a) $Q_{pp}$ only (P(k) block)", C_pp, False),
        (r"(b) $Q_{gg}$ only (growth block)", C_gg, False),
        (r"No cross: $Q_{pp}+Q_{gg}$", C_nox, False),
    ]
    if np.linalg.norm(F_comb - F_dir, ord="fro") / nrm > 1e-6:
        curves.insert(1, ("(d) Direct (EFT Schur)", C_dir, False))

    gd_list = [make_gd(lab, C) for lab, C, _ in curves]
    filled = [f for _, _, f in curves]
    ncur = len(gd_list)
    colors = ["#111111", "#2E7D32", "#1565C0", "#C62828", "#6A1B9A"][:ncur]
    linestyles = ["-", "-.", "-.", "--", (0, (3, 1, 1, 1))][:ncur]

    g = plots.get_subplot_plotter(width_inch=11)
    g.triangle_plot(
        gd_list,
        params=names_3d,
        filled=filled,
        contour_colors=colors,
        legend_labels=[lab for lab, _, _ in curves],
        contour_ls=linestyles,
    )
    plt.suptitle(
        r"Fisher split: $Q_{pp}$ / $Q_{gg}$ / full; no-cross vs full shows $Q_{\rm cross}$ effect "
        r"($Q_{\rm cross}$ is not its own Gaussian)",
        y=1.02,
        fontsize=11,
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
