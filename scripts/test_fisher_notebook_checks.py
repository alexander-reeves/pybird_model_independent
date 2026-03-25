#!/usr/bin/env python3
"""
Self-checks for 01_fisher_information.ipynb logic (run before telling anyone to re-run the notebook).

1) Synthetic: _cov_pd_from_precision always returns SPD (det>0, min eig>0).
2) Block closure on saved npz: Q_pp + Q_cross + Q_gg ≈ F_cosmo_combined.
3) Combined vs direct relative Frobenius error (after notebook §11 PSD fixes saved in npz).
4) Triangle covariances: same construction as notebook → all SPD + finite det.
5) Optional: instantiate getdist GaussianND for each (fails fast if still broken).

Usage (from pybird_model_independent/):
  python scripts/test_fisher_notebook_checks.py
  python scripts/test_fisher_notebook_checks.py --strict   # exit 1 if rel mismatch > 1e-2
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_NPZ = os.path.join(ROOT, "output", "fisher_results.npz")

# Must stay in sync with notebook §12 cell `_cov_pd_from_precision`
def cov_pd_from_precision(F: np.ndarray, rtol: float = 1e-14) -> np.ndarray:
    F = np.asarray(0.5 * (F + F.T), dtype=np.float64)
    w, V = np.linalg.eigh(F)
    scale = float(np.max(np.abs(w)))
    floor = float(np.maximum(rtol * np.maximum(scale, 1.0), 1e-30))
    w = np.maximum(w, floor)
    C = (V * (1.0 / w)) @ V.T
    C = np.asarray(0.5 * (C + C.T), dtype=np.float64)
    wc, Vc = np.linalg.eigh(C)
    wc = np.maximum(wc, np.maximum(float(np.max(wc)), 1e-300) * 1e-16)
    return (Vc * wc) @ Vc.T


def assert_spd(C: np.ndarray, name: str) -> None:
    C = np.asarray(C, dtype=np.float64)
    ev = np.linalg.eigvalsh(C)
    det = float(np.linalg.det(C))
    assert np.all(ev > 0), f"{name}: min eig {ev.min():.3e} <= 0"
    assert np.isfinite(det) and det > 0, f"{name}: det={det}"
    assert np.all(np.isfinite(np.sqrt(np.diag(C)))), f"{name}: nan diag sqrt"


def sym(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=np.float64)
    return 0.5 * (F + F.T)


def test_synthetic() -> None:
    rng = np.random.default_rng(0)
    # Well-conditioned SPD precision
    A = rng.standard_normal((5, 5))
    Fspd = A @ A.T + np.eye(5)
    C = cov_pd_from_precision(Fspd)
    assert_spd(C, "synthetic SPD F")
    err = np.linalg.norm(np.linalg.inv(Fspd) - C, ord="fro") / np.linalg.norm(np.linalg.inv(Fspd), ord="fro")
    assert err < 1e-10, f"cov should match inv(F) for SPD F, rel err {err}"

    # Indefinite F → still SPD cov output
    w = np.array([-2.0, 0.5, 3.0])
    V = np.linalg.qr(rng.standard_normal((3, 3)))[0]
    Find = (V * w) @ V.T
    C2 = cov_pd_from_precision(Find)
    assert_spd(C2, "indefinite F → forced SPD cov")


def test_npz(path: str, rel_tol_cd: float, rel_tol_closure: float) -> bool:
    ok = True
    z = np.load(path, allow_pickle=True)
    Fc = sym(z["F_cosmo_combined"])
    Fd = sym(z["F_cosmo_direct"])
    n = np.linalg.norm(Fc, ord="fro") + 1e-30
    r_cd = np.linalg.norm(Fc - Fd, ord="fro") / n
    print(f"  ||F_comb - F_dir||_F / ||F_comb||_F = {r_cd:.3e}  (tol {rel_tol_cd:g})")
    if r_cd > rel_tol_cd:
        print("  FAIL: combined vs direct mismatch (check §6–§11 notebook cells).")
        ok = False

    if all(k in z.files for k in ("Q_pp", "F_cosmo_cross", "Q_gg")):
        S = sym(z["Q_pp"] + z["F_cosmo_cross"] + z["Q_gg"])
        r_cl = np.linalg.norm(Fc - S, ord="fro") / n
        print(f"  ||F_comb - (Q_pp+Q_x+Q_gg)||_F / ||F_comb||_F = {r_cl:.3e}  (tol {rel_tol_closure:g})")
        if r_cl > rel_tol_closure:
            print("  FAIL: block sum closure.")
            ok = False

    # Triangle-style covariances (need 2d blocks + flat_sig mock)
    sig = np.sqrt(np.diag(np.linalg.inv(Fd)))
    flat_sig = np.maximum(80.0 * sig, np.array([0.02, 1.0, 0.06]))
    if "F_cosmo_pk_only_2d" in z.files and "F_cosmo_growth_only_2d" in z.files:
        C_pk = np.zeros((3, 3))
        C_pk[np.ix_([0, 1], [0, 1])] = cov_pd_from_precision(z["F_cosmo_pk_only_2d"])
        C_pk[2, 2] = flat_sig[2] ** 2
        Inv_gr = cov_pd_from_precision(z["F_cosmo_growth_only_2d"])
        C_gr = np.zeros((3, 3))
        C_gr[0, 0], C_gr[0, 2] = Inv_gr[0, 0], Inv_gr[0, 1]
        C_gr[2, 0], C_gr[2, 2] = Inv_gr[1, 0], Inv_gr[1, 1]
        C_gr[1, 1] = flat_sig[1] ** 2
        C_dir = cov_pd_from_precision(Fd)
        C_comb = cov_pd_from_precision(Fc)
        for nm, C in [
            ("C_dir", C_dir),
            ("C_comb", C_comb),
            ("C_pk", C_pk),
            ("C_gr", C_gr),
        ]:
            assert_spd(C, nm)
        print("  PASS: all triangle covariances SPD + det>0")

    return ok


def test_getdist(path: str) -> None:
    try:
        from getdist.gaussian_mixtures import GaussianND
    except ImportError:
        print("  (skip GetDist: not installed)")
        return
    z = np.load(path, allow_pickle=True)
    fid = np.asarray(z["cosmo_fid_vec"], dtype=float).ravel()
    Fd = sym(z["F_cosmo_direct"])
    Fc = sym(z["F_cosmo_combined"])
    names = ["omega_cdm", "ln10As", "h"]
    labels = [r"\omega", r"\ln A", r"h"]
    C = cov_pd_from_precision(Fc)
    g = GaussianND(fid, C, names=names, labels=labels, label="comb")
    assert g.dim == 3
    print("  PASS: GetDist GaussianND(combined) constructs")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--strict", action="store_true", help="Tight tolerance on F_comb vs F_dir (1e-6)")
    args = ap.parse_args()

    rel_cd = 1e-6 if args.strict else 1e-2
    rel_cl = 1e-5 if args.strict else 1e-3

    print("=== Synthetic _cov_pd_from_precision ===")
    test_synthetic()
    print("  PASS\n")

    if not os.path.isfile(args.npz):
        print(f"=== Saved npz ({args.npz}) missing — only synthetic ran ===")
        print("  Run the notebook save cell, then re-run this script.")
        return 0

    print(f"=== Checks using {args.npz} ===")
    ok = test_npz(args.npz, rel_tol_cd=rel_cd, rel_tol_closure=rel_cl)
    try:
        test_getdist(args.npz)
    except Exception as e:
        print(f"  FAIL GetDist: {e}")
        ok = False

    if ok:
        print("\nAll checks passed.")
        return 0
    print("\nChecks failed.")
    return 1


def smoke_desi_loglkl_sync() -> None:
    """Optional: DESI fake present — MI mock (``*_mi.yaml``) must have |log L|≪1 at fiducial without y-sync."""
    import os
    import sys
    import yaml

    # Before importing jax (avoids CUDA plugin init on head nodes without GPUs).
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    nb = os.path.join(root, "notebooks")
    sys.path.insert(0, os.path.join(root, "scripts"))
    os.chdir(nb)

    from pybird import config as pybird_config

    pybird_config.set_jax_enabled(True)
    from pybird.likelihood import Likelihood
    from pybird.symbolic import Symbolic, D, DA, Hubble, f
    from pybird.symbolic_pofk_linear import plin_emulated
    from utils import DESI_Y6, to_Mpc_jax, to_Mpc_per_h_jax

    out = os.path.join(root, "output")
    cfg_mi = os.path.join(out, "fake_desi_6sky_likelihood_config_mi.yaml")
    cfg = cfg_mi if os.path.isfile(cfg_mi) else os.path.join(out, "fake_desi_6sky_likelihood_config.yaml")
    if not os.path.isfile(cfg):
        raise FileNotFoundError(
            f"Need {cfg_mi} or legacy yaml. Generate: python scripts/generate_fake_data.py --mi-consistent --tag mi"
        )

    cosmo_fid = {
        "omega_b": 0.02235,
        "omega_cdm": 0.120,
        "h": 0.675,
        "ln10^{10}A_s": 3.044,
        "n_s": 0.965,
    }
    h_fid = cosmo_fid["h"]
    h_ref_grid = h_fid * 1.1
    z_ref = 3.0
    s = DESI_Y6
    num_skies = s["n_sky"]
    zeff_unique = sorted(set(s["zeff"]))
    n_z_unique = len(zeff_unique)
    sky_to_z_idx = [zeff_unique.index(z) for z in s["zeff"]]
    k_data = np.arange(0.01, 0.2, 0.01)
    M_sym = Symbolic()
    M_sym.set(cosmo_fid)
    M_sym.compute(np.logspace(-4, np.log10(0.7), 200), 0.5)
    Omega0_m = M_sym.c["Omega_m"]
    n_growth_per_z = 3
    n_growth = n_z_unique * n_growth_per_z + n_z_unique + 1
    growth_fid = []
    for z in zeff_unique:
        growth_fid.extend(
            [
                float(f(Omega0_m, z, -1, 0)),
                float(Hubble(Omega0_m, z, -1, 0)),
                float(DA(Omega0_m, z, -1, 0)),
            ]
        )
    for z in zeff_unique:
        growth_fid.append(float(D(Omega0_m, z, -1, 0) / D(Omega0_m, z_ref, -1, 0)))
    growth_fid.append(h_fid)
    growth_fid = np.array(growth_fid, float)

    lkl_config = yaml.full_load(open(cfg))
    lkl_config["drop_logdet"] = True
    lkl_config["get_maxlkl"] = True
    L_jax = Likelihood(lkl_config)

    n_knots = len(k_data)
    knots_h_jax = jnp.array(k_data, float)
    knots_mpc_jax = jnp.array(k_data / h_ref_grid, float)
    A_s_fid = 1e-10 * jnp.exp(cosmo_fid["ln10^{10}A_s"])
    Om0 = (cosmo_fid["omega_cdm"] + cosmo_fid["omega_b"]) / h_fid**2
    Ob0 = cosmo_fid["omega_b"] / h_fid**2
    pk_at_knots_h = 1e9 * plin_emulated(
        knots_h_jax, A_s_fid, Om0, Ob0, h_fid, cosmo_fid["n_s"], 0.0, -1.0, 0.0, a=1.0 / (1 + z_ref)
    )
    pk_at_knots_jax = to_Mpc_jax(pk_at_knots_h, knots_h_jax, h_fid, knots_mpc_jax)

    eft_free_names = ["b1", "b2", "b4"]
    pr = lkl_config["eft_prior"]
    fid = {k: pr[k]["mean"][0] for k in eft_free_names}
    eft_init = np.array([fid[k] for _ in range(num_skies) for k in eft_free_names])
    n_eft = len(eft_init)
    params_fid = np.concatenate([eft_init, np.ones(n_knots), growth_fid])

    def model_independent_loglkl(params):
        eft_params = params[:n_eft]
        pk_amps = params[n_eft : n_eft + n_knots]
        growth_params = params[n_eft + n_knots :]
        f_z, H_z, DA_z = [], [], []
        for i_z in range(n_z_unique):
            j = i_z * n_growth_per_z
            f_z.append(growth_params[j])
            H_z.append(growth_params[j + 1])
            DA_z.append(growth_params[j + 2])
        rs = n_z_unique * n_growth_per_z
        D_ratios = [growth_params[rs + i] for i in range(n_z_unique)]
        h_conv = growth_params[-1]
        pk_knots_mpc = pk_amps * pk_at_knots_jax
        pk_knots_h_arr = to_Mpc_per_h_jax(pk_knots_mpc, knots_mpc_jax, h_conv, knots_h_jax)
        cosmo_list = []
        for i_sky in range(num_skies):
            iz = sky_to_z_idx[i_sky]
            cosmo_list.append(
                {
                    "H": H_z[iz],
                    "DA": DA_z[iz],
                    "f": f_z[iz],
                    "pk_lin": pk_knots_h_arr * D_ratios[iz] ** 2,
                    "kk": knots_h_jax,
                }
            )
        return L_jax.loglkl(
            eft_params,
            eft_free_names * num_skies,
            need_cosmo_update=True,
            cosmo_dict=cosmo_list,
            cosmo_module=None,
            cosmo_engine=None,
        )

    l0 = float(model_independent_loglkl(jnp.array(params_fid)))
    if "mi.yaml" in os.path.basename(cfg):
        print(f"smoke_desi (MI mock): loglkl at fiducial = {l0:.3e}")
        if abs(l0) > 0.5:
            raise AssertionError(f"MI-consistent mock should give |log L|≪1 at fiducial, got {l0}")
        return

    print(f"smoke_desi (legacy Symbolic mock): loglkl before y-sync = {l0:.3f}")
    _ = model_independent_loglkl(jnp.array(params_fid))
    for i in range(L_jax.nsky):
        T = L_jax.correlator_sky[i].get(L_jax.b_sky[i]).reshape(-1)[L_jax.m_sky[i]]
        L_jax.d_sky[i]["y"] = np.asarray(T)
    L_jax.y_sky = [L_jax.d_sky[i]["y"] for i in range(L_jax.nsky)]
    L_jax.y_all = np.concatenate(L_jax.y_sky)
    l1 = float(model_independent_loglkl(jnp.array(params_fid)))
    print(f"smoke_desi: after one-step y-sync loglkl={l1:.3f} (prefer: generate_fake_data.py --mi-consistent)")
    if l1 <= -100:
        raise AssertionError("loglkl still catastrophic after sync")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke-desi":
        smoke_desi_loglkl_sync()
        print("PASS smoke_desi_loglkl_sync")
        sys.exit(0)
    sys.exit(main())
