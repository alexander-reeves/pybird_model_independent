#!/usr/bin/env python
# Copyright (C) 2024 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Alexander Reeves
"""
Cosmology-independent P(k) reconstruction for DESI 6-sky mock data.

Samples P(k) knot amplitudes + EFT parameters using emcee, with growth
factors and expansion history fixed to the fiducial cosmology (CLASS or Symbolic).

Usage
-----
python scripts/cosmology_independent_analysis.py \
    --likelihood-config output/fake_desi_6sky_likelihood_config.yaml \
    --output-dir output/ \
    [--n-knots 30] [--n-walkers 0] [--n-burn 5000] [--n-steps 50000]
"""

import argparse
import os
import sys

import numpy as np
import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# GPU off for standalone CLI script; enable explicitly if needed
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from pybird import config as pybird_config
pybird_config.set_jax_enabled(True)
from pybird.likelihood import Likelihood
from pybird.symbolic import Symbolic

import emcee

from utils import DESI_Y6, to_Mpc, to_Mpc_per_h_jax


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Cosmology-independent P(k) reconstruction")
    p.add_argument(
        "--likelihood-config",
        default=os.path.join(_REPO_ROOT, "output", "fake_desi_6sky_likelihood_config.yaml"),
        help="Path to the PyBird likelihood YAML produced by generate_fake_data.py",
    )
    p.add_argument("--output-dir", default=os.path.join(_REPO_ROOT, "output"))
    p.add_argument("--n-knots", type=int, default=30,
                   help="Number of log-spaced P(k) knots")
    p.add_argument("--n-walkers", type=int, default=0,
                   help="emcee walkers (0 = 2*n_dim + 2)")
    p.add_argument("--n-burn", type=int, default=5000)
    p.add_argument("--n-steps", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Fiducial cosmology & survey spec ---------------------------------- #
    cosmo_fid = {
        "omega_b":       0.02235,
        "omega_cdm":     0.120,
        "h":             0.675,
        "ln10^{10}A_s":  3.044,
        "n_s":           0.965,
    }
    h_fid   = cosmo_fid["h"]
    s       = DESI_Y6
    num_sky = s["n_sky"]
    zeff    = s["zeff"]
    zeff_unique = sorted(set(zeff))
    sky_to_z_idx = [zeff_unique.index(z) for z in zeff]

    # --- P(k) knots -------------------------------------------------------- #
    n_knots = args.n_knots
    k_min, k_max = 0.0005, 0.35   # h/Mpc
    k_mid = 0.05
    k_low  = np.logspace(np.log10(k_min), np.log10(k_mid),
                         max(n_knots - n_knots // 3, 5), endpoint=False)
    k_high = np.logspace(np.log10(k_mid), np.log10(k_max), n_knots // 3)
    knots_h   = np.concatenate([k_low, k_high])
    knots_mpc = knots_h / h_fid
    knots_h_jax   = jnp.array(knots_h)
    knots_mpc_jax = jnp.array(knots_mpc)

    # Fiducial P(k) at knots (Mpc^3) at z_ref=3
    z_ref = 3.0
    M_sym = Symbolic()
    M_sym.set(cosmo_fid)
    M_sym.compute(knots_h, z_ref)
    pk_at_knots_h   = np.array(M_sym.pk_lin)
    pk_at_knots_mpc = to_Mpc(pk_at_knots_h, knots_h, h_fid, knots_mpc)  # Mpc^3
    pk_at_knots_jax = jnp.array(pk_at_knots_mpc)

    # Fixed growth factors at each unique z (truth)
    growth_truth = {}
    D_ratios_truth = {}
    D_ref_val = M_sym.D  # D at z_ref
    for z in zeff_unique:
        M_sym.compute(knots_h, z)
        growth_truth[z] = {
            "f":   float(M_sym.f),
            "H":   float(M_sym.H),
            "DA":  float(M_sym.DA),
        }
        D_ratios_truth[z] = float(M_sym.D) / float(D_ref_val)

    print(f"P(k) knots: {len(knots_h)} from k={k_min:.4f} to {k_max:.3f} h/Mpc")
    print(f"Unique z_eff: {zeff_unique}")

    # --- Likelihood -------------------------------------------------------- #
    lkl_config = yaml.full_load(open(args.likelihood_config))
    lkl_config["drop_logdet"] = True
    lkl_config["get_maxlkl"]  = True
    L_jax = Likelihood(lkl_config)

    # EFT free params: b1, b2, b4 per sky
    eft_names = ["b1", "b2", "b4"] * num_sky
    n_eft = len(eft_names)

    # Fiducial EFT from config prior means
    b1_fid = lkl_config["eft_prior"]["b1"]["mean"][0]
    b2_fid = lkl_config["eft_prior"]["b2"]["mean"][0]
    b4_fid = lkl_config["eft_prior"]["b4"]["mean"][0]
    eft_init = np.array([b1_fid, b2_fid, b4_fid] * num_sky)

    # --- Log-likelihood ---------------------------------------------------- #
    def loglkl(params):
        eft_params = params[:n_eft]
        pk_amps    = params[n_eft:]   # relative amplitudes (fiducial = 1)

        # Reconstruct P(k) in Mpc^3 then convert to (Mpc/h)^3
        pk_knots_mpc = pk_amps * pk_at_knots_jax
        pk_knots_h_arr = to_Mpc_per_h_jax(pk_knots_mpc, knots_mpc_jax, h_fid, knots_h_jax)

        cosmo_list = []
        for i_sky in range(num_sky):
            i_z = sky_to_z_idx[i_sky]
            z   = zeff_unique[i_z]
            D_ratio = D_ratios_truth[z]
            cosmo_list.append({
                "H":      growth_truth[z]["H"],
                "DA":     growth_truth[z]["DA"],
                "f":      growth_truth[z]["f"],
                "pk_lin": pk_knots_h_arr * D_ratio**2,
                "kk":     knots_h_jax,
            })

        return L_jax.loglkl(eft_params, eft_names,
                            need_cosmo_update=True,
                            cosmo_dict=cosmo_list,
                            cosmo_module=None,
                            cosmo_engine=None)

    loglkl_jit = jax.jit(loglkl)

    # --- Prior + log-prob -------------------------------------------------- #
    def log_prior(params):
        pk_amps = params[n_eft:]
        return jnp.where(jnp.all(pk_amps > 0), 0.0, -jnp.inf)

    def log_prob(params):
        lp = log_prior(params)
        return jnp.where(jnp.isfinite(lp), lp + loglkl_jit(params), -jnp.inf)

    log_prob_scalar = lambda theta: float(log_prob(jnp.array(theta)))

    # --- emcee setup ------------------------------------------------------- #
    n_dim = n_eft + n_knots
    n_walkers = args.n_walkers if args.n_walkers > 0 else n_dim * 2 + 2

    rng = np.random.default_rng(args.seed)
    params_fid = np.concatenate([eft_init, np.ones(n_knots)])
    init_pos = params_fid + 1e-3 * rng.standard_normal((n_walkers, n_dim))
    init_pos[:, n_eft:] = np.abs(init_pos[:, n_eft:])

    print(f"\nn_dim={n_dim}, n_walkers={n_walkers}, burn={args.n_burn}, steps={args.n_steps}")
    print("Running emcee...")

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob_scalar)
    # Burn-in
    state = sampler.run_mcmc(init_pos, args.n_burn, progress=True)
    sampler.reset()
    # Production
    sampler.run_mcmc(state, args.n_steps, progress=True)

    chain = sampler.get_chain(flat=True)
    out_path = os.path.join(args.output_dir, "cosmo_independent_chain.npz")
    np.savez(out_path,
             chain=chain,
             knots_h=knots_h,
             pk_at_knots_mpc=pk_at_knots_mpc,
             params_fid=params_fid,
             n_eft=n_eft,
             n_knots=n_knots)
    print(f"\nSaved chain ({chain.shape[0]} samples) to {out_path}")

    # Quick summary
    pk_amp_samples = chain[:, n_eft:]
    pk_med = np.median(pk_amp_samples, axis=0)
    sigma_amp = np.std(pk_amp_samples, axis=0)
    print(f"P(k) amplitude recovery (median): {pk_med[:5]}")
    print(f"P(k) amplitude sigma (first 5 knots): {sigma_amp[:5]}")


if __name__ == "__main__":
    main()
