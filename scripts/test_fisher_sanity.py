#!/usr/bin/env python
"""
Sanity checks for the model-independent Fisher information pipeline.

Tests:
  1. lkl = 0 at fiducial (fake data matches MI theory at truth)
  2. Finite-difference gradient nonzero for every pk_amp and growth param
  3. Hessian diagonal positive for all physical params
  4. Fisher decomposition: ||J^T F_phys J - F_direct|| / ||F_direct|| < tol
  5. J[pk_amps, h] ~ 0 (1/Mpc knots carry no h information)

Usage:
    python scripts/test_fisher_sanity.py
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "scripts"))

import numpy as np
import yaml
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

# Monkey-patch ShapedArray for CPJ pickle compat (JAX >=0.8 removed named_shape)
_orig_init = jax.core.ShapedArray.__init__
def _patched_init(self, *args, named_shape=None, **kwargs):
    return _orig_init(self, *args, **kwargs)
jax.core.ShapedArray.__init__ = _patched_init

from pybird import config as pybird_config
pybird_config.set_jax_enabled(True)
from pybird.module import *
from pybird.likelihood import Likelihood
from pybird.fake import Fake
from pybird.symbolic import Symbolic, f, Hubble, DA, D
from pybird.symbolic_pofk_linear import plin_emulated
from pybird.jax_special import interp1d as jax_interp1d
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ

from utils import DESI_Y6
from generate_fake_data import write_fake_desi_mi_consistent


# ============================================================================
# Setup
# ============================================================================

cosmo_fid = {
    'omega_b':      0.02235,
    'omega_cdm':    0.120,
    'h':            0.675,
    'ln10^{10}A_s': 3.044,
    'n_s':          0.965,
}
h_fid = cosmo_fid['h']
z_ref = 3.0

s = DESI_Y6
num_skies = s['n_sky']
zeff_list = s['zeff']
zeff_unique = sorted(set(zeff_list))
n_z_unique = len(zeff_unique)
sky_to_z_idx = [zeff_unique.index(z) for z in zeff_list]

M_sym = Symbolic()
M_sym.set(cosmo_fid)
M_sym.compute(np.logspace(-4, np.log10(0.7), 200), 0.5)
Omega0_m = M_sym.c['Omega_m']

# P(k) knots in 1/Mpc (h-independent)
KMIN_MPC, KMAX_MPC, DK_MPC = 0.01, 0.30, 0.01
knots_mpc = np.arange(KMIN_MPC, KMAX_MPC + 0.5 * DK_MPC, DK_MPC)
knots_mpc_jax = jnp.array(knots_mpc)
n_knots = len(knots_mpc)
k_data = knots_mpc * h_fid  # h/Mpc for Fake

# Fiducial P(k) in Mpc^3 from plin_emulated (EH) — matches Fake data
A_s_fid = 1e-10 * float(jnp.exp(cosmo_fid['ln10^{10}A_s']))
Om0 = (cosmo_fid['omega_cdm'] + cosmo_fid['omega_b']) / h_fid**2
Ob0 = cosmo_fid['omega_b'] / h_fid**2
knots_h_fid = knots_mpc_jax * h_fid
pk_h_fid = 1e9 * plin_emulated(
    knots_h_fid, A_s_fid, Om0, Ob0, h_fid, cosmo_fid['n_s'],
    0.0, -1.0, 0.0, a=1.0 / (1 + z_ref),
)
pk_at_knots_mpc_jax = pk_h_fid / h_fid**3  # used by model_independent_loglkl

# CPJ P(k) for the Jacobian (CLASS-based, nearly h-independent at z_ref=3)
M_cpj = CPJ(probe='mpk_lin')
cpj_modes = jnp.array(M_cpj.modes)

def cpj_pk_mpc_at_knots(omega_b, omega_cdm, n_s, lnAs, h, z):
    inp = {
        'omega_b': jnp.array([omega_b]), 'omega_cdm': jnp.array([omega_cdm]),
        'n_s': jnp.array([n_s]), 'ln10^{10}A_s': jnp.array([lnAs]),
        'h': jnp.array([h]), 'z': jnp.array([z]),
    }
    pk_cpj = M_cpj.predict(inp).flatten()
    ilogpk = jax_interp1d(jnp.log(cpj_modes), jnp.log(pk_cpj), fill_value='extrapolate')
    return jnp.exp(ilogpk(jnp.log(knots_mpc_jax)))

pk_at_knots_cpj = cpj_pk_mpc_at_knots(
    cosmo_fid['omega_b'], cosmo_fid['omega_cdm'], cosmo_fid['n_s'],
    cosmo_fid['ln10^{10}A_s'], h_fid, z_ref,
)

# Growth parameters at fiducial
n_growth_per_z = 3
n_growth = n_z_unique * n_growth_per_z + n_z_unique + 1

growth_fid_list = []
for z in zeff_unique:
    growth_fid_list.extend([
        float(f(Omega0_m, z, -1, 0)),
        float(Hubble(Omega0_m, z, -1, 0)),
        float(DA(Omega0_m, z, -1, 0)),
    ])
D_ref = float(D(Omega0_m, z_ref, -1, 0))
for z in zeff_unique:
    growth_fid_list.append(float(D(Omega0_m, z, -1, 0) / D_ref))
growth_fid_list.append(h_fid)
growth_fid = np.array(growth_fid_list)


# ============================================================================
# Generate fake data
# ============================================================================

OUTPUT_DIR = os.path.join(_REPO_ROOT, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

template_cfg = os.path.join(_REPO_ROOT, 'configs', 'fake_desi_6sky_config.yaml')
data_name = 'fake_desi_6sky_test'
config_name = 'fake_desi_6sky_test_likelihood_config'

config_path = os.path.join(OUTPUT_DIR, f'{config_name}.yaml')
h5_path = os.path.join(OUTPUT_DIR, f'{data_name}.h5')

for p in (config_path, h5_path):
    if os.path.isfile(p):
        os.remove(p)

print("Generating fake data...")
F = Fake(
    s['n_sky'], s['zmin'], s['zmax'], s['zeff'],
    s['Veff'], s['degsq'], s['P0'], cosmo_fid,
    likelihood_config_template_file=template_cfg,
    boltzmann='Symbolic', Omega_m_fid=Omega0_m,
    k_arr=k_data,
    nbar_prior=s['nbar_prior'],
    fake_data_filename=data_name,
    path_to_data=OUTPUT_DIR,
    fake_likelihood_config_filename=config_name,
    path_to_config=OUTPUT_DIR,
)
write_fake_desi_mi_consistent(F, cosmo_fid, knots_mpc=knots_mpc)
print("Fake data written.")

lkl_config = yaml.full_load(open(config_path))
lkl_config['drop_logdet'] = True
lkl_config['get_maxlkl'] = True
L_jax = Likelihood(lkl_config)
print("Likelihood initialised.")

# EFT params
eft_free_names = ['b1', 'b2', 'b4']
fiducial_nuisance = F.fiducial_nuisance[0]
eft_init = np.array([fiducial_nuisance[k] for _ in range(num_skies) for k in eft_free_names])
n_eft = len(eft_init)

# Full fiducial parameter vector
pk_amps_fid = np.ones(n_knots)
params_fid = np.concatenate([eft_init, pk_amps_fid, growth_fid])


# ============================================================================
# model_independent_loglkl (must match notebook Cell 11)
# ============================================================================

def model_independent_loglkl(params):
    eft_params = params[:n_eft]
    pk_amps = params[n_eft:n_eft + n_knots]
    growth_params = params[n_eft + n_knots:]

    f_z, H_z, DA_z = [], [], []
    for i_z in range(n_z_unique):
        idx = i_z * n_growth_per_z
        f_z.append(growth_params[idx])
        H_z.append(growth_params[idx + 1])
        DA_z.append(growth_params[idx + 2])

    ratio_start = n_z_unique * n_growth_per_z
    D_ratios = [growth_params[ratio_start + i] for i in range(n_z_unique)]
    h_conv = growth_params[-1]

    pk_mpc = pk_amps * pk_at_knots_mpc_jax
    pk_h = pk_mpc * h_conv**3
    kk_h = knots_mpc_jax * h_conv

    cosmo_list = []
    for i_sky in range(num_skies):
        i_z = sky_to_z_idx[i_sky]
        cosmo_list.append({
            'H': H_z[i_z],
            'DA': DA_z[i_z],
            'f': f_z[i_z],
            'pk_lin': pk_h * D_ratios[i_z]**2,
            'kk': kk_h,
        })

    return L_jax.loglkl(eft_params, eft_free_names * num_skies,
                        need_cosmo_update=True, cosmo_dict=cosmo_list,
                        cosmo_module=None, cosmo_engine=None)


# ============================================================================
# cosmo_to_observables (must match notebook Cell 7)
# ============================================================================

def cosmo_to_observables(cosmo_params):
    omega_cdm, lnAs, h = cosmo_params[0], cosmo_params[1], cosmo_params[2]
    omega_b = cosmo_fid['omega_b']
    n_s = cosmo_fid['n_s']
    w0, wa = -1.0, 0.0

    # P(k) at fixed 1/Mpc knots via CPJ (Mpc^3, nearly h-independent)
    pk_mpc = cpj_pk_mpc_at_knots(omega_b, omega_cdm, n_s, lnAs, h, z_ref)
    pk_amps = pk_mpc / pk_at_knots_cpj

    Omega_m = (omega_cdm + omega_b) / h**2
    growth_list, D_values = [], []
    for z in zeff_unique:
        growth_list.extend([
            f(Omega_m, z, w0, wa),
            Hubble(Omega_m, z, w0, wa),
            DA(Omega_m, z, w0, wa),
        ])
        D_values.append(D(Omega_m, z, w0, wa))
    D_ref_val = D(Omega_m, z_ref, w0, wa)
    for Dz in D_values:
        growth_list.append(Dz / D_ref_val)
    growth_list.append(h)
    return jnp.concatenate([pk_amps, jnp.array(growth_list)])


cosmo_fid_vec = jnp.array([cosmo_fid['omega_cdm'], cosmo_fid['ln10^{10}A_s'], cosmo_fid['h']])


# ============================================================================
# TEST 1: lkl = 0 at fiducial
# ============================================================================

print("\n" + "=" * 60)
print("TEST 1: Log-likelihood at fiducial")
print("=" * 60)

lkl_fid = float(model_independent_loglkl(jnp.array(params_fid)))
print(f"  lkl(fiducial) = {lkl_fid:.6e}")
assert abs(lkl_fid) < 1.0, f"FAIL: |lkl| = {abs(lkl_fid):.3e} > 1.0"
print("  PASS")


# ============================================================================
# TEST 2: Finite-difference gradient nonzero for every pk_amp and growth param
# ============================================================================

print("\n" + "=" * 60)
print("TEST 2: Finite-difference gradient (pk_amps + growth)")
print("=" * 60)

eps = 1e-5
idx_pk = np.arange(n_eft, n_eft + n_knots)
idx_growth = np.arange(n_eft + n_knots, len(params_fid))

zero_grad_params = []

for label, indices in [("pk_amp", idx_pk), ("growth", idx_growth)]:
    for i, idx in enumerate(indices):
        p_plus = np.array(params_fid, dtype=float)
        p_minus = np.array(params_fid, dtype=float)
        p_plus[idx] += eps
        p_minus[idx] -= eps
        grad = (float(model_independent_loglkl(jnp.array(p_plus)))
                - float(model_independent_loglkl(jnp.array(p_minus)))) / (2 * eps)
        if abs(grad) < 1e-10:
            zero_grad_params.append(f"{label}[{i}] (param idx {idx})")
            print(f"  WARNING: zero gradient for {label}[{i}] (idx={idx}), grad={grad:.3e}")
        else:
            print(f"  {label}[{i:2d}] (idx={idx:3d}): grad = {grad:+.6e}")

if zero_grad_params:
    print(f"\n  FAIL: {len(zero_grad_params)} parameters with zero gradient:")
    for p in zero_grad_params:
        print(f"    {p}")
    assert False, f"{len(zero_grad_params)} parameters have zero gradient"
else:
    print("  PASS: all gradients nonzero")


# ============================================================================
# TEST 3: Hessian diagonal positive for physical params
# ============================================================================

print("\n" + "=" * 60)
print("TEST 3: Hessian diagonal (physical params only)")
print("=" * 60)

print("  Computing Hessian (this may take a few minutes)...")
F_hess = np.asarray(
    -np.array(jax.hessian(model_independent_loglkl)(jnp.array(params_fid))),
    dtype=np.float64,
)
F_hess = 0.5 * (F_hess + F_hess.T)

# Add EFT priors
b2_sigma, b4_sigma = 5.0, 5.0
for i_sky in range(num_skies):
    idx_b2 = i_sky * len(eft_free_names) + eft_free_names.index("b2")
    idx_b4 = i_sky * len(eft_free_names) + eft_free_names.index("b4")
    F_hess[idx_b2, idx_b2] += 1.0 / b2_sigma**2
    F_hess[idx_b4, idx_b4] += 1.0 / b4_sigma**2

diag_pk = np.diag(F_hess)[n_eft:n_eft + n_knots]
diag_growth = np.diag(F_hess)[n_eft + n_knots:]

print(f"  pk_amp diag: min={diag_pk.min():.3e}, max={diag_pk.max():.3e}")
print(f"  growth diag: min={diag_growth.min():.3e}, max={diag_growth.max():.3e}")

n_nonpos_pk = np.sum(diag_pk <= 0)
n_nonpos_gr = np.sum(diag_growth <= 0)
if n_nonpos_pk > 0:
    print(f"  WARNING: {n_nonpos_pk} pk_amp diag entries <= 0")
if n_nonpos_gr > 0:
    print(f"  WARNING: {n_nonpos_gr} growth diag entries <= 0")

if n_nonpos_pk == 0 and n_nonpos_gr == 0:
    print("  PASS: all physical diag entries positive")
else:
    print("  FAIL")


# ============================================================================
# TEST 4: Fisher decomposition J^T F_phys J ~ F_direct
# ============================================================================

print("\n" + "=" * 60)
print("TEST 4: Fisher decomposition consistency")
print("=" * 60)

# Schur-complement: marginalise EFT
idx_eft_arr = np.arange(0, n_eft)
idx_phys_arr = np.concatenate([idx_pk, idx_growth])

F_eft_eft = F_hess[np.ix_(idx_eft_arr, idx_eft_arr)]
F_phys = F_hess[np.ix_(idx_phys_arr, idx_phys_arr)]
F_phys_eft = F_hess[np.ix_(idx_phys_arr, idx_eft_arr)]
F_phys_marg = F_phys - F_phys_eft @ np.linalg.inv(F_eft_eft) @ F_phys_eft.T
F_phys_marg = 0.5 * (F_phys_marg + F_phys_marg.T)

# Jacobian
J = np.array(jax.jacobian(cosmo_to_observables)(cosmo_fid_vec))
F_cosmo_combined = J.T @ F_phys_marg @ J

# Direct cosmo Fisher
def direct_cosmo_loglkl(cosmo_eft_params):
    cosmo_params = cosmo_eft_params[:3]
    eft_params = cosmo_eft_params[3:]
    obs = cosmo_to_observables(cosmo_params)
    pk_amps = obs[:n_knots]
    growth_p = obs[n_knots:]
    full_params = jnp.concatenate([eft_params, pk_amps, growth_p])
    return model_independent_loglkl(full_params)

print("  Computing direct cosmo Hessian...")
cosmo_eft_fid = jnp.concatenate([cosmo_fid_vec, jnp.array(eft_init)])
F_direct_full = np.asarray(
    -np.array(jax.hessian(direct_cosmo_loglkl)(cosmo_eft_fid)), dtype=np.float64
)
F_cosmo_direct = 0.5 * (F_direct_full[:3, :3] + F_direct_full[:3, :3].T)
F_eft_direct = 0.5 * (F_direct_full[3:, 3:] + F_direct_full[3:, 3:].T)
F_cross_direct = F_direct_full[:3, 3:]
F_cosmo_direct_marg = (
    F_cosmo_direct - F_cross_direct @ np.linalg.inv(F_eft_direct) @ F_cross_direct.T
)
F_cosmo_direct_marg = 0.5 * (F_cosmo_direct_marg + F_cosmo_direct_marg.T)

# Compare
norm_dir = np.linalg.norm(F_cosmo_direct_marg, ord='fro')
rel_err = np.linalg.norm(F_cosmo_combined - F_cosmo_direct_marg, ord='fro') / (norm_dir + 1e-30)
print(f"  ||J^T F_phys J - F_direct|| / ||F_direct|| = {rel_err:.3e}")

tol = 0.05
if rel_err < tol:
    print(f"  PASS (< {tol})")
else:
    print(f"  FAIL (> {tol})")

# Print sigma comparison
ev_comb = np.linalg.eigvalsh(F_cosmo_combined)
ev_dir = np.linalg.eigvalsh(F_cosmo_direct_marg)
if ev_comb.min() > 0 and ev_dir.min() > 0:
    s_comb = np.sqrt(np.diag(np.linalg.inv(F_cosmo_combined)))
    s_dir = np.sqrt(np.diag(np.linalg.inv(F_cosmo_direct_marg)))
    print(f"\n  sigma comparison (omega_cdm, ln10As, h):")
    print(f"    Combined: {s_comb}")
    print(f"    Direct:   {s_dir}")
    print(f"    Ratio:    {s_comb / s_dir}")


# ============================================================================
# TEST 5: J[pk_amps, h] ~ 0 (1/Mpc knots carry no h information)
# ============================================================================

print("\n" + "=" * 60)
print("TEST 5: J[pk_amps, h] ~ 0")
print("=" * 60)

J_pk_h = J[:n_knots, 2]  # column 2 = h
max_J_pk_h = float(np.abs(J_pk_h).max())
mean_J_pk_other = float(np.abs(J[:n_knots, :2]).mean())
print(f"  max|J[pk, h]| = {max_J_pk_h:.4e}")
print(f"  mean|J[pk, omega_cdm/lnAs]| = {mean_J_pk_other:.4e}")
ratio = max_J_pk_h / (mean_J_pk_other + 1e-30)
print(f"  ratio = {ratio:.4e}")
print(f"  With CPJ (CLASS-based), J[pk,h] ~ 0 as expected: P(k) in Mpc^3")
print(f"  at fixed k [1/Mpc] is nearly h-independent at z_ref={z_ref}.")
if ratio > 0.05:
    print(f"  WARNING: ratio > 0.05 — check P(k) emulator h-independence!")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"  n_knots = {n_knots}, n_eft = {n_eft}, n_growth = {n_growth}")
print(f"  Total params: {len(params_fid)}")
print(f"  lkl at fiducial: {lkl_fid:.6e}")
print(f"  Zero-gradient params: {len(zero_grad_params)}")
print(f"  Non-positive pk diag: {n_nonpos_pk}")
print(f"  Non-positive growth diag: {n_nonpos_gr}")
print(f"  Fisher decomposition rel err: {rel_err:.3e}")
print(f"  max|J[pk,h]| / mean|J[pk,other]|: {ratio:.3e}")

# Cleanup test files
for p in (config_path, h5_path):
    if os.path.isfile(p):
        os.remove(p)
print(f"\nCleaned up test files.")
print("Done.")
