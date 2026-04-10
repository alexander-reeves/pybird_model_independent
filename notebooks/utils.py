# Copyright (C) 2026 ETH Zurich, Institute for Particle Physics and Astrophysics
# Author: Alexander Reeves
"""
Utility functions for the DESI model-independent P(k) + growth reconstruction.
"""

import numpy as np
from pybird.jax_special import interp1d
from scipy.special import legendre
import jax.numpy as jnp
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ


# =============================================================================
# DESI Y6 survey specification
# 7 sky patches at 6 unique effective redshifts
# Sky 4 and 5 both target z_eff=0.930 (two ELG samples in the same shell)
# =============================================================================
DESI_Y6 = {
    'n_sky': 7,
    'zmin':  [0.1,   0.4,   0.6,  0.8,  0.8,  1.1,  0.8],
    'zmax':  [0.4,   0.6,   0.8,  1.1,  1.1,  1.6,  2.1],
    'zeff':  [0.295, 0.510, 0.706, 0.930, 0.930, 1.317, 1.491],
    'Veff':  np.array([4., 8., 12., 15., 8., 12., 4.]) * 1.e9,   # Mpc^3
    'degsq': [14000] * 7,
    'P0':    np.array([9.2, 8.9, 8.9, 8.4, 8.4, 2.9, 5.0]) * 1.e3,  # (Mpc/h)^3
    'nbar_prior': [3.e-4, 3.e-4, 3.e-4, 3.e-4, 3.e-4, 2.e-3, 1.e-4],
}


# =============================================================================
# Unit conversion: (Mpc/h)^3 <-> Mpc^3
# =============================================================================

def to_Mpc(pk_mpc_h, kk_mpc_h, h, kk_out_mpc=None):
    """Convert P(k) from (Mpc/h)^3 to Mpc^3 via log-space interpolation.

    Parameters
    ----------
    pk_mpc_h  : array_like  P(k) in (Mpc/h)^3
    kk_mpc_h  : array_like  k in h/Mpc
    h         : float       dimensionless Hubble constant
    kk_out_mpc: array_like or None  output k grid in 1/Mpc; if None uses kk_mpc_h/h
    """
    ilogpk = interp1d(np.log(kk_mpc_h), np.log(pk_mpc_h), fill_value='extrapolate')
    if kk_out_mpc is not None:
        return np.exp(ilogpk(np.log(kk_out_mpc / h))) / h**3
    else:
        return np.exp(ilogpk(np.log(kk_mpc_h / h))) / h**3


def to_Mpc_per_h(pk_mpc, kk_mpc, h, kk_out_mpc_h=None):
    """Convert P(k) from Mpc^3 to (Mpc/h)^3 via log-space interpolation.

    Parameters
    ----------
    pk_mpc      : array_like  P(k) in Mpc^3
    kk_mpc      : array_like  k in 1/Mpc
    h           : float       dimensionless Hubble constant
    kk_out_mpc_h: array_like or None  output k grid in h/Mpc; if None uses kk_mpc*h
    """
    ilogpk = interp1d(np.log(kk_mpc), np.log(pk_mpc), fill_value='extrapolate')
    if kk_out_mpc_h is not None:
        return np.exp(ilogpk(np.log(kk_out_mpc_h * h))) * h**3
    else:
        return np.exp(ilogpk(np.log(kk_mpc * h))) * h**3


def to_Mpc_jax(pk_mpc_h, kk_mpc_h, h, kk_out_mpc=None):
    """JAX-compatible version of to_Mpc (uses JAX-traceable interp1d from jax_special)."""
    ilogpk = interp1d(jnp.log(kk_mpc_h), jnp.log(pk_mpc_h), fill_value='extrapolate')
    if kk_out_mpc is not None:
        return jnp.exp(ilogpk(jnp.log(kk_out_mpc / h))) / h**3
    else:
        return jnp.exp(ilogpk(jnp.log(kk_mpc_h / h))) / h**3


def to_Mpc_per_h_jax(pk_mpc, kk_mpc, h, kk_out_mpc_h=None):
    """JAX-compatible version of to_Mpc_per_h (uses JAX-traceable interp1d from jax_special)."""
    ilogpk = interp1d(jnp.log(kk_mpc), jnp.log(pk_mpc), fill_value='extrapolate')
    if kk_out_mpc_h is not None:
        return jnp.exp(ilogpk(jnp.log(kk_out_mpc_h * h))) * h**3
    else:
        return jnp.exp(ilogpk(jnp.log(kk_mpc * h))) * h**3


# =============================================================================
# Gaussian shot-noise covariance for galaxy P(k) multipoles
# =============================================================================

def get_cov(kk, ipklin, b1, f1, Vs=3.e9, nbar=3.e-4):
    """Gaussian covariance matrix for P_ell P_ell'.

    Parameters
    ----------
    kk     : 1-D array  k-bins in h/Mpc
    ipklin : callable   interpolator for the linear P(k) in (Mpc/h)^3
    b1     : float      linear bias
    f1     : float      growth rate f
    Vs     : float      survey volume in (Mpc/h)^3
    nbar   : float      number density in (h/Mpc)^3

    Returns
    -------
    cov : (3*nk, 3*nk) array  block covariance [ell=0,2,4] x [ell=0,2,4]
    """
    dk = np.concatenate((kk[1:] - kk[:-1], [kk[-1] - kk[-2]]))
    Nmode = 4 * np.pi * kk**2 * dk * (Vs / (2 * np.pi)**3)
    mu_arr = np.linspace(0., 1., 200)
    k_mesh, mu_mesh = np.meshgrid(kk, mu_arr, indexing='ij')
    legendre_mesh = np.array([legendre(2 * l)(mu_mesh) for l in range(3)])
    legendre_ell_mesh = np.array([(2 * (2 * l) + 1) * legendre(2 * l)(mu_mesh) for l in range(3)])
    pkmu_mesh = (b1 + f1 * mu_mesh**2)**2 * ipklin(k_mesh)
    integrand = np.einsum('k,km,lkm,pkm->lpkm',
                          1. / Nmode,
                          (pkmu_mesh + 1 / nbar)**2,
                          legendre_ell_mesh,
                          legendre_ell_mesh)
    cov_diagonal = 2 * np.trapz(integrand, x=mu_arr, axis=-1)
    return np.block([[np.diag(cov_diagonal[i, j]) for i in range(3)] for j in range(3)])


