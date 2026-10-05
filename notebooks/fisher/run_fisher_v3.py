"""Fisher information decomposition v3 — development script.

Sections map 1:1 to the cells of 01_fisher_v3.ipynb. Run inside the pybird
jax_env container (see submit instructions in the notebook header).

v3 design (fixes v1's inflated growth information):
  - P(k) parametrized on the EFT emulator's native 80 knots (interpolation-free,
    full required k-range [1e-4, 0.7] h/Mpc; v1's [0.005, 0.35] knots caused
    silent extrapolation of 31/80 emulator inputs -> chi2=8.23 at fiducial and
    an indefinite "Fisher").
  - Physical template: P(k) in Mpc^3 at z_ref=5 on fixed knots in 1/Mpc
    (h-independent there); a FREE unit-conversion parameter h_conv (mapped to h)
    carries the Mpc^3 -> (Mpc/h)^3 conversion in the growth/AP sector.
  - Data = the notebook's own direct model at fiducial (exact expansion point).
  - No hidden priors; pos_pinv only inside Schur complements; fisher_to_cov
    (large-variance floor) for every Fisher -> covariance conversion.
  - The final plot shows both exact chain orderings: p(P(k))p(growth|P(k))
    and p(growth)p(P(k)|growth), with the conditional terms distinguished.

v3.1 (2026-09-04) fixes two defects of v3.0, both of which corrupted the growth
sector specifically:
  1. CUSP. pybird's emulator interpolates its P_lin input onto the 80 knots with a
     piecewise-LINEAR interpolant, and our template sits exactly on those knots at
     the expansion point, so the log-likelihood curvature along h_conv is one-sided
     (~20% jump; mi_model.py carries the same note). Patched to cubic here too.
  2. NOISE INVERSION. The sector marginals are Schur complements, i.e. they DIVIDE
     by the block being marginalized. The autodiff Hessian is PSD only to ~1e-7 of
     its largest eigenvalue, and pos_pinv(rtol=1e-12) inverted that noise with
     weight ~1e7: v3.0's "growth marginal" had eigenvalues -134 and -1.2, and its
     (omega_cdm, h) projection -24.5, i.e. it was not a probability distribution at
     all. Every Fisher is now PSD-clipped before it is marginalized or plotted, and
     the pseudo-inverse threshold sits above the measured Hessian noise.

Physics that the fixed run makes explicit (Gate 5a): the MI physical block has two
degeneracies, so neither sector marginal can contain A_s or h --
  (i)  a * D(z)^2 invariant (EXACT)      -> lnAs flat in the P(k) marginal,
  (ii) h_conv <-> a rigid dilation of the free template (near-exact: only ~1e-5 of the
       h_conv information survives marginalizing the template, sigma 0.0019 -> 0.52)
       -> h unconstrained in the GROWTH marginal, which collapses to an Omega_m band.
All of the h is in the cross term, which is why growth|P(k) (green) carries it.
"""

# %% [cell 1] imports and setup ------------------------------------------------
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import time
from copy import deepcopy
from collections import defaultdict

import numpy as np
import matplotlib
try:
    get_ipython()  # in a notebook: keep the inline backend so figures embed
except NameError:
    matplotlib.use('Agg')  # script mode: headless
import matplotlib.pyplot as plt
import yaml
import h5py
import jax
import jax.numpy as jnp
from scipy.interpolate import interp1d as scipy_interp1d

jax.config.update("jax_enable_x64", True)

from pybird import config
config.set_jax_enabled(True)
from pybird.likelihood import Likelihood
from pybird.fake import Fake, get_cov_gauss
from pybird.symbolic import D as symD, f as symf, Hubble as symHubble, DA as symDA

# --- emulator input interpolation: cubic, not piecewise-linear ----------------
# Emulator.make_params interpolates log P_lin(kk) onto the 80 knots with a piecewise
# LINEAR interpolant. Our template sits exactly ON those knots at h_conv = h_fid, so an
# infinitesimal shift of the grid picks up a one-sided slope and the log-likelihood has a
# ~20% jump in curvature along h_conv AT the expansion point: jax.hessian then returns a
# one-sided second derivative for every h_conv entry of the Fisher. Cubic interpolation is
# C^2, identical at the nodes (chi2(fid) unchanged) and differs off-node only at
# O(dk^2 P''), the emulator's own training-interpolation error. Same monkeypatch as
# mi_model.py, so the Fisher and the sampled analysis share exactly one model.
from pybird import emulator as _pb_emu
from pybird.jax_special import interp1d as _pb_interp1d

def _make_params_smooth(self, kk, pk, f=1.0, time=False, ir=False, pca=False):
    pk_max = jnp.max(pk)
    self.pk_max = jnp.array(pk_max)
    ilogpk = _pb_interp1d(jnp.log(kk), jnp.log(pk / self.pk_max.reshape(-1, 1)),
                          axis=-1, kind='cubic')
    logpk = ilogpk(self.logknots)
    if ir:
        self.params = jnp.concatenate([logpk, jnp.array([[pk_max]]), jnp.array([[f]])], axis=1)
    else:
        self.params = logpk
    return logpk

_pb_emu.Emulator.make_params = _make_params_smooth

from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

rootdir = "../.."
output_path = os.path.join(rootdir, "output")
figdir = os.path.join(output_path, "fisher_v3")
os.makedirs(figdir, exist_ok=True)

# %% [cell 3] fiducial cosmology, survey, CPJ ----------------------------------
cosmo_fid = {
    'omega_b': 0.02235,
    'omega_cdm': 0.120,
    'h': 0.675,
    'ln10^{10}A_s': 3.044,
    'n_s': 0.965,
}
h_true = cosmo_fid['h']
Om_fid = (cosmo_fid['omega_cdm'] + cosmo_fid['omega_b']) / h_true**2
z_ref = 5.0  # P_lin(k [1/Mpc], z_ref) in Mpc^3 is ~h-independent here

DESI_Y6 = {
    'n_sky': 7,
    'zmin': [.1, .4, .6, .8, .8, 1.1, .8],
    'zmax': [.4, .6, .8, 1.1, 1.1, 1.6, 2.1],
    'zeff': [0.295, 0.51, 0.706, 0.930, 0.930, 1.317, 1.491],
    'Veff': np.array([4., 8., 12., 15., 8., 12., 4.]) * 1.e9,
    'degsq': [14000] * 7,
    'P0': np.array([9.2, 8.9, 8.9, 8.4, 8.4, 2.9, 5.]) * 1.e3,
    'nbar_prior': [3.e-4, 3.e-4, 3.e-4, 3.e-4, 3.e-4, 2.e-3, 1.e-4],
}
num_skies = DESI_Y6['n_sky']
zeff_list = DESI_Y6['zeff']
zeff_unique = sorted(set(zeff_list))
n_z_unique = len(zeff_unique)
sky_to_z_idx = [zeff_unique.index(z) for z in zeff_list]

cpj_pklin = CPJ(probe='mpk_lin')
cpj_k_modes = jnp.array(cpj_pklin.modes)  # 1/Mpc

def pklin_cpj(k_mpc, omega_b, omega_cdm, h, lnAs, n_s, z):
    """P_lin in Mpc^3 at k in 1/Mpc from CosmoPower-JAX (log-log interp onto k_mpc)."""
    input_dict = {
        'omega_b': jnp.atleast_1d(omega_b), 'omega_cdm': jnp.atleast_1d(omega_cdm),
        'h': jnp.atleast_1d(h), 'ln10^{10}A_s': jnp.atleast_1d(lnAs),
        'n_s': jnp.atleast_1d(n_s), 'z': jnp.atleast_1d(z),
    }
    pk = cpj_pklin.predict(input_dict)
    return jnp.exp(jnp.interp(jnp.log(k_mpc), jnp.log(cpj_k_modes), jnp.log(pk)))

log(f"CPJ ready; Om_fid={Om_fid:.4f}")

# %% [cell 5] knots = emulator's native 80 knots; physical template ------------
knots_h = np.load(os.path.join(rootdir, 'pybird', 'emu_data', 'knots.npy'))  # h/Mpc
n_knots = len(knots_h)
knots_mpc = jnp.array(knots_h * h_true)  # FIXED physical grid in 1/Mpc

template_mpc = pklin_cpj(knots_mpc, cosmo_fid['omega_b'], cosmo_fid['omega_cdm'],
                         h_true, cosmo_fid['ln10^{10}A_s'], cosmo_fid['n_s'], z_ref)
template_mpc = jnp.array(template_mpc)

# h-independence of the Mpc^3 template (the user's central physical point)
template_mpc_h110 = pklin_cpj(knots_mpc, cosmo_fid['omega_b'], cosmo_fid['omega_cdm'],
                              h_true * 1.1, cosmo_fid['ln10^{10}A_s'], cosmo_fid['n_s'], z_ref)
resid_h = np.abs(np.array(template_mpc_h110 / template_mpc) - 1)
log(f"{n_knots} emulator knots, k=[{knots_h.min():.1e}, {knots_h.max():.2f}] h/Mpc")
log(f"Template h-independence: max |dP/P| for 10% h shift = {resid_h.max():.2e} "
    f"(median {np.median(resid_h):.2e})")

# %% [cell 7] cosmology -> observable mappings ---------------------------------
def cosmo_to_amps(cosmo_params_3d):
    """[omega_cdm, lnAs, h] -> P(k) amplitude ratios at the FIXED physical knots (1/Mpc).
    The h column is ~0 by construction (P_lin in Mpc^3 at z_ref is h-free)."""
    omega_cdm, lnAs, h = cosmo_params_3d
    pk = pklin_cpj(knots_mpc, cosmo_fid['omega_b'], omega_cdm, h, lnAs,
                   cosmo_fid['n_s'], z_ref)
    return pk / template_mpc

def cosmo_to_growth(cosmo_params_2d):
    """[omega_cdm, h] -> [f, H/H0, DA*H0 per z, D(z)/D(z_ref) per z, h_conv].
    f/H/DA/D depend on h only via Omega_m; h_conv = h (identity) carries the
    Mpc^3 -> (Mpc/h)^3 unit conversion."""
    omega_cdm, h = cosmo_params_2d
    Om = (omega_cdm + cosmo_fid['omega_b']) / h**2
    out = []
    for z in zeff_unique:
        out.extend([symf(Om, z, -1., 0.), symHubble(Om, z, -1., 0.), symDA(Om, z, -1., 0.)])
    D_ref = symD(Om, z_ref, -1., 0.)
    for z in zeff_unique:
        out.append(symD(Om, z, -1., 0.) / D_ref)
    out.append(h)  # h_conv
    return jnp.array(out)

def cosmo_to_observables(cosmo_params_3d):
    return jnp.concatenate([cosmo_to_amps(cosmo_params_3d),
                            cosmo_to_growth(jnp.array([cosmo_params_3d[0], cosmo_params_3d[2]]))])

cosmo_fid_vec = jnp.array([cosmo_fid['omega_cdm'], cosmo_fid['ln10^{10}A_s'], cosmo_fid['h']])
cosmo_fid_2d = jnp.array([cosmo_fid['omega_cdm'], cosmo_fid['h']])

growth_fid = np.array(cosmo_to_growth(cosmo_fid_2d))
n_growth = len(growth_fid)  # 6*3 + 6 + 1 = 25
growth_names = ([n for z in zeff_unique for n in
                 (f'f(z={z:.2f})', f'H/H0(z={z:.2f})', f'DA*H0(z={z:.2f})')]
                + [f'D(z={z:.2f})/D(z_ref)' for z in zeff_unique] + ['h_conv'])

# %% [cell 9] shared model builder: (amps, growth) -> per-sky pybird inputs ----
def build_cosmo_dicts(amps, growth):
    """MI parameters -> list of pybird cosmo_dicts (one per sky).
    Template Mpc^3 on fixed 1/Mpc knots; conversion to the data's h-units via
    h_conv: kk = knots_mpc / h_conv [h/Mpc], pk = pk_mpc * h_conv^3 [(Mpc/h)^3]
    (same physical points -> no interpolation; exact emulator grid at fiducial)."""
    f_z = [growth[i * 3] for i in range(n_z_unique)]
    H_z = [growth[i * 3 + 1] for i in range(n_z_unique)]
    DA_z = [growth[i * 3 + 2] for i in range(n_z_unique)]
    D_ratios = [growth[3 * n_z_unique + i] for i in range(n_z_unique)]
    h_conv = growth[-1]

    kk_h = knots_mpc / h_conv
    pk_h = amps * template_mpc * h_conv**3

    dicts = []
    for i_sky in range(num_skies):
        i_z = sky_to_z_idx[i_sky]
        dicts.append({
            "H": H_z[i_z], "DA": DA_z[i_z], "f": f_z[i_z],
            "kk": kk_h, "pk_lin": pk_h * D_ratios[i_z]**2,
        })
    return dicts

def build_direct_cosmo_dicts(cosmo_params_3d):
    """Direct model = MI model composed with the cosmology mapping (exactly)."""
    amps = cosmo_to_amps(cosmo_params_3d)
    growth = cosmo_to_growth(jnp.array([cosmo_params_3d[0], cosmo_params_3d[2]]))
    return build_cosmo_dicts(amps, growth)

# %% [cell 11] fake DESI Y6 data = direct model at fiducial --------------------
# The data are generated by pybird's internal CPJ path, configured so that it is
# EXACTLY the notebook's direct model: `cpj_pk_on_knots` evaluates P_lin on the 80
# emulator knots (P_h(k) = P_Mpc(k h) h^3, linear log-log interpolation from the
# CosmoPower grid, as in cosmo_to_amps) and `cpj_z_ref` evaluates CosmoPower at z_ref
# and rescales with the symbolic growth factor D(z)^2/D(z_ref)^2 (as in build_cosmo_dicts).
# Growth/AP (f, H, D_A) come from the same symbolic functions in both routes. Verified in
# robustness/fake_cpj_options_test.py: chi2(internal - explicit direct model) = 4e-9.
template_configfile = os.path.join(rootdir, 'data', 'fake', 'likelihood_config',
                                   'fake_fast_pk_desi_y6.yaml')
fake_data_name = 'fake_desi_y6_fisher_v3'
fake_config_name = 'fake_desi_y6_fisher_v3_config'

lkl_cfg = yaml.full_load(open(template_configfile))
lkl_cfg['cpj_pk_on_knots'] = True
lkl_cfg['cpj_z_ref'] = float(z_ref)

s = DESI_Y6
F = Fake(s['n_sky'], s['zmin'], s['zmax'], s['zeff'], s['Veff'], s['degsq'], s['P0'],
         cosmo_fid, likelihood_config=lkl_cfg,
         boltzmann='CPJ', Omega_m_fid=Om_fid, kmin=0.01, kmax=0.2, dk=0.01,
         nbar_prior=s['nbar_prior'],
         fake_data_filename=fake_data_name, path_to_data=output_path,
         fake_likelihood_config_filename=fake_config_name, path_to_config=output_path)
fiducial_nuisance = F.fiducial_nuisance[0]
log(f"Fake initialized. fiducial EFT: {fiducial_nuisance}")
F.set()  # writes the data (Gaussian covariance, EFT priors centred on the truth) and the config
log("Fake data (internal CPJ path = direct model at fiducial) written.")

# %% [cell 13] likelihood + the two model functions ----------------------------
lkl_config = yaml.full_load(open(os.path.join(output_path, f'{fake_config_name}.yaml')))
lkl_config['get_maxlkl'] = True
L_jax = Likelihood(lkl_config)

eft_free_names = ["b1", "b2", "b4"]
eft_init = np.array([fiducial_nuisance[k] for _ in range(num_skies) for k in eft_free_names])
n_eft = len(eft_init)
eft_names_flat = eft_free_names * num_skies

def model_independent_loglkl(params):
    """EFT (21) + P(k) amps at 80 physical knots + growth (24 + h_conv). NO priors."""
    eft = params[:n_eft]
    amps = params[n_eft:n_eft + n_knots]
    growth = params[n_eft + n_knots:]
    return L_jax.loglkl(eft, eft_names_flat, need_cosmo_update=True,
                        cosmo_dict=build_cosmo_dicts(amps, growth),
                        cosmo_module=None, cosmo_engine=None)

def direct_cosmo_loglkl(params):
    """EFT (21) + [omega_cdm, lnAs, h]; factors exactly through the MI model."""
    eft = params[:n_eft]
    return L_jax.loglkl(eft, eft_names_flat, need_cosmo_update=True,
                        cosmo_dict=build_direct_cosmo_dicts(params[n_eft:]),
                        cosmo_module=None, cosmo_engine=None)

params_fid = np.concatenate([eft_init, np.ones(n_knots), growth_fid])
cosmo_eft_fid = jnp.concatenate([jnp.array(eft_init), cosmo_fid_vec])

# ===== Gate 1: both likelihoods reproduce the data at fiducial =====
chi2_direct = -2 * float(direct_cosmo_loglkl(cosmo_eft_fid))
chi2_mi = -2 * float(model_independent_loglkl(jnp.array(params_fid)))
log(f"[Gate 1] chi2(fid): direct = {chi2_direct:.3e}, model-independent = {chi2_mi:.3e} "
    f"(v1: 0.093 / 8.231)")
log("[Gate 1] " + ("PASS" if max(abs(chi2_direct), abs(chi2_mi)) < 1e-6 else "FAIL"))

# %% [cell 15] Fisher matrices -------------------------------------------------
log("Computing model-independent Fisher (126 params)...")
F_full = -np.array(jax.hessian(model_independent_loglkl)(jnp.array(params_fid)))
log(f"F_full done, shape {F_full.shape}")

log("Computing direct Fisher (24 params)...")
F_direct_full = -np.array(jax.hessian(direct_cosmo_loglkl)(cosmo_eft_fid))
log("F_direct_full done")

idx_eft = np.arange(n_eft)
idx_pk = np.arange(n_eft, n_eft + n_knots)
idx_growth = np.arange(n_eft + n_knots, len(params_fid))

# --- Gauss-Newton reference: J^T P J of the profiled model vector -------------
# Exactly PSD by construction, and equal to the Hessian above at a zero-residual
# expansion point up to the EFT-prior and analytic-marginalization log-det terms
# (which the Hessian has and this does not). Used only to MEASURE how much of the
# Hessian's non-PSD-ness is numerical noise (Gate 2b); the science numbers below all
# come from the Hessian, as before.
def mi_model_vector(params):
    model_independent_loglkl(params)  # side effect: fills correlator_sky / b_sky
    return jnp.concatenate([L_jax.correlator_sky[i].get(L_jax.b_sky[i]).reshape(-1)[L_jax.m_sky[i]]
                            for i in range(L_jax.nsky)])

log("Computing Gauss-Newton reference Fisher...")
_J_gn = np.array(jax.jacfwd(mi_model_vector)(jnp.array(params_fid)))
F_full_gn = _J_gn.T @ np.array(L_jax.p_all) @ _J_gn
F_full_gn = 0.5 * (F_full_gn + F_full_gn.T)
log("F_full_gn done")

# %% [cell 17] Fisher arithmetic utilities -------------------------------------
def sym(M):
    """Symmetric part (an autodiff Hessian is symmetric only up to roundoff)."""
    return 0.5 * (M + M.T)

def noise_level(Fm):
    """Relative size of the numerical noise in a Hessian. The most negative eigenvalue
    and the antisymmetric part are two independent estimates of the same quantity."""
    Fs = sym(Fm)
    w = np.linalg.eigvalsh(Fs)
    return max(max(-w.min(), 0.0) / np.abs(w).max(),
               np.linalg.norm(Fm - Fs) / np.linalg.norm(Fs))

def psd_clip(Fm, rtol=0.0):
    """Symmetrize and clip eigenvalues below rtol*|max| to zero.

    A Fisher matrix is PSD by construction, so the negative eigenvalues of an autodiff
    Hessian are pure noise. Leaving them in is what broke v3.0: every sector marginal is
    a Schur complement, which DIVIDES by the block being marginalized, so a noise
    eigenvalue at 1e-7 of the maximum was inverted with weight 1e7. The resulting
    "growth marginal" had eigenvalues -134 and -1.2 and its (omega_cdm, h) projection
    -24.5 -- not a probability distribution. Everything is PSD-clipped here instead."""
    w, V = np.linalg.eigh(sym(Fm))
    w = np.where(w > rtol * np.abs(w).max(), w, 0.0)
    return (V * w) @ V.T

# Pseudo-inverse threshold for every marginalization: above the measured Hessian noise
# (~1e-7 relative, Gate 2 below) and far below the smallest physically meaningful
# eigenvalue. Gate 6 shows the answers are flat in it over four decades.
RTOL_MARG = 1e-8

def pos_pinv(Fm, rtol=RTOL_MARG):
    """Pseudo-inverse keeping only positive eigenvalues. Use ONLY for inverting nuisance
    blocks inside Schur complements: a direction the data do not constrain gets zero
    weight and so correctly contributes nothing to the marginalization."""
    w, V = np.linalg.eigh(sym(Fm))
    good = w > rtol * (float(np.abs(w).max()) or 1.0)
    return (V * np.where(good, 1.0 / np.where(good, w, 1.0), 0.0)) @ V.T

def fisher_to_cov(Fm, rtol=1e-10, big_var=1e10):
    """Fisher -> covariance. Tiny/negative eigenvalues (unconstrained directions) are
    floored to a LARGE variance, never zero (v1 used pos_pinv here, which collapsed
    unconstrained directions to delta functions -> spuriously tight contours)."""
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

log(f"Hessian noise level: F_full = {noise_level(F_full):.2e}, "
    f"F_direct_full = {noise_level(F_direct_full):.2e}; pinv threshold {RTOL_MARG:.0e}")

# ===== Gate 2b: how much of that is numerical? compare with Gauss-Newton =====
_sc = np.linalg.norm(psd_clip(F_full)) / np.linalg.norm(F_full_gn)
log(f"[Gate 2b] Gauss-Newton reference: min_eig/|max_eig| = "
    f"{np.linalg.eigvalsh(F_full_gn).min()/np.abs(np.linalg.eigvalsh(F_full_gn)).max():+.2e} "
    f"(PSD by construction); ||F_hess - F_GN||/||F_hess|| = "
    f"{np.linalg.norm(psd_clip(F_full) - _sc*F_full_gn)/np.linalg.norm(F_full):.2e} "
    f"(EFT priors + marg log-det are in the Hessian only)")

# EFT-marginalized physical-sector Fisher and its three blocks. The whole information
# split is built out of these, so that BOTH chain orderings are exactly additive.
idx_phys = np.concatenate([idx_pk, idx_growth])
F_phys_marg = schur_marg(F_full, idx_phys, idx_eft)
A_blk = F_phys_marg[:n_knots, :n_knots]      # template amplitudes
G_blk = F_phys_marg[n_knots:, n_knots:]      # growth/AP + h_conv
C_blk = F_phys_marg[:n_knots, n_knots:]      # cross-sector
A_pinv, G_pinv = pos_pinv(A_blk), pos_pinv(G_blk)
A_marg = psd_clip(A_blk - C_blk @ G_pinv @ C_blk.T)   # template, growth marginalized
G_marg = psd_clip(G_blk - C_blk.T @ A_pinv @ C_blk)   # growth, template marginalized
F_pk_marginal, F_growth_marginal = A_marg, G_marg

# cross-check: marginalizing {EFT, other sector} out of F_full in one step must agree
for nm, blk, idx_k, idx_m in [('pk', A_marg, idx_pk, np.concatenate([idx_eft, idx_growth])),
                              ('growth', G_marg, idx_growth, np.concatenate([idx_eft, idx_pk]))]:
    one_step = schur_marg(F_full, idx_k, idx_m)
    log(f"[check] {nm} marginal, two-step vs one-step Schur: "
        f"{np.linalg.norm(blk - one_step)/np.linalg.norm(blk):.2e}")

# ===== Gate 2: everything that is marginalized or plotted is PSD =====
for nm, M in [('F_full (clipped)', psd_clip(F_full)), ('F_phys_marg', F_phys_marg),
              ('A_marg  = P(k) marginal', A_marg), ('G_marg  = growth marginal', G_marg)]:
    w = np.linalg.eigvalsh(sym(M))
    log(f"[Gate 2] {nm:26s}: min_eig/|max_eig| = {w.min()/np.abs(w).max():+.2e}")

# %% [cell 19] Jacobians & projections -----------------------------------------
J_pk = np.array(jax.jacobian(cosmo_to_amps)(cosmo_fid_vec))          # [80 x 3]
J_growth_2d = np.array(jax.jacobian(cosmo_to_growth)(cosmo_fid_2d))  # [25 x 2]
J_full = np.array(jax.jacobian(cosmo_to_observables)(cosmo_fid_vec)) # [105 x 3]

# growth depends on (omega_cdm, h) only -> [25 x 3] with an identically zero lnAs column
J_growth_3d = np.zeros((n_growth, 3))
J_growth_3d[:, 0] = J_growth_2d[:, 0]   # d/d omega_cdm
J_growth_3d[:, 2] = J_growth_2d[:, 1]   # d/d h

ratio_h_As = np.linalg.norm(J_pk[:, 2]) / np.linalg.norm(J_pk[:, 1])
log(f"amps Jacobian ||h col|| / ||As col|| = {ratio_h_As:.3e} (should be ~0)")

# ---------------------------------------------------------------------------
# The six objects the figure shows, all 3x3 in (omega_cdm, lnAs, h) and all built from
# the SAME three blocks of F_phys_marg, so that both chain orderings are exact:
#
#   Direct              the full 3-parameter LCDM fit
#   Combined            J^T F_phys_marg J, the whole MI Fisher projected on LCDM
#   P(k) marginal       template alone, the 25 growth parameters marginalized out
#   Growth marginal     growth/AP alone, the 80 template amplitudes marginalized out
#   Growth | P(k)       the conditional factor of p(a, g) = p(a) p(g | a)
#   P(k) | Growth       the conditional factor of the other ordering, p(g) p(a | g)
#
#   F_combined = F_pk_marg + F_growth|pk = F_growth_marg + F_pk|growth      (Gate 5)
#
# The two orderings are alternative *partitions* of the same information: each assigns
# all of the cross-sector information to its own conditional, so a marginal from one
# ordering must never be added to a conditional from the other.
# ---------------------------------------------------------------------------
F_cosmo_combined = J_full.T @ F_phys_marg @ J_full
F_cosmo_direct_marg_3d = schur_marg(F_direct_full, np.arange(n_eft, n_eft + 3),
                                    np.arange(n_eft))

F_pk_marg_3d     = psd_clip(J_pk.T @ A_marg @ J_pk)
F_growth_marg_3d = psd_clip(J_growth_3d.T @ G_marg @ J_growth_3d)
Jg_corr = J_growth_3d + G_pinv @ C_blk.T @ J_pk
Ja_corr = J_pk + A_pinv @ C_blk @ J_growth_3d
F_growth_given_pk = psd_clip(Jg_corr.T @ G_blk @ Jg_corr)
F_pk_given_growth = psd_clip(Ja_corr.T @ A_blk @ Ja_corr)

# 2d restrictions kept for the saved results (pk has no h response, growth no lnAs one)
F_cosmo_from_pk_only_2d = F_pk_marg_3d[np.ix_([0, 1], [0, 1])]
F_cosmo_from_growth_only_2d = F_growth_marg_3d[np.ix_([0, 2], [0, 2])]

# ===== Gate 3: chain-rule identity F_direct ~= J^T F_phys_marg J =====
rel = (np.linalg.norm(F_cosmo_combined - F_cosmo_direct_marg_3d)
       / np.linalg.norm(F_cosmo_direct_marg_3d))
log(f"[Gate 3] ||F_combined - F_direct|| / ||F_direct|| = {rel:.3e}")

# ===== Gate 4: no marginal or conditional beats Direct =====
# NOTE on flooring: with big_var=1e10 a floored flat direction leaks eps^2 * big_var into
# every parameter it has any eigenvector overlap with. big_var=1.0 keeps the leak
# negligible, and a reported sigma of O(1) simply means "unconstrained".
def sig3(F, big_var=1.0):
    return np.sqrt(np.diag(fisher_to_cov(F, big_var=big_var)))

sig_direct = sig3(F_cosmo_direct_marg_3d)
pieces = [('Direct (full)', F_cosmo_direct_marg_3d), ('Combined', F_cosmo_combined),
          ('P(k) marginal', F_pk_marg_3d), ('Growth+AP marginal', F_growth_marg_3d),
          ('Growth+AP | P(k)', F_growth_given_pk), ('P(k) | Growth+AP', F_pk_given_growth)]
ok4 = True
for nm, F in pieces[2:]:
    s_ = sig3(F)
    ok4 &= bool(np.all(s_ >= sig_direct * 0.999))
log("[Gate 4] " + ("PASS (every sub-piece is looser than Direct)" if ok4
                   else "FAIL <- a sub-piece beats Direct (impossible)"))

# %% [cell 21] where does the h information live? ------------------------------
# ---------------------------------------------------------------------------
# Where does the h information live?  The answer is fixed by two degeneracies of the
# MI physical block, both checked numerically below (Gate 5a).
#
#   (i)  a * D(z)^2 invariant -- EXACT. Raising every template amplitude by dln a and
#        lowering every D(z)/D(z_ref) by dln a / 2 leaves every model spectrum
#        unchanged, so the ABSOLUTE normalization -- i.e. A_s -- is not measurable
#        from the template alone: lnAs is exactly flat in the P(k) marginal.
#   (ii) units bridge -- NEAR-exact. h_conv only rescales the grid the template is
#        handed to the emulator on, k[h/Mpc] = k[1/Mpc] / h_conv with P *= h_conv^3,
#        which with all 80 knot amplitudes free is a rigid dilation of the template,
#        dln a_j = (3 + dlnP/dlnk|_j) dln h_conv, and the amplitudes can follow it.
#        Only ~1e-5 of the h_conv information survives marginalizing the template
#        (sigma(h_conv) 0.0019 -> 0.52), so h is unconstrained in the growth marginal
#        for every practical purpose, and that marginal collapses to an Omega_m band.
#        The residual is not physics: the emulator sees the template only after it has
#        been re-interpolated onto its own fixed knots and normalized by max(pk), and
#        those two steps are what stop the family from closing exactly.
#        (mi_model.py makes the same statement for units='mpc'.)
#
# Hence NEITHER sector marginal contains h: the standard-ruler calibration is the
# product of a template scale and a dilation, and lives entirely in the cross term.
# It appears in whichever conditional the chain ordering puts second.
# ---------------------------------------------------------------------------
dlnP_dlnk = np.gradient(np.log(np.array(template_mpc)), np.log(np.array(knots_mpc)))
n_phys = len(idx_phys)
i_hc_phys = n_phys - 1          # h_conv within the phys ordering [amps, growth]
i_hc_growth = n_growth - 1

v_aD = np.zeros(n_phys)                       # (i) a * D^2
v_aD[:n_knots] = 1.0                          #     d a_j       (a_fid = 1)
for i in range(n_z_unique):
    v_aD[n_knots + 3 * n_z_unique + i] = -0.5 * growth_fid[3 * n_z_unique + i]
v_dil = np.zeros(n_phys)                      # (ii) h_conv <-> dilation
v_dil[:n_knots] = 3.0 + dlnP_dlnk             #     d a_j per dln h_conv
v_dil[i_hc_phys] = growth_fid[-1]             #     d h_conv per dln h_conv

log("--- Gate 5a: the two degeneracies that empty the sector marginals ---")
w_phys = np.linalg.eigvalsh(sym(F_phys_marg))
_vv = v_aD / np.linalg.norm(v_aD)
_q = float(_vv @ F_phys_marg @ _vv)
log(f"  a * D^2 invariant: v^T F v = {_q:.3e} = {_q/w_phys.max():.1e} of the largest "
    f"eigenvalue -> EXACT (smallest eigenvalue of the P(k) marginal: "
    f"{np.linalg.eigvalsh(sym(A_marg)).min():+.1e})")
_f_cond = float(G_blk[i_hc_growth, i_hc_growth])
_f_marg = float(G_marg[i_hc_growth, i_hc_growth])
log(f"  h_conv <-> template dilation: information {_f_cond:.3e} with the template fixed, "
    f"{_f_marg:.3e} with it free -> {_f_marg/_f_cond:.1e} survives, "
    f"sigma(h_conv) {1/np.sqrt(_f_cond):.4f} -> {1/np.sqrt(_f_marg):.4f}")
log("  (these two numbers are exact projections of the Fisher; the analytic dilation "
    "vector v_dil below is only a diagnostic -- its (3 + dlnP/dlnk) uses a finite "
    "difference across the BAO wiggles and is not accurate enough to test flatness with)")

# ===== Gate 5b: both chain orderings are exactly additive =====
for nm, Fm, Fc in [('p(a) p(g|a)  [template first]', F_pk_marg_3d, F_growth_given_pk),
                   ('p(g) p(a|g)  [growth first]  ', F_growth_marg_3d, F_pk_given_growth)]:
    r = np.linalg.norm(Fm + Fc - F_cosmo_combined) / np.linalg.norm(F_cosmo_combined)
    log(f"[Gate 5b] {nm}: ||marg + cond - combined|| / ||combined|| = {r:.2e}")

# --- the sigma table -------------------------------------------------------
log("--- sigma summary, EFT always marginalized (sigma ~ O(1) means UNCONSTRAINED) ---")
log(f"{'':24s} {'omega_cdm':>11s} {'lnAs':>11s} {'h':>11s}")
for nm, F in pieces:
    s_ = sig3(F)
    log(f"{nm:24s} {s_[0]:11.2e} {s_[1]:11.2e} {s_[2]:11.2e}")

def report_combos(F, pnames, label):
    """Eigen-structure of a Fisher: which parameter combinations are constrained (and
    how well), and which are exactly flat."""
    w, V = np.linalg.eigh(sym(F))
    scale = np.abs(w).max()
    log(f"{label}:")
    for i in range(len(w) - 1, -1, -1):
        combo = " ".join(f"{V[j, i]:+.3f}*{pnames[j]}" for j in range(len(pnames)))
        if w[i] > 1e-8 * scale:
            log(f"   constrained: sigma({combo}) = {1/np.sqrt(w[i]):.3e}")
        else:
            log(f"   FLAT:        {combo}")

pn = ['ocdm', 'lnAs', 'h']
for nm, F in pieces:
    report_combos(F, pn, nm)

# fractional Omega_m precision of the growth-marginal band (propagate the covariance,
# dropping the exactly-flat direction -- NOT 1/sqrt(g F g), which understates the width)
gOm3 = np.array([1 / h_true**2, 0.0,
                 -2 * (cosmo_fid['omega_cdm'] + cosmo_fid['omega_b']) / h_true**3])
sig_Om = np.sqrt(gOm3 @ pos_pinv(F_growth_marg_3d) @ gOm3)
log(f"Growth marginal = an Omega_m band: sigma(Om)/Om = {sig_Om/Om_fid:.3f}")

# h_conv: unconstrained once the template is free, well measured once it is fixed
sig_hconv_cond = 1.0 / np.sqrt(F_phys_marg[i_hc_phys, i_hc_phys])
_cov_g = fisher_to_cov(G_marg, big_var=1.0)
log(f"sigma(h_conv): conditional on the template = {sig_hconv_cond:.4f}, "
    f"marginal over it = {np.sqrt(_cov_g[i_hc_growth, i_hc_growth]):.4f} (unconstrained)")


# --- product of the two sector marginals vs Direct --------------------------
# If the sectors were independent this would equal Direct. The gap IS the cross-sector
# information, i.e. everything that distinguishes the two chain orderings.
F_prod3 = F_pk_marg_3d + F_growth_marg_3d
sig_prod = sig3(F_prod3)
log("--- product of the two sector marginals vs Direct ---")
for i, nm in enumerate(['omega_cdm', 'lnAs', 'h']):
    log(f"sigma({nm:9s}): product = {sig_prod[i]:.2e}   direct = {sig_direct[i]:.2e}   "
        f"ratio = {sig_prod[i]/sig_direct[i]:.1f}x")
report_combos(F_prod3, pn, "Product of the sector marginals")

# ===== Gate 6: stability of every number in the pinv threshold =====
log("--- Gate 6: sensitivity to the pseudo-inverse threshold ---")
log(f"{'rtol':>8s} {'rank(A)':>8s} {'sig(Om)/Om':>11s} {'sig_h(g|pk)':>12s} {'sig_ocdm(pk marg)':>18s}")
for _r in [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]:
    _Ap, _Gp = pos_pinv(A_blk, _r), pos_pinv(G_blk, _r)
    _Am = psd_clip(A_blk - C_blk @ _Gp @ C_blk.T)
    _Gm = psd_clip(G_blk - C_blk.T @ _Ap @ C_blk)
    _Fg = psd_clip(J_growth_3d.T @ _Gm @ J_growth_3d)
    _Fa = psd_clip(J_pk.T @ _Am @ J_pk)
    _Jgc = J_growth_3d + _Gp @ C_blk.T @ J_pk
    _Fgc = psd_clip(_Jgc.T @ G_blk @ _Jgc)
    _rank = int((np.linalg.eigvalsh(sym(A_blk)) > _r * np.abs(np.linalg.eigvalsh(sym(A_blk))).max()).sum())
    log(f"{_r:8.0e} {_rank:8d} {np.sqrt(gOm3 @ pos_pinv(_Fg) @ gOm3)/Om_fid:11.3f} "
        f"{sig3(_Fgc)[2]:12.3e} {sig3(_Fa)[0]:18.3e}")

# %% [cell 23] triangle plot ---------------------------------------------------
# The plotting lives in make_fisher_fig.py, which is also runnable standalone against the
# saved .npz -- so the paper figure and these diagnostics cannot drift apart.
from make_fisher_fig import make_figures

F_fig = {'F_cosmo_direct': F_cosmo_direct_marg_3d, 'F_cosmo_combined': F_cosmo_combined,
         'F_pk_marg_3d': F_pk_marg_3d, 'F_growth_marg_3d': F_growth_marg_3d,
         'F_growth_given_pk': F_growth_given_pk, 'F_pk_given_growth': F_pk_given_growth}
make_figures(F_fig, figdir, paper_figdir=os.path.join(rootdir, 'paper', 'figs'), log=log)

np.savez(os.path.join(figdir, 'fisher_v3_results.npz'),
         F_full=F_full, F_full_gn=F_full_gn, F_direct_full=F_direct_full,
         F_phys_marg=F_phys_marg, A_blk=A_blk, G_blk=G_blk, C_blk=C_blk,
         F_pk_marginal=F_pk_marginal, F_growth_marginal=F_growth_marginal,
         F_cosmo_direct=F_cosmo_direct_marg_3d, F_cosmo_combined=F_cosmo_combined,
         F_pk_marg_3d=F_pk_marg_3d, F_growth_marg_3d=F_growth_marg_3d,
         F_growth_given_pk=F_growth_given_pk, F_pk_given_growth=F_pk_given_growth,
         F_pk_only_2d=F_cosmo_from_pk_only_2d,
         F_growth_only_2d=F_cosmo_from_growth_only_2d, F_prod3=F_prod3,
         # v3.0 names kept so the paper-number pipeline keeps working
         F_red_chain=F_pk_marg_3d, F_green_chain=F_growth_given_pk,
         F_pk_given_growth_chain=F_pk_given_growth,
         J_pk=J_pk, J_growth_2d=J_growth_2d, J_growth_3d=J_growth_3d, J_full=J_full,
         v_flat_aD=v_aD, v_flat_dilation=v_dil,
         params_fid=params_fid, growth_fid=growth_fid,
         knots_h=knots_h, template_mpc=np.array(template_mpc))
log("Results saved. DONE.")
