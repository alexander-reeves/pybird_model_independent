"""Fisher information split for P(k) alone vs P(k)+B(k) -- development script.

Sections map 1:1 to the code cells of 06_fisher_pk_bk.ipynb (built by
`python3 build_fisher_pb_nb.py`); run inside the pybird jax_env container with
`sbatch exec_fisher_pb.sbatch`.

Same model-independent (MI) parametrization as 01_fisher_v3 (v3.1):
  * P_lin template in Mpc^3 at z_ref=5 on the 80 emulator knots (fixed 1/Mpc grid), free
    amplitudes a_j; growth/AP block [f, H/H0, D_A H0 per z, D(z)/D(z_ref) per z, h_conv];
  * the direct LCDM model is the exact composition of the MI model with the cosmology maps;
  * the data are the notebook's own direct model at the fiducial (chi2(fid) = 0, Gate 1).

What is new here:
  * the likelihood is pybird-dev's (the bispectrum JAX code lives only there, see
    PLAN_merge_bispectrum.md), imported by putting the pybird-dev checkout FIRST on sys.path;
  * the fake DESI-Y6-like data vector is P0+P2+P4 (k <= 0.2 h/Mpc, `MULTIPOLE`=3 as in the
    v3 P(k)-only Fisher) AND the tree-level bispectrum monopole B0 (0.02 <= k_i <= 0.10
    h/Mpc, closed ordered triangles, analytic PPP covariance, no P-B cross covariance),
    written from the direct model by a small re-implementation of `Fake.set()` (pybird-dev's
    own `Fake.set()` would run its Boltzmann path, which is not the notebook's model). Both
    likelihoods carry the SAME P(k) multipoles, so the P(k) data are a strict subset of the
    P(k)+B(k) data and every width ratio below is attributable to B(k) alone;
  * the EFT basis is 'eth' (2211.17130 app. D.4), the only one the bispectrum accepts. For
    the power spectrum it is the eftoflss basis relabelled (Bird.setBias: b1=Bb1, b2=Bb2,
    b3=Bb3+15Bb8, b4=Bb5, cct=-Bc1, cr1=f Bc2 - f^2/2 Bc4, cr2=-f^2/2 Bc3, ce0=Be1,
    ce1=Be2+ce2/2), so the P(k) loops come from the SAME 80-knot emulator as v3 (with_emu),
    interpolation-free on the knots, with the cubic input-interpolation patch of v3.1. The
    emulator path of pybird-dev asks for the exact-time coefficients explicitly; the EdS
    values (G1=1, Y1=0, G1t=3/7, V12t=1/7) that the MI tree hard-codes are passed;
  * TWO likelihoods read the same file: P(k) only ('bPk') and P(k)+B(k) ('bPk,bBk'), and
    every Fisher object is computed for both. The deliverable is the growth/AP sector
    (f, H/H0, D_A H0 at every z) for P vs P+B, as corner plots, both with the template
    fixed (the conditional G, i.e. what a fixed-shape analysis gives) and with the template
    marginalized (G_marg, the truly model-independent statement).

Set `with_bk_tree_level: False` + a `bk_loop_matrix_path` per sky once the one-loop
bispectrum matrices are available; nothing else in this file depends on tree level.
"""

# %% [cell 1] imports and setup ------------------------------------------------
import os, sys, time
# Autodiff Hessian reductions are non-deterministic on GPU, and every sector marginal below
# is a Schur complement that DIVIDES by the block being marginalized, so that noise is
# amplified: on GH200 the Hessian came out PSD only to 1.6e-4 (Gate 3 at 2%, Gate 4 failing),
# against 1e-7 on CPU. run_fisher_v3.py pins the CPU for the same reason. Override with
# JAX_PLATFORMS=cuda to measure the difference.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from copy import deepcopy
from collections import defaultdict

# pybird-dev (bispectrum JAX code) must shadow the editable pybird_model_independent install
PYBIRD_DEV = os.environ.get('PYBIRD_DEV', '/capstor/store/cscs/swissai/a0158/areeves/pybird-dev')
sys.path.insert(0, PYBIRD_DEV)

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
assert os.path.realpath(config.__file__).startswith(os.path.realpath(PYBIRD_DEV)), config.__file__
config.set_jax_enabled(True)
from pybird.likelihood import Likelihood
from pybird.fake import Fake
from pybird.utils_cov import get_cov_gauss, get_cov_b_PPP
from pybird.symbolic import D as symD, f as symf, Hubble as symHubble, DA as symDA
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ

# --- emulator input interpolation: cubic, not piecewise-linear (v3.1 fix) ----------------
# Emulator.make_params interpolates log P_lin(kk) onto the 80 knots with a piecewise-LINEAR
# interpolant; the template sits exactly ON those knots at h_conv = h_fid, so the curvature
# along h_conv would jump by ~20% at the expansion point. Same patch as run_fisher_v3.py and
# mi_model.py (pybird-dev's make_params has the identical signature).
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

from fisher_utils import (RTOL_MARG, sym, noise_level, psd_clip, pos_pinv, fisher_to_cov,
                          fisher_to_cov_capped, report_combos, schur_marg, sigmas,
                          sector_split, cosmo_pieces)

t0 = time.time()
def log(msg): print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

SMOKE = os.environ.get('SMOKE', '0') == '1'   # 2 skies, no figures: pipeline check only
rootdir = "../.."
output_path = os.path.join(rootdir, "output")
figdir = os.path.join(output_path, "fisher_pb" + ("_smoke" if SMOKE else ""))
os.makedirs(figdir, exist_ok=True)
log(f"pybird from {os.path.dirname(config.__file__)}; jax devices: {jax.devices()}")

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
if SMOKE:
    DESI_Y6 = {k: (v[:2] if isinstance(v, (list, np.ndarray)) else 2) for k, v in DESI_Y6.items()}
num_skies = DESI_Y6['n_sky']
zeff_list = list(DESI_Y6['zeff'])
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

log(f"CPJ ready; Om_fid={Om_fid:.4f}; {num_skies} skies, z_eff unique = {zeff_unique}")

# %% [cell 5] knots = emulator's native 80 knots; physical template (as v3) --
knots_h = np.load(os.path.join(PYBIRD_DEV, 'pybird', 'emu_data', 'knots.npy'))  # h/Mpc
n_knots = len(knots_h)
knots_mpc = jnp.array(knots_h * h_true)  # FIXED physical grid in 1/Mpc
kk_fid_h = knots_h

template_mpc = jnp.array(pklin_cpj(knots_mpc, cosmo_fid['omega_b'], cosmo_fid['omega_cdm'],
                                   h_true, cosmo_fid['ln10^{10}A_s'], cosmo_fid['n_s'], z_ref))

template_mpc_h110 = pklin_cpj(knots_mpc, cosmo_fid['omega_b'], cosmo_fid['omega_cdm'],
                              h_true * 1.1, cosmo_fid['ln10^{10}A_s'], cosmo_fid['n_s'], z_ref)
resid_h = np.abs(np.array(template_mpc_h110 / template_mpc) - 1)
log(f"{n_knots} emulator knots, k=[{knots_h.min():.1e}, {knots_h.max():.2f}] h/Mpc")
log(f"Template h-independence: max |dP/P| for 10% h shift = {resid_h.max():.2e} "
    f"(median {np.median(resid_h):.2e})")

# %% [cell 7] cosmology -> observable mappings (identical to v3) ---------------
def cosmo_to_amps(cosmo_params_3d):
    """[omega_cdm, lnAs, h] -> P(k) amplitude ratios at the FIXED physical knots (1/Mpc)."""
    omega_cdm, lnAs, h = cosmo_params_3d
    pk = pklin_cpj(knots_mpc, cosmo_fid['omega_b'], omega_cdm, h, lnAs, cosmo_fid['n_s'], z_ref)
    return pk / template_mpc

def cosmo_to_growth(cosmo_params_2d):
    """[omega_cdm, h] -> [f, H/H0, DA*H0 per z, D(z)/D(z_ref) per z, h_conv]."""
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
n_growth = len(growth_fid)  # 3*n_z + n_z + 1
growth_names = ([n for z in zeff_unique for n in
                 (f'f(z={z:.2f})', f'H/H0(z={z:.2f})', f'DA*H0(z={z:.2f})')]
                + [f'D(z={z:.2f})/D(z_ref)' for z in zeff_unique] + ['h_conv'])
idx_f  = [3 * i for i in range(n_z_unique)]
idx_H  = [3 * i + 1 for i in range(n_z_unique)]
idx_DA = [3 * i + 2 for i in range(n_z_unique)]

# %% [cell 9] shared model builder: (amps, growth) -> per-sky pybird inputs ----
def build_cosmo_dicts(amps, growth):
    """MI parameters -> list of pybird cosmo_dicts (one per sky), in the data's h-units via
    h_conv: kk = knots_mpc / h_conv [h/Mpc], pk = P_mpc * h_conv^3 [(Mpc/h)^3] (same physical
    points -> no interpolation; exact emulator grid at fiducial). The four exact-time
    coefficients are the EdS values the MI tree hard-codes (pybird-dev's emulator path
    requires them explicitly)."""
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
        dicts.append({"H": H_z[i_z], "DA": DA_z[i_z], "f": f_z[i_z],
                      "kk": kk_h, "pk_lin": pk_h * D_ratios[i_z]**2,
                      "G1": 1.0, "Y1": 0.0, "G1t": 3.0 / 7.0, "V12t": 1.0 / 7.0})
    return dicts

def build_direct_cosmo_dicts(cosmo_params_3d):
    """Direct model = MI model composed with the cosmology mapping (exactly)."""
    amps = cosmo_to_amps(cosmo_params_3d)
    growth = cosmo_to_growth(jnp.array([cosmo_params_3d[0], cosmo_params_3d[2]]))
    return build_cosmo_dicts(amps, growth)

# %% [cell 11] fake DESI Y6 P0+P2+B0 data = direct model at fiducial ----------
# Likelihood configuration: pybird-dev's Abacus 'ELG_pk_bktree' template ('eth' basis, the
# only basis the bispectrum supports), adapted to the multi-sky fake survey. Every
# projection effect (binning, window) is off, as in the v3 P(k) Fisher.
template_configfile = os.path.join(PYBIRD_DEV, 'data', 'abacus', 'likelihood_config', 'ELG_pk_bktree.yaml')
lkl_cfg = yaml.safe_load(open(template_configfile))
for key in ['data_path', 'data_file', 'nd', 'sky']:
    lkl_cfg.pop(key, None)
# P(k) multipoles. 3 = P0+P2+P4, matching the v3 P(k)-only Fisher; 2 = P0+P2, the setting
# every bispectrum configuration shipped with pybird uses. The hexadecapole is applied to BOTH
# likelihoods, not to the P(k)-only one alone: the whole comparison rests on the P(k) data
# being a strict SUBSET of the P(k)+B(k) data (that is what makes Gate 7, F_PB - F_P PSD,
# both true and meaningful, and what lets a width ratio be read as "what B(k) adds"). Giving
# P(k) a multipole that P(k)+B(k) does not have would break the nesting and let the P(k)-only
# column be tighter than P(k)+B(k) on some directions, which is not a bispectrum statement at
# all. Nothing in the bispectrum code indexes the P(k) multipole count -- bk_Common passes Nl
# straight through to Common and no bk_* module reads self.co.Nl -- so Nl=3 is safe here.
MULTIPOLE = int(os.environ.get('MULTIPOLE', '3'))

lkl_cfg.update({
    'output': 'bPk,bBk', 'multipole': MULTIPOLE,
    'with_emu': True, 'with_resum': True, 'with_ap': True,
    'with_binning': False, 'with_survey_mask': False,
    'bk_multipole': [{'l': 0, 'mu_i': 1}],
    'with_bk_tree_level': True,   # <- False + per-sky 'bk_loop_matrix_path' once the loop matrices exist
    'with_bk_resum': True, 'with_bk_ap': True,
    'with_bk_binning': False, 'with_bk_survey_mask': False,
})
lkl_cfg['write'].update({'save': False, 'fake': False, 'plot': False, 'show': False})

# fiducial EFT parameters (pybird-dev demo/fake_bispectrum.ipynb): Bb1, Bb2, Bb5 non-zero
eft_all = ("Bb1 Bb2 Bb3 Bb5 Bb8 Bc1 Bc2 Bc3 Bc4 Bd1 Be1 Be2 Be5 ce2 cr4 cr6").split()
fiducial_nuisance = {name: 0.0 for name in eft_all}
fiducial_nuisance.update({"Bb1": 1.8, "Bb2": 0.4, "Bb5": 0.2})

# bispectrum triangles: closed ordered triangles with 0.02 <= k_i <= 0.10 h/Mpc
k_bk = np.arange(0.02, 0.101, 0.01)
k123 = np.array([(k1, k2, k3) for i, k1 in enumerate(k_bk) for j, k2 in enumerate(k_bk[i:], start=i)
                 for k3 in k_bk[j:] if k1 + k2 >= k3 - 1e-12]).T
fake_data_name = 'fake_desi_y6_fisher_pb' + ('_smoke' if SMOKE else '')
fake_config_name = fake_data_name + '_config'

s = DESI_Y6
F = Fake(s['n_sky'], s['zmin'], s['zmax'], s['zeff'], s['Veff'], s['degsq'], s['P0'],
         cosmo_fid, likelihood_config=lkl_cfg, fiducial_nuisance=fiducial_nuisance,
         boltzmann='Symbolic', Omega_m_fid=Om_fid, kmin=0.01, kmax=0.2, dk=0.01,
         nbar_prior=s['nbar_prior'], k123=k123, k123_eff=k123, bk_binsize=0.01,
         fake_data_filename=fake_data_name, path_to_data=output_path,
         fake_likelihood_config_filename=fake_config_name, path_to_config=output_path)
log(f"Fake initialized: {len(F.kd)} P(k) bins per multipole, {k123.shape[1]} triangles per sky, "
    f"nbar = {np.array(F.nbar).round(6).tolist()}")

def fake_set_from_cosmo_dicts(F, cosmo_dicts, prior_center_on_truth=True):
    """`Fake.set()` with the model evaluated from explicit pybird cosmo_dicts (the notebook's
    direct model at the fiducial) instead of Fake's internal Boltzmann call. Everything else
    -- analytic Gaussian P(k) covariance, PPP B(k) covariance, zero P-B cross covariance,
    EFT priors centred on the truth, the written yaml -- is the pybird-dev code path."""
    d = defaultdict(dict)
    for i in range(F.n_sky):
        F.e[i].compute(cosmo_dict=cosmo_dicts[i])
        bpk, bbk = F.e[i].get(deepcopy(F.fiducial_nuisance[i]))
        bpk, bbk = np.asarray(bpk), np.asarray(bbk)
        b1, f1 = F.fiducial_nuisance[i]['Bb1'], float(F.e[i].bird.f)
        ipk_lin = scipy_interp1d(np.asarray(F.e[i].bird.kin), np.asarray(F.e[i].bird.Pin))
        F.io.write_common(d[f'sky_{i+1}'], F.zmin[i], F.zmax[i], F.zeff[i], F.Omega_m_fid, F.H_fid[i], F.D_fid[i])
        cov_pk = get_cov_gauss(F.kd, F.dk, ipk_lin, b1, f1, Vs=F.Veff[i], nbar=F.nbar[i], n_mult=F.c['multipole'])
        F.io.write_pk(d[f'sky_{i+1}'], F.c['multipole'], F.kd, bpk, cov_pk, nsims=-1,
                      survey_mask_arr_p=None, survey_mask_mat_kp=None, binsize=F.dk)
        cov_bk = get_cov_b_PPP(F.k123_eff[i], F.bk_binsize, ipk_lin, b1, f1, Vs=F.Veff[i], nbar=F.nbar[i])
        cov_pkbk = np.zeros((bpk.size, bbk.size))
        F.io.write_bk(d[f'sky_{i+1}'], bbk, k123=F.k123[i], k123_eff=F.k123_eff[i], cov_bk=cov_bk,
                      cov_pkbk=cov_pkbk, bk_multipole=F.c.get('bk_multipole'), nsims=-1)
    with h5py.File(F.path_to_file, 'w') as hf:
        F.save_dict_to_hdf5(hf, d)
    F.c['data_path'], F.c['data_file'] = F.path, F.fake_data_filename
    if prior_center_on_truth:
        for name in F.c['eft_prior']:
            if name not in F.fiducial_nuisance[0]:
                continue
            F.c['eft_prior'][name]['mean'] = [F.fiducial_nuisance[i][name] for i in range(F.n_sky)]
            if 'range' in F.c['eft_prior'][name]:
                F.c['eft_prior'][name]['range'] = [F.c['eft_prior'][name]['range'][0]] * F.n_sky
    with open(F.path_to_config, 'w') as file:
        yaml.dump(F.c, file)

fake_set_from_cosmo_dicts(F, build_direct_cosmo_dicts(cosmo_fid_vec))
log(f"Fake P0+P2+B0 data (direct model at fiducial) written to {F.path_to_file}")

# %% [cell 13] two likelihoods (P, P+B) + the model functions; Gate 1 ----------
cfg_pb = yaml.full_load(open(os.path.join(output_path, f'{fake_config_name}.yaml')))
cfg_pb['get_maxlkl'] = True
cfg_p = deepcopy(cfg_pb)
cfg_p['output'] = 'bPk'
for name in ['Bd1', 'Be5']:   # bispectrum-only EFT parameters (tree level)
    cfg_p['eft_prior'].pop(name)

L = {'P': Likelihood(cfg_p, verbose=False), 'P+B': Likelihood(cfg_pb, verbose=False)}
for tag, Li in L.items():
    n_data = [len(Li.y_sky[i]) for i in range(Li.nsky)]
    eft_list = Li.correlator_sky[0].eft_parameters_list
    assert set(Li.c['eft_prior']) == set(eft_list), (tag, set(Li.c['eft_prior']) ^ set(eft_list))
    log(f"[{tag}] data points per sky {n_data} (total {sum(n_data)}); EFT: "
        f"{len(Li.bg_name)} marginalized {Li.bg_name}, non-marginalized {Li.bng_name + ['Bb1']}")

eft_free_names = ["Bb1", "Bb2", "Bb5"]
eft_init = np.array([fiducial_nuisance[k] for _ in range(num_skies) for k in eft_free_names])
n_eft = len(eft_init)
eft_names_flat = [f"{k}_{i+1}" for i in range(num_skies) for k in eft_free_names]

def make_loglkls(Li):
    def model_independent_loglkl(params):
        """EFT (3/sky) + P(k) amps at 80 physical knots + growth (3 n_z + n_z + h_conv). NO priors
        beyond the EFT ones of the likelihood configuration."""
        eft = params[:n_eft]
        amps = params[n_eft:n_eft + n_knots]
        growth = params[n_eft + n_knots:]
        return Li.loglkl(eft, eft_names_flat, need_cosmo_update=True, cosmo_module=None,
                         cosmo_dict=build_cosmo_dicts(amps, growth))
    def direct_cosmo_loglkl(params):
        """EFT (3/sky) + [omega_cdm, lnAs, h]; factors exactly through the MI model."""
        return Li.loglkl(params[:n_eft], eft_names_flat, need_cosmo_update=True, cosmo_module=None,
                         cosmo_dict=build_direct_cosmo_dicts(params[n_eft:]))
    return model_independent_loglkl, direct_cosmo_loglkl

loglkl_mi, loglkl_direct = {}, {}
for tag, Li in L.items():
    loglkl_mi[tag], loglkl_direct[tag] = make_loglkls(Li)

params_fid = np.concatenate([eft_init, np.ones(n_knots), growth_fid])
cosmo_eft_fid = jnp.concatenate([jnp.array(eft_init), cosmo_fid_vec])
n_params = len(params_fid)
idx_eft = np.arange(n_eft)
idx_pk = np.arange(n_eft, n_eft + n_knots)
idx_growth = np.arange(n_eft + n_knots, n_params)

# ===== Gate 1: all four likelihoods reproduce the data at fiducial =====
chi2_fid = {}
for tag in L:
    chi2_fid[tag] = (-2 * float(loglkl_direct[tag](cosmo_eft_fid)), -2 * float(loglkl_mi[tag](jnp.array(params_fid))))
    log(f"[Gate 1] {tag:4s} chi2(fid): direct = {chi2_fid[tag][0]:.3e}, model-independent = {chi2_fid[tag][1]:.3e}")
log("[Gate 1] " + ("PASS" if max(abs(v) for c in chi2_fid.values() for v in c) < 1e-6 else "FAIL"))

# --- the unprofiled model vector (all EFT coefficients explicit) and the EFT priors ------
# Defined here rather than in cell 15 so that jacobian_pb.py, which executes cells 1-13 of
# this file verbatim, uses exactly the same model code. See cell 15 for why it exists.
def make_model_vector(Li):
    """Concatenated, masked theory vector for all skies, as a function of
    [all EFT (n_eft_sky per sky), 80 amplitudes, 25 growth]. No marginalization inside."""
    # pybird-dev lists 'Be1' TWICE for output 'bPk,bBk' (correlator.py appends the stochastic
    # block without an "if not already present" check, unlike the counterterm block just above
    # it). A duplicated name is harmless for the likelihood, which builds a dict, but here it
    # would create a phantom parameter whose column of J is identically zero -- and, worse, an
    # ambiguous name->index map that made the P vs P+B comparison select the phantom column for
    # one data set and the real one for the other (Gate 7 failed at -8.8e-1 before this).
    raw = list(Li.correlator_sky[0].eft_parameters_list)
    eft_names_sky = list(dict.fromkeys(raw))
    if len(eft_names_sky) != len(raw):
        dup = [n for n in eft_names_sky if raw.count(n) > 1]
        log(f"  note: de-duplicated pybird's eft_parameters_list ({len(raw)} -> "
            f"{len(eft_names_sky)}); repeated: {dup}")
    n_e = len(eft_names_sky)

    def model_vector(params):
        eft = params[:n_e * num_skies].reshape(num_skies, n_e)
        amps = params[n_e * num_skies:n_e * num_skies + n_knots]
        growth = params[n_e * num_skies + n_knots:]
        dicts = build_cosmo_dicts(amps, growth)
        out = []
        for i in range(num_skies):
            Li.correlator_sky[i].compute(cosmo_dict=dicts[i], cosmo_module=None)
            bias = {nm: eft[i, j] for j, nm in enumerate(eft_names_sky)}
            out.append(Li.correlator_sky[i].get(bias, concatenated=True)[Li.m_sky[i]])
        return jnp.concatenate(out)

    return model_vector, eft_names_sky


def eft_prior_precision(Li, eft_names_sky):
    """Diagonal prior precision for the EFT coefficients, per sky, from the likelihood config.
    Every coefficient with a 'gauss' or 'marg_gauss' prior contributes 1/range^2; a 'flat' one
    (here Bb1) contributes nothing. These are the SAME priors the Hessian route applies
    analytically, now written out explicitly because the parameters are explicit."""
    n_e = len(eft_names_sky)
    # full-length: the amplitudes and the growth parameters carry NO prior here, exactly as in
    # the Hessian route, so the physical block is constrained by the data alone.
    prec = np.zeros(n_e * num_skies + n_knots + n_growth)
    for j, nm in enumerate(eft_names_sky):
        pr = Li.c['eft_prior'].get(nm, {})
        if pr.get('type') in ('gauss', 'marg_gauss'):
            rng = np.atleast_1d(pr['range']).astype(float)
            for i in range(num_skies):
                prec[i * n_e + j] = 1.0 / (rng[i] if len(rng) > 1 else rng[0])**2
    return prec

# %% [cell 15] Fisher matrices for P and P+B -----------------------------------
F_full, F_direct_full = {}, {}
for tag in L:
    t1 = time.time()
    F_full[tag] = -np.array(jax.hessian(loglkl_mi[tag])(jnp.array(params_fid)))
    log(f"[{tag}] MI Fisher ({n_params} params) done in {time.time()-t1:.0f}s; "
        f"Hessian noise level {noise_level(F_full[tag]):.2e}")
    t1 = time.time()
    F_direct_full[tag] = -np.array(jax.hessian(loglkl_direct[tag])(cosmo_eft_fid))
    log(f"[{tag}] direct Fisher ({n_eft + 3} params) done in {time.time()-t1:.0f}s; "
        f"noise level {noise_level(F_direct_full[tag]):.2e}")

# ===== Gate 7: adding B(k) can only add information =====
# The P(k) data are a subset of the P(k)+B(k) data with zero cross-covariance, so
# F(P+B) - F(P) = F(B) + (priors of the Bk-only EFT parameters) must be PSD, up to the
# Hessian noise and the analytic-marginalization log-det term.
dF = psd_clip(F_full['P+B']) - psd_clip(F_full['P'])
w_dF = np.linalg.eigvalsh(sym(dF))
log(f"[Gate 7] min eig(F_PB - F_P) / max eig = {w_dF.min()/w_dF.max():+.2e} "
    f"(PSD up to noise: {'PASS' if w_dF.min()/w_dF.max() > -1e-4 else 'FAIL'})")

# --- Gauss-Newton Fisher: the object the marginalization is actually done with -------
# WHY THIS EXISTS. The autodiff Hessian above is PSD only to ~1e-7 of its largest eigenvalue,
# which in these (fractional) units is an ABSOLUTE floor of ~0.1 -- and the flattest directions
# of the 126-parameter Hessian are 91-100% concentrated in the 80 template amplitudes. So the
# amplitude block has real eigenvalues down at the noise floor, and marginalizing it means
# inverting noise. Measured consequences, before this fix: one-step vs nested marginalization
# to the AP block disagreed by 2.5-50% and moved erratically with the pseudo-inverse threshold;
# a ridge-regularized marginalization disagreed with the nested Schur chain by up to 6x on
# per-redshift F_AP; and the "flat gauge direction" that appeared in the AP block was an
# artifact (the explicit template-dilation direction has n^T F n = 766, i.e. sigma = 0.036 --
# well measured, because AP acts on the full anisotropic redshift-space spectrum including
# loops and resummation, not just on P_lin).
#
# WHAT THIS IS. J^T P J + (EFT priors), with J the Jacobian of the UNPROFILED model vector with
# respect to EVERY parameter -- all EFT coefficients written out explicitly rather than
# analytically marginalized inside the likelihood, plus the 80 amplitudes and the 25 growth
# parameters. It is PSD by construction with no noise floor, so every marginalization below is
# well posed. At a zero-residual expansion point (our data ARE the model at the fiducial) it is
# the exact Fisher information; it differs from the Hessian only by the log-determinant term of
# the analytic marginalization, which is a property of the profiled likelihood and not of the
# information (Gate 2b measures that difference).
F_gn, gn_idx, gn_names = {}, {}, {}
for tag, Li in L.items():
    t1 = time.time()
    mv, eft_names_sky = make_model_vector(Li)
    n_e = len(eft_names_sky)
    n_all = n_e * num_skies + n_knots + n_growth
    x0 = np.concatenate([np.tile([fiducial_nuisance[nm] for nm in eft_names_sky], num_skies),
                         np.ones(n_knots), growth_fid])
    # sanity: the unprofiled model vector at the fiducial must reproduce the data exactly
    # relative residual: the bispectrum entries are O(1e8), so an absolute norm says nothing
    y = np.array(Li.y_all)
    resid = np.abs(np.array(mv(jnp.array(x0))) - y) / np.maximum(np.abs(y), 1e-30)
    Jm = np.array(jax.jacfwd(mv)(jnp.array(x0)))
    Fm = Jm.T @ np.array(Li.p_all) @ Jm
    Fm = 0.5 * (Fm + Fm.T) + np.diag(eft_prior_precision(Li, eft_names_sky))
    F_gn[tag] = Fm
    gn_names[tag] = eft_names_sky
    gn_idx[tag] = dict(eft=np.arange(n_e * num_skies),
                       pk=np.arange(n_e * num_skies, n_e * num_skies + n_knots),
                       growth=np.arange(n_e * num_skies + n_knots, n_all))
    w = np.linalg.eigvalsh(sym(Fm))
    log(f"[{tag}] Gauss-Newton Fisher ({n_all} params: {n_e}x{num_skies} EFT + {n_knots} amps "
        f"+ {n_growth} growth) in {time.time()-t1:.0f}s; max RELATIVE residual at fiducial "
        f"{resid.max():.2e}; min_eig/max_eig = {w.min()/w.max():+.2e}")

# ===== Gate 8: the Gauss-Newton Fisher is PSD, so marginalizing it is well posed =====
for tag in L:
    w = np.linalg.eigvalsh(sym(F_gn[tag]))
    ok = w.min() / w.max() > -1e-12
    log(f"[Gate 8] {tag:4s} GN Fisher PSD: min_eig/max_eig = {w.min()/w.max():+.2e} "
        f"{'PASS' if ok else 'FAIL'}")

# ===== Gate 7 (on the GN Fisher): adding B(k) can only add information =====
# Comparable only on the parameters the two data vectors share: P(k) has no Bd1/Be5.
# The per-sky stride of F_gn is the DE-DUPLICATED name list. Indexing with pybird's raw
# list (17 names for P+B, 'Be1' twice) put every EFT index after the duplicate off by one
# per sky -- that, not physics, is why this gate first failed at -1.14.
sh_p, sh_pb = gn_names['P'], gn_names['P+B']
common = [n for n in sh_pb if n in sh_p]
sel = lambda names, idx: np.concatenate([
    np.array([i * len(names) + names.index(n) for i in range(num_skies) for n in common]),
    idx['pk'], idx['growth']])
A_ = F_gn['P'][np.ix_(sel(sh_p, gn_idx['P']), sel(sh_p, gn_idx['P']))]
B_ = F_gn['P+B'][np.ix_(sel(sh_pb, gn_idx['P+B']), sel(sh_pb, gn_idx['P+B']))]
w_dF = np.linalg.eigvalsh(sym(B_ - A_))
log(f"[Gate 7] GN: min eig(F_PB - F_P) / max eig = {w_dF.min()/w_dF.max():+.2e} "
    f"({'PASS' if w_dF.min()/w_dF.max() > -1e-10 else 'FAIL'})")

# %% [cell 17] sector split (EFT marginalized): template / growth / cross ------
# The split, and everything downstream, now uses the PSD Gauss-Newton Fisher.
split = {tag: sector_split(F_gn[tag], gn_idx[tag]['eft'], gn_idx[tag]['pk'], gn_idx[tag]['growth'])
         for tag in L}
split_hess = {tag: sector_split(F_full[tag], idx_eft, idx_pk, idx_growth) for tag in L}
for tag in L:
    a, b = split[tag]['F_phys_marg'], split_hess[tag]['F_phys_marg']
    log(f"[Gate 2b] {tag:4s} ||F_phys(GN) - F_phys(Hessian)|| / ||F_phys(GN)|| = "
        f"{np.linalg.norm(a-b)/np.linalg.norm(a):.2e} (EFT-marginalization log-det is in the "
        f"Hessian only)")
for tag in L:
    for nm, M in [('F_phys_marg', split[tag]['F_phys_marg']), ('A_marg (template, growth marg.)', split[tag]['A_marg']),
                  ('G_marg (growth, template marg.)', split[tag]['G_marg'])]:
        w = np.linalg.eigvalsh(sym(M))
        log(f"[Gate 2] {tag:4s} {nm:32s}: min_eig/|max_eig| = {w.min()/np.abs(w).max():+.2e}")

# ===== Gate 9: the marginalization is CONVERGED =====
# The question this whole section turns on is whether marginalizing 80 free template
# amplitudes out of the AP block is a well-posed operation. Two independent checks, both of
# which FAILED on the Hessian and must pass on the Gauss-Newton Fisher:
#   (a) nesting: marginalizing in stages (EFT -> amplitudes -> f, D, h_conv) must agree with
#       going straight from the full Fisher to the 12 AP parameters in one step;
#   (b) stability: the answer must not move when the pseudo-inverse threshold is swept.
idx_ap_g = np.array([i for iz in range(n_z_unique) for i in (idx_H[iz], idx_DA[iz])])
for tag in L:
    gi = gn_idx[tag]
    n_all = F_gn[tag].shape[0]
    ap_full = gi['growth'][idx_ap_g]
    rest = np.array([i for i in range(n_all) if i not in set(ap_full.tolist())])
    Dap = np.diag(growth_fid[idx_ap_g])
    ref = None
    log(f"[Gate 9] {tag:4s} marginalizing {n_all - len(ap_full)} parameters down to the 12 AP:")
    for rt in [1e-6, 1e-8, 1e-10, 1e-12]:
        one = Dap @ schur_marg(F_gn[tag], ap_full, rest, rt) @ Dap
        sp = sector_split(F_gn[tag], gi['eft'], gi['pk'], gi['growth'], rt)
        Gf = np.diag(growth_fid) @ sp['G_marg'] @ np.diag(growth_fid)
        others = np.array([i for i in range(n_growth) if i not in set(idx_ap_g.tolist())])
        multi = schur_marg(Gf, idx_ap_g, others, rt)
        if ref is None:
            ref = one
        sig = np.sort(sigmas(one, big_var=1e6))
        log(f"          rtol {rt:.0e}: nested vs one-step {np.linalg.norm(one-multi)/np.linalg.norm(one):.2e}"
            f"   drift vs rtol=1e-6 {np.linalg.norm(one-ref)/np.linalg.norm(ref):.2e}"
            f"   sigma(AP) {sig[0]:.4f}..{sig[-1]:.4f}")

# --- a WELL-POSED model-independent template: the smoothness prior --------------------
# Gate 9 above is not a numerical failure to be tuned away, it is a statement about the model:
# with 80 unconstrained amplitudes the AP parameters are genuinely degenerate with the
# template, so their marginal depends on where one declares a direction "unconstrained" and no
# amount of numerical care makes it well defined. (This is why the sampled analysis in
# mi_model.py does not use 80 free knots either.)
#
# The fix is a PRIOR, and the one adopted here is the project's own: a Gaussian prior on the
# SECOND DIFFERENCES of ln a with weight LAMBDA_SMOOTH, plus a broad amplitude prior
# SIGMA_LNA. It penalizes jaggedness that no linear power spectrum has, while leaving the
# broadband and the BAO wiggles free (the second-difference operator has the smooth AND the
# linear-in-ln k modes in its null space). With it, A + P_a is positive definite, the Schur
# complement is exact, and Gate 10 confirms the answer is converged and prior-insensitive.
LAMBDA_SMOOTH = float(os.environ.get('LAMBDA_SMOOTH', '200.'))
SIGMA_LNA = float(os.environ.get('SIGMA_LNA', '0.5'))

def amp_prior_precision(n, lam=LAMBDA_SMOOTH, sig=SIGMA_LNA):
    """Precision matrix for the 80 template amplitudes: lam * D2^T D2 + I / sig^2, with D2 the
    second-difference operator on the (log-spaced) knots. a_fid = 1, so to first order a prior
    on ln a is a prior on a - 1 and this is the Fisher-level precision directly."""
    D2 = np.zeros((n - 2, n))
    for i in range(n - 2):
        D2[i, i], D2[i, i + 1], D2[i, i + 2] = 1.0, -2.0, 1.0
    return lam * (D2.T @ D2) + np.eye(n) / sig**2

P_amp = amp_prior_precision(n_knots)
split_sm, growth_ln_sm = {}, {}
for tag in L:
    gi = gn_idx[tag]
    Fp = F_gn[tag].copy()
    Fp[np.ix_(gi['pk'], gi['pk'])] += P_amp
    split_sm[tag] = sector_split(Fp, gi['eft'], gi['pk'], gi['growth'])
    Dg = np.diag(growth_fid)
    growth_ln_sm[tag] = {'template fixed': psd_clip(Dg @ split_sm[tag]['G_blk'] @ Dg),
                         'template free (smoothness prior)':
                             psd_clip(Dg @ split_sm[tag]['G_marg'] @ Dg)}

# ===== Gate 10: with the prior, the marginalization IS well posed =====
log(f"[Gate 10] template smoothness prior lambda={LAMBDA_SMOOTH:g}, sigma_lna={SIGMA_LNA:g}")
others_g = np.array([i for i in range(n_growth) if i not in set(idx_ap_g.tolist())])
for tag in L:
    gi = gn_idx[tag]
    Fp = F_gn[tag].copy(); Fp[np.ix_(gi['pk'], gi['pk'])] += P_amp
    n_all = Fp.shape[0]
    ap_full = gi['growth'][idx_ap_g]
    rest = np.array([i for i in range(n_all) if i not in set(ap_full.tolist())])
    Dap = np.diag(growth_fid[idx_ap_g])
    base = None
    for rt in [1e-6, 1e-8, 1e-10, 1e-12]:
        one = Dap @ schur_marg(Fp, ap_full, rest, rt) @ Dap
        sp = sector_split(Fp, gi['eft'], gi['pk'], gi['growth'], rt)
        multi = schur_marg(np.diag(growth_fid) @ sp['G_marg'] @ np.diag(growth_fid),
                           idx_ap_g, others_g, rt)
        if base is None: base = one
        sig = np.sort(sigmas(one, big_var=1e6))
        log(f"          {tag:4s} rtol {rt:.0e}: nested vs one-step "
            f"{np.linalg.norm(one-multi)/np.linalg.norm(one):.2e}   drift {np.linalg.norm(one-base)/np.linalg.norm(base):.2e}"
            f"   sigma(AP) {sig[0]:.4f}..{sig[-1]:.4f}")
# prior sensitivity: the answer must not be created by the prior
for lam in [50., 200., 800.]:
    Pa = amp_prior_precision(n_knots, lam=lam)
    row = []
    for tag in L:
        gi = gn_idx[tag]
        Fp = F_gn[tag].copy(); Fp[np.ix_(gi['pk'], gi['pk'])] += Pa
        sp = sector_split(Fp, gi['eft'], gi['pk'], gi['growth'])
        Gl = psd_clip(np.diag(growth_fid) @ sp['G_marg'] @ np.diag(growth_fid))
        c = fisher_to_cov(schur_marg(Gl, idx_ap_g, others_g), big_var=1e6)
        fap = [np.sqrt(c[2*k,2*k] + c[2*k+1,2*k+1] + 2*c[2*k,2*k+1]) for k in range(n_z_unique)]
        row.append(f"{tag} F_AP " + " ".join(f"{x:.4f}" for x in fap))
    log(f"          lambda={lam:6.0f}: " + " | ".join(row))

# %% [cell 19] cosmology projections; Gates 3, 4, 5b ---------------------------
J_pk = np.array(jax.jacobian(cosmo_to_amps)(cosmo_fid_vec))          # [80 x 3]
J_growth_2d = np.array(jax.jacobian(cosmo_to_growth)(cosmo_fid_2d))  # [n_growth x 2]
J_growth_3d = np.zeros((n_growth, 3))
J_growth_3d[:, 0] = J_growth_2d[:, 0]
J_growth_3d[:, 2] = J_growth_2d[:, 1]

pieces = {}
for tag in L:
    F_direct_3 = schur_marg(F_direct_full[tag], np.arange(n_eft, n_eft + 3), np.arange(n_eft))
    pieces[tag] = cosmo_pieces(split[tag], J_pk, J_growth_3d, F_direct_3)
    rel = (np.linalg.norm(pieces[tag]['Combined'] - F_direct_3) / np.linalg.norm(F_direct_3))
    log(f"[Gate 3] {tag:4s} ||F_combined - F_direct|| / ||F_direct|| = {rel:.3e}")
    sig_direct = sigmas(F_direct_3)
    worst = min(((sigmas(Fm) / sig_direct).min(), nm) for nm, Fm in pieces[tag].items()
                if nm not in ('Direct',))
    log(f"[Gate 4] {tag:4s} " + ("PASS" if worst[0] >= 0.999 else "FAIL") +
        f" (tightest sub-piece vs Direct: {worst[1]} at {worst[0]:.4f}x; "
        f"1.0 = exactly as tight, < 1 impossible except for numerical noise)")
    for nm, Fm, Fc in [('p(a) p(g|a)', 'P(k) marginal', 'Growth+AP | P(k)'),
                       ('p(g) p(a|g)', 'Growth+AP marginal', 'P(k) | Growth+AP')]:
        r = np.linalg.norm(pieces[tag][Fm] + pieces[tag][Fc] - pieces[tag]['Combined']) / np.linalg.norm(pieces[tag]['Combined'])
        log(f"[Gate 5b] {tag:4s} {nm}: ||marg + cond - combined|| / ||combined|| = {r:.2e}")

log("--- cosmology sigmas (omega_cdm, lnAs, h); EFT marginalized; O(1) = unconstrained ---")
log(f"{'':24s} " + " ".join(f"{tag+' '+p:>14s}" for p in ['ocdm', 'lnAs', 'h'] for tag in L))
for nm in pieces['P']:
    row = [sigmas(pieces[tag][nm])[i] for i in range(3) for tag in L]
    log(f"{nm:24s} " + " ".join(f"{v:14.3e}" for v in row))

# %% [cell 21] the growth/AP sector: P vs P+B -----------------------------------
# Two conditionings of the growth block, both with the EFT parameters marginalized:
#   G_blk  = growth | template  (template held at the fiducial: the usual fixed-shape AP/RSD
#            measurement, the one a known-cosmology analysis performs);
#   G_marg = growth, template marginalized (the model-independent statement: the isotropic
#            dilation shared by all redshifts is degenerate with a rigid dilation of the free
#            template and with h_conv, so only F_AP = H D_A and the RELATIVE isotropic scales
#            between redshifts survive -- the corner plots show that directly).
# Fractional (ln) parametrization so that every redshift and both AP parameters are on the
# same footing: F_ln = diag(g_fid) F diag(g_fid).
growth_ln = {}
for tag in L:
    Dg = np.diag(growth_fid)
    growth_ln[tag] = {'template fixed': psd_clip(Dg @ split[tag]['G_blk'] @ Dg),
                      'template free': psd_clip(Dg @ split[tag]['G_marg'] @ Dg)}

# Two DIFFERENT thresholds, kept separate because conflating them corrupts the comparison.
#
# COV_CAP exists only so that a covariance exists at all: the template-free block has
# directions that are flat for practical purposes, and a relative eigenvalue threshold does
# not catch them (they are tiny in absolute terms yet far above rtol * lambda_max, giving
# widths of 1e4-1e5). The cap must be LARGE ENOUGH not to truncate anything real: capping at
# 1.0 pulled sigma(f) at z=1.491 from its true 3.00 down to 0.71 for P(k) while leaving
# P(k)+B(k) at 0.78, which inverted the comparison at exactly the redshift where the
# bispectrum helps most, and broke the ordering that Gate 7 guarantees. At 5.0 nothing in the
# template-fixed block is truncated (0 capped directions) and the ordering holds; verified in
# make_fisher_pb_fig._check_ordering for every block that is plotted.
COV_CAP = 5.0
# MEASURED is the physical threshold: a fractional width worse than 100% is no measurement.
# It is used for COUNTING measured directions and for flagging table entries, never for
# building a covariance.
MEASURED = 1.0

def growth_sigma_table(which, sigma_max=COV_CAP):
    """Fractional 1-sigma widths of f, H/H0, DA*H0 (and F_AP = H*DA, the AP anisotropy,
    which is invariant under an isotropic dilation) per redshift."""
    out = {}
    for tag in L:
        cov = fisher_to_cov_capped(growth_ln[tag][which], sigma_max)
        sig = np.sqrt(np.diag(cov))
        fap = [np.sqrt(max(cov[iH, iH] + cov[iD, iD] + 2 * cov[iH, iD], 0.0))
               for iH, iD in zip(idx_H, idx_DA)]
        out[tag] = {'f': sig[idx_f], 'H': sig[idx_H], 'DA': sig[idx_DA],
                    'FAP': np.array(fap),
                    'h_conv': sig[-1], 'D': sig[3 * n_z_unique:3 * n_z_unique + n_z_unique]}
    return out

# A width is only a statement about the DATA if it does not move when the cap is relaxed.
# Where the block is degenerate (the template-free case) it does move -- the width is then a
# statement about the truncation -- so those entries are flagged with '~' and no ratio is
# quoted for them. `growth_spectrum.png` carries the cap-free version of the comparison.
sig_tab = {which: growth_sigma_table(which) for which in ['template fixed', 'template free']}
sig_tab_loose = {which: growth_sigma_table(which, sigma_max=5 * COV_CAP)
                 for which in ['template fixed', 'template free']}

def _cell(a, b, a_loose, b_loose, measured=MEASURED, tol=0.10):
    """'P  P+B  ratio'. A width that shifts by more than `tol` when the cap is relaxed 5x is
    truncation-limited, not data-limited: it is prefixed '~' and its ratio is suppressed."""
    def fmt(v, v_loose):
        if abs(v_loose - v) > tol * v:      # still growing with the cap: not a measurement
            return f"~{v:6.3g}", True
        if v >= measured:                   # resolved, but too wide to be a measurement
            return f"{v:7.3g}", True
        return f"{v:7.4f}", False
    fa, la = fmt(a, a_loose)
    fb, lb = fmt(b, b_loose)
    r = "   -  " if (la or lb) else f"{b/a:5.2f}x"
    return f"{fa:>7s} {fb:>7s} {r:>6s}"
for which in sig_tab:
    log(f"--- growth/AP sector, {which}: fractional sigma (P / P+B / ratio); "
        f"'~' = still growing with the cap; no ratio quoted unless BOTH are measured "
        f"(< {MEASURED:g}) ---")
    log(f"{'z':>6s} " + " ".join(f"{q:>23s}" for q in ['f', 'H/H0', 'DA*H0', 'F_AP=H*DA']))
    for iz, z in enumerate(zeff_unique):
        log(f"{z:6.3f} " + "  ".join(
            _cell(sig_tab[which]['P'][q][iz], sig_tab[which]['P+B'][q][iz],
                  sig_tab_loose[which]['P'][q][iz], sig_tab_loose[which]['P+B'][q][iz])
            for q in ['f', 'H', 'DA', 'FAP']))
    log("h_conv: " + _cell(sig_tab[which]['P']['h_conv'], sig_tab[which]['P+B']['h_conv'],
                           sig_tab_loose[which]['P']['h_conv'],
                           sig_tab_loose[which]['P+B']['h_conv']))

# Which COMBINATIONS survive when the template is marginalized? The per-parameter sigmas
# above cannot answer that -- the block is dominated by a few near-flat directions -- so the
# eigen-structure of the AP sub-block (ln H, ln DA at every z) is printed directly. The
# common isotropic dilation is degenerate with a rigid dilation of the free template and with
# h_conv (Gate 5a), so what survives is the anisotropy at each z and the RELATIVE isotropic
# scales between redshifts.
# Cap-free summary: eigenvalues of a Fisher need no truncation, so counting how many
# directions of the growth block are measured better than SIGMA_MAX is a statement about the
# data alone. This is what growth_spectrum.png plots.
log("--- growth/AP directions measured better than "
    f"{MEASURED:g} (fractional), of {n_growth} ---")
for which in ['template fixed', 'template free']:
    row = []
    for tag in L:
        w = np.linalg.eigvalsh(sym(growth_ln[tag][which]))
        row.append(f"{tag} {int((w > 1.0 / MEASURED**2).sum()):d}")
    log(f"  {which:15s}: " + ", ".join(row))

ap_idx = np.array([i for iz in range(n_z_unique) for i in (idx_H[iz], idx_DA[iz])])
ap_names = [n for z in zeff_unique for n in (f'lnH({z:.2f})', f'lnDA({z:.2f})')]
for which in ['template fixed', 'template free']:
    for tag in L:
        cov_ap = fisher_to_cov_capped(growth_ln[tag][which], COV_CAP)[np.ix_(ap_idx, ap_idx)]
        log(f"AP sub-block eigen-structure, {which}, {tag} "
            f"(best-measured first, coefficients |v|>0.15):")
        report_combos(np.linalg.inv(cov_ap), ap_names, MEASURED, log=log, top=4)

# the two degeneracies of the MI block (v3 Gate 5a) survive the bispectrum? B_tree ~ P_lin^2
# is invariant under a -> a e^s, D -> D e^{-s/2} exactly, and a rigid template dilation is
# still absorbed by h_conv, so both sector marginals stay empty of A_s and h for P+B too.
for tag in L:
    G_blk, G_marg = split[tag]['G_blk'], split[tag]['G_marg']
    log(f"[Gate 5a] {tag:4s} h_conv information: template fixed {G_blk[-1, -1]:.3e}, free "
        f"{G_marg[-1, -1]:.3e} -> {G_marg[-1, -1]/G_blk[-1, -1]:.1e} survives; "
        f"lnAs in P(k) marginal: sigma = {sigmas(pieces[tag]['P(k) marginal'])[1]:.2f}")

# %% [cell 23] figures + saved results ----------------------------------------
results = dict(zeff_unique=np.array(zeff_unique), growth_fid=growth_fid, growth_names=np.array(growth_names),
               idx_f=np.array(idx_f), idx_H=np.array(idx_H), idx_DA=np.array(idx_DA),
               params_fid=params_fid, knots_h=knots_h, kk_fid_h=kk_fid_h, template_mpc=np.array(template_mpc),
               J_pk=J_pk, J_growth_3d=J_growth_3d, k123=k123, chi2_fid=np.array([chi2_fid[t] for t in L]))
for tag, key in [('P', 'p'), ('P+B', 'pb')]:
    results[f'F_full_{key}'] = F_full[tag]
    results[f'F_direct_full_{key}'] = F_direct_full[tag]
    for nm, M in split[tag].items():
        results[f'{nm}_{key}'] = M
    for nm, M in pieces[tag].items():
        results[f"cosmo_{nm.replace(' ', '_').replace('|', 'given').replace('(', '').replace(')', '').replace('+', '')}_{key}"] = M
    for which, M in growth_ln[tag].items():
        results[f"growth_ln_{which.replace(' ', '_')}_{key}"] = M
    results[f'F_gn_{key}'] = F_gn[tag]
    for nm, M in split_sm[tag].items():
        results[f'sm_{nm}_{key}'] = M
    for which in sig_tab:
        for q, v in sig_tab[which][tag].items():
            results[f"sig_{q}_{which.replace(' ', '_')}_{key}"] = v
np.savez(os.path.join(figdir, 'fisher_pb_results.npz'), **results)
log(f"Results saved to {figdir}/fisher_pb_results.npz")

if not SMOKE:
    from make_fisher_pb_fig import make_figures
    make_figures(results, figdir, log=log)
log("DONE.")
