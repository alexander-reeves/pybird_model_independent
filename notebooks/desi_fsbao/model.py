"""Model-independent (MI) EFTofLSS fit of DESI DR1 full shape + post-reconstruction BAO.

The model
---------
The pybird likelihood (80-knot loop emulator, AP, survey window, joint FS+BAO covariance)
depends on cosmology only through the physical block phi, sampled in log. Its blocks, in this
order (`MIModel.idx[name]` indexes each in the sampled vector x = [eft, phi]):

  lna    n_amp   the linear power spectrum at z_N, the farthest effective redshift:
                 P_lin(k, z_N) = a(k) T(k), T the fiducial P_lin(k, z_N) in Mpc^3 on the
                 emulator's 80 knots in 1/Mpc. ln a lives on log-spaced nodes, is cubic in ln k
                 onto the knots and constant outside the nodes. a(k) carries the SHAPE and the
                 AMPLITUDE at z_N.
  f      N       growth rate f(z_i) at each unique effective redshift
  H      N       H(z_i)/H0
  DA     N       D_A(z_i) H0
  D      N-1     D(z_i)/D(z_N) for the N-1 nearer redshifts; D(z_N)/D(z_N) = 1 by definition
  hconv  1       units bridge: the template is in 1/Mpc, the data in h/Mpc; kk = k*/h_conv
  bao    n_bao   post-reconstruction BAO dilations, alpha_par and alpha_per per anisotropic sample,
                 alpha_iso per isotropic one (BGS and QSO in DR1)

so that P_lin(k, z_i) = a(k) T(k) [D(z_i)/D(z_N)]^2. The data measure the linear amplitude at
each redshift, i.e. N numbers; the model has exactly N amplitude parameters (the normalization
of a and the N-1 ratios). The legacy parametrization (../mi_model.py) referred the ratios to
z_ref = 5 and floated N of them, so one combination (ln a +1 everywhere, ln D -1/2 everywhere)
was an exact flat direction held only by the prior.

The priors (Gaussian in phi, centred on the fiducial cosmology)
--------------------------------------------------------------
  shape       ln a minus its node mean: width s_lna per node plus the smoothness penalty
              smooth_lambda * sum (second differences of ln a)^2 (blind to a constant and a tilt)
  amplitudes  u_i = ln of the linear amplitude at z_i = (node mean of ln a) + 2 ln D(z_i)/D(z_N),
              u_N = node mean of ln a: independent widths s_lnP, plus the common s_lna/sqrt(n_amp)
              that the node prior puts on the mean
  f, H, DA    s_lng;   h_conv  s_lnh;   alphas  s_lnbao
With s_lnP = 2 s_lng this is exactly the prior the legacy z_ref = 5 model implied for every
quantity the likelihood depends on (shape, u_i, f, H, DA, h_conv, alphas; gates.py G5): the two
models then have the same posterior on all of them, and only the redundant direction is gone.
Note that the per-node prior must NOT act on the node mean here: with z_ref = z_N that mean is
the amplitude at z_N, and s_lna per node would hold it at s_lna/sqrt(n_amp) = 0.065, i.e. a 3%
prior on sigma_8(z_N) in place of the 30% the legacy model had.

The direct model
----------------
The direct model is the MI likelihood composed with the map theta -> phi(theta) (`phi_ln`),
for a selectable parameter set (lcdm3 = (omega_cdm, ln10^10 A_s, h), lcdm5, w0wa7). The linear
spectrum comes from CosmoPower (LCDM) at z_early = 5, where dark energy is negligible, and is
grown to z_N with the early-normalized growth factor of pybird.symbolic, so that w0/wa enter
through growth and geometry only:
    P_lin(k, z; theta) = P_CPJ(k, z_early; theta) [D_e(z; theta) / D_e(z_early; theta, LCDM)]^2.
This is exactly the direct model of the legacy pipeline (checked by gates.py).
"""
import os
from copy import deepcopy

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from pybird import config as _pb_config
_pb_config.set_jax_enabled(True)
from pybird.likelihood import Likelihood
from pybird.symbolic import (f as sym_f, Hubble as sym_H, DA as sym_DA, _D as sym_D_early,
                             comoving_distance as sym_DM, rs_drag as sym_rd, c_light as C_LIGHT)
from pybird import emulator as _pb_emu
from pybird.jax_special import interp1d as _pb_interp1d
from cosmopower_jax.cosmopower_jax import CosmoPowerJAX as CPJ

from sampling import log

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

# ------------------------------------------------------------------------------------------
# Emulator input: cubic instead of piecewise-linear interpolation of ln P_lin onto the 80 knots.
# PyBird's own linear interpolant has a one-sided slope at the knots, which put a ~20% jump in
# the curvature of the log-likelihood along h_conv exactly at h_conv = h_fid. Cubic is C^2 and
# identical at the knots. Applied as a monkeypatch so the installed pybird is untouched.
# ------------------------------------------------------------------------------------------
def _make_params_cubic(self, kk, pk, f=1.0, time=False, ir=False, pca=False):
    pk_max = jnp.max(pk)
    self.pk_max = jnp.array(pk_max)
    logpk = _pb_interp1d(jnp.log(kk), jnp.log(pk / self.pk_max.reshape(-1, 1)), axis=-1, kind='cubic')(self.logknots)
    self.params = jnp.concatenate([logpk, jnp.array([[pk_max]]), jnp.array([[f]])], axis=1) if ir else logpk
    return logpk

_pb_emu.Emulator.make_params = _make_params_cubic


# ------------------------------------------------------------------------------------------
# Cosmological parametrizations of the direct model
# ------------------------------------------------------------------------------------------
COSMO_KEYS = ['omega_cdm', 'omega_b', 'n_s', 'lnAs', 'h', 'w0', 'wa', 'fEDE', 'log10z_c', 'thetai_scf']
COSMO_MODELS = {'lcdm3': ['omega_cdm', 'lnAs', 'h'],
                'lcdm5': ['omega_cdm', 'omega_b', 'n_s', 'lnAs', 'h'],
                'w0wa7': ['omega_cdm', 'omega_b', 'n_s', 'lnAs', 'h', 'w0', 'wa'],
                # early dark energy through the CosmoPower ede-v2 emulators (ede.py); n_s fixed
                'ede7': ['omega_cdm', 'omega_b', 'lnAs', 'h', 'fEDE', 'log10z_c', 'thetai_scf'],
                # the same emulators at their LCDM limit (fEDE = 0.001): the EDE engine's own LCDM
                'lcdm4e': ['omega_cdm', 'omega_b', 'lnAs', 'h']}
EDE_KEYS = ['fEDE', 'log10z_c', 'thetai_scf']
EDE_ENGINE = {'ede7', 'lcdm4e'}                  # models whose linear spectrum and r_d come from ede.py
EDE_FIXED = {'fEDE': 0.001, 'log10z_c': 3.562, 'thetai_scf': 2.83}
COSMO_TEX = {'omega_cdm': r'\omega_{\rm cdm}', 'omega_b': r'\omega_b', 'n_s': r'n_s',
             'lnAs': r'\ln(10^{10}A_s)', 'h': r'h', 'w0': r'w_0', 'wa': r'w_a',
             'fEDE': r'f_{\rm EDE}', 'log10z_c': r'\log_{10} z_c', 'thetai_scf': r'\theta_i'}
# flat boxes: CosmoPower's training range for the template parameters, DESI 2024 for (w0, wa), the
# ede-v2 training box for EDE (whose lnAs range is also narrower)
COSMO_BOX = {'omega_cdm': (0.08, 0.16), 'lnAs': (2.0, 4.0), 'h': (0.5, 0.85),
             'omega_b': (0.0197, 0.0238), 'n_s': (0.84, 1.10), 'w0': (-3.0, 1.0), 'wa': (-3.0, 2.0),
             'fEDE': (0.001, 0.5), 'log10z_c': (3.0, 4.3), 'thetai_scf': (0.1, 3.1)}
EDE_BOX = {'lnAs': (2.5, 3.5)}
BBN_OMEGA_B = (0.02218, 0.00055)     # Gaussian prior on omega_b of DESI 2024 (Schoeneberg 2024)
COSMO_SCALES = {'omega_cdm': 0.005, 'omega_b': 0.0005, 'n_s': 0.01, 'lnAs': 0.05, 'h': 0.01, 'w0': 0.1, 'wa': 0.3,
                'fEDE': 0.05, 'log10z_c': 0.2, 'thetai_scf': 0.5}


class PkLinCPJ:
    """P_lin(k [1/Mpc], z) in Mpc^3 from CosmoPower-JAX, log-log interpolated onto k."""

    def __init__(self):
        self.cpj = CPJ(probe='mpk_lin')
        self.k_modes = jnp.array(self.cpj.modes)

    def __call__(self, k_mpc, c, z):
        d = {'omega_b': jnp.atleast_1d(c['omega_b']), 'omega_cdm': jnp.atleast_1d(c['omega_cdm']),
             'h': jnp.atleast_1d(c['h']), 'ln10^{10}A_s': jnp.atleast_1d(c['lnAs']),
             'n_s': jnp.atleast_1d(c['n_s']), 'z': jnp.atleast_1d(z)}
        return jnp.exp(jnp.interp(jnp.log(k_mpc), jnp.log(self.k_modes), jnp.log(self.cpj.predict(d))))


def Omega_m(c):
    """(omega_cdm + omega_b) / h^2 (massless neutrinos throughout, as the growth and BAO maps)."""
    return (c['omega_cdm'] + c['omega_b']) / c['h']**2


def rd_h(c):
    """h r_d in Mpc/h (analytic fit of 2404.03002)."""
    return sym_rd(h=c['h'], omega_m=c['omega_cdm'] + c['omega_b'], omega_b=c['omega_b'])


class MIModel:
    """The model-independent and the direct likelihood on one pybird Likelihood.

    data : dict from settings.load_data (config, skies, zeff per sky, BAO fiducials)
    prior: {'s_lna', 'smooth_lambda', 's_lnP', 's_lng', 's_lnh', 's_lnbao'} (module docstring)
    nodes: {'k_range': (kmin, kmax) in h/Mpc at h_fid, 'spacing': in ln k}
    """

    PRIOR_DEFAULT = {'s_lna': 0.5, 'smooth_lambda': 200.0, 's_lnP': 0.6, 's_lng': 0.3, 's_lnh': 0.05, 's_lnbao': 0.3}

    def __init__(self, data, cosmo_fid, prior=None, nodes=None, z_early=5.0, get_maxlkl=True, verbose=True):
        self.cosmo_fid = {'omega_cdm': cosmo_fid['omega_cdm'], 'omega_b': cosmo_fid['omega_b'], 'n_s': cosmo_fid['n_s'],
                          'lnAs': cosmo_fid['ln10^{10}A_s'], 'h': cosmo_fid['h'], 'w0': -1.0, 'wa': 0.0}
        self.h_fid = self.cosmo_fid['h']
        self.z_early = z_early
        self.prior = dict(self.PRIOR_DEFAULT, **(prior or {}))

        # ---- redshifts: z_N is the farthest unique effective redshift, the growth reference
        self.skies = list(data['skies'])
        self.zeff_sky = list(data['zeff'])
        self.num_skies = len(self.skies)
        self.z = sorted(set(self.zeff_sky))
        self.N = len(self.z)
        self.z_N = self.z[-1]
        self.sky_to_z = [self.z.index(z) for z in self.zeff_sky]

        # ---- the pybird likelihood (data, window, covariance, EFT priors)
        cfg = deepcopy(data['cfg']); cfg['get_maxlkl'] = get_maxlkl
        self.L = Likelihood(cfg, verbose=verbose)
        assert self.L.nsky == self.num_skies
        self.eft_free = [p for p, pr in cfg['eft_prior'].items() if pr['type'] in ('flat', 'gauss', 'lognormal')]
        self.eft_names_flat = self.eft_free * self.num_skies
        self.eft_labels = [f"{n}_{s}" for s in self.skies for n in self.eft_free]
        self.n_eft = len(self.eft_names_flat)

        # ---- BAO block (ragged: 1 dilation for an isotropic sample, 2 for an anisotropic one)
        self.bao = [dict(b) for b in data['bao']] if data.get('bao') else []
        self.bao_iso = [bool(b['iso']) for b in self.bao]
        self.bao_slice, n = [], 0
        for iso in self.bao_iso:
            self.bao_slice.append(slice(n, n + (1 if iso else 2))); n += 1 if iso else 2
        self.n_bao = n
        assert not self.bao or [b['zeff'] for b in self.bao] == self.z

        # ---- template: fiducial P_lin(k, z_N) in Mpc^3 on the 80 knots (fixed grid in 1/Mpc)
        self.pklin = PkLinCPJ()
        self.knots_h = np.load(os.path.join(ROOT, 'pybird', 'emu_data', 'knots.npy'))
        self.knots_mpc = jnp.array(self.knots_h * self.h_fid)
        self.lnk_knots = jnp.log(self.knots_mpc)
        self.T_knots = self.plin_zN(self.knots_mpc, self.cosmo_fid)

        # ---- amplitude nodes: log-spaced in 1/Mpc over nodes['k_range'] (h/Mpc at h_fid)
        nodes = dict({'k_range': (1e-4, 0.7), 'spacing': 0.15}, **(nodes or {}))
        lo, hi = np.log(nodes['k_range'][0] * self.h_fid), np.log(nodes['k_range'][1] * self.h_fid)
        self.n_amp = int(np.round((hi - lo) / nodes['spacing'])) + 1
        self.nodes_lnk = jnp.array(np.linspace(lo, hi, self.n_amp))
        self.nodes_mpc = jnp.exp(self.nodes_lnk)
        self.nodes_h = np.array(self.nodes_mpc) / self.h_fid
        self.T_nodes = self.plin_zN(self.nodes_mpc, self.cosmo_fid)
        self.node_settings = nodes

        # ---- parameter layout
        sizes = [('eft', self.n_eft), ('lna', self.n_amp), ('f', self.N), ('H', self.N), ('DA', self.N),
                 ('D', self.N - 1), ('hconv', 1), ('bao', self.n_bao)]
        self.idx, i0 = {}, 0
        for name, n in sizes:
            self.idx[name] = np.arange(i0, i0 + n); i0 += n
        self.n_mi = i0
        self.n_phys = self.n_mi - self.n_eft
        self.pidx = {k: v - self.n_eft for k, v in self.idx.items() if k != 'eft'}   # into phi
        zl = [f'{z:.2f}' for z in self.z]
        self.phys_names = ([f'ln a({k:.3g})' for k in self.nodes_h] + [f'ln f({z})' for z in zl] + [f'ln H/H0({z})' for z in zl]
                           + [f'ln DA H0({z})' for z in zl] + [f'ln D({z})/D({zl[-1]})' for z in zl[:-1]] + ['ln h_conv']
                           + [nm for b, iso in zip(self.bao, self.bao_iso) for nm in
                              ([f"ln alpha_iso({b['zeff']:.2f})"] if iso else [f"ln alpha_par({b['zeff']:.2f})", f"ln alpha_per({b['zeff']:.2f})"])])
        self.names = self.eft_labels + self.phys_names

        # ---- direct model: baseline lcdm3; set_cosmo_model switches. extra_loglike(c) is an external
        # likelihood on the cosmology (e.g. cmb.loglike), part of the direct model's prior and therefore
        # of both the direct chains and the projections
        self.extra_loglike = None
        self.set_cosmo_model('lcdm3')

        # ---- prior: Gaussian in phi, centred on the fiducial
        self.phi_fid = np.asarray(self.phi_ln_full(self.cosmo_fid))
        assert np.abs(self.phi_fid[self.pidx['lna']]).max() < 1e-12       # a = 1 at the fiducial
        self.prior_mean = self.phi_fid.copy()
        self.prior_prec = self._prior_precision()

        if verbose:
            log(f"MIModel: {self.num_skies} skies {self.skies}, N = {self.N} redshifts {self.z} (reference z_N = {self.z_N}); "
                f"{self.n_eft} EFT ({self.eft_free} per sky) + {self.n_amp} ln a nodes "
                f"[{self.nodes_h[0]:.3g}, {self.nodes_h[-1]:.3g}] h/Mpc + {self.N} f + {self.N} H + {self.N} DA + {self.N - 1} D "
                f"+ h_conv + {self.n_bao} BAO = {self.n_mi} MI parameters; direct {self.n_eft} + {self.n_cosmo}")
            log(f"  prior {self.prior}")

    # ------------------------------------------------------------------------------------
    # prior
    # ------------------------------------------------------------------------------------
    def amplitude_matrix(self):
        """B with u = B phi: u_i = node mean of ln a + 2 ln D(z_i)/D(z_N), u_N = node mean of ln a."""
        B = np.zeros((self.N, self.n_phys))
        B[:, self.pidx['lna']] = 1.0 / self.n_amp
        B[np.arange(self.N - 1), self.pidx['D']] = 2.0
        return B

    def _prior_precision(self):
        p, n, N = self.prior, self.n_amp, self.N
        Pi = np.zeros((self.n_phys, self.n_phys))
        ia = self.pidx['lna']
        # shape: the per-node width on the complement of the constant mode + smoothness
        D2 = np.zeros((n - 2, n))
        for i in range(n - 2): D2[i, i:i + 3] = (1., -2., 1.)
        Pi[np.ix_(ia, ia)] = (np.eye(n) - np.ones((n, n)) / n) / p['s_lna']**2 + p['smooth_lambda'] * D2.T @ D2
        # per-redshift amplitudes: independent s_lnP plus the node prior's common s_lna/sqrt(n)
        B = self.amplitude_matrix()
        Su = p['s_lna']**2 / n * np.ones((N, N)) + p['s_lnP']**2 * np.eye(N)
        Pi += B.T @ np.linalg.inv(Su) @ B
        for blk in ('f', 'H', 'DA'):
            Pi[self.pidx[blk], self.pidx[blk]] = p['s_lng']**-2
        Pi[self.pidx['hconv'], self.pidx['hconv']] = p['s_lnh']**-2
        Pi[self.pidx['bao'], self.pidx['bao']] = p['s_lnbao']**-2
        return Pi

    def prior_widths(self):
        """Per-parameter 1-sigma of the prior (diagonal of the prior covariance)."""
        return np.sqrt(np.diag(np.linalg.inv(self.prior_prec)))

    # ------------------------------------------------------------------------------------
    # cosmology -> physical block (the direct model)
    # ------------------------------------------------------------------------------------
    def set_cosmo_model(self, model, box=None, gauss=None):
        """Free parameters of the direct model: a key of COSMO_MODELS or a list of COSMO_KEYS; the
        others stay at the fiducial (omega_b, n_s) or at LCDM (w0 = -1, wa = 0). `gauss`:
        {key: (mean, sigma)} Gaussian priors (e.g. BBN on omega_b). w0 + wa < 0 when both are free."""
        keys = list(COSMO_MODELS[model]) if isinstance(model, str) else list(model)
        assert all(k in COSMO_KEYS for k in keys), keys
        self.cosmo_model = model if isinstance(model, str) else '+'.join(keys)
        self.cosmo_keys, self.n_cosmo = keys, len(keys)
        self.cosmo_names = [('logA' if k == 'lnAs' else k) for k in keys]
        self.cosmo_labels = [COSMO_TEX[k] for k in keys]
        # the EDE engine carries its three parameters in every cosmology dict (fixed at the LCDM limit
        # unless free); their presence is what routes phi_ln_full through the emulators
        ede = self.cosmo_model in EDE_ENGINE or any(k in EDE_KEYS for k in keys)
        self.cosmo_fixed = dict(self.cosmo_fid, **(EDE_FIXED if ede else {}))
        if ede: self.ede_emulator()                     # build it here, never under a jit trace
        self.cosmo_box = {k: tuple((box or {}).get(k, (EDE_BOX if ede else {}).get(k, COSMO_BOX[k]))) for k in keys}
        self.cosmo_gauss = dict(gauss or {})
        self.theta_fid = np.array([self.cosmo_fixed[k] for k in keys])
        self._lo = jnp.array([self.cosmo_box[k][0] for k in keys]); self._hi = jnp.array([self.cosmo_box[k][1] for k in keys])
        return self

    def theta_to_full(self, theta, keys=None, fixed=None):
        c = dict(self.cosmo_fixed if fixed is None else fixed)
        for i, k in enumerate(self.cosmo_keys if keys is None else keys): c[k] = theta[i]
        return c

    def ede_emulator(self):
        if getattr(self, '_ede', None) is None:
            from ede import EDEEmulator
            self._ede = EDEEmulator()
        return self._ede

    def growth_to_zN(self, c):
        """D_e(z_N; theta) / D_e(z_early; theta, LCDM): growth from the CosmoPower redshift to z_N."""
        Om = Omega_m(c)
        return sym_D_early(Om, 1. / (1. + self.z_N), c['w0'], c['wa']) / sym_D_early(Om, 1. / (1. + self.z_early), -1., 0.)

    def plin_zN(self, k_mpc, c):
        """P_lin(k [1/Mpc], z_N) in Mpc^3 for a full cosmology dict c."""
        return self.pklin(k_mpc, c, self.z_early) * self.growth_to_zN(c)**2

    def phi_ln_full(self, c):
        """ln phi for a full cosmology dict c: [ln a | ln f | ln H/H0 | ln DA H0 | ln D_i/D_N | ln h | ln alpha].
        With EDE keys in c the linear spectrum at z_N and r_d come from the ede-v2 emulators (massive
        neutrinos, sum 0.06 eV, counted in the late-time Omega_m); otherwise from CosmoPower-LCDM at
        z_early grown to z_N."""
        ede = c.get('fEDE') is not None
        if ede:
            E = self.ede_emulator()
            c = dict(c, Omega_m=Omega_m(c) + E.omega_nu() / c['h']**2, rd_h=E.rs_drag(c) * c['h'])
            lna = jnp.log(E.plin(self.nodes_mpc, c, self.z_N) / self.T_nodes)
        else:
            lna = jnp.log(self.plin_zN(self.nodes_mpc, c) / self.T_nodes)
        Om, w0, wa = c.get('Omega_m', Omega_m(c)), c['w0'], c['wa']
        f = jnp.array([sym_f(Om, z, w0, wa) for z in self.z])
        H = jnp.array([sym_H(Om, z, w0, wa) for z in self.z])
        DA = jnp.array([sym_DA(Om, z, w0, wa) for z in self.z])
        DzN = sym_D_early(Om, 1. / (1. + self.z_N), w0, wa)
        D = jnp.array([sym_D_early(Om, 1. / (1. + z), w0, wa) / DzN for z in self.z[:-1]])
        blocks = [lna, jnp.log(f), jnp.log(H), jnp.log(DA), jnp.log(D), jnp.log(jnp.atleast_1d(c['h']))]
        if self.n_bao: blocks.append(jnp.log(self.bao_alphas(c)))
        return jnp.concatenate(blocks)

    def bao_alphas(self, c):
        """alpha_par = (D_H/r_d)/(D_H/r_d)_fid, alpha_per = (D_M/r_d)/(D_M/r_d)_fid, alpha_iso = alpha_par^1/3 alpha_per^2/3,
        in the ragged layout of the data; distances and h r_d in Mpc/h."""
        Om, rdh, out = c.get('Omega_m', Omega_m(c)), c.get('rd_h', rd_h(c)), []
        for b, iso in zip(self.bao, self.bao_iso):
            a_par = C_LIGHT / (100. * sym_H(Om, b['zeff'], c['w0'], c['wa'])) / rdh / b['DH_over_rd_fid']
            a_per = sym_DM(Om, b['zeff'], c['w0'], c['wa']) / rdh / b['DM_over_rd_fid']
            out += [a_par**(1 / 3.) * a_per**(2 / 3.)] if iso else [a_par, a_per]
        return jnp.array(out)

    def phi_ln(self, theta):
        """theta (free vector of the current cosmo model) -> ln phi: the exact composition map."""
        return self.phi_ln_full(self.theta_to_full(theta))

    def phi_ln_keys(self, keys):
        """The composition map as a closure over a fixed parameter list (survives set_cosmo_model)."""
        keys, fixed = list(keys), dict(self.cosmo_fixed)
        return lambda theta: self.phi_ln_full(self.theta_to_full(theta, keys, fixed))

    # ------------------------------------------------------------------------------------
    # physical block -> pybird inputs -> likelihood
    # ------------------------------------------------------------------------------------
    def lna_to_knots(self, lna_nodes):
        """ln a on the nodes -> the 80 knots (cubic in ln k, constant outside the nodes)."""
        q = jnp.clip(self.lnk_knots, self.nodes_lnk[0], self.nodes_lnk[-1])
        return _pb_interp1d(self.nodes_lnk, lna_nodes, kind='cubic')(q)

    def unpack(self, phi_ln):
        """ln phi -> dict of the linear physical quantities."""
        e = jnp.exp(phi_ln)
        return {k: e[v] for k, v in self.pidx.items()} | {'lna': phi_ln[self.pidx['lna']]}

    def cosmo_dicts(self, phi_ln):
        """ln phi -> one pybird cosmo dict per sky. The units bridge relabels the fixed 1/Mpc knots
        as h/Mpc: kk = k*/h_conv and P = a T h_conv^3, so at h_conv = h_fid the emulator gets its own
        knots. The amplitude at z_i is (D_i/D_N)^2 times that at z_N."""
        p = self.unpack(phi_ln)
        hc = p['hconv'][0]
        kk = self.knots_mpc / hc
        pk_zN = jnp.exp(self.lna_to_knots(p['lna'])) * self.T_knots * hc**3
        D = jnp.concatenate([p['D'], jnp.ones(1)])                     # D(z_N)/D(z_N) = 1
        out = []
        for i_sky in range(self.num_skies):
            j = self.sky_to_z[i_sky]
            d = {'H': p['H'][j], 'DA': p['DA'][j], 'f': p['f'][j], 'kk': kk, 'pk_lin': pk_zN * D[j]**2}
            if self.n_bao:
                # the dilations enter as background distances with r_d = 1, so get_alpha_bao_rec
                # returns exactly the sampled alphas (an isotropic alpha scales both distances)
                b, a = self.bao[j], p['bao'][self.bao_slice[j]]
                a_par, a_per = (a[0], a[0]) if self.bao_iso[j] else (a[0], a[1])
                d.update({'DH': a_par * b['DH_over_rd_fid'], 'DM': a_per * b['DM_over_rd_fid'], 'rd': 1.0})
            out.append(d)
        return out

    def loglkl_phi(self, eft, phi_ln):
        return self.L.loglkl(eft, self.eft_names_flat, need_cosmo_update=True, cosmo_dict=self.cosmo_dicts(phi_ln),
                             cosmo_module=None, cosmo_engine=None)

    def loglkl_mi(self, x):
        """x = [eft, ln phi] -> pybird log-likelihood (EFT priors of the config included)."""
        return self.loglkl_phi(x[:self.n_eft], x[self.n_eft:])

    def loglkl_direct(self, x):
        """x = [eft, theta] -> the same likelihood through the composition map."""
        return self.loglkl_phi(x[:self.n_eft], self.phi_ln(x[self.n_eft:]))

    # ------------------------------------------------------------------------------------
    # posteriors in the sampled coordinates
    # ------------------------------------------------------------------------------------
    def logprior_mi(self, x):
        d = x[self.n_eft:] - self.prior_mean
        return -0.5 * d @ jnp.array(self.prior_prec) @ d

    def logpost_mi(self, x):
        return self.loglkl_mi(x) + self.logprior_mi(x)

    def logprior_theta_keys(self, keys=None):
        """Direct-model prior on theta as a closure over a snapshot of the cosmo model: flat in
        cosmo_box, Gaussian terms from cosmo_gauss, w0 + wa < 0 when both are free; -inf outside."""
        keys = list(self.cosmo_keys if keys is None else keys)
        lo = jnp.array([self.cosmo_box[k][0] for k in keys]); hi = jnp.array([self.cosmo_box[k][1] for k in keys])
        gauss = [(keys.index(k), mu, sig) for k, (mu, sig) in self.cosmo_gauss.items() if k in keys]
        extra, fixed = self.extra_loglike, dict(self.cosmo_fixed)
        iw = (keys.index('w0'), keys.index('wa')) if ('w0' in keys and 'wa' in keys) else None
        def _lp(th):
            ok = jnp.all((th > lo) & (th < hi))
            if iw is not None: ok = ok & (th[iw[0]] + th[iw[1]] < 0.)
            lp = sum(-0.5 * ((th[i] - mu) / sig)**2 for i, mu, sig in gauss) if gauss else 0.0
            if extra is not None: lp = lp + extra(self.theta_to_full(th, keys, fixed))
            return jnp.where(ok, lp, -jnp.inf)
        return _lp

    def logpost_direct(self, x):
        lp = self.logprior_theta_keys()(x[self.n_eft:])
        # evaluate at clipped values outside the box so the emulator stays on its manifold
        xc = jnp.concatenate([x[:self.n_eft], jnp.clip(x[self.n_eft:], self._lo, self._hi)])
        return jnp.where(jnp.isfinite(lp), self.loglkl_direct(xc) + lp, -jnp.inf)

    # ------------------------------------------------------------------------------------
    # expansion points, minimizer scales, Gauss-Newton Fisher
    # ------------------------------------------------------------------------------------
    def x_mi_fid(self, eft):
        return np.concatenate([np.asarray(eft), self.phi_fid])

    def x_direct_fid(self, eft):
        return np.concatenate([np.asarray(eft), self.theta_fid])

    def min_scales(self, kind):
        if kind == 'direct':
            return np.concatenate([np.full(self.n_eft, 0.5), [COSMO_SCALES[k] for k in self.cosmo_keys]])
        s = np.full(self.n_phys, 0.05); s[self.pidx['lna']] = 0.1; s[self.pidx['bao']] = 0.02
        return np.concatenate([np.full(self.n_eft, 0.5), s])

    def min_bounds(self, kind):
        """Box for L-BFGS-B: the direct model's flat box; for the MI model +-4 prior sigma."""
        if kind == 'direct':
            return [(None, None)] * self.n_eft + [self.cosmo_box[k] for k in self.cosmo_keys]
        w = 4 * self.prior_widths()
        return [(None, None)] * self.n_eft + list(zip(self.prior_mean - w, self.prior_mean + w))

    def model_vector(self, kind, x):
        """Theory vector in the order of the data vector L.y_all: per sky the masked multipoles
        (profiled EFT parameters included) followed by that sky's BAO alpha(s)."""
        (self.loglkl_direct if kind == 'direct' else self.loglkl_mi)(x)
        L, out = self.L, []
        for i in range(L.nsky):
            v = L.correlator_sky[i].get(L.b_sky[i]).reshape(-1)[L.m_sky[i]]
            if L.c['with_bao_rec']: v = jnp.concatenate([v, jnp.atleast_1d(jnp.asarray(L.alpha_sky[i]))])
            out.append(v)
        return jnp.concatenate(out)

    def gn_fisher(self, kind, x, with_prior=True):
        """J^T P J with J = d(model vector)/dx (forward mode) and P the data precision, plus the
        MI prior precision on phi for kind='mi'. PSD by construction (a preconditioner)."""
        J = np.array(jax.jacfwd(lambda xx: self.model_vector(kind, xx))(jnp.array(x)))
        F = J.T @ np.array(self.L.p_all) @ J
        if kind == 'mi' and with_prior:
            F[self.n_eft:, self.n_eft:] += self.prior_prec
        return 0.5 * (F + F.T)

    # ------------------------------------------------------------------------------------
    # derived quantities
    # ------------------------------------------------------------------------------------
    def sigma8_z0(self, c, R=8.0):
        """sigma_8 today of a full cosmology dict (CosmoPower at z_early grown to z = 0, w0wa growth)."""
        k = self.pklin.k_modes                                                # 1/Mpc
        Om = Omega_m(c)
        g = sym_D_early(Om, 1., c['w0'], c['wa']) / sym_D_early(Om, 1. / (1. + self.z_early), -1., 0.)
        P = self.pklin(k, c, self.z_early) * g**2
        x = k * R / c['h']; W = 3 * (jnp.sin(x) - x * jnp.cos(x)) / x**3
        lk = jnp.log(k); f = k**3 * P * W**2 / (2 * jnp.pi**2)
        return jnp.sqrt(jnp.sum(0.5 * (f[1:] + f[:-1]) * jnp.diff(lk)))

    def sigma_R_zN(self, lna_nodes, hconv, R=8.0, n_ext=60):
        """sigma_R(z_N) [R in Mpc/h] of the free spectrum P = a T, in the h of the fit (h_conv).
        Above the last knot the fiducial shape is continued at the boundary amplitude, exactly as
        the model treats ln a there. Vectorized over the leading axis of lna_nodes / hconv."""
        lna_nodes, hconv = np.atleast_2d(lna_nodes), np.atleast_1d(hconv)
        a = np.exp(np.asarray(jax.vmap(self.lna_to_knots)(jnp.array(lna_nodes))))
        k_ext = np.exp(np.linspace(np.log(float(self.knots_mpc[-1])), np.log(5.0), n_ext))[1:]
        T_ext = np.asarray(self.plin_zN(jnp.array(k_ext), self.cosmo_fid))
        kk = np.concatenate([np.asarray(self.knots_mpc), k_ext])
        P = np.concatenate([a * np.asarray(self.T_knots), a[:, [-1]] * T_ext], axis=1)   # Mpc^3
        kh = kk[None, :] / hconv[:, None]; x = kh * R
        W = 3 * (np.sin(x) - x * np.cos(x)) / x**3
        return np.sqrt(np.trapezoid(P * hconv[:, None]**3 * W**2 * kh**3 / (2 * np.pi**2), np.log(kh), axis=1))
