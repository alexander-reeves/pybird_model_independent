"""The model-independent (MI) parametrization of the templates (config.TEMPLATES): what the EFT likelihood sees instead
of a cosmology.

The linear spectrum at z_N (the farthest redshift, 1.491)
---------------------------------------------------------
The fiducial spectrum is split once into a smooth part and the BAO wiggles, T(k) = T_nw(k) [1 + O(k)] (T_nw: an
Eisenstein-Hu no-wiggle shape times a Gaussian-filtered ratio, nowiggle_eh_filter). Then

    agnostic      P_lin(k, z_N) = a(k) T(k)                                            bao_template = False
    bao_template  P_lin(k, z_N) = a(k) T_nw(k) [1 + (1 + A) exp(-k^2 dSigma^2 / 2) O(alpha_rs k)]    bao_template = True

with ln a(k) a cubic spline on nodes uniform in ln k over [1e-4, 0.7] h/Mpc: 60 nodes (spacing 0.15) for the agnostic
template, fine enough to reshape the wiggles; 21 nodes (0.45) for the BAO template, too coarse to move a wiggle, so there
the BAO position is alpha_rs = (h r_d)/(h r_d)_fid and A, dSigma^2 are the BAO-fit amplitude and damping. k is in 1/Mpc at
h_fid and there is no h_conv: the emulator gets its own knots, kk = knots_mpc / h_fid.

Per redshift there are two dilations, alpha_par(z) = D_H / (r_d,ref (D_H/r_d)_fid) and alpha_per(z) =
D_M / (r_d,ref (D_M/r_d)_fid) with r_d,ref = h_fid r_d,fid (A) or the true h r_d (B), (D/r_d)_fid the
DESI BAO fiducial of the data file. They set the AP of the full shape (H/H0 and D_A H0 are derived
from them) and predict the post-reconstruction BAO:  alpha_BAO = alpha / alpha_rs.  So in B the two
alphas ARE the BAO alphas, and P_lin is the spectrum in sound-horizon units, P(k r_d) / r_d^3. (Frame A: `bao_template`, with
alpha_rs free. Frame B: alpha_rs = 1, i.e. the baseline `bao_template_no_rs`, `growth_from_h` and `agnostic`.)

B is A with one direction removed: with a free broadband, (alpha_rs, every alpha) -> lambda x (alpha_rs, every alpha)
plus a dilation of a(k) leaves the observed multipoles unchanged (the AP volume factor 1/(q_par q_perp^2) cancels the
lambda^3 of the dilated spectrum). Only the dimensionful EFT priors and the finite node basis break it; in A it is held
by the priors on alpha_rs and the alphas.

Growth from the expansion (growth_from_h, alpha_rs = 1 only)
----------------------------------------------------------
Instead of f, the two alphas and D at each redshift (4N - 1 numbers), one function: s(z) = ln[H r_d / (H r_d)_fid] as a cubic
spline in ln(1+z) on z_h = (0, z_1/2, the N data redshifts), and mu = ln[Omega_m (H0 r_d)^2 / fid]. (Without the node at
z_1/2 = 0.15 the D_M of the w0wa reference draws is off by up to 1.5%: nothing else samples H below z_1.) Then (`expansion`)
    alpha_par(z_i) = D_H/r_d over its fiducial from s(z_i);   alpha_per(z_i) from D_M/r_d = int_0^z D_H/r_d dz (flat);
    D(z), f(z) from the growth equation (cosmology.growth_general) with E^2 = E_fid^2 e^{2s} and Omega_m(a) from mu;
    above z_N the dark-energy density E^2 - Omega_m (1+z)^3 is held at its value at z_N.
Assumes GR, scale-independent growth (no massive-neutrino or modified-gravity scale dependence), smooth dark energy, flatness,
and that the dark-energy density does not grow above z_N = 1.49. Any LCDM / w0wa expansion is represented up to the spline.

Parameter vector x = [eft | ln a (n_amp) | ln f (N) | ln alpha_par (N) | ln alpha_per (N) | ln D_i/D_N (N-1) |
A, dSigma^2 (n_env) | ln alpha_rs (A only)]. Priors (Gaussian, centred on the fiducial): shape and amplitudes exactly
as model.MIModel, with the smoothness penalty scaled to the same continuum curvature prior, lambda (0.15 / spacing)^3;
the alphas s_lng, f s_lnf, A s_env, dSigma^2 s_dsig2, alpha_rs s_lnrs.

Direct model
------------
theta -> phi(theta) (`phi_ln_full`): f, D ratios as model.MIModel; the alphas from the background and r_d of the engine
(rd_h_fixed, or the ede-v2 r_d); ln a (and A, dSigma^2) by least squares on the 80 emulator knots of

    A:  ln P(k s; theta) + 3 ln s - ln T_nw(k) - ln[1 + O(k alpha_rs)],   s = h / h_fid
    B:  ln P(k s; theta) + 3 ln s - ln T(k),                              s = r_d,fid / r_d

weighted by the data (ls_weight = the knot Fisher) or uniformly in ln k. The BAO alphas
of the direct model are reproduced exactly, the full shape up to what the node basis cannot represent; `checks.py map`
measures that as a likelihood difference against the true likelihood.
"""
import numpy as np
import jax
import jax.numpy as jnp
from copy import deepcopy

import os
from model import MIModel, PkLinCPJ, Omega_m, rd_h, ROOT, sym_H, sym_DA, sym_DM, C_LIGHT, _pb_interp1d, Likelihood
from cosmology import growth, growth_general, rd_h_fixed
from inference import log


# ------------------------------------------------------------------------------------------
# wiggle / no-wiggle split of the fiducial spectrum (numpy, once)
# ------------------------------------------------------------------------------------------
def eh_nowiggle_T(k_mpc, omega_m, omega_b, h, T_cmb=2.7255):
    """Eisenstein & Hu 1998 no-wiggle transfer function (eqs. 26-31), k in 1/Mpc."""
    th2, fb = (T_cmb / 2.7)**2, omega_b / omega_m
    s = 44.5 * np.log(9.83 / omega_m) / np.sqrt(1 + 10 * omega_b**0.75)                 # Mpc
    ag = 1 - 0.328 * np.log(431 * omega_m) * fb + 0.38 * np.log(22.3 * omega_m) * fb**2
    gam = omega_m / h * (ag + (1 - ag) / (1 + (0.43 * k_mpc * s)**4))
    q = k_mpc / h * th2 / gam
    L0, C0 = np.log(2 * np.e + 1.8 * q), 14.2 + 731 / (1 + 62.5 * q)
    return L0 / (L0 + C0 * q**2)


def nowiggle_eh_filter(lnk, P, cosmo, sigma_dex=0.25):
    """Smooth spectrum P_EH,nw x G[P / P_EH,nw]: the ratio to the Eisenstein-Hu no-wiggle shape (n_s tilt
    included) smoothed with a Gaussian of sigma_dex in log10 k (Vlah et al. 2016, arXiv:1509.02120), on a
    uniform ln k grid (edges padded with the end values). The ratio is flat at large scales, so O -> 0 there."""
    from scipy.ndimage import gaussian_filter1d
    k = np.exp(lnk); om = cosmo['omega_cdm'] + cosmo['omega_b']
    Peh = k**cosmo['n_s'] * eh_nowiggle_T(k, om, cosmo['omega_b'], cosmo['h'])**2
    R = P / Peh
    sm = gaussian_filter1d(np.log(R), sigma_dex * np.log(10.) / (lnk[1] - lnk[0]), mode='nearest')
    return Peh * np.exp(sm)


class Hermite:
    """C^1 cubic Hermite interpolant on a uniform grid (values and spline slopes precomputed), in JAX;
    `outside` beyond the grid. Cheap and smooth in the query point, for O(k alpha_rs)."""

    def __init__(self, x, y, outside=0.0):
        from scipy.interpolate import CubicSpline
        x = np.asarray(x, float)
        self.x0, self.dx, self.n = float(x[0]), float(x[1] - x[0]), len(x)
        self.y, self.d = jnp.array(y), jnp.array(CubicSpline(x, y)(x, 1))
        self.outside = outside

    def __call__(self, xq):
        t = (xq - self.x0) / self.dx
        i = jnp.clip(jnp.floor(t).astype(jnp.int32), 0, self.n - 2); u = t - i
        h00, h10, h01, h11 = 2*u**3 - 3*u**2 + 1, u**3 - 2*u**2 + u, -2*u**3 + 3*u**2, u**3 - u**2
        v = h00 * self.y[i] + h10 * self.dx * self.d[i] + h01 * self.y[i + 1] + h11 * self.dx * self.d[i + 1]
        return jnp.where((t < 0) | (t > self.n - 1), self.outside, v)


def node_grid(k_range_h, spacing, h_fid):
    """Nodes uniform in ln k (1/Mpc) over k_range_h (h/Mpc at h_fid), the spacing rounded to fit."""
    lo, hi = np.log(k_range_h[0] * h_fid), np.log(k_range_h[1] * h_fid)
    n = int(np.round((hi - lo) / spacing)) + 1
    return np.linspace(lo, hi, n)


def weighted_lstsq(S, W, rcond=1e-4):
    """(Pi, N): Pi = (S^T W S)^+ S^T W, the weighted least-squares projection of knot values onto the nodes, through the SVD
    of W^1/2 S, and N the node combinations invisible at the knots (singular values below rcond x the largest), or None.
    The agnostic template has 21 (31 nodes below 0.01 h/Mpc see 13 knots: 20 singular values ~1e-16, one 1.7e-5, the rest
    >= 8.6e-3); a plain solve filled them with round-off amplified to |ln a| ~ 1e7, invisible to the likelihood but not to
    the prior and the flow of the projection. TemplateModel sets them to the MI prior's conditional mean (phi_ln_full).
    The bao_template system is well conditioned: nothing is cut."""
    w, V = np.linalg.eigh(0.5 * (W + W.T)); Wh = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
    U, sv, Vt = np.linalg.svd(Wh @ S, full_matrices=False); keep = sv > rcond * sv[0]
    Pi = (Vt[keep].T / sv[keep]) @ U[:, keep].T @ Wh
    return Pi, (Vt[~keep].T if (~keep).any() else None)


def ls_projector(nodes_lnk, lnk_knots, w):
    """S (knots x nodes): the cubic node interpolant (constant outside the nodes) as a matrix, and the least-squares
    projection of knot values onto the nodes with weights w (weighted_lstsq)."""
    nodes_lnk, q = jnp.asarray(nodes_lnk), jnp.clip(jnp.asarray(lnk_knots), nodes_lnk[0], nodes_lnk[-1])
    S = np.array(jax.vmap(lambda y: _pb_interp1d(nodes_lnk, y, kind='cubic')(q))(jnp.eye(len(nodes_lnk)))).T
    return S, weighted_lstsq(S, np.diag(w))[0]


class TemplateModel(MIModel):
    """model.MIModel with the parametrization of the module docstring. Everything that only needs
    `idx`/`pidx`, the prior, `phi_ln_full` and `cosmo_dicts` (likelihoods, posteriors, direct models,
    GN Fisher, compress.py) is inherited unchanged."""

    PRIOR_DEFAULT = dict(MIModel.PRIOR_DEFAULT, s_lnrs=0.05, s_env=0.3, s_dsig2=25.0, s_lnm=1.0)
    LAMBDA_SPACING = 0.15        # smooth_lambda is quoted at this node spacing (the baseline's)

    def __init__(self, data, cosmo_fid, prior=None, nodes=None, bao_template=False, z_early=5.0, get_maxlkl=True, verbose=True,
                 bao_dm_compat=True, rd_fix=True, ls_weight=None, ls_eps=1e-3, alpha_rs=None, envelope=None, growth_from_h=False):
        # bao_template: alpha_rs free (frame A) and the wiggle envelope (A, dSigma^2); otherwise frame B, no wiggle parameters.
        # alpha_rs / envelope set the two separately (default: both as bao_template)
        alpha_rs = bao_template if alpha_rs is None else alpha_rs; envelope = bao_template if envelope is None else envelope
        self.bao_template = bool(envelope)
        self.rs_free, self.n_env = bool(alpha_rs), (2 if envelope else 0)
        self.gfh = bool(growth_from_h); assert not (self.gfh and self.rs_free), "growth_from_h is defined in the alpha_rs = 1 frame"
        # wiggle envelope E multiplying O(alpha_rs k), the BAO-fit form: E = (1 + A) exp(-k^2 dSigma^2 / 2) (n_env = 2),
        # a BAO amplitude A and a Gaussian damping relative to the fiducial, in k (Mpc^-1); n_env = 0: E = 1
        self.rd_func = rd_h_fixed if rd_fix else rd_h          # r_d of the CPJ (LCDM/w0wa) engine; EDE has its own
        self.cosmo_fid = {'omega_cdm': cosmo_fid['omega_cdm'], 'omega_b': cosmo_fid['omega_b'], 'n_s': cosmo_fid['n_s'],
                          'lnAs': cosmo_fid['ln10^{10}A_s'], 'h': cosmo_fid['h'], 'w0': -1.0, 'wa': 0.0}
        self.h_fid, self.z_early = self.cosmo_fid['h'], z_early
        self.prior = dict(self.PRIOR_DEFAULT, **(prior or {}))

        # ---- redshifts and the pybird likelihood (as MIModel)
        self.skies, self.zeff_sky = list(data['skies']), list(data['zeff'])
        self.num_skies = len(self.skies)
        self.z = sorted(set(self.zeff_sky)); self.N = len(self.z); self.z_N = self.z[-1]
        self.sky_to_z = [self.z.index(z) for z in self.zeff_sky]
        cfg = deepcopy(data['cfg']); cfg['get_maxlkl'] = get_maxlkl
        self.L = Likelihood(cfg, verbose=verbose)
        self.eft_free = [p for p, pr in cfg['eft_prior'].items() if pr['type'] in ('flat', 'gauss', 'lognormal')]
        self.eft_names_flat = self.eft_free * self.num_skies
        self.eft_labels = [f"{n}_{s}" for s in self.skies for n in self.eft_free]
        self.n_eft = len(self.eft_names_flat)
        # BAO fiducials: one per unique redshift (the alphas are per redshift)
        self.bao = [dict(b) for b in data['bao']]; self.bao_iso = [bool(b['iso']) for b in self.bao]
        self.n_bao = sum(1 if i else 2 for i in self.bao_iso)          # data points, not parameters
        assert [b['zeff'] for b in self.bao] == self.z
        self.DH_fid = jnp.array([b['DH_over_rd_fid'] for b in self.bao]); self.DM_fid = jnp.array([b['DM_over_rd_fid'] for b in self.bao])
        # h r_d at the fiducial, Mpc/h: only a normalization of the alphas (any constant works, the direct map uses the
        # same one); kept at model.rd_h so the parametrization does not depend on rd_fix (the two differ by 1e-5)
        self.rdh_fid = float(rd_h(self.cosmo_fid))
        # alpha_per is the accurate D_M ((1+z) c/100 symbolic.DA, as the full-shape AP of the direct model). The
        # direct model's BAO uses symbolic.comoving_distance, which integrates from z = 1e-3 and is low by
        # ~(c/100) 1e-3 Mpc/h: 0.37% at z = 0.295, 0.10% at z = 1.49. With bao_dm_compat the predicted BAO
        # alpha_per carries the same factor, evaluated at the fiducial (the missing piece is ~1e-3 for any
        # cosmology, so the factor is cosmology-independent to ~1e-5), so that the recovery test compares
        # with the existing direct chains like for like. Set False once comoving_distance is fixed.
        Om_fid = self.Om_fid = float(Omega_m(self.cosmo_fid))
        # growth_from_h: the spline nodes, and the grid of the D_M integral (the nodes of symbolic.DA, so the fiducial maps to s = 0)
        self.z_h = [0., 0.5 * self.z[0]] + list(self.z); self.zz_dm = jnp.array([np.linspace(1e-5, z, 30) for z in self.z])
        self.bao_dm_factor = jnp.array([float(sym_DM(Om_fid, z) / (C_LIGHT / 100. * (1 + z) * sym_DA(Om_fid, z))) if bao_dm_compat else 1.0
                                        for z in self.z])

        # ---- template on the 80 knots, and its split
        self.pklin = PkLinCPJ()
        self.knots_h = np.load(os.path.join(ROOT, 'pybird', 'emu_data', 'knots.npy'))
        self.knots_mpc = jnp.array(self.knots_h * self.h_fid); self.lnk_knots = jnp.log(self.knots_mpc)
        self.kk = jnp.array(self.knots_h)
        self.T_knots = self.plin_zN(self.knots_mpc, self.cosmo_fid)
        self._split()

        # ---- broadband nodes, uniform in ln k
        nodes = dict({'k_range': (1e-4, 0.7), 'spacing': 0.6}, **(nodes or {}))
        g = node_grid(nodes['k_range'], nodes['spacing'], self.h_fid)
        self.n_amp = len(g); self.node_spacing = float(g[1] - g[0]); self.node_settings = nodes
        self.nodes_lnk = jnp.array(g); self.nodes_mpc = jnp.exp(self.nodes_lnk)
        self.nodes_h = np.array(self.nodes_mpc) / self.h_fid
        # least-squares projector knots -> nodes, weights d ln k
        # weights of the least-squares map (direct model only): d ln k on the knots, or with ls_weight (an npz with the
        # data Fisher W = J^T C^-1 J of the multipoles + BAO in ln P on the 80 knots, wiggle_fisher_map.py) the data
        # metric, regularized by ls_eps x (its mean eigenvalue scale) x the d ln k weights where the data are blind
        self.ls_w = np.gradient(np.log(self.knots_h)); self.ls_weight = ls_weight
        self.S_knots, Pi = ls_projector(g, self.lnk_knots, self.ls_w)
        Wm = np.diag(self.ls_w)
        if ls_weight:
            Wd = np.load(ls_weight)['W']; Wm = Wd + ls_eps * np.trace(Wd) / self.ls_w.sum() * np.diag(self.ls_w)
        Pi, self.null_nodes = weighted_lstsq(self.S_knots, Wm)           # null_nodes: node combinations invisible at the knots
        self._null_fix = None
        self.Pi_ls = jnp.array(Pi); self.S_knots_j, self.ls_w_j, self.Wm_j = jnp.array(self.S_knots), jnp.array(self.ls_w), jnp.array(Wm)

        # ---- layout
        sizes = [('eft', self.n_eft), ('lna', self.n_amp)] + ([('hz', self.N + 2), ('m', 1)] if self.gfh else
                                                                  [('f', self.N), ('apar', self.N), ('aper', self.N), ('D', self.N - 1)])
        if self.n_env: sizes.append(('env', self.n_env))
        if self.rs_free: sizes.append(('rs', 1))
        self.idx, i0 = {}, 0
        for name, n in sizes:
            self.idx[name] = np.arange(i0, i0 + n); i0 += n
        self.n_mi = i0; self.n_phys = self.n_mi - self.n_eft
        self.pidx = {k: v - self.n_eft for k, v in self.idx.items() if k != 'eft'}
        zl = [f'{z:.2f}' for z in self.z]
        evo = ([f'ln H r_d({z:.2f})' for z in self.z_h] + ['ln Omega_m (H0 r_d)^2'] if self.gfh else
               [f'ln f({z})' for z in zl] + [f'ln alpha_par({z})' for z in zl] + [f'ln alpha_per({z})' for z in zl]
               + [f'ln D({z})/D({zl[-1]})' for z in zl[:-1]])
        self.phys_names = [f'ln a({k:.3g})' for k in self.nodes_h] + evo + self.env_names() + (['ln alpha_rs'] if self.rs_free else [])
        self.names = self.eft_labels + self.phys_names

        self.extra_loglike = None
        self.set_cosmo_model('lcdm')
        # the prior is centred on the fiducial mapped with model.rd_h, whatever rd_fix: the MI model (likelihood and
        # prior) does not depend on the direct model's r_d, so chains sampled before rd_fix existed stay exact
        rd_func, self.rd_func = self.rd_func, rd_h
        self.phi_fid = np.asarray(self.phi_ln_full(self.cosmo_fid))
        self.rd_func = rd_func
        # the fiducial maps to a = 1 up to round-off (~1e-9 with a data-weighted map on many nodes)
        assert np.abs(self.phi_fid[self.pidx['lna']]).max() < 1e-6, self.phi_fid[self.pidx['lna']]
        self.prior_mean = self.phi_fid.copy()
        self.prior_prec = self._prior_precision()
        if self.null_nodes is not None:
            # the map sets the invisible node combinations to the MI prior's conditional mean given everything else (they do
            # not change the likelihood, so p_MI / pi_MI of the projection does not depend on them; this keeps phi(theta)
            # inside the region the flow was trained on): y -> y + E (y - prior mean), E = N K, K = -(N^T Pi_aa N)^-1 N^T Pi_a.
            la, N, Pp = self.pidx['lna'], self.null_nodes, self.prior_prec
            E = np.zeros((self.n_phys, self.n_phys)); E[la, :] = N @ -np.linalg.solve(N.T @ Pp[np.ix_(la, la)] @ N, N.T @ Pp[la, :])
            self._null_fix, self._prior_mean_j = jnp.array(E), jnp.array(self.prior_mean)
        if verbose:
            log(f"TemplateModel ({'bao_template: alpha_rs, A, dSigma^2 free' if self.rs_free else 'agnostic: alphas = BAO alphas'}): "
                f"{self.n_eft} EFT + {self.n_amp} ln a nodes (spacing {self.node_spacing:.3f} in ln k, "
                f"[{self.nodes_h[0]:.3g}, {self.nodes_h[-1]:.3g}] h/Mpc) + "
                + (f"{self.N + 2} H(z) nodes + Omega_m (growth from the expansion)" if self.gfh else f"{self.N} f + {2*self.N} alphas + {self.N-1} D")
                + (f" + {self.n_env} wiggle envelope" if self.n_env else "") + (" + alpha_rs" if self.rs_free else "") + f" = {self.n_mi} MI parameters; no h_conv")
            log(f"  prior {self.prior}; smoothness lambda_eff = {self.lambda_eff:.3g}; h r_d,fid = {self.rdh_fid:.3f} Mpc/h")

    # ------------------------------------------------------------------------------------
    def _split(self, k_taper=(1e-3, 3e-3), k_grid=(1e-5, 9.0)):
        """T = T_nw (1 + O) at the fiducial, O = T/T_nw - 1 with T_nw from `nowiggle_eh_filter` on a fine ln k
        grid (1/Mpc), tapered to exactly 0 below k_taper (no wiggles there), as a Hermite interpolant;
        T_nw := T / (1 + O), so model A at alpha_rs = 1 is exactly T. (pybird's DST split leaves a ~10%
        non-oscillatory residual near the turnover, hence this one.)"""
        k_cpj = np.asarray(self.pklin.k_modes)
        lnk = np.linspace(np.log(k_grid[0]), np.log(k_grid[1]), 8000)
        P = np.asarray(self.plin_zN(jnp.array(np.exp(lnk)), self.cosmo_fid))
        O_raw = P / nowiggle_eh_filter(lnk, P, self.cosmo_fid) - 1.0
        t = np.clip((lnk - np.log(k_taper[0])) / np.log(k_taper[1] / k_taper[0]), 0, 1)
        O = O_raw * t * t * (3 - 2 * t)
        self.split_grid = {'k_mpc': np.exp(lnk), 'P': P, 'O': O, 'O_raw': O_raw, 'k_cpj': k_cpj,
                           'O_at_taper': float(np.abs(O_raw)[lnk < np.log(k_taper[1])].max())}
        self.O = Hermite(lnk, O)
        self.O_of_k = lambda k_mpc: self.O(jnp.log(k_mpc))
        self.T_nw_knots = self.T_knots / (1.0 + self.O_of_k(self.knots_mpc))

    def T_nw(self, k_mpc):
        """The smooth fiducial spectrum at any k (Mpc^3)."""
        return self.plin_zN(k_mpc, self.cosmo_fid) / (1.0 + self.O_of_k(k_mpc))

    @property
    def lambda_eff(self):
        return self.prior['smooth_lambda'] * (self.LAMBDA_SPACING / self.node_spacing)**3

    def _prior_precision(self):
        p, n, N = self.prior, self.n_amp, self.N
        Pi = np.zeros((self.n_phys, self.n_phys)); ia = self.pidx['lna']
        D2 = np.zeros((n - 2, n))
        for i in range(n - 2): D2[i, i:i + 3] = (1., -2., 1.)
        Pi[np.ix_(ia, ia)] = (np.eye(n) - np.ones((n, n)) / n) / p['s_lna']**2 + self.lambda_eff * D2.T @ D2
        if self.gfh:      # the amplitudes below z_N follow from the growth: only the one at z_N (the node mean of ln a) has a prior
            Pi[np.ix_(ia, ia)] += np.ones((n, n)) / n**2 / (p['s_lna']**2 / n + p['s_lnP']**2)
            Pi[self.pidx['hz'], self.pidx['hz']] = p['s_lng']**-2; Pi[self.pidx['m'], self.pidx['m']] = p['s_lnm']**-2
        else:
            B = self.amplitude_matrix()
            Su = p['s_lna']**2 / n * np.ones((N, N)) + p['s_lnP']**2 * np.eye(N)
            Pi += B.T @ np.linalg.inv(Su) @ B
            for blk in ('f', 'apar', 'aper'):
                Pi[self.pidx[blk], self.pidx[blk]] = (p.get('s_lnf', p['s_lng']) if blk == 'f' else p['s_lng'])**-2
        if self.rs_free: Pi[self.pidx['rs'], self.pidx['rs']] = p['s_lnrs']**-2
        if self.n_env: Pi[self.pidx['env'], self.pidx['env']] = np.asarray(self.env_sigmas())**-2
        return Pi

    # ------------------------------------------------------------------------------------
    # physical block -> pybird inputs
    # ------------------------------------------------------------------------------------
    def expansion(self, s, mu):
        """growth_from_h: (f, alpha_par, alpha_per, D(z_i)/D(z_N)) at the data redshifts from s = ln[H r_d / (H r_d)_fid] on the
        nodes z_h and mu = ln[Omega_m (H0 r_d)^2 / fid] (module docstring)."""
        I = _pb_interp1d(jnp.log1p(jnp.array(self.z_h)), s, kind='cubic'); Of, zN = self.Om_fid, self.z_N
        sp = lambda z: I(jnp.ravel(jnp.log1p(z))).reshape(jnp.shape(z))                              # the interpolant takes 1-d queries
        lnE = lambda z: 0.5 * jnp.log(Of * (1. + z)**3 + 1. - Of) + sp(jnp.clip(z, 0., zN))          # H r_d over (H0 r_d)_fid, z <= z_N
        rm = lambda z: Of * jnp.exp(mu) * (1. + z)**3

        def lne2(x):
            z = jnp.exp(-x) - 1.
            return jnp.where(z <= zN, 2. * lnE(z), jnp.log(rm(z) + jnp.exp(2. * lnE(zN)) - rm(zN)))
        Dz, f = growth_general(lne2, lambda x: rm(jnp.exp(-x) - 1.), self.z)
        apar = C_LIGHT / 100. / (jnp.exp(lnE(jnp.array(self.z))) * self.rdh_fid * self.DH_fid)
        y = jnp.exp(-lnE(self.zz_dm))                                                               # trapezoid on symbolic.DA's nodes
        aper = C_LIGHT / 100. * jnp.sum(0.5 * (y[:, 1:] + y[:, :-1]) * jnp.diff(self.zz_dm, axis=1), axis=1) / (self.rdh_fid * self.DM_fid)
        return f, apar, aper, Dz[:-1] / Dz[-1]

    def unpack(self, phi_ln):
        """ln phi -> dict of the linear physical quantities (ln a and the envelope coefficients stay as they are)."""
        e = jnp.exp(phi_ln); out = {k: e[v] for k, v in self.pidx.items()}
        if self.gfh: out.update(zip(('f', 'apar', 'aper', 'D'), self.expansion(phi_ln[self.pidx['hz']], phi_ln[self.pidx['m']][0])))
        out['lna'] = phi_ln[self.pidx['lna']]
        if self.n_env: out['env'] = phi_ln[self.pidx['env']]
        return out

    def env_names(self):
        return ['BAO amplitude A', 'wiggle damping dSigma^2 [Mpc^2]'] if self.n_env else []

    def env_sigmas(self):
        return [self.prior['s_env'], self.prior['s_dsig2']] if self.n_env else []

    def env_E(self, q_mpc, env):
        """The envelope E = (1 + A) exp(-k^2 dSigma^2 / 2) at the knots (exact)."""
        return (1.0 + env[0]) * jnp.exp(-0.5 * self.knots_mpc**2 * env[1])

    def env_cols(self, q_mpc):
        """dE/d(A, dSigma^2) at (0, 0): the columns of the linearized least-squares map."""
        d = (-0.5 * self.knots_mpc**2)[:, None]
        return jnp.concatenate([jnp.ones_like(d), d], 1)

    def spectrum_knots(self, lna_nodes, ln_ars=0.0, env=None):
        """P_lin(knots_mpc, z_N) in Mpc^3 of the MI model: a T_nw [1 + E O(k alpha_rs)] (A), a T (B)."""
        a = jnp.exp(self.lna_to_knots(lna_nodes))
        if not self.rs_free and not self.n_env:
            return a * self.T_knots
        q = self.knots_mpc * jnp.exp(ln_ars); O = self.O_of_k(q)
        if self.n_env: O = O * self.env_E(q, env)
        return a * self.T_nw_knots * (1.0 + O)

    def background(self, p):
        """alphas (linear, per redshift) -> H/H0 and D_A H0 for the AP of the full shape, and the BAO alphas."""
        ars = p['rs'][0] if self.rs_free else 1.0
        z = jnp.array(self.z)
        H = C_LIGHT / 100. / (p['apar'] * self.DH_fid * self.rdh_fid)
        DA = p['aper'] * self.DM_fid * self.rdh_fid / (C_LIGHT / 100. * (1. + z))
        return H, DA, p['apar'] / ars, p['aper'] / ars * self.bao_dm_factor

    def cosmo_dicts(self, phi_ln):
        p = self.unpack(phi_ln)
        pk_zN = self.spectrum_knots(p['lna'], phi_ln[self.pidx['rs']][0] if self.rs_free else 0.0, p.get('env')) * self.h_fid**3
        H, DA, bpar, bper = self.background(p)
        D = jnp.concatenate([p['D'], jnp.ones(1)])
        out = []
        for i_sky in range(self.num_skies):
            j = self.sky_to_z[i_sky]; b = self.bao[j]
            out.append({'H': H[j], 'DA': DA[j], 'f': p['f'][j], 'kk': self.kk, 'pk_lin': pk_zN * D[j]**2,
                        'DH': bpar[j] * b['DH_over_rd_fid'], 'DM': bper[j] * b['DM_over_rd_fid'], 'rd': 1.0})
        return out

    def bao_alphas_mi(self, phi_ln):
        """The post-reconstruction alphas the MI parameters predict, in the ragged layout of the data."""
        _, _, bpar, bper = self.background(self.unpack(phi_ln)); out = []
        for j, iso in enumerate(self.bao_iso):
            out += [bpar[j]**(1/3.) * bper[j]**(2/3.)] if iso else [bpar[j], bper[j]]
        return jnp.array(out)

    # ------------------------------------------------------------------------------------
    # cosmology -> physical block
    # ------------------------------------------------------------------------------------
    def engine(self, c):
        """(c with Omega_m and rd_h, P_lin(k_mpc, z_N) callable) for LCDM/w0wa (CPJ) or EDE (ede-v2)."""
        if c.get('fEDE') is not None:
            E = self.ede_emulator()
            c = dict(c, Omega_m=Omega_m(c) + E.omega_nu(c) / c['h']**2, rd_h=E.rs_drag(c) * c['h'])
            return c, (lambda k: E.plin(k, c, self.z_N, cb=c.get('cb', False)))
        c = dict(c, Omega_m=Omega_m(c), rd_h=self.rd_func(c))
        g2 = self.growth_to_zN(c)**2                                     # once per cosmology (an ODE for w0wa)
        return c, (lambda k: self.pklin(k, c, self.z_early) * g2)

    def lna_target_knots(self, c, plin):
        """ln a on the 80 knots that reproduces the direct spectrum exactly (before the node projection)."""
        ars = c['rd_h'] / self.rdh_fid
        if self.rs_free:
            s = c['h'] / self.h_fid
            return (jnp.log(plin(self.knots_mpc * s)) + 3 * jnp.log(s) - jnp.log(self.T_nw_knots)
                    - jnp.log1p(self.O_of_k(self.knots_mpc * ars)))
        s = (c['h'] / self.h_fid) / ars                                  # = r_d,fid / r_d
        return jnp.log(plin(self.knots_mpc * s)) + 3 * jnp.log(s) - jnp.log(self.T_knots)

    def fit_spectrum(self, c, plin):
        """(ln a on the nodes, envelope coefficients) for the direct spectrum: weighted least squares on the 80 knots
        of the target r(k) (lna_target_knots) on the spline columns and, with an envelope, the columns
        O/(1+O) dE/d(A, dSigma^2), the linearization of ln[1 + E O] - ln[1 + O] (error ~ (E - 1)^2 O^2 / 2 ~ 1e-5)."""
        r = self.lna_target_knots(c, plin)
        if not self.n_env:
            return self.Pi_ls @ r, None
        q = self.knots_mpc * (c['rd_h'] / self.rdh_fid if self.rs_free else 1.0); O = self.O_of_k(q)
        Bm = jnp.concatenate([self.S_knots_j, (O / (1.0 + O))[:, None] * self.env_cols(q)], 1)
        coef = jnp.linalg.solve(Bm.T @ self.Wm_j @ Bm, Bm.T @ (self.Wm_j @ r))
        return coef[:self.n_amp], coef[self.n_amp:]

    def spectrum_residual(self, c):
        """ln P_MI - ln P_direct on the 80 knots after the fit (fit_spectrum), exact (no linearization)."""
        c, plin = self.engine(c)
        r = self.lna_target_knots(c, plin); lna, env = self.fit_spectrum(c, plin)
        q = self.knots_mpc * (c['rd_h'] / self.rdh_fid if self.rs_free else 1.0); O = self.O_of_k(q)
        E = self.env_E(q, env) if self.n_env else 1.0
        return self.lna_to_knots(lna) + jnp.log1p(E * O) - jnp.log1p(O) - r

    def phi_ln_full(self, c):
        c, plin = self.engine(c)
        Om, w0, wa, rdh = c['Omega_m'], c['w0'], c['wa'], c['rd_h']
        lna, env = self.fit_spectrum(c, plin)
        Dz, f = growth(Om, w0, wa, self.z); D = Dz[:-1] / Dz[-1]          # the growth equation: exact for w0wa
        rdh_ref = self.rdh_fid if self.rs_free else rdh
        DH = jnp.array([C_LIGHT / (100. * sym_H(Om, z, w0, wa)) for z in self.z])
        DM = jnp.array([C_LIGHT / 100. * (1. + z) * sym_DA(Om, z, w0, wa) for z in self.z])
        blocks = [lna, jnp.log(f), jnp.log(DH / (rdh_ref * self.DH_fid)), jnp.log(DM / (rdh_ref * self.DM_fid)), jnp.log(D)]
        if self.gfh:      # s on the nodes z_h and mu: H r_d and Omega_m (H0 r_d)^2 of the cosmology over the fiducial's
            s = jnp.array([jnp.log(sym_H(Om, z, w0, wa) / sym_H(self.Om_fid, z)) for z in self.z_h]) + jnp.log(rdh / self.rdh_fid)
            blocks = [lna, s, jnp.atleast_1d(jnp.log(Om / self.Om_fid) + 2. * jnp.log(rdh / self.rdh_fid))]
        if self.n_env: blocks.append(env)
        if self.rs_free: blocks.append(jnp.log(jnp.atleast_1d(rdh / self.rdh_fid)))
        y = jnp.concatenate(blocks)
        return y if self._null_fix is None else y + self._null_fix @ (y - self._prior_mean_j)

    # ------------------------------------------------------------------------------------
    def min_scales(self, kind):
        if kind == 'direct': return super().min_scales('direct')
        s = np.full(self.n_phys, 0.02); s[self.pidx['lna']] = 0.1
        for blk in ('f', 'D', 'm'):
            if blk in self.pidx: s[self.pidx[blk]] = 0.05
        if self.n_env: s[self.pidx['env']] = [0.05, 2.0]
        return np.concatenate([np.full(self.n_eft, 0.5), s])


def sound_horizon_loglkl(MB, eft, c):
    """ln L of the exact spectrum of the cosmology c in the sound-horizon frame of MB (a template with alpha_rs = 1): P(k rho) rho^3
    with rho = r_d,fid / r_d on the knots, MB's AP from the BAO alphas, growth and BAO points; no node representation. It is the
    true likelihood (model.true_loglkl) moved by the gauge T_lam of 6_alpha_rs_degeneracy.ipynb with the EFT parameters relabelled,
    c -> alpha_rs^-n c, i.e. with their Gaussian priors in units of the sound horizon instead of h/Mpc."""
    phi = MB.phi_ln_full(c); d = MB.cosmo_dicts(phi); c2, plin = MB.engine(c)
    rho = (c['h'] / MB.h_fid) / (c2['rd_h'] / MB.rdh_fid)
    P = plin(MB.knots_mpc * rho) * rho**3 * MB.h_fid**3
    D = jnp.concatenate([jnp.exp(phi[MB.pidx['D']]), jnp.ones(1)])
    d = [dict(di, pk_lin=P * D[MB.sky_to_z[i]]**2) for i, di in enumerate(d)]
    return MB.L.loglkl(eft, MB.eft_names_flat, need_cosmo_update=True, cosmo_dict=d, cosmo_module=None, cosmo_engine=None)


# ------------------------------------------------------------------------------------------
# The sound-horizon frame shared by both options (the agnostic option never learns h r_d)
# ------------------------------------------------------------------------------------------
# The sound-horizon frame shared by both template options (the agnostic option never learns h r_d):
# 
#     k~ = k r_d / r_d,fid,    P~(k~, z_N) = P(k~ r_d,fid / r_d, z_N) (r_d,fid / r_d)^3
# 
# The agnostic option (B) is in this frame already, P~ = a(k) T(k); a wiggle-option draw (A) converts with its alpha_rs =
# h r_d / (h r_d)_fid: P~(k) = P_A(k / alpha_rs) / alpha_rs^3 = a(k/alpha_rs) T_nw(k/alpha_rs) [1 + E(k/alpha_rs) O(k)] / alpha_rs^3,
# i.e. the wiggles sit at their fiducial positions. sigma_s8 = sigma on R = 8 Mpc/h_fid in this frame, i.e. on
# R = 8 r_d/r_d,fid Mpc/h_fid physically (DESI's sigma_s8).
def batched(f, x, B=250):
    fb = jax.jit(jax.vmap(f)); out = [fb(jnp.array(x[i:i + B])) for i in range(0, len(x), B)]
    return jax.tree_util.tree_map(lambda *a: np.concatenate([np.asarray(v) for v in a]), *out)


class Frame:
    """Spectra of MI parameters and of cosmologies in sound-horizon units, on a plotting grid and an integration grid."""

    def __init__(self, M):
        self.M, self.hf = M, M.h_fid
        self.kh = np.geomspace(1e-3, 0.5, 300); self.k_plot = jnp.array(self.kh * self.hf)        # 1/Mpc
        self.k_int = jnp.geomspace(1e-4, 5.0, 1500)
        lnk = np.linspace(np.log(1e-5), np.log(9.0), 4000)                                     # T, T_nw tabulated once
        T = np.asarray(M.plin_zN(jnp.array(np.exp(lnk)), M.cosmo_fid))
        self.lnk_tab, self.lnT = jnp.array(lnk), jnp.array(np.log(T))
        self.lnTnw = jnp.array(np.log(T / (1.0 + np.asarray(M.O_of_k(jnp.array(np.exp(lnk)))))))

    def _a(self, lna, k):
        M = self.M
        return jnp.exp(_pb_interp1d(M.nodes_lnk, lna, kind='cubic')(jnp.clip(jnp.log(k), M.nodes_lnk[0], M.nodes_lnk[-1])))

    def mi(self, ph, k):
        """P~(k) [Mpc^3] of the physical block ph."""
        M, P = self.M, self.M.pidx
        if not M.rs_free and not M.n_env:
            return self._a(ph[P['lna']], k) * jnp.exp(jnp.interp(jnp.log(k), self.lnk_tab, self.lnT))
        ars = jnp.exp(ph[P['rs']][0]) if M.rs_free else 1.0; q = k / ars
        # the envelope is held at its value at the last knot (the model's spectrum ends there; with dSigma^2 < 0 the Gaussian
        # would blow up in the sigma_s8 integral to 5/Mpc)
        qe = jnp.minimum(q, M.knots_mpc[-1])
        E = (1.0 + ph[P['env']][0]) * jnp.exp(-0.5 * qe**2 * ph[P['env']][1]) if M.n_env else 1.0
        Tnw = jnp.exp(jnp.interp(jnp.log(q), self.lnk_tab, self.lnTnw))
        return self._a(ph[P['lna']], q) * Tnw * (1.0 + E * M.O_of_k(k)) / ars**3

    def sigma_s8(self, P_int):
        x = self.k_int * 8.0 / self.hf; Wt = 3 * (jnp.sin(x) - x * jnp.cos(x)) / x**3
        f = self.k_int**3 * P_int * Wt**2 / (2 * jnp.pi**2); lk = jnp.log(self.k_int)
        return jnp.sqrt(jnp.sum(0.5 * (f[1:] + f[:-1]) * jnp.diff(lk)))
