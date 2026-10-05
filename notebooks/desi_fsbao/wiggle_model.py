"""The wiggle/no-wiggle MI model: few broadband nodes, two dilations per redshift shared by the full
shape and the post-reconstruction BAO, and (variant A) a sound-horizon rescaling of the wiggles.

The model
---------
The linear spectrum at z_N is split once, at the DESI fiducial, into a smooth part and the wiggles
(DST method of 1003.3999, as in pybird/cosmo.py): T(k) = T_nw(k) [1 + O(k)]. The MI spectrum is

    A (rs_free=True):   P_lin(k, z_N) = a(k) T_nw(k) [1 + O(k alpha_rs)]
    B (rs_free=False):  P_lin(k, z_N) = a(k) T(k)                         (alpha_rs = 1)

with ln a cubic on `nodes['spacing']`-spaced nodes in ln k (0.6 -> 16 nodes: too coarse to slide a BAO
wiggle, see ../bao_basis_scan.py). k is in 1/Mpc at h_fid and there is NO h_conv: the emulator gets
its own knots, kk = knots_mpc / h_fid.

Per redshift there are two dilations, alpha_par(z) = D_H / (r_d,ref (D_H/r_d)_fid) and alpha_per(z) =
D_M / (r_d,ref (D_M/r_d)_fid) with r_d,ref = h_fid r_d,fid (A) or the true h r_d (B), (D/r_d)_fid the
DESI BAO fiducial of the data file. They set the AP of the full shape (H/H0 and D_A H0 are derived
from them) and predict the post-reconstruction BAO:  alpha_BAO = alpha / alpha_rs.  So in B the two
alphas ARE the BAO alphas, and P_lin is the spectrum in sound-horizon units, P(k r_d) / r_d^3.

Why B is A with one direction removed: with a free broadband, (alpha_rs, every alpha) -> lambda x
(alpha_rs, every alpha) plus a dilation of a(k) leaves the observed multipoles unchanged (the AP volume
factor 1/(q_par q_perp^2) cancels the lambda^3 of the dilated spectrum; the wiggles stay put because
both their AP and their alpha_rs scale by lambda). Only the dimensionful EFT priors and the finite
node basis break it. In A it is held by the priors on alpha_rs and the alphas alone (wiggle_gates.py
measures it).

Parameter vector x = [eft | ln a (n_amp) | ln f (N) | ln alpha_par (N) | ln alpha_per (N) |
ln D_i/D_N (N-1) | ln alpha_rs (A only)]. Priors (Gaussian in ln, centred on the fiducial): shape
and amplitudes exactly as model.MIModel, with the smoothness penalty scaled to the same continuum
curvature prior, lambda (0.15 / spacing)^3; the alphas s_lng, f s_lnf (default s_lng); alpha_rs s_lnrs.

Direct model
------------
theta -> phi(theta) (`phi_ln_full`): f, D ratios as model.MIModel; the alphas from the background and
r_d of the engine (the analytic r_d of model.rd_h, or the ede-v2 r_d); ln a by least squares on the
80 emulator knots (weights d ln k) of

    A:  ln P(k s; theta) + 3 ln s - ln T_nw(k) - ln[1 + O(k alpha_rs)],   s = h / h_fid
    B:  ln P(k s; theta) + 3 ln s - ln T(k),                              s = r_d,fid / r_d

which is what the model must carry to reproduce the direct spectrum. The BAO alphas of the direct
model are reproduced exactly (wiggle_gates.py W2); the full shape up to what the node basis cannot
represent (a residual wiggle amplitude/damping mismatch; W4 measures it as a likelihood difference).
"""
import numpy as np
import jax
import jax.numpy as jnp
from copy import deepcopy

from model import (MIModel, PkLinCPJ, Omega_m, rd_h, ROOT, sym_f, sym_H, sym_DA, sym_DM, sym_D_early, C_LIGHT,
                   _pb_interp1d, Likelihood)
from sampling import log
import os


# ------------------------------------------------------------------------------------------
# wiggle / no-wiggle split of the fiducial spectrum (numpy, once)
# ------------------------------------------------------------------------------------------
def nowiggle_dst(k_mpc, P_mpc, n_lo=120, n_hi=240):
    """Smooth spectrum by the discrete-sine-transform method (Hamann et al. 2010, arXiv:1003.3999;
    the pybird/CLASS-PT implementation, 2004.10607): the BAO bump is cut out of the odd and even DST
    harmonics of ln(k P) on a linear grid to 7/Mpc and interpolated over. Input log-log extrapolated.
    Returns (kp, P_nw(kp)) on the linear grid, 1/Mpc and Mpc^3."""
    from scipy.fft import dst
    from scipy.interpolate import interp1d
    kp = np.linspace(1e-7, 7., 2**16)
    ilogpk = interp1d(np.log(k_mpc), np.log(P_mpc), fill_value='extrapolate')
    harmonics = dst(np.log(kp) + ilogpk(np.log(kp)), type=2, norm='ortho')
    odd, even = harmonics[::2], harmonics[1::2]
    nn = np.arange(odd.shape[0]); keep = np.delete(nn, np.arange(n_lo, n_hi))
    s_odd = interp1d(keep, odd[keep], kind='cubic')(nn)
    s_even = interp1d(keep, even[keep], kind='cubic')(nn)
    smooth = dst(np.stack([s_odd, s_even], 1).reshape(-1), type=3, norm='ortho')
    return kp, np.exp(smooth) / kp


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


def rd_h_fixed(c, Neff=3.04):
    """h r_d in Mpc/h, DESI 2024 (2404.03002) eq. 2.5 with the omega_b exponent -0.13. pybird.symbolic.rs_drag
    (model.rd_h) has +0.13: d ln r_d / d ln omega_b = +0.09 instead of -0.17 (a numerical sound-horizon integral gives
    -0.169), i.e. r_d off by 0.65% per BBN sigma of omega_b, the wrong way."""
    om = c['omega_cdm'] + c['omega_b']
    return 147.05 * (om / 0.1432)**-0.23 * (Neff / 3.04)**-0.1 * (c['omega_b'] / 0.02236)**-0.13 * c['h']


def node_grid(k_range_h, spacing, h_fid):
    """Nodes uniform in ln k (1/Mpc) over k_range_h (h/Mpc at h_fid), the spacing rounded to fit."""
    lo, hi = np.log(k_range_h[0] * h_fid), np.log(k_range_h[1] * h_fid)
    n = int(np.round((hi - lo) / spacing)) + 1
    return np.linspace(lo, hi, n)


def ls_projector(nodes_lnk, lnk_knots, w):
    """S (knots x nodes): the cubic node interpolant (constant outside the nodes) as a matrix, and
    Pi = (S^T W S)^-1 S^T W, the weighted least-squares projection of knot values onto the nodes."""
    nodes_lnk, q = jnp.asarray(nodes_lnk), jnp.clip(jnp.asarray(lnk_knots), nodes_lnk[0], nodes_lnk[-1])
    S = np.array(jax.vmap(lambda y: _pb_interp1d(nodes_lnk, y, kind='cubic')(q))(jnp.eye(len(nodes_lnk)))).T
    return S, np.linalg.solve(S.T @ (w[:, None] * S), S.T * w)


class WiggleModel(MIModel):
    """model.MIModel with the parametrization of the module docstring. Everything that only needs
    `idx`/`pidx`, the prior, `phi_ln_full` and `cosmo_dicts` (likelihoods, posteriors, direct models,
    GN Fisher, compress.py) is inherited unchanged."""

    PRIOR_DEFAULT = dict(MIModel.PRIOR_DEFAULT, s_lnrs=0.05, s_env=0.3, s_dsig2=25.0)
    LAMBDA_SPACING = 0.15        # smooth_lambda is quoted at this node spacing (the baseline's)

    def __init__(self, data, cosmo_fid, prior=None, nodes=None, rs_free=False, z_early=5.0, get_maxlkl=True, verbose=True,
                 bao_dm_compat=True, n_env=0, k_env=0.1, rd_fix=True, ls_weight=None, ls_eps=1e-3, env_type='log'):
        self.rs_free, self.n_env, self.k_env = bool(rs_free), int(n_env), float(k_env)
        # wiggle envelope E multiplying O(alpha_rs k):
        #   'log'   E = 1 + sum_j e_j x^j, x = ln(alpha_rs k / k_env)                     (n_env terms)
        #   'gauss' E = (1 + A) exp(-k^2 dSigma^2 / 2) (n_env = 2), or exp(-k^2 dSigma^2 / 2) (n_env = 1):
        #           the BAO-fit form, a BAO amplitude A and a Gaussian damping relative to the fiducial, in k (Mpc^-1)
        assert env_type in ('log', 'gauss') and (env_type == 'log' or self.n_env in (1, 2)), (env_type, n_env)
        self.env_type = env_type
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
        Om_fid = float(Omega_m(self.cosmo_fid))
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
            Pi = np.linalg.solve(self.S_knots.T @ Wm @ self.S_knots, self.S_knots.T @ Wm)
        self.Pi_ls = jnp.array(Pi); self.S_knots_j, self.ls_w_j, self.Wm_j = jnp.array(self.S_knots), jnp.array(self.ls_w), jnp.array(Wm)

        # ---- layout
        sizes = [('eft', self.n_eft), ('lna', self.n_amp), ('f', self.N), ('apar', self.N), ('aper', self.N), ('D', self.N - 1)]
        if self.n_env: sizes.append(('env', self.n_env))
        if self.rs_free: sizes.append(('rs', 1))
        self.idx, i0 = {}, 0
        for name, n in sizes:
            self.idx[name] = np.arange(i0, i0 + n); i0 += n
        self.n_mi = i0; self.n_phys = self.n_mi - self.n_eft
        self.pidx = {k: v - self.n_eft for k, v in self.idx.items() if k != 'eft'}
        zl = [f'{z:.2f}' for z in self.z]
        self.phys_names = ([f'ln a({k:.3g})' for k in self.nodes_h] + [f'ln f({z})' for z in zl] + [f'ln alpha_par({z})' for z in zl]
                           + [f'ln alpha_per({z})' for z in zl] + [f'ln D({z})/D({zl[-1]})' for z in zl[:-1]]
                           + self.env_names() + (['ln alpha_rs'] if self.rs_free else []))
        self.names = self.eft_labels + self.phys_names

        self.extra_loglike = None
        self.set_cosmo_model('lcdm3')
        # the prior is centred on the fiducial mapped with model.rd_h, whatever rd_fix: the MI model (likelihood and
        # prior) does not depend on the direct model's r_d, so chains sampled before rd_fix existed stay exact
        rd_func, self.rd_func = self.rd_func, rd_h
        self.phi_fid = np.asarray(self.phi_ln_full(self.cosmo_fid))
        self.rd_func = rd_func
        # the fiducial maps to a = 1 up to round-off (~1e-9 with a data-weighted map on many nodes)
        assert np.abs(self.phi_fid[self.pidx['lna']]).max() < 1e-6, self.phi_fid[self.pidx['lna']]
        self.prior_mean = self.phi_fid.copy()
        self.prior_prec = self._prior_precision()
        if verbose:
            log(f"WiggleModel ({'A: alpha_rs free' if self.rs_free else 'B: alpha_rs = 1, alphas = BAO alphas'}): "
                f"{self.n_eft} EFT + {self.n_amp} ln a nodes (spacing {self.node_spacing:.3f} in ln k, "
                f"[{self.nodes_h[0]:.3g}, {self.nodes_h[-1]:.3g}] h/Mpc) + {self.N} f + {2*self.N} alphas + {self.N-1} D"
                + (f" + {self.n_env} wiggle envelope" if self.n_env else "") + (" + alpha_rs" if self.rs_free else "") + f" = {self.n_mi} MI parameters; no h_conv")
            log(f"  prior {self.prior}; smoothness lambda_eff = {self.lambda_eff:.3g}; h r_d,fid = {self.rdh_fid:.3f} Mpc/h")

    # ------------------------------------------------------------------------------------
    def _split(self, k_taper=(1e-3, 3e-3), k_grid=(1e-5, 9.0)):
        """T = T_nw (1 + O) at the fiducial, O = T/T_nw - 1 with T_nw from `nowiggle_eh_filter` on a fine ln k
        grid (1/Mpc), tapered to exactly 0 below k_taper (no wiggles there), as a Hermite interpolant;
        T_nw := T / (1 + O), so model A at alpha_rs = 1 is exactly T. The DST split (pybird's) is kept for
        comparison only: it leaves a ~10% non-oscillatory residual near the turnover."""
        from scipy.interpolate import interp1d
        k_cpj = np.asarray(self.pklin.k_modes)
        lnk = np.linspace(np.log(k_grid[0]), np.log(k_grid[1]), 8000)
        P = np.asarray(self.plin_zN(jnp.array(np.exp(lnk)), self.cosmo_fid))
        O_raw = P / nowiggle_eh_filter(lnk, P, self.cosmo_fid) - 1.0
        t = np.clip((lnk - np.log(k_taper[0])) / np.log(k_taper[1] / k_taper[0]), 0, 1)
        O = O_raw * t * t * (3 - 2 * t)
        kp, Pnw = nowiggle_dst(k_cpj, np.asarray(self.plin_zN(jnp.array(k_cpj), self.cosmo_fid)))
        O_dst = P / np.exp(interp1d(np.log(kp[1:]), np.log(Pnw[1:]), bounds_error=False, fill_value='extrapolate')(lnk)) - 1
        self.split_grid = {'k_mpc': np.exp(lnk), 'P': P, 'O': O, 'O_raw': O_raw, 'O_dst': O_dst, 'k_cpj': k_cpj,
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
    def unpack(self, phi_ln):
        """ln phi -> dict of the linear physical quantities (ln a and the envelope coefficients stay as they are)."""
        e = jnp.exp(phi_ln); out = {k: e[v] for k, v in self.pidx.items()}
        out['lna'] = phi_ln[self.pidx['lna']]
        if self.n_env: out['env'] = phi_ln[self.pidx['env']]
        return out

    def env_basis(self, q_mpc):
        """(len(q), n_env): powers of x = ln(q / k_env), q the wavenumber in the frame of the wiggles."""
        x = jnp.log(q_mpc / self.k_env)
        return jnp.stack([x**j for j in range(self.n_env)], 1)

    def env_names(self):
        if self.env_type == 'gauss':
            return (['BAO amplitude A'] if self.n_env == 2 else []) + ['wiggle damping dSigma^2 [Mpc^2]']
        return [f'wiggle envelope e{j}' for j in range(self.n_env)]

    def env_sigmas(self):
        p = self.prior
        if self.env_type == 'gauss':
            return ([p['s_env']] if self.n_env == 2 else []) + [p['s_dsig2']]
        return [p['s_env']] * self.n_env

    def env_E(self, q_mpc, env):
        """The envelope E at the knots (exact); q = alpha_rs k (log type), k = knots (gauss type)."""
        if self.env_type == 'gauss':
            k2 = self.knots_mpc**2
            return (1.0 + (env[0] if self.n_env == 2 else 0.0)) * jnp.exp(-0.5 * k2 * env[-1])
        return 1.0 + self.env_basis(q_mpc) @ env

    def env_cols(self, q_mpc):
        """dE/d(env) at env = 0: the columns of the linearized least-squares map."""
        if self.env_type == 'gauss':
            d = (-0.5 * self.knots_mpc**2)[:, None]
            return jnp.concatenate([jnp.ones_like(d), d], 1) if self.n_env == 2 else d
        return self.env_basis(q_mpc)

    def spectrum_knots(self, lna_nodes, ln_ars=0.0, env=None):
        """P_lin(knots_mpc, z_N) in Mpc^3 of the MI model: a T_nw [1 + E O(k alpha_rs)], E = 1 + sum_j e_j x^j."""
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
            c = dict(c, Omega_m=Omega_m(c) + E.omega_nu() / c['h']**2, rd_h=E.rs_drag(c) * c['h'])
            return c, (lambda k: E.plin(k, c, self.z_N))
        c = dict(c, Omega_m=Omega_m(c), rd_h=self.rd_func(c))
        return c, (lambda k: self.plin_zN(k, c))

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
        O/(1+O) x^j, the linearization of ln[1 + (1 + sum e_j x^j) O] - ln[1 + O] (error (e O)^2/2 ~ 1e-5)."""
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
        f = jnp.array([sym_f(Om, z, w0, wa) for z in self.z])
        DzN = sym_D_early(Om, 1. / (1. + self.z_N), w0, wa)
        D = jnp.array([sym_D_early(Om, 1. / (1. + z), w0, wa) / DzN for z in self.z[:-1]])
        rdh_ref = self.rdh_fid if self.rs_free else rdh
        DH = jnp.array([C_LIGHT / (100. * sym_H(Om, z, w0, wa)) for z in self.z])
        DM = jnp.array([C_LIGHT / 100. * (1. + z) * sym_DA(Om, z, w0, wa) for z in self.z])
        blocks = [lna, jnp.log(f), jnp.log(DH / (rdh_ref * self.DH_fid)), jnp.log(DM / (rdh_ref * self.DM_fid)), jnp.log(D)]
        if self.n_env: blocks.append(env)
        if self.rs_free: blocks.append(jnp.log(jnp.atleast_1d(rdh / self.rdh_fid)))
        return jnp.concatenate(blocks)

    # ------------------------------------------------------------------------------------
    def min_scales(self, kind):
        if kind == 'direct': return super().min_scales('direct')
        s = np.full(self.n_phys, 0.02); s[self.pidx['lna']] = 0.1; s[self.pidx['f']] = 0.05; s[self.pidx['D']] = 0.05
        if self.n_env: s[self.pidx['env']] = 0.05 if self.env_type == 'log' else np.r_[[0.05] * (self.n_env - 1), 2.0]
        return np.concatenate([np.full(self.n_eft, 0.5), s])

    def gauge_shift(self, phi_ln, lam):
        """A only: the near-symmetry (alpha_rs, every alpha) -> lam (alpha_rs, every alpha), with ln a
        re-fitted to the dilated spectrum lam^3 a(lam k) T_nw(lam k) / T_nw(k) (least squares on the knots)."""
        assert self.rs_free
        phi = jnp.asarray(phi_ln); ll = jnp.log(lam)
        q = jnp.clip(self.lnk_knots + ll, self.nodes_lnk[0], self.nodes_lnk[-1])
        lna_dil = _pb_interp1d(self.nodes_lnk, phi[self.pidx['lna']], kind='cubic')(q)
        target = lna_dil + jnp.log(self.T_nw(self.knots_mpc * lam) / self.T_nw_knots) + 3 * ll
        phi = phi.at[self.pidx['lna']].set(self.Pi_ls @ target)
        for blk in ('apar', 'aper', 'rs'): phi = phi.at[self.pidx[blk]].add(ll)
        return phi
