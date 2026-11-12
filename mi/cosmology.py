"""Cosmology outside the EFT likelihood: linear growth for w0waCDM, the EDE emulator, the sound horizon, theta_s.

Growth. pybird.symbolic._D is Heath's integral, which solves the growth equation only for w = -1: for w0wa its f is wrong by
-21% at z = 0.1 at the DESI+CMB best fit. `growth` solves u'' + (2 + P) u' + (1 + P - Q) u = 0 for u = D/a in x = ln a
(P = 2 + dlnE/dx, Q = 3/2 Om(a), u = 1, u' = 0 at a = 1e-3) by Chebyshev collocation on 33 nodes (one linear solve; relative
error <= 4e-6 in f against RK4). LCDM given as concrete numbers keeps pybird's integral, exact there. `growth_general` solves
the same equation for any expansion history E(a) (the growth_from_h template: growth derived from a spline of H(z)).

EDE. The CosmoPower `ede-v2` emulators (ACT DR6 extended models, arXiv:2503.14454; github.com/cosmopower-organization/ede)
in pure JAX, from the `_v2_plain.npz` weights in data/emulators/cosmopower_ede/ (not in git): PKL_v2 = log10[k^3 P_lin] on
geomspace(5e-4, 10, 1000)/Mpc (PKLCB_v2: cdm + baryons only), DER_v2 index 0 = 100 theta_s, 1 = sigma_8, 13 = r_d [Mpc].
Neutrinos: 3 degenerate species of m_ncdm eV each (0.02 unless given; the emulators cover 0 to 0.33), so the same engine
is the neutrino-mass model `mnu` in its LCDM limit.

Sound horizon. rd_h_fixed is DESI 2024's fit (2404.03002 eq. 2.5); pybird.symbolic.rs_drag has the omega_b exponent with the
wrong sign (+0.13), 0.65% in r_d per BBN sigma of omega_b.

theta_s. 100 r_s(z_rec)/D_M(z_rec) in CLASS's definition with massless neutrinos (as the Planck PR4 early-universe
anchor 100 theta_s = 1.040268 +- 0.000246 of Lemos & Lewis 2023, data/planck_early_lcdm). LCDM: a cubic polynomial in
(omega_b, omega_cdm, h) fitted to CLASS (data/mi_chains/theta_s_poly.npz, checks.py theta; within 9e-6 of CLASS off the
grid); w0wa: the same r_s with D_M(z_rec) of the w0wa background (within 1e-5 of CLASS). EDE engine (ede, mnu): that
polynomial times the ede-v2 emulator's ratio to its own LCDM limit with 3 x 0.02 eV, the mass of the Planck chains the
anchor was made from, so only ratios come from the emulator (its absolute value there is 3.6e-4 = 1.5 sigma off CLASS).
The anchor assumes standard physics before recombination: for EDE it is a consistency test of the projection, not a CMB fit.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from pybird import config as _pb_config
_pb_config.set_jax_enabled(True)
from pybird.symbolic import f as _f_heath, _D as _D_heath

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THETA_S = (1.040268, 0.000246)                 # 100 theta_s of the Planck PR4 early-universe Gaussian (tau marginalized)


# ------------------------------------------------------------------------------------------ growth
A0, NCHEB = 1e-3, 32
X0 = np.log(A0)
_t = np.cos(np.pi * np.arange(NCHEB + 1) / NCHEB)                       # Chebyshev points; t = 1 is today, t = -1 is a = A0
_c = np.r_[2., np.ones(NCHEB - 1), 2.] * (-1.) ** np.arange(NCHEB + 1)
_Dt = np.outer(_c, 1 / _c) / (_t[:, None] - _t[None, :] + np.eye(NCHEB + 1)); _Dt -= np.diag(_Dt.sum(1))
_X = X0 + (_t + 1) / 2 * (0. - X0); _DX = _Dt * 2 / (0. - X0); _DX2 = _DX @ _DX
_W = (-1.) ** np.arange(NCHEB + 1); _W[0] *= 0.5; _W[-1] *= 0.5


def _interp_matrix(x):
    """Barycentric interpolation from the collocation nodes to the points x = ln a (concrete)."""
    d = (2 * (np.asarray(x, float) - X0) / (0. - X0) - 1)[:, None] - _t[None, :]; on = np.abs(d) < 1e-14
    M = _W / np.where(on, 1.0, d); M /= M.sum(1, keepdims=True); r = on.any(1); M[r] = on[r]
    return M


def uses_ode(w0, wa):
    """False only for LCDM given as concrete numbers; free (traced) w0, wa always take the collocation solution."""
    try: return not (float(w0) == -1.0 and float(wa) == 0.0)
    except Exception: return True


def _solve(Om, w0, wa):
    """u = D/a and u' = du/dln a on the collocation nodes."""
    a = jnp.exp(_X); rm = Om * a**-3; rde = (1 - Om) * a**(-3 * (1 + w0 + wa)) * jnp.exp(-3 * wa * (1 - a)); e2 = rm + rde
    return _solve_pq(2 - 1.5 * (rm + (1 + w0 + wa * (1 - a)) * rde) / e2, 1.5 * rm / e2)


def _solve_pq(P, Q):
    """u = D/a and u' on the collocation nodes for P = 2 + dlnE/dln a and Q = 3/2 Omega_m(a) given there."""
    L = _DX2 + (2 + P)[:, None] * _DX + jnp.diag(1 + P - Q)
    L = L.at[-1].set(jnp.eye(NCHEB + 1)[-1]).at[-2].set(_DX[-1]); b = jnp.zeros(NCHEB + 1).at[-1].set(1.0)
    u = jnp.linalg.solve(L, b)
    return u, _DX @ u


def growth(Om, w0, wa, zs, ode=None):
    """(D(z), f(z) = dlnD/dlna) at the redshifts zs (concrete), D = a in matter domination. ode=None: uses_ode(w0, wa)."""
    zs = [float(z) for z in np.atleast_1d(zs)]
    if not (uses_ode(w0, wa) if ode is None else ode):
        return jnp.array([_D_heath(Om, 1. / (1. + z), w0, wa) for z in zs]), jnp.array([_f_heath(Om, z, w0, wa) for z in zs])
    x = -np.log1p(np.array(zs)); M = _interp_matrix(x); u, du = _solve(Om, w0, wa); ue = M @ u
    return jnp.exp(x) * ue, 1 + (M @ du) / ue


def growth_general(lne2, rm, zs):
    """(D(z), f(z)) at the concrete redshifts zs for any expansion history: lne2(x) = ln E^2 and rm(x) = Omega_m (1+z)^3 in the
    same units, JAX functions of x = ln a over [ln 1e-3, 0]. Matter domination at a = 1e-3, GR, smooth dark energy, scale-
    independent growth: P = 2 + (1/2) dlnE^2/dx (autodiff), Q = (3/2) rm / E^2."""
    X = jnp.array(_X); x = -np.log1p(np.array([float(z) for z in np.atleast_1d(zs)])); M = _interp_matrix(x)
    u, du = _solve_pq(2 + 0.5 * jax.vmap(jax.grad(lne2))(X), 1.5 * rm(X) / jnp.exp(jax.vmap(lne2)(X))); ue = M @ u
    return jnp.exp(x) * ue, 1 + (M @ du) / ue


def growth_ratio(Om, w0, wa, z, z_early):
    """D(z; w0, wa) / D(z_early; LCDM): CosmoPower's LCDM spectrum at z_early grown to z."""
    if not uses_ode(w0, wa):
        return _D_heath(Om, 1. / (1. + float(z)), w0, wa) / _D_heath(Om, 1. / (1. + float(z_early)), -1., 0.)
    return growth(Om, w0, wa, z, True)[0][0] / growth(Om, -1., 0., z_early, True)[0][0]


# ------------------------------------------------------------------------------------------ sound horizon
def rd_h_fixed(c, Neff=3.04):
    """h r_d in Mpc/h, DESI 2024 (2404.03002) eq. 2.5 with the omega_b exponent -0.13. pybird.symbolic.rs_drag
    (model.rd_h) has +0.13: d ln r_d / d ln omega_b = +0.09 instead of -0.17 (a numerical sound-horizon integral gives
    -0.169), i.e. r_d off by 0.65% per BBN sigma of omega_b, the wrong way."""
    om = c['omega_cdm'] + c['omega_b']
    return 147.05 * (om / 0.1432)**-0.23 * (Neff / 3.04)**-0.1 * (c['omega_b'] / 0.02236)**-0.13 * c['h']


# ------------------------------------------------------------------------------------------ theta_s
POLY_FILE = os.path.join(ROOT, 'data', 'mi_chains', 'theta_s_poly.npz')
OMEGA_R = 2.4728e-5 * (1. + 0.22711 * 3.044)                                 # photons + 3.044 massless nu, T_cmb 2.7255


def _feats(ob, oc, h, xp=np):
    u = [(ob - 0.02223) / 0.002, (oc - 0.1192) / 0.01, (h - 0.675) / 0.075]
    cols = [xp.ones_like(u[0])] + list(u)
    for i in range(3):
        for j in range(i, 3): cols.append(u[i] * u[j])
    for i in range(3):
        for j in range(i, 3):
            for k in range(j, 3): cols.append(u[i] * u[j] * u[k])
    return xp.stack(cols, axis=-1)


_COEF = None
def theta_s_lcdm(ob, oc, h):
    global _COEF
    if _COEF is None: _COEF = np.load(POLY_FILE)['coef']
    return _feats(ob, oc, h, xp=jnp) @ jnp.array(_COEF)


def z_rec(ob, om):
    """Hu & Sugiyama (1996) fit to the redshift of photon decoupling."""
    g1 = 0.0783 * ob**-0.238 / (1. + 39.5 * ob**0.763)
    g2 = 0.560 / (1. + 21.1 * ob**1.81)
    return 1048. * (1. + 0.00124 * ob**-0.738) * (1. + g1 * om**g2)


def D_M(z, h, om, w0=-1., wa=0., n=4000):
    """Comoving distance [Mpc] to z for flat w0wa with radiation (massless neutrinos), trapezoid in ln(1+z)."""
    Om, Orad = om / h**2, OMEGA_R / h**2
    x = jnp.linspace(0., jnp.log1p(z), n); zz = jnp.expm1(x); a = 1. / (1. + zz)
    de = (1. - Om - Orad) * a**(-3. * (1. + w0 + wa)) * jnp.exp(-3. * wa * (1. - a))
    E = jnp.sqrt(Orad * (1. + zz)**4 + Om * (1. + zz)**3 + de)
    f = (1. + zz) / E
    return 2997.92458 / h * jnp.sum(0.5 * (f[1:] + f[:-1]) * jnp.diff(x))


def theta_s(c):
    """100 theta_s for a full cosmology dict (omega_b, omega_cdm, h, w0, wa); massless neutrinos."""
    ob, oc, h = c['omega_b'], c['omega_cdm'], c['h']
    th = theta_s_lcdm(ob, oc, h)
    if c.get('fEDE') is not None:
        E = ede_emulator()
        return th * E.derived(c)[0] / E.derived(dict(c, **LCDM_LIMIT))[0]
    if isinstance(c['w0'], (int, float)) and c['w0'] == -1. and isinstance(c['wa'], (int, float)) and c['wa'] == 0.:
        return th
    zr = z_rec(ob, ob + oc)
    return th * D_M(zr, h, ob + oc) / D_M(zr, h, ob + oc, c['w0'], c['wa'])


# ------------------------------------------------------------------------------------------ EDE
EMU_DIR = os.path.join(ROOT, 'data', 'emulators', 'cosmopower_ede')
DEFAULTS = {'r': 0.0, 'm_ncdm': 0.02, 'N_ur': 0.00441, 'tau_reio': 0.054}
DEG_NCDM = 3
LCDM_LIMIT = {'fEDE': 0.001, 'log10z_c': 3.562, 'thetai_scf': 2.83, 'm_ncdm': 0.02}     # of the EDE engine: no EDE, sum m_nu = 0.06 eV
K_PKL = np.geomspace(5e-4, 10., 1000)                     # 1/Mpc


class CPNet:
    """A CosmoPower dense network (cosmopower_NN) from a `_plain.npz` file, in JAX."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        # float64 numpy constants (not jnp arrays): safe to build anywhere, including under a trace
        grab = lambda f: [np.asarray(z[f'{f}.{i}'], float) for i in range(int(z[f'{f}.n']))]
        self.W, self.b, self.alpha, self.beta = grab('weights_'), grab('biases_'), grab('alphas_'), grab('betas_')
        self.pmean, self.pstd = np.asarray(z['param_train_mean'], float), np.asarray(z['param_train_std'], float)
        self.fmean, self.fstd = np.asarray(z['feature_train_mean'], float), np.asarray(z['feature_train_std'], float)
        self.parameters = [str(p) for p in z['parameters']]

    def __call__(self, x):
        """x: inputs in the order of self.parameters (last axis) -> raw outputs (log10 for P(k), DER)."""
        h = (x - self.pmean) / self.pstd
        for W, b, a, be in zip(self.W[:-1], self.b[:-1], self.alpha, self.beta):
            y = h @ W + b
            h = (be + (1. - be) * jax.nn.sigmoid(a * y)) * y
        return (h @ self.W[-1] + self.b[-1]) * self.fstd + self.fmean

    def inputs(self, c):
        return jnp.stack([jnp.asarray(c[p], dtype=float) for p in self.parameters])

    def training_box(self):
        lo, hi = self.pmean - np.sqrt(3) * self.pstd, self.pmean + np.sqrt(3) * self.pstd
        return {p: (float(l), float(h)) for p, l, h in zip(self.parameters, lo, hi)}


def class_inputs(c, z=None):
    """Our cosmology dict (omega_cdm, omega_b, n_s, lnAs, h, fEDE, log10z_c, thetai_scf) -> emulator inputs."""
    d = dict(DEFAULTS)
    d.update({'fEDE': c['fEDE'], 'log10z_c': c['log10z_c'], 'thetai_scf': c['thetai_scf'], 'ln10^{10}A_s': c['lnAs'],
              'n_s': c['n_s'], 'H0': 100. * c['h'], 'omega_b': c['omega_b'], 'omega_cdm': c['omega_cdm']})
    if 'm_ncdm' in c: d['m_ncdm'] = c['m_ncdm']
    if z is not None: d['z_pk_save_nonclass'] = z
    return d


class EDEEmulator:
    def __init__(self, emu_dir=EMU_DIR):
        self.pkl = CPNet(os.path.join(emu_dir, 'PKL_v2_plain.npz'))
        self.pklcb = CPNet(os.path.join(emu_dir, 'PKLCB_v2_plain.npz'))
        self.der = CPNet(os.path.join(emu_dir, 'DER_v2_plain.npz'))
        self.lnk = np.log(K_PKL)

    def plin(self, k_mpc, c, z, cb=False):
        """P_lin(k [1/Mpc], z) in Mpc^3 of the total matter (cb: of cdm + baryons, what galaxies trace), log-log interpolated;
        below the emulator's kmin = 5e-4/Mpc the spectrum is continued as a power law with the slope of its first two modes."""
        net = self.pklcb if cb else self.pkl
        lp = net(net.inputs(class_inputs(c, z))) * np.log(10.) - 3. * self.lnk     # ln P
        lk = jnp.log(k_mpc)
        slope = (lp[1] - lp[0]) / (self.lnk[1] - self.lnk[0])
        inside = jnp.interp(lk, self.lnk, lp)
        return jnp.exp(jnp.where(lk < self.lnk[0], lp[0] + slope * (lk - self.lnk[0]), inside))

    def derived(self, c):
        return 10.**self.der(self.der.inputs(class_inputs(c)))

    def rs_drag(self, c):
        """r_d in Mpc."""
        return self.derived(c)[13]

    @staticmethod
    def omega_nu(c=None):
        return DEG_NCDM * (c or {}).get('m_ncdm', DEFAULTS['m_ncdm']) / 93.14


_EDE = None
def ede_emulator():
    """One shared EDEEmulator (its networks are numpy constants: safe to build anywhere)."""
    global _EDE
    if _EDE is None: _EDE = EDEEmulator()
    return _EDE
