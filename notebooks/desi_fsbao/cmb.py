"""The CMB marginalized over late-time physics, as a Gaussian likelihood on the direct model's parameters.

Planck PR4 TTTEEE + lowE + lensing with the late-time effects marginalized (empirical lensing spline +
ISW template; Lemos & Lewis 2023, arXiv:2302.12911, chains in data/planck_early_lcdm/). Its Gaussian in
[ln10^10 A_s, n_s, 100 theta_s, omega_b, omega_cdm, tau] was derived in ../planck_early_lcdm_gaussian.ipynb
with theta_s in CLASS's definition (100 r_s(z_rec)/D_M(z_rec), massless neutrinos, the same for our model);
tau is marginalized here, since nothing else in the analysis sees it.

theta_s of the direct model:
  LCDM  a degree-3 polynomial in (omega_b, omega_cdm, h) fitted to a CLASS grid (cached in
        data/emulators/theta_s_poly.npz; residual 1e-6 against sigma(100 theta_s) = 2.5e-4)
  w0wa  the same r_s, with D_M(z_rec) of the w0wa background: theta_s = theta_s^LCDM D_M^LCDM / D_M^w0wa,
        both integrated here with radiation (photons + 3.044 massless neutrinos), z_rec from Hu & Sugiyama
        (1996). Validated against CLASS fluid w0wa runs by cmb_check.py.
Valid for any model that leaves the universe before recombination LCDM (w0wa, a late fifth force); NOT
for EDE, which changes r_s and the damping tail.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

from model import ROOT

NAMES = ['lnAs', 'n_s', 'theta_s', 'omega_b', 'omega_cdm']                 # after marginalizing tau
_MU6 = np.array([3.027556, 0.964835, 1.040268, 0.022232, 0.119202, 0.048633])
_SD6 = np.array([0.017520, 0.004695, 0.000246, 0.000146, 0.001303, 0.008329])
_CR6 = np.array([[1.0000, -0.0443, -0.0017, -0.0312, 0.1056, 0.9349], [-0.0443, 1.0000, 0.2747, 0.5224, -0.7106, 0.0889],
                 [-0.0017, 0.2747, 1.0000, 0.2347, -0.2833, 0.0341], [-0.0312, 0.5224, 0.2347, 1.0000, -0.6059, 0.0496],
                 [0.1056, -0.7106, -0.2833, -0.6059, 1.0000, -0.0717], [0.9349, 0.0889, 0.0341, 0.0496, -0.0717, 1.0000]])
MU = _MU6[:5]
COV = (_CR6 * np.outer(_SD6, _SD6))[:5, :5]                                  # marginal over tau = drop it
PREC = np.linalg.inv(COV)
POLY_FILE = os.path.join(ROOT, 'data', 'emulators', 'theta_s_poly.npz')
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


def _class_theta_s(a):
    from classy import Class
    ob, oc, h = a; c = Class(); c.set({'omega_b': ob, 'omega_cdm': oc, 'h': h}); c.compute()
    d = c.get_current_derived_parameters(['z_rec', 'rs_rec', 'da_rec'])
    v = 100.0 * d['rs_rec'] / (d['da_rec'] * (1.0 + d['z_rec'])); c.struct_cleanup(); c.empty()
    return v


def build_poly(n_proc=32):
    """Fit the theta_s polynomial on a CLASS grid and cache it (run once, ~2 min)."""
    import multiprocessing
    from itertools import product
    os.environ['OMP_NUM_THREADS'] = '1'
    grid = list(product(np.linspace(0.0195, 0.0245, 9), np.linspace(0.095, 0.150, 13), np.linspace(0.55, 0.82, 15)))
    with multiprocessing.get_context('spawn').Pool(n_proc) as pool: vals = pool.map(_class_theta_s, grid, chunksize=8)
    X, y = np.array(grid), np.array(vals)
    coef, *_ = np.linalg.lstsq(_feats(*X.T), y, rcond=None)
    res = y - _feats(*X.T) @ coef
    os.makedirs(os.path.dirname(POLY_FILE), exist_ok=True)
    np.savez(POLY_FILE, coef=coef, grid=X, theta=y, rms=res.std(), max=np.abs(res).max())
    return coef, res


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
    if isinstance(c['w0'], (int, float)) and c['w0'] == -1. and isinstance(c['wa'], (int, float)) and c['wa'] == 0.:
        return th
    zr = z_rec(ob, ob + oc)
    return th * D_M(zr, h, ob + oc) / D_M(zr, h, ob + oc, c['w0'], c['wa'])


def loglike(c):
    """ln L_CMB for a full cosmology dict (needs lnAs, n_s, omega_b, omega_cdm, h, w0, wa)."""
    d = jnp.array([c['lnAs'], c['n_s'], theta_s(c), c['omega_b'], c['omega_cdm']]) - jnp.array(MU)
    return -0.5 * d @ jnp.array(PREC) @ d
