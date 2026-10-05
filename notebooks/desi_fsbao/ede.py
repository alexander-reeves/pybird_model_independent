"""Early dark energy (axion-like, n = 3; fEDE, log10 z_c, theta_i) from the CosmoPower `ede-v2`
emulators of the ACT DR6 extended-models analysis (arXiv:2503.14454, also Poulin et al. 2025,
arXiv:2505.08051), https://github.com/cosmopower-organization/ede, in pure JAX.

Files (the `_v2_plain.npz` weights, no TensorFlow) in data/emulators/cosmopower_ede/:
  PKL_v2    log10[k^3 P_lin(k, z)] on k = geomspace(5e-4, 10, 1000) 1/Mpc, P in Mpc^3 (total matter)
  DER_v2    10^output = 17 derived parameters; index 1 sigma8, 4 Neff, 12 ra_star, 13 rs_drag [Mpc]
Inputs: fEDE, log10z_c, thetai_scf, ln10^{10}A_s, n_s, H0, omega_b, omega_cdm, r, m_ncdm, N_ur
(+ z for P(k), + tau_reio for DER). Neutrinos as class_sz 'ede-v2': N_ncdm = 1 with deg_ncdm = 3,
m_ncdm = 0.02 eV each (sum 0.06), N_ur = 0.00441 (N_eff = 3.044). The training box (uniform, read
off the stored normalization) is printed by `training_box()`: fEDE [0, 0.5], log10 z_c [3, 4.3],
theta_i [0.1, 3.1], H0 [40, 100], omega_cdm [0.08, 0.2], omega_b [0.019, 0.025], n_s [0.8, 1.2],
ln10^10 A_s [2.5, 3.5], z [0, 20].
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

from model import ROOT

EMU_DIR = os.path.join(ROOT, 'data', 'emulators', 'cosmopower_ede')
DEFAULTS = {'r': 0.0, 'm_ncdm': 0.02, 'N_ur': 0.00441, 'tau_reio': 0.054}
DEG_NCDM = 3
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
    if z is not None: d['z_pk_save_nonclass'] = z
    return d


class EDEEmulator:
    def __init__(self, emu_dir=EMU_DIR):
        self.pkl = CPNet(os.path.join(emu_dir, 'PKL_v2_plain.npz'))
        self.der = CPNet(os.path.join(emu_dir, 'DER_v2_plain.npz'))
        self.lnk = np.log(K_PKL)

    def plin(self, k_mpc, c, z):
        """P_lin(k [1/Mpc], z) in Mpc^3, log-log interpolated; below the emulator's kmin = 5e-4/Mpc the
        spectrum is continued as a power law with the slope of its first two modes."""
        lp = self.pkl(self.pkl.inputs(class_inputs(c, z))) * np.log(10.) - 3. * self.lnk     # ln P
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
    def omega_nu():
        return DEG_NCDM * DEFAULTS['m_ncdm'] / 93.14
