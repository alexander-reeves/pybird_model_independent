"""Timing of the cosmology -> MI-parameter map of Ag245F: the least-squares projection alone (fit_spectrum given the
spectrum), the CosmoPower spectrum, the full map phi(theta), and one MI likelihood, per cosmology, jitted and vmapped
over 1000 draws of the lcdm5 true-likelihood chain.      SCRIPT=wiggle_time_map.py sbatch exec_wiggle.sbatch
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import jax, jax.numpy as jnp
import settings as S
import wiggle_settings as W
from model import COSMO_MODELS
from wiggle_direct import rdfix_files
from sampling import log

cv = 'lcdm5'; sv = S.resolve(cv); keys = COSMO_MODELS[sv['cosmo_model']]
M = W.build(W.resolve('Ag245F', cv), verbose=False)[0]; M.set_cosmo_model(sv['cosmo_model'], gauss=sv['cosmo_gauss'])
fx = dict(M.cosmo_fixed)
x = np.load(rdfix_files(cv, 'exact')[1]['chain_direct'])['x']; x = x.reshape(-1, x.shape[-1])
x = jnp.array(x[np.random.default_rng(0).choice(len(x), 1000, replace=False)]); e, th = x[:, :M.n_eft], x[:, M.n_eft:]

def full(t): return M.engine(M.theta_to_full(t, keys, fx))
def target(t):
    c, plin = full(t); return M.lna_target_knots(c, plin)
def proj(t):
    c, plin = full(t); return jnp.concatenate(M.fit_spectrum(c, plin))
phi = lambda t: M.phi_ln_full(M.theta_to_full(t, keys, fx))
lik = lambda e_, t: M.loglkl_phi(e_, phi(t))

def timeit(name, f, *a, n=5):
    g = jax.jit(jax.vmap(f)); jax.block_until_ready(g(*a))            # compile
    t0 = time.perf_counter()
    for _ in range(n): jax.block_until_ready(g(*a))
    dt = (time.perf_counter() - t0) / n / len(a[0])
    log(f"{name:55s} {dt * 1e6:10.2f} us per cosmology"); return dt

log(f"devices {jax.devices()}; {len(th)} cosmologies, vmapped")
t_spec = timeit("spectrum + target r(k) on the 80 knots (CosmoPower)", target, th)
t_proj = timeit("spectrum + target + least-squares projection", proj, th)
t_phi = timeit("full map phi(theta) (+ f, D, alphas, alpha_rs)", phi, th)
t_lik = timeit("full map + MI likelihood (pybird one loop)", lik, e, th)
log(f"=> the projection itself costs {(t_proj - t_spec) * 1e6:.2f} us per cosmology "
    f"({(t_proj - t_spec) / t_lik:.1e} of one likelihood evaluation)")
