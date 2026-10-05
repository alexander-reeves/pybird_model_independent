"""Checks of the CosmoPower ede-v2 emulators (ede.py) before using them as a direct model:
LCDM limit against CLASS (classy 3.3.4, same neutrinos) and against our CosmoPower-LCDM template,
the size of the EDE effect on P_lin(k, z_N) and r_d, and JAX differentiability."""
import os, sys, time
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
from ede import EDEEmulator, DEFAULTS, DEG_NCDM
from model import PkLinCPJ, rd_h, sym_D_early, Omega_m
from sampling import log

E = EDEEmulator()
log("PKL training box: " + ", ".join(f"{k} [{a:.4g}, {b:.4g}]" for k, (a, b) in E.pkl.training_box().items()))
log("DER training box (tau_reio): " + str(E.der.training_box()['tau_reio']))
fid = {'omega_cdm': 0.12, 'omega_b': 0.02237, 'h': 0.6736, 'lnAs': 3.044, 'n_s': 0.9649, 'w0': -1., 'wa': 0.}
lcdm = dict(fid, fEDE=0.001, log10z_c=3.562, thetai_scf=2.83)
zN = 1.491
k = np.geomspace(1e-3, 0.5, 200)                                    # 1/Mpc
P_e = np.asarray(E.plin(jnp.array(k), lcdm, zN))

# CLASS, same neutrinos as the emulator training
from classy import Class
C = Class()
C.set({'omega_b': fid['omega_b'], 'omega_cdm': fid['omega_cdm'], 'h': fid['h'], 'ln10^{10}A_s': fid['lnAs'], 'n_s': fid['n_s'],
       'N_ncdm': 1, 'deg_ncdm': DEG_NCDM, 'm_ncdm': DEFAULTS['m_ncdm'], 'N_ur': DEFAULTS['N_ur'], 'tau_reio': DEFAULTS['tau_reio'],
       'output': 'mPk', 'P_k_max_1/Mpc': 12., 'z_max_pk': 3.})
C.compute()
P_c = np.array([C.pk_lin(kk, zN) for kk in k]); rd_c = C.rs_drag()
r = P_e / P_c
log(f"LCDM limit (fEDE=0.001) vs CLASS at z_N={zN}: P ratio in [{r.min():.4f}, {r.max():.4f}], rms dev {np.sqrt(np.mean((r-1)**2)):.4f}; "
    f"r_d emu {float(E.rs_drag(lcdm)):.3f} vs CLASS {rd_c:.3f} Mpc; sigma8 emu {float(E.derived(lcdm)[1]):.4f} vs CLASS {C.sigma8():.4f}")
# our CosmoPower-LCDM route (z = 5 grown to z_N) and the analytic r_d
cpj = PkLinCPJ()
g = sym_D_early(Omega_m(fid), 1 / (1 + zN)) / sym_D_early(Omega_m(fid), 1 / 6.)
P_t = np.asarray(cpj(jnp.array(k), fid, 5.0)) * float(g)**2
r2 = P_e / P_t
log(f"LCDM limit vs our template route (CPJ z=5 x D^2): ratio in [{r2.min():.4f}, {r2.max():.4f}]; r_d analytic fit {float(rd_h(fid))/fid['h']:.3f} Mpc")

# the EDE effect at fixed (omega_cdm, omega_b, h, A_s, n_s)
for fe in (0.05, 0.1, 0.2):
    c = dict(lcdm, fEDE=fe)
    re = np.asarray(E.plin(jnp.array(k), c, zN)) / P_e
    log(f"fEDE={fe}: P(k,z_N)/P_LCDM in [{re.min():.3f}, {re.max():.3f}] (k=0.01: {np.interp(0.01, k, re):.3f}, k=0.2: {np.interp(0.2, k, re):.3f}); "
        f"r_d {float(E.rs_drag(c)):.2f} Mpc ({100*(float(E.rs_drag(c))/float(E.rs_drag(lcdm))-1):+.2f}%)")

# differentiability and speed
f = jax.jit(lambda fe: jnp.sum(jnp.log(E.plin(jnp.array(k), dict(lcdm, fEDE=fe), zN))))
gfe = jax.grad(lambda fe: jnp.log(E.plin(jnp.array([0.1]), dict(lcdm, fEDE=fe), zN))[0])(0.1)
f(0.1); t0 = time.time(); [f(0.1).block_until_ready() for _ in range(100)]
log(f"d ln P(k=0.1/Mpc)/d fEDE at fEDE=0.1: {float(gfe):+.3f}; jitted P_lin evaluation {1e3*(time.time()-t0)/100:.2f} ms")
