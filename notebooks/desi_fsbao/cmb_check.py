"""Builds the theta_s polynomial (if absent) and checks cmb.py against CLASS: LCDM off the fitting grid and
w0wa (CLASS fluid, w0_fld / wa_fld) for theta_s, then what the Gaussian implies for h and Omega_m."""
import os, sys
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
import cmb
from sampling import log

if __name__ == '__main__':
    if not os.path.exists(cmb.POLY_FILE):
        coef, res = cmb.build_poly(); log(f"theta_s polynomial built: rms residual {res.std():.2e}, max {np.abs(res).max():.2e}")
    z = np.load(cmb.POLY_FILE); log(f"theta_s polynomial: {len(z['theta'])} CLASS points, rms {float(z['rms']):.2e}, max {float(z['max']):.2e}; "
                                    f"sigma(100 theta_s) of the CMB {np.sqrt(cmb.COV[2, 2]):.2e}")
    from classy import Class
    def class_theta(ob, oc, h, w0=None, wa=None):
        C = Class(); p = {'omega_b': ob, 'omega_cdm': oc, 'h': h}
        if w0 is not None: p.update({'Omega_Lambda': 0., 'w0_fld': w0, 'wa_fld': wa})
        C.set(p); C.compute(); d = C.get_current_derived_parameters(['z_rec', 'rs_rec', 'da_rec'])
        v = 100 * d['rs_rec'] / (d['da_rec'] * (1 + d['z_rec'])); C.struct_cleanup(); C.empty(); return v
    rng = np.random.default_rng(1)
    pts = [(rng.uniform(0.020, 0.024), rng.uniform(0.10, 0.14), rng.uniform(0.60, 0.78)) for _ in range(6)]
    d = [float(cmb.theta_s_lcdm(*p)) - class_theta(*p) for p in pts]
    log(f"LCDM, off-grid points: max |ours - CLASS| = {np.abs(d).max():.2e}")
    for w0, wa in [(-1., 0.), (-0.8, -0.6), (-0.6, -1.4), (-1.2, 0.5), (-0.45, -1.8)]:
        ob, oc, h = 0.0223, 0.119, 0.66
        c = {'omega_b': ob, 'omega_cdm': oc, 'h': h, 'w0': w0, 'wa': wa}
        ours, cl = float(cmb.theta_s(c)), class_theta(ob, oc, h, w0, wa)
        log(f"w0wa ({w0:+.2f}, {wa:+.2f}): ours {ours:.6f}  CLASS {cl:.6f}  diff {ours-cl:+.2e} ({(ours-cl)/np.sqrt(cmb.COV[2,2]):+.3f} sigma)")
    # what the Gaussian alone says in LCDM: draw, solve theta_s(omega_b, omega_cdm, h) for h
    x = rng.multivariate_normal(cmb.MU, cmb.COV, 4000)
    def solve_h(t, ob, oc):
        h = 0.67
        for _ in range(30): h = h - (float(cmb.theta_s_lcdm(ob, oc, h)) - t) / float(jax.grad(lambda hh: cmb.theta_s_lcdm(ob, oc, hh))(h))
        return h
    hs = np.array([solve_h(t, ob, oc) for t, ob, oc in x[:800, 2:5]]); Om = (x[:800, 3] + x[:800, 4]) / hs**2
    log(f"CMB (late-time marginalized) alone, LCDM: h = {hs.mean():.4f} +- {hs.std():.4f}, Omega_m = {Om.mean():.4f} +- {Om.std():.4f}, "
        f"omega_m = {(x[:,3]+x[:,4]).mean():.4f} +- {(x[:,3]+x[:,4]).std():.4f}")
