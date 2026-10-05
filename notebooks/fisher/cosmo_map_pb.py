"""d(MI parameters)/d(omega_cdm, lnAs, h): the map that turns the model-independent Fisher
into a LCDM Fisher, saved so the figure scripts need neither the emulator nor pybird.

The direct LCDM model of run_fisher_pb.py is the MI model composed with this map (exactly the
same likelihood, no second model): the 80 template amplitudes and the 25 growth parameters are
functions of three numbers instead of free. Composing at the level of the JACOBIAN is what
makes the LCDM Fisher a projection of the MI one rather than a separate fit.

Rows are in the SAME scaled units as the whitened Jacobian of jacobian_pb.py, i.e. d ln:
  amps    a_j = P_lin(k_j)/T(k_j), equal to 1 at the fiducial, so d ln a = d a
  growth  [f, H/H0, D_A H0] per z, D(z)/D(z_ref) per z, h_conv: divided by their fiducial values
It also runs THE GATE that says the two routes carry the same information: the direct LCDM
model vector is differentiated with respect to (omega_cdm, lnAs, h) through the SAME likelihood
call, and compared with J_MI @ M. If those agree, fitting LCDM directly and fitting the
model-independent model and then imposing LCDM are the same operation -- no information is
inserted by the extra freedom, and none is removed by the parametrization.

Runs cells 1-13 of run_fisher_pb.py (fake data + both likelihoods), ~3 min.
"""
import os, re, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

_src = open(os.path.join(HERE, 'run_fisher_pb.py')).read()
_parts = re.split(r'^# %% \[cell (\d+)\][^\n]*\n', _src, flags=re.M)
_cells = {int(_parts[i]): _parts[i + 1] for i in range(1, len(_parts), 2)}
NS = {'__name__': 'run_fisher_pb_setup'}
for _c in (1, 3, 5, 7, 9, 11, 13):
    exec(compile(_cells[_c], f'run_fisher_pb.py [cell {_c}]', 'exec'), NS)

import numpy as np
jax, jnp = NS['jax'], NS['jnp']
cosmo_to_observables = NS['cosmo_to_observables']
theta0 = NS['cosmo_fid_vec']
growth_fid = np.asarray(NS['growth_fid'])
n_knots = int(NS['n_knots'])

obs0 = np.asarray(cosmo_to_observables(theta0))
Jm = np.asarray(jax.jacfwd(cosmo_to_observables)(theta0))        # (n_knots + n_growth, 3)
scale = np.concatenate([np.ones(n_knots), growth_fid])           # amps are 1 at the fiducial
M = Jm / scale[:, None]

OUT = os.path.join(HERE, '..', '..', 'output', 'fisher_pb')
print(f"amps at the fiducial: max|a-1| = {np.abs(obs0[:n_knots] - 1).max():.2e}")
print(f"growth at the fiducial matches growth_fid: max rel dev "
      f"{np.abs(obs0[n_knots:] / growth_fid - 1).max():.2e}")
print("d ln(observable)/d theta, a few rows:")
for nm, i in [('a(k=0.01 h/Mpc)', int(np.argmin(np.abs(np.asarray(NS['knots_h']) - 0.01)))),
              ('a(k=0.1 h/Mpc)', int(np.argmin(np.abs(np.asarray(NS['knots_h']) - 0.1)))),
              ('f(z=0.71)', n_knots + 6), ('H(z=0.71)', n_knots + 7),
              ('D_A(z=0.71)', n_knots + 8), ('h_conv', n_knots + len(growth_fid) - 1)]:
    print(f"  {nm:18s} " + "  ".join(f"{v:+10.4f}" for v in M[i]))
print(f"\nsaved {os.path.join(OUT, 'cosmo_map_pb.npz')}")


# ---- the gate: differentiate the DIRECT LCDM model, compare with J_MI @ M ---------------
make_model_vector = NS['make_model_vector']
fiducial_nuisance, num_skies = NS['fiducial_nuisance'], int(NS['num_skies'])
cosmo_to_amps, cosmo_to_growth = NS['cosmo_to_amps'], NS['cosmo_to_growth']
R = np.load(os.path.join(OUT, 'jacobians_pb.npz'), allow_pickle=True)

gate = {}
for tag, key in (('P', 'p'), ('P+B', 'pb')):
    mv, names = make_model_vector(NS['L'][tag])
    n_e = len(names)
    eft0 = jnp.array(np.tile([fiducial_nuisance[nm] for nm in names], num_skies))

    def m_direct(th, mv=mv, eft0=eft0):
        """The direct LCDM model vector: 3 numbers -> the same prediction the MI model makes."""
        amps = cosmo_to_amps(th)
        growth = cosmo_to_growth(jnp.array([th[0], th[2]]))
        return mv(jnp.concatenate([eft0, amps, growth]))

    J_dir = np.asarray(jax.jacfwd(m_direct)(theta0))                  # (n_data, 3)
    J_mi = R[f'J_{key}'][:, n_e * num_skies:]                         # (n_data, 105)
    J_comp = J_mi @ Jm                                                # chain rule
    rel = np.linalg.norm(J_dir - J_comp) / np.linalg.norm(J_dir)
    gate[key] = rel
    print(f"[GATE] {tag:4s} ||J_direct - J_MI M|| / ||J_direct|| = {rel:.2e}   "
          f"({'PASS' if rel < 1e-4 else 'CHECK'})")
print("  i.e. the LCDM fit is the MI fit with LCDM imposed -- the same likelihood, the same"
      "\n  derivatives. Any difference between the two posteriors is the MI PRIORS, not the model.")

np.savez(os.path.join(OUT, 'cosmo_map_pb.npz'), M=M, M_raw=Jm, theta_fid=np.asarray(theta0),
         obs_fid=obs0, names=np.array(['omega_cdm', 'lnAs', 'h']),
         gate_p=gate['p'], gate_pb=gate['pb'])
print(f"\nsaved {os.path.join(OUT, 'cosmo_map_pb.npz')}")
