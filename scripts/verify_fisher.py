#!/usr/bin/env python3
"""
Verify and recompute the Fisher information decomposition.
Loads Fisher blocks from fisher_results.npz and recomputes:
  - Jacobian with corrected cosmo_to_observables
  - 3×3 combined Fisher F_comb = Jᵀ F_phys J and exact block sum Q_pp + Q_cross + Q_gg
  - (a)(b) as 2×2 slices of Q_pp and Q_gg (not additive with cross — do not expect (a)+(b)=(c))
  - Sigma table and triangle plot

Run from pybird_model_independent/ root:
  source /cluster/project/refregier/areeves/.venvs/pybird-mi/bin/activate
  JAX_PLATFORMS=cpu python scripts/verify_fisher.py
"""
import os
import sys
sys.path.insert(0, 'scripts')

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from pybird import config as pybird_config
pybird_config.set_jax_enabled(True)
from pybird.symbolic import Symbolic, f, Hubble, DA, D
from pybird.symbolic_pofk_linear import plin_emulated

from utils import DESI_Y6, to_Mpc_jax

PROJECT_ROOT = os.path.abspath('.')
OUTPUT_DIR   = os.path.join(PROJECT_ROOT, 'output')

# ── Fiducial cosmology ────────────────────────────────────────────────────────
cosmo_fid = {
    'omega_b':      0.02235,
    'omega_cdm':    0.120,
    'h':            0.675,
    'ln10^{10}A_s': 3.044,
    'n_s':          0.965,
}
h_true = cosmo_fid['h']
h_fid  = h_true
z_ref = 3.0

s           = DESI_Y6
zeff_list   = s['zeff']
zeff_unique = sorted(set(zeff_list))
n_z_unique  = len(zeff_unique)

KMIN, KMAX, DK = 0.01, 0.2, 0.01
k_data = np.arange(KMIN, KMAX, DK)

n_growth_per_z  = 3
n_growth_ratios = n_z_unique
n_growth        = n_z_unique * n_growth_per_z + n_growth_ratios + 1

cosmo_fid_vec = jnp.array([cosmo_fid['omega_cdm'], cosmo_fid['ln10^{10}A_s'], cosmo_fid['h']])

# ── Load saved Fisher matrices ────────────────────────────────────────────────
npz = np.load(os.path.join(OUTPUT_DIR, 'fisher_results.npz'), allow_pickle=True)
h_ref_grid = float(npz['h_ref_grid']) if 'h_ref_grid' in npz.files else h_true * 1.1

F_pk_block        = npz['F_pk_block']
F_growth_block    = npz['F_growth_block']
F_pk_marginal     = npz['F_pk_marginal']
F_growth_marginal = npz['F_growth_marginal']
F_cosmo_combined  = npz['F_cosmo_combined']
F_cosmo_direct    = npz['F_cosmo_direct']
F_phys_marg       = npz['F_phys_marg']
n_knots           = F_pk_block.shape[0]

if 'knots_h' in npz.files:
    knots_h = np.asarray(npz['knots_h'], dtype=float).ravel()
else:
    knots_h = np.asarray(k_data, dtype=float)
if len(knots_h) != n_knots:
    raise ValueError(
        f"k grid mismatch: F_pk_block has n_knots={n_knots}, knots_h has {len(knots_h)}. "
        "Re-run the notebook to refresh fisher_results.npz, or set KMIN,KMAX,DK to match it."
    )

knots_h_jax = jnp.array(knots_h)
knots_mpc_jax = jnp.array(knots_h / h_ref_grid)

M_sym = Symbolic()
M_sym.set(cosmo_fid)
M_sym.compute(knots_h, 0.5)
Omega0_m = M_sym.c['Omega_m']

omega_b_np = cosmo_fid['omega_b']
A_s_fid = 1e-10 * jnp.exp(cosmo_fid['ln10^{10}A_s'])
Om0 = (cosmo_fid['omega_cdm'] + omega_b_np) / h_true**2
Ob0 = omega_b_np / h_true**2
pk_at_knots_h = 1e9 * plin_emulated(
    knots_h_jax, A_s_fid, Om0, Ob0, h_fid, cosmo_fid['n_s'],
    0.0, -1.0, 0.0, a=1.0 / (1 + z_ref),
)
pk_at_knots_jax = to_Mpc_jax(pk_at_knots_h, knots_h_jax, h_fid, knots_mpc_jax)
pk_at_knots_mpc = np.array(pk_at_knots_jax)
knots_mpc = np.array(knots_mpc_jax)

print(f"Loaded Fisher matrices from {OUTPUT_DIR}/fisher_results.npz")
print(f"  h_ref_grid = {h_ref_grid:.4f}, Omega_m(fid) = {Omega0_m:.4f}, n_knots = {n_knots}")
print(f"  F_phys_marg: {F_phys_marg.shape}")

# ── cosmo_to_observables (corrected) ─────────────────────────────────────────
def cosmo_to_observables(cosmo_params):
    """Map (omega_cdm, ln10As, h) -> (pk_amps [n_knots], growth block).

    P(k): plin_emulated and Mpc conversion use cosmo h (emu / notebook §3).
    Growth: Omega_m = omega_m/h^2; h_conv = h.
    """
    omega_cdm, lnAs, h = cosmo_params[0], cosmo_params[1], cosmo_params[2]
    omega_b = cosmo_fid['omega_b']
    n_s     = cosmo_fid['n_s']
    mnu, w0, wa = 0.0, -1.0, 0.0

    A_s = 1e-10 * jnp.exp(lnAs)

    Om_pk = (omega_cdm + omega_b) / h**2
    Ob_pk = omega_b / h**2
    pk_h  = 1e9 * plin_emulated(
        knots_h_jax, A_s, Om_pk, Ob_pk, h, n_s, mnu, w0, wa,
        a=1.0 / (1 + z_ref)
    )
    pk_mpc  = to_Mpc_jax(pk_h, knots_h_jax, h, knots_mpc_jax)
    pk_amps = pk_mpc / pk_at_knots_jax

    # Growth: uses varying h; no A_s dependence
    Omega_m = (omega_cdm + omega_b) / h**2

    growth_list, D_values = [], []
    for z in zeff_unique:
        growth_list.extend([
            f(Omega_m, z, w0, wa),
            Hubble(Omega_m, z, w0, wa),
            DA(Omega_m, z, w0, wa),
        ])
        D_values.append(D(Omega_m, z, w0, wa))

    D_ref_val = D(Omega_m, z_ref, w0, wa)
    for Dz in D_values:
        growth_list.append(Dz / D_ref_val)

    growth_list.append(h)  # h_conv
    return jnp.concatenate([pk_amps, jnp.array(growth_list)])


# ── Verify at fiducial ────────────────────────────────────────────────────────
obs_fid = cosmo_to_observables(cosmo_fid_vec)
print(f"\nFiducial check:")
print(f"  pk_amps min={float(obs_fid[:n_knots].min()):.4f}, max={float(obs_fid[:n_knots].max()):.4f}  (expect ~1)")
print(f"  h_conv  = {float(obs_fid[-1]):.4f}  (expect {h_fid})")

# ── Jacobian ──────────────────────────────────────────────────────────────────
print("\nComputing Jacobian d(observables)/d(omega_cdm, ln10As, h)...")
J = np.array(jax.jacobian(cosmo_to_observables)(cosmo_fid_vec))
print(f"  Jacobian shape: {J.shape}  (75 observables x 3 cosmo params)")

J_pk     = J[:n_knots, :]
J_growth = J[n_knots:, :]

print(f"\n  Structural checks:")
print(f"  max|J(pk_amps,  h)|    = {np.abs(J_pk[:, 2]).max():.2e}  (small at high z_ref)")
print(f"  max|J(growth, ln10As)| = {np.abs(J_growth[:, 1]).max():.2e}  (should be 0)")
print(f"  d(h_conv)/d(h)         = {J[-1, 2]:.4f}  (should be 1.0)")

# ── Fisher projections: exact 3×3 block sum + 2×2 block slices for (a)(b) ───
J_pk_2d = J_pk[:, :2]
J_growth_2d = J_growth[:, [0, 2]]

F_pp, F_gg = F_pk_block, F_growth_block
F_pg = F_phys_marg[:n_knots, n_knots:]
F_gp = F_phys_marg[n_knots:, :n_knots]
Q_pp = J_pk.T @ F_pp @ J_pk
Q_pg = J_pk.T @ F_pg @ J_growth
Q_gp = J_growth.T @ F_gp @ J_pk
Q_gg = J_growth.T @ F_gg @ J_growth
F_cosmo_cross = Q_pg + Q_gp
F_cosmo_combined_new = J.T @ F_phys_marg @ J
F_cosmo_block_sum = Q_pp + F_cosmo_cross + Q_gg
_nc = np.linalg.norm(F_cosmo_combined_new, ord='fro') + 1e-30
print(f"\nBlock closure ||F_comb-(Q_pp+Q_cross+Q_gg)||_F = "
      f"{np.linalg.norm(F_cosmo_combined_new - F_cosmo_block_sum, ord='fro'):.3e} (rel {_nc:.3e})")

F_cosmo_pk_only_2d = Q_pp[:2, :2]
F_cosmo_growth_only_2d = Q_gg[np.ix_([0, 2], [0, 2])]

# Regularise combined if needed
ev_co = np.linalg.eigvalsh(F_cosmo_combined_new)
if ev_co.min() < 0:
    F_cosmo_combined_new += (abs(ev_co.min()) + 1e-12) * np.eye(3)
    ev_co = np.linalg.eigvalsh(F_cosmo_combined_new)

ev_pk = np.linalg.eigvalsh(F_cosmo_pk_only_2d)
ev_gr = np.linalg.eigvalsh(F_cosmo_growth_only_2d)
ev_di = np.linalg.eigvalsh(F_cosmo_direct)

print(f"\nFisher eigenvalue checks:")
print(f"  F_pk_only_2d     min eigval: {ev_pk.min():.3e}  (should be > 0)")
print(f"  F_growth_only_2d min eigval: {ev_gr.min():.3e}  (should be > 0)")
print(f"  F_combined_new   min eigval: {ev_co.min():.3e}  (should be > 0)")
print(f"  F_direct         min eigval: {ev_di.min():.3e}  (should be > 0)")

# ── Sigma table ───────────────────────────────────────────────────────────────
s_pk = np.sqrt(np.diag(np.linalg.inv(F_cosmo_pk_only_2d)))
s_gr = np.sqrt(np.diag(np.linalg.inv(F_cosmo_growth_only_2d)))
s_co = np.sqrt(np.diag(np.linalg.inv(F_cosmo_combined_new)))
s_di = np.sqrt(np.diag(np.linalg.inv(F_cosmo_direct)))

results = {
    '(a) Q_pp slice':   np.array([s_pk[0], s_pk[1], np.nan]),
    '(b) Q_gg slice': np.array([s_gr[0], np.nan,   s_gr[1]]),
    '(c) Combined':    s_co,
    '(d) Direct':      s_di,
}

print(f"\nSigma comparison table:")
print(f"  {'':20s}  {'omega_cdm':>12}  {'ln10As':>12}  {'h':>12}")
print("  " + "-"*62)
for label, sig in results.items():
    row = "  ".join(f"{s:>12.5f}" if np.isfinite(s) else f"{'---':>12}" for s in sig)
    print(f"  {label:<20s}  {row}")

print(f"\n  Ratios (sigma / sigma_direct):")
print(f"  {'':20s}  {'omega_cdm':>12}  {'ln10As':>12}  {'h':>12}")
print("  " + "-"*62)
for label in ['(a) Q_pp slice', '(b) Q_gg slice', '(c) Combined']:
    ratios = results[label] / results['(d) Direct']
    row = "  ".join(f"{r:>12.3f}" if np.isfinite(r) else f"{'---':>12}" for r in ratios)
    print(f"  {label:<20s}  {row}")

print("""
  Expected physics:
    (c) Combined should match (d) Direct (same Jᵀ F_phys J).
    (a) and (b) are only the Q_pp and Q_gg block slices; (a)+(b) does not equal (c) when Q_cross ≠ 0.
    ln10As: block (b) has no A_s row — pad for triangle; combined still constrains A_s via P(k) block + cross.
""")

# ── Triangle plot ─────────────────────────────────────────────────────────────
try:
    from getdist import MCSamples, plots

    sigma_combined = np.sqrt(np.diag(np.linalg.inv(F_cosmo_combined_new)))
    pad = 5.0

    # P(k)-only 3D: h direction wide
    F_pk_3d = np.zeros((3, 3))
    F_pk_3d[:2, :2] = F_cosmo_pk_only_2d
    F_pk_3d[2, 2]   = (pad * sigma_combined[2])**(-2)

    # Growth-only 3D: ln10As direction wide
    F_gr_3d = np.zeros((3, 3))
    F_gr_3d[np.ix_([0, 2], [0, 2])] = F_cosmo_growth_only_2d
    F_gr_3d[1, 1] = (pad * sigma_combined[1])**(-2)

    rng = np.random.default_rng(2026)
    fid_np  = np.array(cosmo_fid_vec)
    n_samp  = 20000
    names_3d  = ['omega_cdm', 'ln10As', 'h']
    labels_3d = [r'\omega_{\rm cdm}', r'\ln(10^{10}A_s)', r'h']
    colors    = ['#000000', '#43A047', '#E53935', '#1E88E5']

    gd_list, gd_labels = [], []
    for label, F, col in [
        ('(d) Direct',           F_cosmo_direct,       colors[0]),
        ('(c) Combined',         F_cosmo_combined_new, colors[1]),
        ('(a) P(k) $Q_{pp}$',    F_pk_3d,              colors[2]),
        ('(b) Growth $Q_{gg}$', F_gr_3d,              colors[3]),
    ]:
        ev = np.linalg.eigvalsh(F)
        if ev.min() < 0:
            F = F + (abs(ev.min()) + 1e-8) * np.eye(F.shape[0])
        try:
            samps = rng.multivariate_normal(fid_np, np.linalg.inv(F), n_samp)
            gd_list.append(MCSamples(samples=samps, names=names_3d,
                                     labels=labels_3d, label=label))
            gd_labels.append(label)
        except Exception as e:
            print(f"  {label}: failed ({e})")

    g = plots.get_subplot_plotter(width_inch=10)
    g.triangle_plot(gd_list, params=names_3d,
                    filled=[True, False, False, False],
                    contour_colors=colors[:len(gd_list)],
                    legend_labels=gd_labels)
    plt.suptitle('Information Decomposition: DESI 6-sky Mock\n'
                 r'(a)(b) = $Q_{pp}$, $Q_{gg}$ slices; combined includes $Q_{\rm cross}$)',
                 y=1.02, fontsize=13)
    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, 'fisher_triangle.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Triangle plot saved to {outpath}")
except Exception as e:
    print(f"Triangle plot failed: {e}")

# ── Save updated results ──────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, 'fisher_results.npz')
np.savez(out_path,
         F_full            = npz['F_full'],
         F_phys_marg       = F_phys_marg,
         F_pk_block        = npz['F_pk_block'],
         F_growth_block    = npz['F_growth_block'],
         F_pk_marginal     = F_pk_marginal,
         F_growth_marginal = F_growth_marginal,
         F_cosmo_pk_only   = F_cosmo_pk_only_2d,
         F_cosmo_growth_only = F_cosmo_growth_only_2d,
         F_cosmo_cross     = F_cosmo_cross,
         F_cosmo_block_sum = F_cosmo_block_sum,
         Q_pp              = Q_pp,
         Q_gg              = Q_gg,
         F_cosmo_combined  = F_cosmo_combined_new,
         F_cosmo_direct    = F_cosmo_direct,
         J                 = J,
         params_fid        = npz['params_fid'],
         knots_h           = knots_h,
         knots_mpc         = knots_mpc,
         pk_at_knots_mpc   = pk_at_knots_mpc,
         h_ref_grid        = np.array(h_ref_grid),
         zeff_unique       = np.array(zeff_unique),
         cosmo_fid_vec     = np.array(cosmo_fid_vec))
print(f"Saved updated results to {out_path}")
