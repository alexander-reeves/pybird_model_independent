"""Jacobians of the P(k) and P(k)+B(k) model vectors, for the square-root Fisher analysis in
robust_marg_pb.py, plus direct finite-difference tests of the model's exact symmetries.

WHY A SEPARATE SCRIPT. Every marginalization in run_fisher_pb.py goes through a Fisher MATRIX:
either the autodiff Hessian (PSD only to ~1e-7 of its top eigenvalue) or the Gauss-Newton
J^T P J, which is PSD but, being formed explicitly, SQUARES the condition number -- so its
eigenvalues below ~1e-8 of the maximum are already unreliable in double precision.
Marginalizing 80 free template amplitudes out of the geometry needs exactly those small
eigenvalues, and Gate 9 showed the answer moving by 25x with the pseudo-inverse threshold.
Here the Jacobian itself is kept; robust_marg_pb.py whitens it and works with its SVD, whose
singular values carry the SQUARE ROOT of the condition number (twice the usable dynamic range).

WHAT IT RUNS. Cells 1-13 of run_fisher_pb.py, verbatim (setup, template, cosmology mappings,
model builder, fake data, the two likelihoods, the model-vector machinery) -- there is no copy
of the model code here, so the two scripts cannot drift apart. Then, for P and for P+B:
  * J = d(model vector)/d(theta), theta = [all EFT coefficients per sky, 80 template
    amplitudes, 25 growth parameters], by forward-mode JVPs in chunks (memory-bounded);
  * the data precision matrix and the explicit EFT prior precision;
  * FINITE-DIFFERENCE tests of the three exact symmetries of the continuum model
      g1  ruler:      h_conv -> q h_conv, D_A -> q D_A, H -> H/q at every z (template FIXED)
      g2  amplitude:  a_j -> a_j e^s, D(z) -> D(z) e^(-s/2)
      g3  template dilation: template k-axis stretched by lambda, h_conv -> lambda h_conv
    g1 is exact because pybird-dev's AP carries the full volume factor, 1/(q_perp^2 q_par) for
    P(k) and (q_par q_perp^2)^-2 for B(k): the h/Mpc ruler length (h_conv) and a common
    rescaling of every distance are then the same parameter. Delta chi^2 = dm^T P dm along each
    measures how far the DISCRETIZED model (80 knots, emulator re-sampling, AP interpolation,
    fixed-scale IR resummation) is from that symmetry, i.e. how much spurious "information" the
    numerics feed into a direction the physics says is unmeasurable.
Output: ../../output/fisher_pb/jacobians_pb.npz
"""
import os, re, sys, time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.chdir(HERE)

# ---- execute cells 1-13 of run_fisher_pb.py in one namespace ---------------------------
_src = open(os.path.join(HERE, 'run_fisher_pb.py')).read()
_parts = re.split(r'^# %% \[cell (\d+)\][^\n]*\n', _src, flags=re.M)
_cells = {int(_parts[i]): _parts[i + 1] for i in range(1, len(_parts), 2)}
NS = {'__name__': 'run_fisher_pb_setup'}
for _c in (1, 3, 5, 7, 9, 11, 13):
    exec(compile(_cells[_c], f'run_fisher_pb.py [cell {_c}]', 'exec'), NS)

np, jnp, jax, log = NS['np'], NS['jnp'], NS['jax'], NS['log']
L = NS['L']
make_model_vector, eft_prior_precision = NS['make_model_vector'], NS['eft_prior_precision']
num_skies, n_knots, n_growth, n_z_unique = (NS['num_skies'], NS['n_knots'], NS['n_growth'],
                                            NS['n_z_unique'])
growth_fid = np.asarray(NS['growth_fid'])
idx_f, idx_H, idx_DA = NS['idx_f'], NS['idx_H'], NS['idx_DA']
fiducial_nuisance = NS['fiducial_nuisance']
knots_h, knots_mpc = np.asarray(NS['knots_h']), NS['knots_mpc']
template_mpc = np.asarray(NS['template_mpc'])
pklin_cpj, cosmo_fid, h_true, z_ref = NS['pklin_cpj'], NS['cosmo_fid'], NS['h_true'], NS['z_ref']
figdir = NS['figdir']
CHUNK = int(os.environ.get('JAC_CHUNK', '16'))


def jacobian_chunked(f, x, chunk=CHUNK):
    """Forward-mode Jacobian, `chunk` tangent directions at a time. Memory is one
    linearization plus `chunk` tangents, instead of all n at once as with jax.jacfwd."""
    y, f_lin = jax.linearize(f, x)
    n = x.shape[0]
    eye = jnp.eye(n, dtype=x.dtype)
    blocks = [np.asarray(jax.vmap(f_lin)(eye[s:s + chunk])).T for s in range(0, n, chunk)]
    return np.asarray(y), np.concatenate(blocks, axis=1)


def template_at(k_mpc):
    return np.asarray(pklin_cpj(k_mpc, cosmo_fid['omega_b'], cosmo_fid['omega_cdm'], h_true,
                                cosmo_fid['ln10^{10}A_s'], cosmo_fid['n_s'], z_ref))


# exact log-slope of the template function at the knots (for the linear g3 direction)
def _lnT(lnk):
    return jnp.log(pklin_cpj(jnp.exp(lnk), cosmo_fid['omega_b'], cosmo_fid['omega_cdm'], h_true,
                             cosmo_fid['ln10^{10}A_s'], cosmo_fid['n_s'], z_ref))
dlnT_dlnk = np.asarray(jax.jvp(_lnT, (jnp.log(knots_mpc),), (jnp.ones(n_knots),))[1])

out = dict(knots_h=knots_h, template_mpc=template_mpc, dlnT_dlnk=dlnT_dlnk, h_true=h_true,
           zeff_unique=np.asarray(NS['zeff_unique']), growth_fid=growth_fid,
           idx_f=np.asarray(idx_f), idx_H=np.asarray(idx_H), idx_DA=np.asarray(idx_DA),
           n_knots=n_knots, n_growth=n_growth, n_z_unique=n_z_unique, num_skies=num_skies,
           sky_to_z_idx=np.asarray(NS['sky_to_z_idx']))

# no-wiggle version of the fiducial template (for the band-recovery figure)
try:
    from pybird.cosmo import get_smooth_wiggle
    _k, _pk, _pnw, _pw = get_smooth_wiggle(knots_h, template_mpc * h_true**3, h=None,
                                           method='gaussian')
    pnw = np.interp(np.log(knots_h), np.log(np.asarray(_k)), np.asarray(_pnw)) / h_true**3
    out['template_nw_mpc'] = pnw
    log("no-wiggle template from pybird's gaussian log-k filter")
except Exception as e:                                          # noqa: BLE001
    log(f"pybird get_smooth_wiggle unavailable ({e!r}); no-wiggle template not saved")

models = {}
for tag, key in [('P', 'p'), ('P+B', 'pb')]:
    Li = L[tag]
    t1 = time.time()
    mv, names = make_model_vector(Li)
    n_e = len(names)
    x0 = np.concatenate([np.tile([fiducial_nuisance[nm] for nm in names], num_skies),
                         np.ones(n_knots), growth_fid])
    y0, J = jacobian_chunked(mv, jnp.array(x0))
    y = np.asarray(Li.y_all)
    rel = np.max(np.abs(y0 - y) / np.maximum(np.abs(y), 1e-30))
    log(f"[{tag}] J: {J.shape[0]} data x {J.shape[1]} params ({n_e}x{num_skies} EFT + "
        f"{n_knots} amps + {n_growth} growth) in {time.time()-t1:.0f}s; "
        f"max relative residual at fiducial {rel:.1e}")
    out.update({f'J_{key}': J, f'P_{key}': np.asarray(Li.p_all),
                f'prior_{key}': eft_prior_precision(Li, names), f'x0_{key}': x0,
                f'y_{key}': y, f'eft_names_{key}': np.array(names), f'n_e_{key}': n_e,
                f'nsky_data_{key}': np.array([len(Li.y_sky[i]) for i in range(Li.nsky)])})
    models[tag] = (mv, names, x0)

# which entries of the P+B vector are P(k): per sky the vector is [P multipoles ; B0]
n_p = out['nsky_data_p']; n_pb = out['nsky_data_pb']
isB = np.concatenate([np.r_[np.zeros(n_p[i], bool), np.ones(n_pb[i] - n_p[i], bool)]
                      for i in range(num_skies)])
out['isB_pb'] = isB
same = np.allclose(out['y_pb'][~isB], out['y_p'], rtol=0, atol=0)
log(f"P entries of the P+B data vector identical to the P-only vector: {same}")

# ---- finite-difference symmetry tests ---------------------------------------------------
log("--- exact-symmetry tests: Delta chi^2 of the DISCRETIZED model along directions the "
    "continuum model leaves exactly invariant ---")
sym_tests = {}
for tag, key in [('P', 'p'), ('P+B', 'pb')]:
    mv, names, x0 = models[tag]
    P = out[f'P_{key}']
    n_e = len(names)
    o_a, o_g = n_e * num_skies, n_e * num_skies + n_knots
    ih = o_g + n_growth - 1
    iH = np.array([o_g + i for i in idx_H]); iD = np.array([o_g + i for i in idx_DA])
    iDr = np.array([o_g + 3 * n_z_unique + i for i in range(n_z_unique)])
    m0 = np.asarray(mv(jnp.array(x0)))

    def dchi2(x):
        d = np.asarray(mv(jnp.array(x))) - m0
        return float(d @ P @ d)

    rows = []
    for eps in (1e-3, 1e-2):
        q = np.exp(eps)
        x = x0.copy(); x[ih] *= q                                   # h_conv alone (reference)
        c_h = dchi2(x)
        x = x0.copy(); x[ih] *= q; x[iD] *= q; x[iH] /= q           # g1 ruler
        c_g1 = dchi2(x)
        x = x0.copy(); x[o_a:o_g] *= q; x[iDr] /= np.sqrt(q)        # g2 amplitude
        c_g2 = dchi2(x)
        x = x0.copy()                                               # g3 template dilation
        x[o_a:o_g] = q**-3 * template_at(knots_mpc / q) / template_mpc
        x[ih] *= q
        c_g3 = dchi2(x)
        x = x0.copy(); x[iD] *= q                                   # D_A alone at every z
        c_da = dchi2(x)
        rows.append((eps, c_h, c_g1, c_g2, c_g3, c_da))
        log(f"  {tag:4s} eps={eps:.0e}:  h_conv alone {c_h:.3e} | g1 ruler {c_g1:.3e} "
            f"(= {c_g1/c_h:.1e} of h_conv alone) | g2 amplitude {c_g2:.1e} | g3 template "
            f"dilation {c_g3:.3e} (= {c_g3/c_h:.1e}) | D_A alone {c_da:.3e}")
    (e1, h1, g11, _, g31, _), (e2, h2, g12, _, g32, _) = rows
    log(f"  {tag:4s} linearity (expect 100 for a smooth quadratic): h_conv {h2/h1:.1f}, "
        f"g1 {g12/max(g11,1e-300):.1f}, g3 {g32/max(g31,1e-300):.1f};  "
        f"constraint the numerics alone would imply on the ruler: sigma(ln q) = "
        f"{e1/np.sqrt(max(g11,1e-300)):.3g}  (vs {e1/np.sqrt(h1):.2g} for h_conv alone)")
    sym_tests[key] = np.array(rows)
    out[f'sym_tests_{key}'] = np.array(rows)

np.savez(os.path.join(figdir, 'jacobians_pb.npz'), **out)
log(f"saved {os.path.join(figdir, 'jacobians_pb.npz')}")
