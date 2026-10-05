"""Build 07_robust_pk_bk.ipynb: the validated P(k) vs P(k)+B(k) analysis, end to end.

The notebook runs the scripts (they stay the source of truth) and shows their figures:
  jacobian_pb.py      model-vector Jacobians + exact-symmetry tests   (cached npz reused)
  hmc_prior_bands.py  the HMC model as a Fisher, validated on the MI chain; band recovery
  corner_audit_pb.py  the all-z AP corner + the prior audit
  bao_basis_scan.py   why the MI model is not yet a BAO method, and the basis that makes it one
Executed in place by exec_robust_nb.sbatch.  python3 build_robust_nb.py  rewrites the notebook
with cleared outputs.
"""
import json

FIG = '../../output/fisher_pb/'


def md(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.strip('\n')}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': text.strip('\n')}


def show(*names, width=1000):
    return code("from IPython.display import Image, display\n" + "\n".join(
        f"display(Image(filename='{FIG}{n}', width={width}))" for n in names))


cells = [
md(r"""
# P(k) vs P(k)+B(k): the validated, properly marginalized analysis

This notebook supersedes the geometry results of `06_fisher_pk_bk.ipynb`. Three things went
wrong there, and each is fixed here.

1. **A numerical artifact posed as a measurement.** With pybird's AP volume factor, rescaling
   $h_{\rm conv}$, every $D_A$ and $1/H$ together is an **exact** symmetry of the continuum model
   ($h_{\rm conv}$ is the ruler length in Mpc/$h$). Interpolation breaks it at $\sigma \approx 0.05$,
   and the Fisher matrix read that breaking as information. The per-redshift $H$ and $D_A$ of
   06 (~5%, redshift-independent, "halved by B") were that artifact. Here the data's response along
   the exact symmetries is projected out; priors are left free to constrain those directions.
2. **The wrong model.** 06 freed 80 emulator knots with no prior, a degenerate model the HMC never
   samples. Here the Fisher uses the HMC's model exactly: 60 ln-$a$ nodes (spacing 0.15 in ln $k$)
   and the `setups.COMMON` priors ($\sigma_{\ln a}=0.5$, $\lambda=200$, $\sigma_{\ln g}=0.3$,
   $\sigma_{\ln h}=0.05$). It reproduces the HMC chain's per-node widths to **5%**.
3. **Marginalization precision.** Every marginal comes from the whitened Jacobian (SVD or a
   well-conditioned inverse with the priors), never from an explicit $J^TPJ$ at the noise floor.

Superseded figures are in `output/fisher_pb/superseded/` with a README saying why.
"""),
code(r"""
import os
os.chdir(os.path.dirname(os.path.abspath('07_robust_pk_bk.ipynb')))
if not os.path.exists('../../output/fisher_pb/jacobians_pb.npz'):
    %run jacobian_pb.py
else:
    print('using cached ../../output/fisher_pb/jacobians_pb.npz  (delete it to recompute, ~5 min)')
"""),
md(r"""
## 1. The Fisher reproduces the HMC, and the band recovery

v3's own P(k)-only Fisher, put into the HMC's node basis with the HMC's priors, is compared
node-by-node with the standard deviation of the chain (`hmc_v3_p5/chain_mi.npz`). The same priors
are then applied to the P and P+B systems.

Expected: Fisher/chain ≈ 1.05, and the bispectrum barely moves the template bands (≈0.11 → 0.10
per node in the data range) because they are already prior-limited. Note: the absolute $\ln D_A$,
$\ln H$ table this script prints does **not** project the ruler symmetry, so it includes the
artifact; use section 2 for geometry. (Not projecting is right for the chain comparison itself: the
chain samples the same discretized model.)
"""),
code("%run hmc_prior_bands.py"),
show('hmc_prior_bands.png', width=850),
md(r"""
## 2. The all-z AP corner, and are the priors fair?

`corner_ap_all_z_mi.png` is the 12-parameter ($\ln H$, $\ln D_A$ at six redshifts) corner in the HMC
model, with the symmetry breaking projected out of the data. The widths (~4.7%, nearly the same at
every $z$ and for both data sets) are the **ruler**: the direction "every $D_A$ up, every $H$ down"
has $\sigma = 0.165$ in the posterior against $0.156$ from the prior alone, so the data add nothing
to it. Projecting that one direction out (`..._rulerfree.png`) leaves what the data measure,
~1.5–2%.

The audit then asks whether the priors are fair:
- **Effective data-constrained dof** per block: EFT all; template 14 of 60; $h_{\rm conv}$ 0.2 of 1.
- **Real cosmologies are allowed:** a Planck-1σ shift in $\omega_{\rm cdm}$ costs prior
  $\chi^2 = 0.06$, 5σ costs 1.3 (sanity checks: constant or linear ln $a$ give exactly zero
  smoothness penalty).
- **But the ruler is loose:** a 1% dilation of the template costs only $\chi^2 = 0.38$, so the
  prior lets the BAO feature slide by 1.6%.
"""),
code("%run corner_audit_pb.py"),
show('corner_ap_all_z_mi.png', 'corner_ap_all_z_mi_rulerfree.png', width=900),
md(r"""
## 3. Why it is not yet a BAO method, and the basis that makes it one

A BAO oscillation spans 0.60 in ln $k$; the HMC's nodes are 0.15 apart, about four per wiggle, so
the template can slide its own BAO feature. Coarsening the nodes past the BAO period makes the
wiggle shape rigid, which is exactly what a BAO fit assumes, while the broadband stays free. The
BAO parameters $D_M/r_d = \ln D_A - \ln h_{\rm conv}$ and $D_H/r_d = -\ln H - \ln h_{\rm conv}$
are invariant under the ruler symmetry, so they are measurable without any $h$ prior.

Expected at $z=0.71$: $\sigma(D_M/r_d)$ 2.1% → 0.71% and $\sigma(D_H/r_d)$ 2.4% → 1.3% as the
spacing goes 0.15 → ≥0.4, saturating at the fixed-template answer. $F_{\rm AP}$ needs no ruler and
does not move. The bispectrum's gain is on $f$ (×~2), not on the geometry.
"""),
code("%run bao_basis_scan.py"),
show('bao_summary_pb.png', width=1100),
md(r"""
## Summary

| | HMC model (60 nodes) | BAO-rigid basis (16 nodes) |
|---|---|---|
| $\sigma(D_M/r_d)$, $z=0.71$, P → P+B | 2.13% → 1.94% | 0.71% → 0.69% |
| $\sigma(D_H/r_d)$ | 2.37% → 2.17% | 1.30% → 1.26% |
| $\sigma(F_{\rm AP})$ | 1.62% → 1.59% | 1.58% → 1.55% |
| $\sigma(f)$ | 18.5% → 9.7% | 18.5% → 9.6% |

- Absolute $H(z)$ and $D_A(z)$ are set by the $h_{\rm conv}$ and growth priors in this model; no
  data can fix them.
- The bispectrum improves the growth rate by about a factor two and the geometry by a few percent.
- The bispectrum here is tree-level; one-loop is `with_bk_tree_level: False` plus
  `bk_loop_matrix_path` in `run_fisher_pb.py`, then `jacobian_pb.py` again.
"""),
]

nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                                  'name': 'python3'},
                                   'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 5}
for i, c in enumerate(nb['cells']):
    c['id'] = f'c{i:02d}'
json.dump(nb, open('07_robust_pk_bk.ipynb', 'w'), indent=1)
print(f"wrote 07_robust_pk_bk.ipynb: {len(cells)} cells")
