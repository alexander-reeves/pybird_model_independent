# Model-independent EFTofLSS: notebooks

The linear power spectrum, the growth and the distances are fitted freely to DESI DR1 full shape + BAO; cosmologies are
then fitted by projecting that model-independent (MI) posterior. Two templates for the linear spectrum:
**agnostic** (a free a(k) T(k) on 60 nodes, no BAO template) and **bao_template** (a(k) on 21 nodes times the fiducial BAO
wiggles, moved by α_rs, with an amplitude and a damping). Every DESI-alone fit uses BBN ω_b and n_s ~ N(0.9649, 0.042).

| notebook | what |
|---|---|
| [`1_fisher.ipynb`](1_fisher.ipynb) | Fisher forecast of both templates on DESI DR1 FS+BAO, against the chains |
| [`2_sampling.ipynb`](2_sampling.ipynb) | the MI chains, the representation test, recovery of direct ΛCDM / w0waCDM / EDE fits, projections with and without θ_s |
| [`3_sectors.ipynb`](3_sectors.ipynb) | ΛCDM from shape \| evolution and from background \| perturbations |
| [`4_master_plot.ipynb`](4_master_plot.ipynb) | the reconstructed P(k) of both templates, and the master figure with and without θ_s |
| [`5_fisher_pk_bk.ipynb`](5_fisher_pk_bk.ipynb) | P(k) vs P(k)+B(k) Fisher forecast on a DESI-Y6 mock (the original 60-node model) |
| [`6_alpha_rs_degeneracy.ipynb`](6_alpha_rs_degeneracy.ipynb) | is α_rs of the BAO template degenerate with an isotropic BAO dilation? the gauge T_λ, what breaks it, Fisher; data and noiseless mock (`checks.py degeneracy`) |
| [`7_alpha_rs_comparison.ipynb`](7_alpha_rs_comparison.ipynb) | the BAO template with α_rs free vs α_rs = 1 (`bao_template_no_rs`, the baseline): chains, MI posterior, representation, recovery, ΛCDM / w0wa, sectors, P(k), Fisher; and why the two differ for w0waCDM: the units of the EFT priors, with the direct fits in either units and α_rs = 1 with the priors adapted per cosmology |
| [`8_validation.ipynb`](8_validation.ipynb) | the baseline against direct fits: ΛCDM, w0waCDM, EDE on DESI DR1 without and with θ_s; noiseless mocks with a w0waCDM and an EDE truth; neutrino mass on the data and on a mock |
| [`9_growth_from_expansion.ipynb`](9_growth_from_expansion.ipynb) | `growth_from_h`: f, D and the α derived from a spline of H(z) and the growth equation instead of free per redshift; where that is valid; the data's χ² test of it; the prior-volume shift of the growth marginals |
| [`10_y6_pk_bk.ipynb`](10_y6_pk_bk.ipynb) | the reconstruction on the DESI-Y6 mock with the power spectrum alone and with the tree-level bispectrum: matched best fits and Fisher (no chains: too slow through pybird-dev) |

Notebooks 1–4 and 6–10 read only `../data/mi_chains*/` (numpy, matplotlib, getdist); the code that produced it is in
[`mi/`](../mi/README.md). Notebook 5 and its code are in [`mi/pk_bk/`](../mi/pk_bk/README.md).

## Getting the chains

The notebooks read the exported chains and products in `data/mi_chains*/` (about 200 MB), which are not in git. Unzip the
archive `mi_chains_<date>.zip` in the repository root:

    cd pybird_model_independent
    unzip mi_chains_2026-11-11.zip

It creates `data/mi_chains/` (DESI DR1), `data/mi_chains_mock_w0wa/`, `data/mi_chains_mock_ede/`, `data/mi_chains_mock_mnu/`
(the noiseless mocks) and `data/mi_chains_y6_p/`, `data/mi_chains_y6_pb/` (the DESI-Y6 mock). Three small inputs of the
pipeline are in git already (`data/mi_chains/knot_fisher.npz`, `theta_s_poly.npz`, `planck_early_lcdm_samples.npz`); the
archive holds identical copies. The notebooks then need only numpy, matplotlib and getdist.

To make the products yourself instead: run the pipeline (`../mi/README.md`), then `python run.py export`.
