# Fisher forecasts: P(k) vs P(k)+B(k) in the model-independent parametrization

These are forecasts for a DESI-Y6-like mock. They ask how much the tree-level bispectrum adds to the model-independent
(MI) reconstruction of the linear spectrum, the growth and the geometry. The bispectrum JAX code lives in the
**pybird-dev** tree, which `run_fisher_pb.py` puts first on `sys.path` (`PYBIRD_DEV` overrides the location); see
`PLAN_merge_bispectrum.md` for how to make it permanent here.

**Start with the two presentation notebooks, `08_growth_pk_bk.ipynb` and `09_bao_pk_bk.ipynb`.**

- **`08`:** linear P(k) recovery (11.3% → 10.2% per node when B is added), the growth sector (σ(f) 18.5% → 9.7% at
  z = 0.71), and the ΛCDM recovery fitted directly and through the MI parametrization (agreement 1–3%).
- **`09`:** the alternative BAO pipeline: a free template, then + smoothness prior (D_M/r_d 2.1%), then + the BAO phase
  prior (0.88%, against 0.65% for standard pre-reconstruction BAO on the same mock).

## Files

| file | role |
|---|---|
| `build_meeting_nbs.py` → `08_growth_pk_bk.ipynb`, `09_bao_pk_bk.ipynb` (`exec_meeting_nbs.sbatch`; scripts `meeting_growth_pb.py`, `meeting_bao_pb.py`, `meeting_bao_overlay_pb.py`, `meeting_bao_triangle_free.py`, `cosmo_map_pb.py`, `cosmo_fisher_pb.py`, `prior_decomposition_pb.py`) | The presentation notebooks above, plus the prior audit. The smoothness penalty λ is the only prior block that informs ΛCDM. The MI prior is not flat along ΛCDM directions, so it must be divided out when projecting. Figures in `output/fisher_pb/meeting/` |
| `jacobian_pb.py` + `robust_marg_pb.py` (`exec_jacobian_pb.sbatch`, `exec_robust_pb.sbatch`) | Square-root Fisher analysis of P vs P+B, the version to quote. The model Jacobians are cached in `output/fisher_pb/jacobians_pb.npz` (~5 min to rebuild). Finite-difference tests check the three exact symmetries (ruler h_conv ↔ uniform AP; amplitude a·D²; template dilation), which are then projected out. Every marginal comes from the SVD of the whitened Jacobian |
| `mi_prior_fisher.py` | The Fisher system of the model the HMC samples (60 nodes + the HMC priors), validated against a real chain to 5%. It also holds the BAO phase prior (`phase_modes`) and the wide-prior recommendation (`PRIORS_WIDE`) |
| `build_robust_nb.py` → `07_robust_pk_bk.ipynb` (`exec_robust_nb.sbatch`; scripts `hmc_prior_bands.py`, `corner_audit_pb.py`, `bao_basis_scan.py`) | The audit version: per-node widths vs the chain, the all-z AP corner with the ruler symmetry projected out, and the node-spacing scan (per-z BAO at 0.7% / 1.3% once the basis cannot slide the wiggles) |
| `run_fisher_pb.py` → `06_fisher_pk_bk.ipynb` (`build_fisher_pb_nb.py`, `exec_fisher_pb.sbatch`) | The original P vs P+B Fisher. Its fixed-template per-z H and D_A numbers were an artifact of the ruler symmetry, superseded by `robust_marg_pb.py` |
| `run_fisher_v3.py` → `01_fisher_v3.ipynb` (`sync_fisher_nb.py`, `exec_fisher.sbatch`), `make_fisher_fig.py`, `information_story.py` (`information_story.sbatch`), `fisher_v3_information_split.md` | The P(k)-only Fisher (v3.1) that the P+B work extends. It writes the mock `output/fake_desi_y6_fisher_v3.h5` and `output/fisher_v3/fisher_v3_results.npz`, which `robust_marg_pb.py` and `hmc_prior_bands.py` read |
| `fisher_utils.py`, `make_fisher_pb_fig.py` | Shared Fisher linear algebra (`psd_clip`, `pos_pinv`, `schur_marg`, …) and the P vs P+B figures |
| `bao_se07_forecast.py`, `exec_meeting_figs.sbatch`, `exec_hmc_prior.sbatch`, `exec_corner_audit.sbatch` | One-off forecast and launchers for single scripts |

## Things to know before trusting a number

- **Exact symmetries.** The continuum model has three directions the data cannot see:
  - `g1`, the ruler: h_conv, every D_A up, every H down;
  - `g2`, the amplitude: a up, D down;
  - `g3`, a template dilation.

  Discretization breaks them at σ ≈ 0.05, and a Fisher matrix reads that breaking as information. So the data's
  response along them is projected out (`mi_prior_fisher.build(project_data=True)`). Absolute H(z) and D_A(z) are not
  measurable; D_M/r_d, D_H/r_d and F_AP are.
- **Never form JᵀPJ explicitly** for the marginals: it squares the condition number. Whiten the Jacobian and use its
  SVD (`robust_marg_pb.py`).
- **Run on the CPU** (`JAX_PLATFORMS=cpu`, x64): GPU autodiff reductions are not deterministic, and that noise is
  amplified in the sector marginals.

Outputs go to `output/fisher_pb/` and `output/fisher_v3/` (not in git). These files lived in `notebooks/` until
2026-10-05. Their paths were updated for this folder. `07`, `08` and `09` were re-executed here; `01` and `06` are the
executed records from before the move, so their code cells still show `rootdir = ".."`. Rebuild and execute them with
their launchers to refresh.
