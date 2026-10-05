# Model-independent analysis of DESI DR1 full shape + BAO

The linear power spectrum is reconstructed from the data instead of being computed from a cosmology. Any
cosmological model is then tested by projecting the model-independent (MI) posterior onto it, without rerunning
the EFT likelihood.

**Fiducial model: `Ag245F`.** Read in this order:

1. [`pklin_note/pklin_model.pdf`](pklin_note/pklin_model.pdf): the P_lin model, every factor labelled.
2. [`WIGGLE_SETUP.md`](WIGGLE_SETUP.md): data, all free parameters and priors, sampling, the projection, and the
   validation.
3. [`../../results/Ag245F_parameter_recovery.pdf`](../../results/Ag245F_parameter_recovery.pdf): the parameter-recovery
   figures.

## The model in one paragraph

P_lin(k, z_i) = [D(z_i)/D(z_N)]² · a(k) · T_nw(k) · [1 + (1 + A) e^{−k²ΔΣ²/2} O(α_rs k)], reconstructed at
z_N = 1.491.

- **Broadband:** a(k) is a cubic spline in ln k on 21 nodes. That is too coarse to move a wiggle, so the BAO position
  comes only from α_rs and the α's.
- **Template:** T_nw and O are the no-wiggle part and the wiggle pattern of the fiducial spectrum.
- **Wiggles:** α_rs rescales the wiggles; A and ΔΣ² are the usual BAO-fit amplitude and damping.
- **Per redshift:** f, D(z_i)/D(z_N), and two dilations α_∥, α_⊥ that set the full-shape AP. The post-reconstruction
  BAO points are predicted as α/α_rs, so there is no separate BAO α and no h_conv.
- **Size:** 59 sampled parameters, 12 of them EFT; the other 42 EFT coefficients are profiled analytically.

**Projection onto a cosmology θ:** p(θ|d) ∝ p_MI(φ(θ)|d) / π_MI(φ(θ)) · π(θ).

- f, D, the α's and α_rs follow from θ directly.
- a(k), A and ΔΣ² follow from a least-squares fit of θ's linear spectrum, weighted by the data's Fisher information on
  the 80 emulator knots: the `F` in the name. The fit uses θ only, never the data.
- p_MI is a normalizing-flow ensemble trained on the MI chain.

## Validation: recovery of direct fits

Each cosmology is fitted twice: directly with the full EFT likelihood (the reference), and through the MI posterior.
The table gives the largest shift over all parameters, in units of the reference σ; the pass mark is 0.3σ, with widths
within 20%.

| cosmology | parameters | flow estimate | width ratios | Gaussian estimate |
|---|---|---|---|---|
| ΛCDM | ω_cdm, ln10¹⁰A_s, h | 0.18σ | 0.97–1.03 | 0.18σ |
| ΛCDM | + ω_b (BBN prior), n_s | 0.26σ | 0.97–1.02 | 0.25σ |
| w0waCDM | + w0, wa | 0.20σ | 1.00–1.05 | 0.66σ |
| EDE | ω_cdm, ω_b, A_s, h, f_EDE, log10 z_c, θ_i | 0.11σ | 0.96–1.02 | 0.19σ |

Source: `output/desi_fsbao/wiggle_Ag245F/recovery.log` (2026-10-04). The MI chain has 32 × 2000 draws, max R̂ 1.005 and
min ESS 6405. Use the flow estimate: the α_rs near-symmetry makes the MI posterior non-Gaussian along h–w0–wa.

## Files

| group | files |
|---|---|
| likelihood, data, sampler | `model.py` (`MIModel`: pybird likelihood, parameter layout, priors, direct models lcdm3 / lcdm5 / w0wa7 / ede7), `settings.py` (data, fiducial cosmology, priors, named cosmologies), `sampling.py` (L-BFGS-B, whitened NUTS with BlackJAX), `compress.py` (flow ensemble, projected posterior), `sample.py` (best fits and chains), `ede.py` (CosmoPower `ede-v2` in JAX), `cmb.py` (Planck PR4 early-universe Gaussian, optional) |
| the model | `wiggle_model.py` (`WiggleModel`, the least-squares map θ → MI, `rd_h_fixed`), `wiggle_settings.py` (variant names → settings) |
| runs | `wiggle_sample.py` (MI chain), `wiggle_recovery.py` (projections and recovery tests), `wiggle_direct.py` (true-likelihood reference chains), `wiggle_direct_marg.py` (references with A, ΔΣ² free), `wiggle_fisher_map.py` (the data weights of the `F` map), `wiggle_results_figs.py` (figures + results PDF) |
| design study | `wiggle_analysis.py` → `wiggle_analysis.ipynb` (`build_wiggle_nb.py`, `exec_wiggle_nb.sbatch`): why this model, from the outputs of `wiggle_gates.py`, `wiggle_envelope.py`, `wiggle_truth.py`, `wiggle_frame.py`, `wiggle_reweight_ref.py`, `flow_study.py`; the map rule and its cost in `wiggle_map_rules.py`, `wiggle_compare_marg.py`, `wiggle_time_map.py` |
| checks | `ede_check.py` (ede-v2 vs CLASS: P_lin ±0.2%, r_d 0.02%), `cmb_check.py` (`cmb.py` vs CLASS) |
| launchers | `exec_wiggle.sbatch` (`SCRIPT=… ARGS=…`), `exec_wiggle_nb.sbatch`, `exec_sample.sbatch`, `exec_pair.sbatch`, `exec_seq.sbatch` |

## Running it

Everything runs on CSCS Alps GH200 nodes through SLURM. The launchers hard-code the container
(`--environment=~/.edf/pybird-jax.toml`), the venv (`$HOME/pybird/jax_env`, with jax, blackjax, optax, getdist,
cosmopower_jax and this repository's `pybird` installed editable) and absolute paths; adapt those three lines
elsewhere.

- **Data:** `data/desi_dr1_fs_bao/desi_dr1_kp_fs_bao.h5` with `likelihood_config/desi_dr1_fs_bao.yaml`, built from
  DESI's public DR1 FS+BAO likelihood files.
- **EDE only:** the ede-v2 weights (`*_v2_plain.npz`, from
  [cosmopower-organization/ede](https://github.com/cosmopower-organization/ede)) go in `data/emulators/cosmopower_ede/`.
- **Outputs:** everything goes to `output/desi_fsbao/`, which is not in git.

```bash
cd notebooks/desi_fsbao
SCRIPT=wiggle_fisher_map.py sbatch exec_wiggle.sbatch                             # data weights of the F map (once)
SCRIPT=wiggle_sample.py ARGS="--wiggle Ag245" sbatch exec_wiggle.sbatch           # MI chain, ~1 h on 4 GPUs
SCRIPT=wiggle_direct.py ARGS="lcdm5 exact" sbatch exec_wiggle.sbatch              # a reference chain (per cosmology)
SCRIPT=wiggle_recovery.py ARGS="--wiggle Ag245F --est gauss+c5" sbatch exec_wiggle.sbatch   # projections + tests
WIGGLE=Ag245F SCRIPT=wiggle_results_figs.py sbatch exec_wiggle.sbatch             # figures + results PDF
```

Separate list items in `ARGS` with `+`, not `,`: `sbatch --export` splits at commas.

## Variant names

`A` / `B` = α_rs free / fixed to 1. `g2` = the Gaussian BAO envelope (A, ΔΣ²); `e2` = the older log envelope. Digits
= node spacing × 100 (`245` → 0.45 in ln k, 21 nodes). Suffixes:

- `N`: full shape only;
- `W`: widened priors;
- `F`: data-weighted least-squares map;
- `M`: A and ΔΣ² marginalized in the projection.

`F` and `M` change only the projection, so `Ag245F` and `Ag245FM` share the MI chain `wiggle_Ag245`.

`Ag245WF` and `Ag245WFM` (widened priors; A and ΔΣ² predicted or marginalized) are studies of the priors and of the
information carried by the wiggle amplitude. They are not the fiducial; see `WIGGLE_SETUP.md`.

## Known issues in shared `pybird` code (worked around here, not fixed upstream)

- **`pybird.symbolic.rs_drag`:** the ω_b exponent has the wrong sign, +0.13 instead of −0.13, which is 0.65% in r_d
  per BBN σ of ω_b. The scripts here use `wiggle_model.rd_h_fixed`.
- **`pybird.symbolic.comoving_distance`:** it integrates from z = 10⁻³, so D_M is 0.37% low at z = 0.295. The BAO
  points copy that offset through a fixed per-z factor.
- **The original 106-parameter MI model:** its direct likelihood (`model.MIModel` with 60 nodes) differs from the true
  CosmoPower likelihood by 0.5–0.9σ in ω_cdm. The reference chains here use the true likelihood (`wiggle_direct.py …
  exact`).
