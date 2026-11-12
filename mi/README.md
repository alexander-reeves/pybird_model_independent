# DESI DR1 full shape + BAO, model-independent: the pipeline

The linear power spectrum, the growth and the distances are fitted freely to the data (the model-independent, MI,
posterior). A cosmology θ is then fitted by projecting that posterior,
p(θ|d) ∝ p_MI(φ(θ)|d) / π_MI(φ(θ)) · π(θ) = L_MI(φ(θ)) · π(θ), without running the EFT likelihood again.

## The projection density (`inference.py`, `config.DENSITY`)

What the projection needs is the likelihood of the MI parameters with the sampled EFT parameters marginalized, L_MI(φ).
Two estimates, both fitted once per template by `run.py recovery` and reused:

* **normalizing flow** (default, `fit_ensemble`): 10 RealNVP flows estimate the MI posterior p_MI(φ|d) from the chain;
  divided by the MI prior it is L_MI(φ). The training stops on whole held-out chains.
* **score regression** (back-up and cross-check, `fit_score`): fits log L_MI(φ) from its gradient. By Fisher's identity
  ∇ log L_MI(φ) = E_{e|φ}[∇_φ log L(e, φ)], so the gradient at each chain draw (one autodiff gradient, ~90 s for 96k
  draws on 4 GH200) is an unbiased noisy sample of it. Terms with zero mean given φ (∇_e log L × (1, φ),
  e_k ∂_j log L + δ_jk) are subtracted as control variates, and a quadratic + MLP is fitted to the gradients; 20 fits on
  different held-out chains are averaged. No density is estimated and no prior is divided out.

Recovery of the direct fits on the baseline `bao_template_no_rs`, with the flows of 2026-11-11 (before the whole-chain stopping,
which brings the fit-to-fit scatter of the flow down from 0.15–0.40σ to 0.15–0.24σ on ΛCDM) (max |shift| of the posterior mean in direct-fit σ, and
the range of width ratios; w0waCDM against the direct fit with the EFT priors in sound-horizon units):

| | ΛCDM | w0waCDM | EDE |
|---|---|---|---|
| flow | 0.15 (0.93–1.00) | 0.16 (0.93–1.00); refits 0.15–0.29 | 0.34 (0.89–1.00); refit 0.22 |
| score regression | 0.11 (0.99–1.01) | 0.11 (0.97–1.00) | 0.26 (0.95–1.00) |

The flow is 5–8% too narrow in ln A_s, h, w0, wa. It does worse where the template has more freedom than the baseline:
w0waCDM 0.59σ (widths up to 1.22) for `bao_template` (α_rs free) and 0.47σ for `agnostic`, against 0.25σ with score
regression in both.

`run.py recovery` reports both (and a Gaussian); `run.py project` uses `config.DENSITY` (`--density score` for the back-up).

## What is fitted (`config.py`)

**Two templates for the linear spectrum at z_N = 1.491** (`TEMPLATES`):

| template | P_lin(k, z_N) | free | idea |
|---|---|---|---|
| `agnostic` | a(k) · T(k) | ln a(k) on 60 nodes | the fiducial spectrum times a free function fine enough to reshape the BAO wiggles: nothing assumed about them |
| `bao_template` | a(k) · T_nw(k) · [1 + (1 + A) e^{−k²ΔΣ²/2} O(α_rs k)] | ln a(k) on 21 nodes, α_rs, A, ΔΣ² | a smooth broadband (too coarse to move a wiggle) times the fiducial BAO wiggles O, moved by α_rs = h r_d/(h r_d)_fid, with a BAO-fit amplitude A and damping ΔΣ² |

Two more templates derive from `bao_template`: `bao_template_no_rs` (α_rs = 1, **the baseline**: the α are the BAO α and
P_lin is in sound-horizon units; it is the same likelihood with the EFT priors in units of r_d instead of h⁻¹Mpc, which moves
h, w0, wa of w0waCDM by 0.5σ against the standard direct fit, while α_rs free reproduces it: `7_alpha_rs_comparison.ipynb`) and `growth_from_h` (the baseline with f, D and the α derived from a spline of H(z) r_d and
Ω_m (H0 r_d)² through the growth equation; `templates.py` docstring, `9_growth_from_expansion.ipynb`).

Both also have, at each of the 6 redshifts: the growth rate f, the growth ratio D(z_i)/D(z_N), and two dilations
α_∥, α_⊥ that set both the full-shape AP and the post-reconstruction BAO points. 12 EFT parameters are sampled and 42
profiled. The MI prior (`MI_PRIOR`) is Gaussian in the log of each quantity around the DESI fiducial; every entry is
commented there. The P_lin model is written out in `pklin_note/pklin_model.pdf`.

**Cosmological models** (`MODELS`), fitted directly with the full likelihood (the references) and through the MI posterior:
`lcdm` (ω_cdm, ω_b, n_s, A_s, h), `w0wa` (+ w0, wa), `ede` (ω_cdm, ω_b, A_s, h, f_EDE, log10 z_c, θ_i; n_s fixed), `mnu`
(lcdm + m_ν, the mass of each of three degenerate neutrinos, on the same emulators as `ede`).
**Priors on every DESI-alone fit:** BBN ω_b ~ N(0.02218, 0.00055) and n_s ~ N(0.9649, 0.042) (10 × Planck). The CMB
angle 100θ_s ~ N(1.040268, 0.000246) is added only in the "+ θ_s" projections and in `run.py reference --theta-s`.

## Files

| file | what |
|---|---|
| `config.py` | the templates, models, priors, sampler settings and output paths: start here |
| `run.py` | the pipeline, one step per subcommand (`python run.py -h`) |
| `run.sbatch`, `pipeline.sh` | one step as a SLURM job; the whole pipeline as dependent SLURM jobs |
| `daint.sbatch`, `daint_queue.sh` | the same step on CSCS Daint; a loop that feeds job files to its 30-minute debug partition one at a time |
| `templates.py` | the MI parametrization of both templates, and the sound-horizon frame they share |
| `model.py` | the pybird likelihood, the cosmological models, the true likelihood of the references |
| `cosmology.py` | w0wa growth (ODE), the EDE emulator, r_d, θ_s |
| `inference.py` | L-BFGS-B, NUTS, the projection densities: score regression and normalizing flows |
| `checks.py` | validation that is not part of the pipeline: the map vs the true likelihood, θ_s and EDE vs CLASS, the truth of a noiseless mock (`truth`), the α_rs gauge of bao_template (`degeneracy`, `mock`, notebook 6), α_rs = 1 against the true likelihood with exact loops and wide EFT priors (`exact_repr`) and with the EFT priors adapted per cosmology (`adapted`, notebook 7), growth_from_h against the direct-fit draws (`growth`), whether the MI best fit is the maximum of the posterior (`bestfit`, notebook 9) |
| `plots.py` | the notebooks' figures, from `data/mi_chains/` only |

## Running it

The steps, in order (each skips what already exists):

```
python run.py reference --model lcdm             # direct fits (lcdm first: w0wa and ede start from it)
python run.py reference --model w0wa
python run.py reference --model ede
python run.py chain --template agnostic --seed 1 # MI chains: one per seed and template; more seeds = more draws
python run.py chain --template bao_template --seed 1
python run.py recovery --template agnostic       # projections vs the references (fits the densities the next two reuse)
python run.py project --template agnostic        # lcdm, w0wa without / with theta_s
python run.py sectors --template agnostic        # lcdm from shape | evolution, background | perturbations
python run.py fisher                             # (and recovery/project/sectors for bao_template)
python run.py bands
python run.py export                             # -> data/mi_chains/, what the notebooks read
python run.py notebooks --execute
```

**On a SLURM cluster**, edit the marked block of `run.sbatch` (account, partition, environment) and either submit steps
one by one, `ARGS="chain --template agnostic --seed 1" sbatch run.sbatch`, or everything at once: `bash pipeline.sh`.
That submits each step as its own job with SLURM dependencies, so the three references, the two templates and their seeds
run in parallel on separate nodes. Logs: `output/desi_dr1/logs/`.

**Hardware.** The same code runs on any of these; only `run.sbatch`'s block changes:

| node | `#SBATCH` | `HARDWARE` | chains |
|---|---|---|---|
| several GPUs (CSCS GH200: 4) | as is | `""` | spread over the GPUs |
| one GPU | `--gpus=1` | `""` | all 32 on the one GPU |
| CPU only | `--cpus-per-task=64` | `"--platform cpu --cpus 16"` | 16 groups of 2 chains in parallel |

`--cpus N` splits the CPU into N JAX devices; keep `--chains` (default 32) a multiple of N. On CPUs a chain step is
several times slower than on a GH200: lower `--draws` and add seeds instead (each seed is an independent job). The Fisher
step always runs on the CPU. Typical times on 4 GH200: reference 30–95 min, agnostic chain (500 draws) 55 min,
bao_template chain (1200 draws) 85 min, recovery 45–75 min (flow ~6 min, score regression ~15 min, 9 projections), the
rest minutes.

**Software:** python ≥ 3.10 with jax (x64), blackjax, optax, cosmopower_jax, getdist, h5py, nbclient, and this repository's
`pybird` installed editable. EDE needs the ede-v2 weights (`*_v2_plain.npz`, github.com/cosmopower-organization/ede) in
`data/emulators/cosmopower_ede/`. The notebooks need only numpy, matplotlib and getdist.

## Other data: noiseless mocks, the Y6 mock, short jobs

* `MI_MOCK=<name>` in the environment runs every step on a noiseless mock (`config.MOCKS`: lcdm, w0wa, ede, mnu) instead of
  the data, in `output/desi_dr1_mock_<name>/` and `data/mi_chains_mock_<name>/`. `MI_MOCK=w0wa python checks.py truth` makes
  it first (the true likelihood's prediction at the truth; covariance, priors, sampler unchanged). The θ_s prior of a mock is
  centred on its truth.
* `MI_DATA=y6_p` or `y6_pb` runs on the DESI-Y6 mock of notebook 5, with the power spectrum alone or with the tree-level
  bispectrum, through pybird-dev (`PYBIRD_DEV`, default: a checkout next to this repository): `run.py chain --draws 0` (the
  best fit) and `run.py fisher`. Chains are not practical there (0.3–0.5 s per leapfrog step; one process per GPU, since
  pybird-dev's spline does not run under pmap).
* A chain or a reference that does not fit in one job: `--warmup W --draws N` first, then `--extend --draws N` as often as
  needed (the adapted step sizes, mass matrices and last positions are in the chain file).
* `run.py reference --sound-horizon`: the direct fit with the EFT priors in sound-horizon units, the like-for-like
  comparison of the baseline template (it matters for w0waCDM only).

## Known issues in the shared `pybird` code (worked around here)

- `symbolic.rs_drag` has the ω_b exponent with the wrong sign (0.65% in r_d per BBN σ): `cosmology.rd_h_fixed`.
- `symbolic.comoving_distance` integrates from z = 10⁻³ (D_M 0.37% low at z = 0.295): the BAO points carry the same factor.
- `symbolic._D`/`f` (Heath's integral) are exact only for w = −1: `cosmology.growth` solves the growth equation.
