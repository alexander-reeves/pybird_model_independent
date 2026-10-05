# MI analysis of DESI DR1 FS+BAO: the BAO-template model (`Ag245F`)

State as of 2026-10-04. The P_lin model is written out with every factor labelled in
`pklin_note/pklin_model.pdf`. The results are in `output/desi_fsbao/wiggle_figs/Ag245F/Ag245F_results.pdf`, and the
study behind these choices in `wiggle_analysis.ipynb`.

## Data and likelihood

- **Full shape:** DESI DR1 P0 and P2 of BGS, LRG1, LRG2, LRG3, ELG2 and QSO (z_eff = 0.295, 0.510, 0.706, 0.919, 1.317,
  1.491), over 0.02 < k < 0.20 h/Mpc, with the survey window.
- **Post-reconstruction BAO:** α_iso for BGS and QSO, α_∥ and α_⊥ for the other four samples, so 10 BAO points.
- **Data vector:** one joint FS+BAO covariance; 442 data points in total.
- **Theory:** pybird one-loop EFT in the west-coast basis, IR-resummed through the 80-knot loop emulator. Its input is
  P_lin on 80 knots between 10⁻⁴ and 0.7 h/Mpc. AP is included; k_M = 0.7 and k_R = 0.25 h/Mpc.

## Free parameters: 59 sampled, 42 profiled

| block | number | parameters | prior |
|---|---|---|---|
| EFT, sampled | 12 | b1 and c2 per sample | b1 flat, unbounded; c2 ~ N(0, 5) |
| EFT, profiled | (42) | b3, cct, cr1, cr2, ce0, ce1, ce2 per sample | N(0, 5); ce0 ~ N(0, 2). Set analytically to their maximum a posteriori at every step (pybird `get_maxlkl`). c4 = 0 |
| broadband | 21 | ln a(k) on 21 nodes | shape σ = 0.5 per node plus a smoothness penalty λ = 7.78; amplitudes, see below |
| wiggles | 3 | ln α_rs, A, ΔΣ² | ln α_rs ~ N(0, 0.05); A ~ N(0, 0.3); ΔΣ² ~ N(0, 25 Mpc²) |
| growth | 11 | ln f(z_i) at the 6 redshifts; ln D(z_i)/D(z_N) at the 5 redshifts below z_N | ln f ~ N(fid, 0.3); amplitudes, see below |
| geometry | 12 | ln α_∥(z_i), ln α_⊥(z_i) at the 6 redshifts | N(fid, 0.3) each |

**Amplitude prior.** u_i = ⟨ln a⟩ + 2 ln D(z_i)/D(z_N), with u_N = ⟨ln a⟩, has covariance (0.5²/21) 11ᵀ + 0.6² I.

There is no h_conv and no separate BAO α: the BAO points are predicted.

## The model

P_lin(k, z_i) = [D(z_i)/D(z_N)]² · a(k) · T_nw(k) · [1 + (1 + A) · exp(−k² ΔΣ²/2) · O(α_rs k)]

- **z_N = 1.491** (QSO) is the one redshift where P_lin is reconstructed.
- **k** is in Mpc⁻¹ at the fiducial h.
- **Template.** T = T_nw (1 + O) is the fiducial (Planck 2018 / DESI) linear spectrum at z_N.
  - T_nw is the Eisenstein–Hu no-wiggle shape times T/T_EH smoothed with a 0.25 dex Gaussian.
  - O = T/T_nw − 1 is the BAO wiggle pattern: at most 6.6%, zero below 3×10⁻³ Mpc⁻¹.
- **a(k)** is a cubic spline in ln k on 21 nodes over 10⁻⁴–0.7 h/Mpc (spacing 0.443). That is too coarse to follow or slide
  a wiggle, so the BAO position comes only from α_rs and the α's.
- **α_rs = (h r_d)/(h r_d)_fid** rescales the wiggles only.
- **A and ΔΣ²** are the BAO-fit amplitude and damping, relative to the fiducial; ΔΣ² may be negative. The nonlinear
  damping comes from pybird's IR resummation. Without A, real spectra cannot be represented.

**Per redshift.**
- **α_∥(z) = D_H/D_H,fid and α_⊥(z) = D_M/D_M,fid:** distances in Mpc/h, with D_fid ≡ (D/r_d)_fid,DESI · (h r_d)_fid. They
  set the AP of the full spectrum.
- **f(z):** the growth rate in the RSD.
- **D(z)/D(z_N):** the amplitude relative to z_N.
- **BAO points:** α_BAO = α / α_rs, i.e. (D/r_d)/(D/r_d)_fid. BGS and QSO use α_iso = α_∥^(1/3) α_⊥^(2/3). For the BAO
  points only, α_⊥ is multiplied by b(z) = 0.9964–0.9990 to copy pybird's comoving-distance offset.
- **f and D are independent;** no consistency relation is imposed.

**α_rs is close to prior-limited.** Scaling α_rs and every α together, with a dilation of a(k), barely changes the data
(Δχ² ≤ 0.6 for 3%). The data fix D/r_d = α/α_rs. α_rs is kept explicit so that the EFT priors act in h/Mpc, as in a direct
fit. Fixing α_rs = 1 costs 0.4σ in h and w0 for w0waCDM.

## Sampling

- **Sampler:** NUTS (BlackJAX), whitened by the Gauss–Newton Fisher at the MAP.
- **Run:** 32 chains × 2000 draws after 500 warmup steps, tree depth ≤ 8, target acceptance 0.8; 52 minutes on 4 GH200.
- **Diagnostics:** acceptance 0.888, 0.38% divergent, max R̂ 1.005, min ESS 6405.
- **Fit:** MAP χ² = 284.2 for 442 points; the direct ΛCDM fit has 332.3.

## Projection onto a cosmological model θ

p(θ | d) ∝ p_MI(φ(θ) | d) / π_MI(φ(θ)) · π(θ).

- **φ(θ)** holds the cosmology's own f, D ratios, α's and α_rs. The α_rs uses the DESI r_d fit with the correct ω_b
  exponent (−0.13), or the ede-v2 r_d for EDE.
- **(ln a, A, ΔΣ²)** come from a least-squares fit, on the 80 knots, of θ's spectrum in the h_fid frame. It is weighted by
  W = JᵀC⁻¹J, with J = ∂(data vector)/∂ ln P_knots at the fiducial, plus a 10⁻³ regulariser uniform in ln k.
- **A and ΔΣ² are predicted by θ, not sampled.** Marginalizing them would discard real information: for ΛCDM with ω_b and
  n_s free it widens ω_cdm by 24% and n_s by 29%, and for EDE it shifts ω_cdm, h and f_EDE by 0.37–0.44σ.
- **p_MI** is a flow ensemble (10 RealNVP flows, 8 layers × 128, learning rate 10⁻⁴). The Gaussian estimate is not
  enough for this model; see below.
- **π(θ):** flat boxes, plus ω_b ~ N(0.02218, 0.00055) (BBN) wherever ω_b is free; n_s is fixed for EDE.

## Validation

The reference is a direct NUTS chain of the true likelihood for each model: the CosmoPower (ΛCDM, w0wa) or ede-v2 (EDE)
P_lin at the emulator knots, the same EFT, AP and BAO, the corrected r_d and the same priors. Shifts are the largest over
all parameters, in units of the reference σ. The pass criterion is |shift| < 0.3σ with widths within 20%.

| model | Ag245F, flow | Ag245F, Gaussian | widths (flow) | Ae245F (e₀, e₁ envelope), flow |
|---|---|---|---|---|
| ΛCDM (ω_cdm, A_s, h) | 0.18 | 0.18 | 0.97–1.03 | 0.21 |
| + ω_b (BBN), n_s | 0.26 | 0.25 | 0.97–1.02 | 0.14 |
| w0waCDM | 0.20 | **0.66** | 1.00–1.05 | 0.12 |
| EDE (n_s fixed) | 0.11 | 0.19 | 0.96–1.02 | 0.21 |

**Use the flows.** The Gaussian estimate misses w0waCDM along h, w0, wa. The α_rs near-symmetry makes the MI posterior
non-Gaussian, while the map itself is accurate there.

## Known limitations

- **ΔΣ² prior.** Its width (25 Mpc²) is larger than physical cosmologies need: ΛCDM gives −1.5 ± 4.4 Mpc², the posterior
  7 ± 21. The negative tail amplifies the wiggle term above 0.3 Mpc⁻¹, beyond the data range. A ~10 Mpc² prior, or O set
  to zero above the data range, would remove that; either needs a new chain.
- **pybird bugs.** The BAO points inherit pybird's D_M offset (z_min = 10⁻³ in `comoving_distance`). `rs_drag` is
  corrected only in the new scripts.

## Variant: wide priors, A and ΔΣ² marginalized (`Ag245WFM`, 2026-10-05)

- **Same model; MI chain `Ag245W`** with every binding prior widened: s_lnP 0.6→2, shape s_lna 0.5→2, ln f 0.3→1, A
  0.3→1 and ΔΣ² 25→50 Mpc². The α's, α_rs and the smoothness are unchanged.
- **Projection.** A and ΔΣ² are integrated out (flow on the other 45 MI parameters, divided by their marginal prior), as
  nuisances of a BAO fit. The cosmology still predicts f, D, the α's, α_rs and a(k).
- **Result.** The absolute P_lin amplitude is not measured (only f·√a and b₁·√a are); the shape is measured to 4–7% between
  0.02 and 0.12 h/Mpc. The largest flow shifts against the true-likelihood chains are 0.20 / 0.39 / 0.60 / 0.59σ (ΛCDM /
  +ω_b, n_s / w0wa / EDE), with widths up to ×1.28 and ×1.41. That is the information the wiggle amplitude and damping carry.
  With the old priors and A, ΔΣ² marginalized (`Ag245FM`) the shifts are 0.24 / 0.55 / 0.44 / 0.57σ.
- **Details** (with every prior in one table): `output/desi_fsbao/wiggle_figs/Ag245WFM/Ag245WFM_results.pdf`.

## Files

- **Model:** `wiggle_model.py` (`WiggleModel`, `env_type='gauss'`); the variant name `Ag245F` is resolved by
  `wiggle_settings.py`.
- **MI chain:** `output/desi_fsbao/wiggle_Ag245/chain_mi.npz`.
- **Projections:** `wiggle_recovery.py --wiggle Ag245F`, writing to `output/desi_fsbao/wiggle_Ag245F/`.
- **Figures:** `WIGGLE=Ag245F SCRIPT=wiggle_results_figs.py sbatch exec_wiggle.sbatch`.
- **Reference chains:** `wiggle_direct.py <model> exact`, writing to `output/desi_fsbao/exact_*`.
