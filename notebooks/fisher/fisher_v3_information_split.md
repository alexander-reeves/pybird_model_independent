# Fisher information splitting: what changed in v3, and where the information physically lives

*Companion note to `01_fisher_v3.ipynb` (v3.1, re-executed 2026-09-04; paired script
`run_fisher_v3.py`, synced with `sync_fisher_nb.py`, results in `output/fisher_v3/`).
Supersedes the analysis in `01_fisher.ipynb` (v1).*

## TL;DR

The v1 plot showed the growth sector "knowing" more about h than the full fit — impossible, and
traced to four concrete bugs. v3 rebuilds the analysis so that the split is exact:

- **P(k) template (red)**: measures **pure ω_cdm** at σ = 2.5×10⁻³ (≈ the full fit). Carries
  **zero h by construction** — because it is stored in Mpc³ at fixed wavenumbers in 1/Mpc, the
  one representation in which P(k) has no h dependence.
- **Growth/AP given the template (green)**: carries **all of the h information** (σ(h) = 0.010
  alone → 0.0037 jointly) and **all of the A_s calibration**.
- **red + green = joint = direct fit**, exactly (verified to 6×10⁻¹⁴): the split is the chain
  rule of information (marginal × conditional), not an ad-hoc conditioning or a lossy marginal
  split. The reverse ordering, purple (growth marginal) + yellow (P(k) | growth), is equally
  exact (6×10⁻⁵) and is now plotted alongside it — see "Reading the final triangle plots".

## The parametrization (v3)

Model-independent (MI) parameters, all fed to the same pybird fast-emulator likelihood:

| sector | parameters | h content |
|---|---|---|
| P(k) | 80 amplitudes × template P(k*, z_ref = 5) in **Mpc³** at fixed k* in **1/Mpc** (k* = the EFT emulator's native 80 knots × h_true) | **none** (verified: 10% shift in h moves the template by ≤ 0.1%) |
| growth/AP | f(z), H/H₀(z), D_A·H₀(z), D(z)/D(z_ref) at the 6 unique z's — all dimensionless, responding to h only via Ω_m = ω_m/h² | via Ω_m |
| units | **h_conv, free**: kk = k*/h_conv [h/Mpc], P = P_Mpc·h_conv³ [(Mpc/h)³] (same physical points — no interpolation; at fiducial, kk is exactly the emulator grid) | **h_conv = h identically** |
| EFT | b1, b2, b4 per sky free; the rest analytically marginalized | — |

The "direct" ΛCDM fit is defined as the *exact composition* of this model with the cosmology
mappings (amps(ω_cdm, A_s), growth(Ω_m), h_conv = h), so the chain rule guarantees the projected
MI Fisher reproduces the direct one (Gate 3: agreement to 1.7×10⁻⁵).

## What was wrong in v1

1. **Knots off the emulator grid.** P(k) was parametrized on 38 ad-hoc knots spanning
   [0.005, 0.35] h/Mpc, but the EFT emulator compresses its input onto 80 fixed knots spanning
   [1e-4, 0.7] h/Mpc, and the JAX path silently **extrapolates** out-of-range input (no k-range
   check). 31/80 emulator inputs were off the training manifold → χ²(fiducial) = 8.23 instead of
   0, so the "Fisher" (−Hessian) was indefinite (eigenvalues to −7×10³). Everything downstream
   was contaminated.
2. **h_conv frozen, h response faked.** v1 *had* the unit-conversion parameter but froze it
   (h_conv = 1.1·h_true, hard-coded), so the likelihood had no units degree of freedom. To
   compensate, the cosmology mapping multiplied H and D_A by h/h_ref factors — but the
   likelihood's H, D_A inputs are **dimensionless** (AP uses qpar = H_fid/H): their true h
   response is only via Ω_m. The Jacobian therefore claimed h sensitivity in matrix entries
   whose curvature actually measures Ω_m — manufacturing h information in the growth projection.
   (The two rescalings also used inconsistent reference h's, so the mapping's fiducial did not
   match the Fisher's expansion point.)
3. **Hidden prior.** A σ = 0.5 Gaussian prior on the P(k) amplitudes existed only in the MI path,
   quasi-pinning the template and artificially decoupling the growth marginal.
4. **`pos_pinv` as Fisher→covariance.** Unconstrained directions were given *zero* variance
   instead of a huge one, collapsing degenerate bands into spuriously tight ellipses. (Related
   trap fixed along the way: flooring a flat direction at variance 10¹⁰ leaks ε²·10¹⁰ into any
   parameter with even a ~10⁻⁶ eigenvector overlap — flat directions must be reported as flat.)

v3 additionally regenerates the fake data **from the very model being differentiated**, so both
likelihoods peak at the fiducial to machine precision (χ² = 1×10⁻²⁰ direct, 6×10⁻⁸ MI).

## What was wrong in v3.0 (fixed in v3.1, 2026-09-04)

Both defects hit the growth sector specifically, which is why the *marginals* — not the chain
split above — were the part that looked wrong.

1. **A cusp at the expansion point.** `Emulator.make_params` interpolates the P_lin input onto
   the 80 knots with a piecewise-**linear** interpolant, and the template sits exactly *on*
   those knots at h_conv = h_fid. An infinitesimal grid shift therefore picks up a one-sided
   slope, and the log-likelihood curvature along h_conv jumps by ~20% *at* the expansion point:
   `jax.hessian` returned a one-sided second derivative for every h_conv entry of the Fisher.
   `mi_model.py` had already monkeypatched this to cubic interpolation; the Fisher notebook was
   the last piece of the pipeline still running with the kink. Now patched there too, so the
   Fisher and the sampled analysis share exactly one model. (It moves the headline σ's by less
   than a percent — the split, not the direct fit, was what it distorted.)
2. **Noise inversion in the marginals.** Every sector marginal is a Schur complement, i.e. it
   *divides* by the block being marginalized. The autodiff Hessian is PSD only to ~1.8×10⁻⁷ of
   its largest eigenvalue, and `pos_pinv(rtol=1e-12)` inverted that noise with weight ~10⁷. The
   v3.0 "growth marginal" came out with eigenvalues **−134 and −1.2**, and its (ω_cdm, h)
   projection **−24.5** — not a probability distribution at all, which is where the invisible
   purple contour and σ(h_conv) = 8×10⁴ came from. Gate 2 had been reporting the symptom
   (min_eig/|max_eig| = −7.8×10⁻⁴, four orders worse than every other matrix) without failing.
   Fix: `psd_clip` every Fisher before it is marginalized or plotted, and set the pseudo-inverse
   threshold (10⁻⁸) above the measured noise. New **Gate 6** scans that threshold over four
   decades: σ_h(growth|P(k)) is stable to 10⁻⁴ and σ(ω_cdm) of the P(k) marginal to 10⁻³.

A Gauss–Newton reference Fisher JᵀPJ (PSD by construction) is now computed alongside the
Hessian as **Gate 2b**; the two agree to 6.2×10⁻², the difference being the EFT priors and the
analytic-marginalization log-det, which only the Hessian contains.

## The physical reality: why the information sits where it sits

The key organizing fact: **h-independence of P(k) is a statement about coordinates.** The same
spectrum is h-free in (Mpc³, 1/Mpc) at high z, and strongly h-dependent in ((Mpc/h)³, h/Mpc) —
through nothing but the relabeling k → k/h, P → P·h³. The data live in h-units; the physical
template lives in Mpc. Every parameter's "storage location" follows from this:

- **ω_cdm → the template (red).** The shape of P(k) in 1/Mpc — the equality turnover
  k_eq ∝ ω_m, the BAO scale r_d(ω_b, ω_cdm) in Mpc, the baryon suppression — is pure physical
  shape, measurable with no knowledge of h, growth, or units. This survives *full*
  marginalization over everything else: σ(ω_cdm) = 2.54×10⁻³ vs 2.46×10⁻³ for the full fit.
- **h → the units bridge, meaningful only given the template.** h is measured by comparing a
  known physical scale (the template's BAO/turnover, in Mpc) with where those features appear in
  the data (in h/Mpc): the dilation that reconciles them is h_conv = h — the standard-ruler
  measurement. Two verified consequences:
  - *Given* the template, growth/AP measures h_conv to 0.002–0.004: all the h.
  - *Without* the template (growth fully marginal), h vanishes: a shift of h_conv is
    indistinguishable from a template dilation δln a_i = (3 + dlnP/dlnk)·δln h_conv, which
    80 free knot amplitudes can simply follow. A free-form template is a ruler of unknown
    length. Measured exactly (v3.1): only **1.4×10⁻⁵** of the h_conv information survives
    marginalizing the template — σ(h_conv) 0.0019 → 0.52. The residual 10⁻⁵ is not physics:
    the emulator sees the template only after re-interpolating it onto its own fixed knots and
    normalizing by max(P), and those two steps are what stop the dilation family from closing
    exactly. (The earlier "Fisher-metric cosine 0.977" between the two responses is *not* the
    right diagnostic and is no longer quoted: its (3 + dlnP/dlnk) is a finite difference taken
    across the BAO wiggles on 80 knots, nowhere near accurate enough to test a degeneracy at
    the 10⁻⁵ level. The two information numbers above are exact projections of the Fisher.)
    The fully-marginal growth sector retains only a weak Ω_m band (σ(Ω_m)/Ω_m ≈ 11%, from
    anisotropic AP + RSD + relative D(z) evolution — everything isotropic is absorbed by the
    template).

  **This is the one place where the intuition "h enters only through the growth sector, so the
  growth marginal must hold all the h" fails.** It is true that the amplitude Jacobian's h
  column is ~0, so every h response is routed through growth/h_conv. But *marginalizing* the
  template destroys the ruler, so the growth marginal cannot hold h; only *conditioning* on the
  template preserves it. Marginal and conditional are different questions, and the h lives in
  the cross term between the sectors, not in either sector alone.
- **A_s → the last link of the chain.** The data only ever see amps·D(z)²·h_conv³ (plus fσ₈).
  The template's absolute amplitude alone is therefore unmeasurable (A_s is exactly flat in the
  P(k) marginal — degenerate with the free D-ratios), and A_s is recovered only once h (the
  h_conv³ units factor) and Ω_m (the D's) are known.

## The exact decomposition (what "red + green = gray" means)

For the Gaussian posterior over (amps a, growth g) with Fisher blocks A, C, G (EFT already
marginalized), the joint factorizes exactly as marginal × conditional, which projects onto
cosmology θ = (ω_cdm, ln10¹⁰A_s, h) as

F_direct = J_pkᵀ (A − CG⁺Cᵀ) J_pk  +  (J_g + G⁺CᵀJ_pk)ᵀ G (J_g + G⁺CᵀJ_pk)
         =        red (template alone, marginal)  +  green (growth/AP given the template)

verified additive to 5.5×10⁻¹⁴. The ordering (template first) is not a choice of convenience: it
is the unique ordering whose first factor is h-free, which is exactly what storing the template
in Mpc buys. Two decompositions that do **not** work, and why:

- **Naive conditional splits overcount**: "P(k) with growth held *fixed at the fiducial*" gives
  σ(ω_cdm) = 1.2×10⁻³ — tighter than the full fit — because fixing growth fixes Ω_m *and* h,
  i.e. ω_m = Ω_m h², so the conditioned sector already implies ω_cdm. (This is *not* the same
  object as the chain conditional G⁺CᵀJ_pk-corrected term above, which is a legitimate factor
  of an exact decomposition. v3.1 drops the fixed-at-fiducial variants from the notebook and
  the saved `.npz` — they were diagnostics only, and nothing in `paper/make_numbers.py` reads
  them.)
- **Products of marginals undercount**: red-marginal ⊗ purple-marginal matches the full fit on
  ω_cdm (1.0×) but is 9.7× weaker on h and has no A_s at all — the calibration information
  lives in the cross-correlation between the sectors, which only the joint (or either chain
  split) retains.

## Reading the final triangle plots

`make_fisher_fig.py` writes three figures from the same six matrices (and is runnable
standalone against the saved `.npz`, so the plotting can be iterated without re-executing the
notebook):

| file | curves | window |
|---|---|---|
| `triangle_v3.png` | the four of the paper caption: Direct, MI-projected, template alone, growth \| template | direct-fit scale |
| `triangle_v3_split.png` | all six: both sector marginals **and** both chain conditionals | direct-fit scale |
| `triangle_v3_wide.png` | the same six | wide enough that the two *marginals* are visible — at direct-fit scale both are wider than the frame, which is why the purple looked like a stray line in v3.0 |

All six use the same variance floor for their flat directions, so they are directly comparable.
The list below describes the six-curve versions; the paper figure is their first, third and
fifth entries.

- Gray (Direct, filled) = blue (Combined, the joint projection): the rigorous identity.
- Red (P(k) marginal): vertical ω_cdm band at full-fit precision; flat in A_s and h.
- Purple dotted (growth marginal): a pure Ω_m band, σ(Ω_m)/Ω_m = 11%; h unconstrained along it
  (σ = 0.52 vs 0.0037 for the full fit).
- Green (growth | P(k)): one very elongated, highly correlated 3-D ellipsoid
  (r(ω,lnA_s) = −0.96, r(ω,h) = +0.96, r(lnA_s,h) = −0.92) containing all the h and A_s info.
- Yellow dashed (P(k) | growth): sits essentially **on top of Direct**. That is the correct
  answer, not a duplicated curve — the growth marginal is nearly empty, so the reverse ordering
  hands almost the entire joint to its conditional.

Red + green = Combined, and purple + yellow = Combined. Green and yellow are alternative,
ordering-dependent allocations of the same cross-information and must never be added together.
- Their combination happens in 3-D, not panel by panel — the cascade is:

| step | information used | yields |
|---|---|---|
| 1 | red: template shape in Mpc | ω_cdm = 2.5×10⁻³ |
| 2 | green combo #1 (σ = 1.8×10⁻³, mostly ω–h): Ω_m + standard-ruler dilation | h: 0.010 → 0.0037 |
| 3 | green combo #2 (σ = 4.5×10⁻³, mixing h–lnA_s): amplitude = amps·D²·h_conv³ + fσ₈ | lnA_s: 0.126 → 0.045 |

Numerically: green alone gives σ(lnA_s) = 0.126, σ(h) = 0.010; adding *only* red's ω_cdm band
reproduces the full fit exactly (0.0447, 0.0037). Intersecting 2-D panels can never show this,
because each panel has marginalized away the dimension the cascade routes through — most
dramatically in the A_s panels.

## Sanity gates (all pass, printed in the notebook)

1. χ²(fiducial) ≈ 0 for both the direct and MI likelihoods (2×10⁻²¹ / 6×10⁻⁸).
2. Every Fisher that is marginalized or plotted is PSD to ~10⁻¹⁶ after `psd_clip` (2), and the
   Gauss–Newton reference agrees with the Hessian to 6.2×10⁻² (2b). **In v3.0 this gate printed
   its own failure — the growth marginal at −7.8×10⁻⁴ — without failing; it now clips instead.**
3. ‖F_combined − F_direct‖/‖F_direct‖ = 1.7×10⁻⁵ (chain-rule identity).
4. No marginal *or* conditional sub-piece beats the full fit on any parameter (the inequality
   that v1's green contour violated).
5. (a) The two degeneracies that empty the sector marginals: a·D² invariant exact to 8×10⁻¹⁰ of
   the top eigenvalue, h_conv ↔ template dilation to 1.4×10⁻⁵. (b) Both chain orderings additive,
   to 5.5×10⁻¹⁴ (template first) and 6.0×10⁻⁵ (growth first).
6. Every quoted number stable against the pseudo-inverse threshold over four decades.
