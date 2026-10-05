"""Build the two presentation notebooks from the scripts that do the work.

  08_growth_pk_bk.ipynb   the linear P(k) recovery, the growth sector, the LCDM recovery
  09_bao_pk_bk.ipynb      the AP / BAO parameters: no template assumption, then a phase prior

The scripts stay the source of truth (meeting_growth_pb.py, meeting_bao_pb.py,
mi_prior_fisher.py); the notebooks run them and explain each step. Rebuild with
    python3 build_meeting_nbs.py
then execute in place with exec_meeting_nbs.sbatch.
"""
import json

FIG = '../../output/fisher_pb/meeting/'


def md(t):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': t.strip('\n')}


def code(t):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': t.strip('\n')}


def show(*names, width=1000):
    return code("from IPython.display import Image, display\n" + "\n".join(
        f"display(Image(filename='{FIG}{n}', width={width}))" for n in names))


SETUP = r"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath('.')) if os.path.basename(os.getcwd()) != 'fisher'
         else os.getcwd())
sys.path.insert(0, os.getcwd())
import numpy as np
R = np.load('../../output/fisher_pb/jacobians_pb.npz', allow_pickle=True)
print(f"redshift bins: {np.round(R['zeff_unique'], 3).tolist()}   ({int(R['num_skies'])} skies, "
      f"sky -> z index {R['sky_to_z_idx'].tolist()})")
print(f"free parameters per data set: {R['J_p'].shape[1]} (P)   {R['J_pb'].shape[1]} (P+B)")
print(f"   = per-sky EFT ({len(R['eft_names_p'])} P / {len(R['eft_names_pb'])} P+B names x "
      f"{int(R['num_skies'])} skies) + {int(R['n_knots'])} template amplitudes + "
      f"{int(R['n_growth'])} growth/geometry")
print(f"data vector: {R['J_p'].shape[0]} bins (P0,P2,P4)   {R['J_pb'].shape[0]} bins (+B0: "
      f"{int(R['isB_pb'].sum())} triangles)")
print(f"exact-symmetry check, delta chi^2 along g1/g2/g3 for a 1% move "
      f"(0 = exact): {np.round(R['sym_tests_pb'][0][:3], 4).tolist()}")
"""

# =======================================================================================
nb08 = [
md(r"""
# P(k) + B(k) without a cosmological model: the linear spectrum, the growth rate, and $\Lambda$CDM

**What this is.** A Fisher forecast for a DESI-Y6-like mock, comparing what the power spectrum
alone measures with what the power spectrum plus the tree-level bispectrum measure — in a model
where the linear power spectrum is *not* assumed to be $\Lambda$CDM.

**The model.** Instead of a cosmology, the linear spectrum is a free function on a grid,
$$P_{\rm lin}(k) = a(k)\,T(k),$$
with $T$ a fixed fiducial template (CosmoPower, Mpc$^3$, $z_{\rm ref}=5$, the 80 emulator knots)
and $\ln a$ free on 60 nodes, cubic-splined in $\ln k$. Alongside it sit the growth and geometry
parameters, free at every redshift: $f$, $H/H_0$, $D_A H_0$, the growth ratios $D(z)/D(z_{\rm ref})$,
and $h_{\rm conv}$ (the Mpc$/h$ ruler). Every EFT coefficient is free per sky. This is exactly the
model our HMC samples (`mi_model.py` + `setups.COMMON`), including its priors, so the Fisher here
can be checked against a real chain — it reproduces the chain's per-node widths to 5%.

**Three results, one covariance matrix.** Sections 4–6 are three different linear functionals of
the same inverse Fisher: the spectrum itself, the growth sector, and — composing with the
cosmology map — $\Lambda$CDM.

The AP / BAO parameters get their own notebook, `09_bao_pk_bk.ipynb`.
"""),
md(r"""
## 1. The mock data

Seven sky patches over six effective redshifts, DESI-Y6-like volumes and number densities. The
data vector is **noiseless**: it is the model evaluated at the fiducial cosmology, so the Fisher
is evaluated at the truth and no realization noise enters. Covariance is Gaussian and analytic
(`utils_cov.get_cov_gauss` for P, `get_cov_b_PPP` for B); no window, no binning effects, and zero
P–B cross-covariance.

* **P(k)**: monopole, quadrupole **and hexadecapole**, $0.01 < k < 0.2\ h/$Mpc, $\Delta k = 0.01$.
* **B(k)**: tree-level monopole on closed triangles, $0.02 < k < 0.10\ h/$Mpc.

The *same* P(k) configuration is used in both likelihoods, so every width ratio below is
attributable to the bispectrum alone.
"""),
code(SETUP),
md(r"""
## 2. The likelihood

`pybird-dev` in the `eth` basis — the only basis that supports the bispectrum, and for the power
spectrum a relabelling of the usual one ($b_1 = \mathrm{Bb1}$, $b_2 = \mathrm{Bb2}$,
$b_4 = \mathrm{Bb5}$, …). The one-loop P(k) is evaluated by the **emulator** on the same 80 knots,
which is what makes a free-form template affordable; the bispectrum is tree-level for now
(one-loop needs `with_bk_tree_level: False` plus the loop matrices, then a re-run of
`jacobian_pb.py`).

Per sky the free EFT parameters are Bb1, Bb2, Bb5 and the stochastic terms; the rest
(Bb3, Bb8, Bc1–Bc4, Be1, Be2, ce2, cr4, cr6, and Bd1, Be5 for B) carry the analytic-marginalization
Gaussian priors, which enter the Fisher as extra rows rather than being integrated out — the same
information either way, but it keeps them visible.
"""),
md(r"""
## 3. From the likelihood to a Fisher matrix

1. **Model vector.** $m(\theta)$ = the stacked P(k) (and B(k)) prediction for all skies,
   $\theta$ = [EFT per sky, 80 $\ln a$, 25 growth/geometry].
2. **Jacobian.** $J = \partial m/\partial\theta$ by forward-mode JAX (`jacobian_pb.py`, chunked,
   on CPU — the GPU Hessian was noisy at the level we need). Cached in `jacobians_pb.npz`.
3. **Whiten.** $W = L^{T} J S$ with $L$ the Cholesky factor of the precision and $S$ the parameter
   scaling, so every parameter is measured in fractional units. **All marginals come from the SVD
   of $W$, never from an explicit $J^{T}PJ$** — forming the normal matrix squares the condition
   number, and marginalizing 80 template amplitudes out of the geometry needs exactly the small
   singular values.
4. **Nodes.** The 80 knots are mapped onto the 60 spline nodes the HMC actually samples.
5. **Priors.** The HMC's own: $\sigma_{\ln a} = 0.5$ per node, a smoothness penalty
   $\lambda = 200$ on second differences of $\ln a$, $\sigma_{\ln g} = 0.3$ on each growth
   quantity, $\sigma_{\ln h} = 0.05$ on $h_{\rm conv}$.
6. **Exact symmetries.** The continuum model has three directions along which the data cannot
   change at all — the ruler $g_1$ ($h_{\rm conv}$, every $D_A$ up, every $H$ down), the amplitude
   $g_2$, and the template dilation $g_3$. Discretization breaks them at the $5\%$ level, and a
   Fisher matrix reads that breaking as information. The data's response along $g_1$ and $g_2$ is
   therefore projected out; the priors are left untouched, because constraining those directions
   is exactly what a prior is for.

The cell below runs the analysis (seconds, from the cached Jacobians) and prints the numbers the
figures show.
"""),
code(r"""
if not os.path.exists('../../output/fisher_pb/jacobians_pb.npz'):
    %run jacobian_pb.py     # ~5 min: re-derives the model Jacobians
%run meeting_growth_pb.py
"""),
md(r"""
## 4. Result 1 — the linear power spectrum

Every other parameter is marginalized: the EFT coefficients of all seven skies, the growth rate,
the distances, $h_{\rm conv}$. Top: the recovered spectrum. Bottom: the width of that band.

The uncertainty is **fractional** (per cent of $P_{\rm lin}$, i.e. $\sigma(\ln P)$) because the
spectrum falls by two decades across the plot — a band in Mpc$^3$ would be invisible at high $k$
and meaningless to compare between scales. The shaded columns mark the $k$ ranges where the mock
actually has data; outside them the band relaxes onto the prior, as it must.

**Which priors.** The solid curves use the wider template priors recommended in section 7
($\sigma_{\ln a} = 2$, $\lambda = 50$); the dotted ones are the tighter priors the chains have
been run with so far. The tight priors were supplying a large part of the band — 10.2% against
23.1% — so the solid curves are what the *data* say.

**What to look at.** Inside the P(k) window P(k) alone constrains the template to **27.4%** per
node and P(k)+B(k) to **23.1%**: a ratio of **0.84**, reaching **0.78** at $k \simeq 0.13\ h/$Mpc
where the bispectrum breaks the bias–amplitude degeneracy best (bottom panel). With the tighter
priors that gain looked like only 0.90, because the prior was doing the work the bispectrum could
otherwise do. The bispectrum still helps the template much less than it helps the growth rate —
section 5 — but the effect is real and largest exactly where B(k) has data.
"""),
show('fig_bands_pb.png', width=880),
md(r"""
## 5. Result 2 — the growth sector

At $z = 0.71$, everything else marginalized. **The bispectrum halves $\sigma(f)$** — 18.5% → 9.7% —
and the gain holds at every redshift except the sparsest bin. That is the headline: with a free
template, P(k) alone barely measures the growth rate, because the amplitude of the spectrum and
the linear bias are degenerate; the bispectrum breaks that degeneracy directly.

The $H$ and $D_A$ widths in this corner are *not* a measurement: the ruler direction $g_1$ contains
no amplitudes, so no template prior and no data can fix it, and those widths are set by the
$h_{\rm conv}$ and growth priors. The combinations that *are* measured — $D_M/r_d$, $D_H/r_d$,
$F_{\rm AP}$ — are the subject of notebook 09.
"""),
show('fig_growth_corner.png', width=720),
show('fig_growth_vs_z.png', width=980),
md(r"""
## 6. Result 3 — $\Lambda$CDM recovery, direct vs through the model-independent likelihood

This is the check that the model-independent detour neither invents nor destroys information.
$\Lambda$CDM is fitted two ways:

* **direct** — the ordinary full-shape $\Lambda$CDM fit: three parameters, 80 template amplitudes
  and 25 growth parameters determined by them;
* **through the MI model** — the same data and the same likelihood, but routed through the
  model-independent parametrisation (60 spline nodes), then restricted to $\Lambda$CDM.

Two gates are printed above. First, the direct $\Lambda$CDM model vector is differentiated
*through the same likelihood call* and compared with the MI Jacobian composed with the cosmology
map: they agree to $1.4\times10^{-6}$, so the two routes are the same model, not two models that
happen to look alike. Second, the 60-node basis represents the $\Lambda$CDM template response to
1–2%. The resulting $\Lambda$CDM errors agree to 1–3% — nothing is added, nothing is lost.

**One caveat, and it matters for compressing a real chain.** The MI *prior* is not flat along the
$\Lambda$CDM directions: on its own, with no data at all, it corresponds to
$\sigma(\omega_{\rm cdm}) = 0.0035$ and $\sigma(\ln 10^{10}A_s) = 0.046$ — comparable to what the
data themselves deliver. A per-node width of 0.5 is loose for *one* node but tight for a coherent
shift of all sixty, which is exactly what a cosmological parameter produces. So compressing an MI
posterior onto $\Lambda$CDM **without dividing out that induced prior** tightens $\omega_{\rm cdm}$
by a factor 2 and $A_s$ by 2.5 artificially (third row of the table). The curves below use the
corrected route.
"""),
show('fig_cosmo_pb.png', width=760),
md(r"""
## 7. The priors, and which of them is doing work

Every prior in the model, and what it is for:

| block | width | acts on | why it is there |
|---|---|---|---|
| $\sigma_{\ln a}$ | 0.5 | $\ln a$ at each of the 60 nodes | keep the sampler in a sane region |
| $\lambda$ | 200 | $\|D_2 \ln a\|^2$, second differences between nodes | keep $\ln a$ smooth — the loop emulator misbehaves on jagged input |
| $\sigma_{\ln g}$ | 0.3 | $\ln$ of every growth quantity ($f$, $H$, $D_A$, $D$-ratios) | keep them positive and $O(1)$ |
| $\sigma_{\ln h}$ | 0.05 | $\ln h_{\rm conv}$ | $h_{\rm conv}$ is *exactly* unconstrained by the data (the $g_1$ symmetry) |

None of them was meant to carry information. The cell below turns each one off and on in turn.

**The result is unambiguous: $\lambda$ is the block that matters, and $\sigma_{\ln a}$ does
essentially nothing to $\Lambda$CDM.** With $\lambda$ alone, $\sigma(\omega_{\rm cdm})$ goes from
0.0016 to 0.0010 and $\sigma(\ln 10^{10}A_s)$ from 0.0325 to 0.0139 — the entire effect. Turning
$\sigma_{\ln a}$ from 0.5 to 10 changes the $\Lambda$CDM errors not at all. The reason is that
$A_s$ is a *constant* shift of $\ln a$, which costs the smoothness penalty nothing directly, but
$\lambda$ pins the curvature of $\ln a$ in $\ln k$ and so breaks the degeneracy between the
template shape and the cosmological parameters.

$\sigma_{\ln a}$ is not innocent either, just in a different place: it sets much of the *reported*
P(k) band (section 4), which is why the band in that figure is shown for both settings.

**Recommended for the next chain: $\sigma_{\ln a} = 2$, $\lambda = 50$.** The node-to-node
roughness the prior still allows goes from 0.066 to 0.135 — still smooth, which was $\lambda$'s
actual job. The MI results barely move ($\sigma(f)$ 9.7% → 10.0%, $D_M/r_d$ with the phase prior
0.85% → 1.03%); the P(k) band widens to what the data actually constrain.

**Two warnings from the same audit.**

1. No practical $\lambda$ makes the induced $\Lambda$CDM prior negligible ($\lambda = 5$ still
   leaves $A_s$ at 0.76 of its data-only width, and $\lambda = 0$ allows a roughness of 5). So
   widening is not by itself a fix: an MI posterior compressed onto $\Lambda$CDM **must** divide
   out the induced prior. Done that way it matches the direct fit at any prior width, which is
   what section 6 shows.
2. $\lambda$ multiplies *unnormalized* second differences, so for a smooth function its strength
   scales as (node spacing)$^4$. The 16-node "BAO-rigid" basis of notebook 07 therefore carries a
   smoothness prior ~250× stronger than the same $\lambda$ at spacing 0.15. The phase prior of
   notebook 09 does not have this problem: it is stated directly as a fraction and always applied
   on the 60-node basis.
"""),
code("%run prior_decomposition_pb.py"),
show('fig_prior_decomposition.png', width=1150),
md(r"""
## Summary

| | P(k) | P(k)+B(k) |
|---|---|---|
| fractional $1\sigma$ on $P_{\rm lin}$ per node, in the data window | 27.4% | 23.1% |
| $\sigma(f)/f$ at $z=0.71$ | 18.5% | **9.7%** |
| $\sigma(F_{\rm AP})$ at $z=0.71$ | 1.62% | 1.59% |
| $\sigma(\omega_{\rm cdm})$ | 0.0023 | 0.0016 |
| $\sigma(\ln 10^{10}A_s)$ | 0.040 | 0.033 |
| $\sigma(h)$ | 0.0037 | 0.0034 |
| $\Lambda$CDM through the MI model / direct | 0.99–1.03 | 0.99–1.02 |

(The $P_{\rm lin}$ row uses the recommended wider priors; with the tighter priors run so far it
reads 11.3% / 10.2%, most of which is prior. Everything else is insensitive to the choice.)

The bispectrum's information goes almost entirely into the **growth rate and the amplitude**, not
into the geometry. That is expected: at tree level $B \propto P^2$, so it fixes the bias–amplitude
degeneracy, while the AP parameters come from the shape of the anisotropy, which P(k) already
carries.

**Caveats.** Tree-level bispectrum, Gaussian covariance, no window, zero P–B cross-covariance, a
noiseless mock at the fiducial, and a Fisher (precision, not accuracy — a bias test with a mock
generated at a shifted cosmology is still to do).
"""),
]

# =======================================================================================
nb09 = [
md(r"""
# An alternative BAO pipeline: AP parameters without assuming a template

**The question.** A standard BAO analysis takes the *shape and phase* of the BAO wiggles from a
fiducial cosmology and lets the broadband float. Can we instead leave the whole linear spectrum
free — no fiducial cosmology anywhere — and still measure the distance scale? And does the
bispectrum help?

**What is measurable, and why.** The model has an exact symmetry $g_1$: rescale $h_{\rm conv}$,
every $D_A$ and every $1/H$ by the same factor and no observable changes at all (pybird's AP
carries the full volume factor, so the Mpc$/h$ ruler length and a common rescaling of all
distances are literally the same parameter). Absolute $D_A$ and $H$ are therefore *not* measurable
— exactly as in a real BAO analysis, which measures distances only in units of $r_d$. What is
invariant under $g_1$, and so measurable, is
$$\alpha_\perp \equiv \ln D_A - \ln h_{\rm conv} = D_M/r_d, \qquad
  \alpha_\parallel \equiv -\ln H - \ln h_{\rm conv} = D_H/r_d,$$
where the "ruler" is the template's own BAO feature. $F_{\rm AP} = D_M/D_H$ needs no ruler at all.

**The three stages of this notebook.**

1. **No assumptions.** The template is completely free. The wiggles can move with it, so the ruler
   floats: $D_M/r_d$ is not measurable. What survives is $F_{\rm AP}$, the distance *ratios*
   between redshifts, and $f$.
2. **The model we actually sample.** Adding the smoothness prior of the HMC gets $D_M/r_d$ to
   ~2%, but the template can still slide its own wiggles by 3% — so that "2%" is largely the
   prior's ruler, not the data's.
3. **A phase prior.** Forbid the template from moving the wiggles, while leaving the broadband
   and the wiggle *amplitude* completely free. That is precisely the assumption a standard BAO
   analysis makes, and it brings $D_M/r_d$ to 0.9% — pre-reconstruction BAO precision on the same
   mock.

The data, likelihood and Jacobian are the same as in `08_growth_pk_bk.ipynb`; see there for how
the Fisher is built.
"""),
code(SETUP),
md(r"""
## 1. Stage A — no assumptions at all

All 80 template knots free, **no prior on the template whatsoever**, everything marginalized
(the numbers come from `robust_marg_pb.py`, which does this case with an SVD because the system is
genuinely degenerate and no covariance matrix exists along the flat directions).
"""),
code(r"""
S = np.load('../../output/fisher_pb/robust_pb_results.npz')
z = S['zeff']; iz = 2
print(f"Fully free 80-knot template, no prior, everything marginalized (z = {z[iz]:.3f}):\n")
print(f"{'':34s}{'P(k)':>12s}{'P(k)+B(k)':>12s}")
rows = [('D_M/r_d  (absolute distance)', 'alpha_perp'), ('F_AP = D_M/D_H', 'F_AP'),
        ('isotropic distance vs z=0.93', 'alpha_iso_rel'), ('growth rate f', 'f')]
for lab, key in rows:
    a, b = S[f'geo_free_free_p_{key}'][iz], S[f'geo_free_free_pb_{key}'][iz]
    fmt = lambda v: ('not measured' if v > 1 else f'{100*v:.2f}%')
    print(f"  {lab:32s}{fmt(a):>12s}{fmt(b):>12s}")
print(f"\n  BAO wiggle detection S/N          {S['snr_free_p_marg'][0]:11.1f}"
      f"{S['snr_free_pb_marg'][0]:12.1f}")
print("\nThe distance SCALE is unmeasurable: with a free template the wiggles move with it, so"
      "\nthe model carries its own ruler. What survives is the AP ratio and the distances"
      "\nRELATIVE to one redshift -- and the bispectrum improves both by ~40%, while turning f"
      "\nfrom unmeasured into a 19% measurement.")
"""),
md(r"""
## 2. Stage B — the model we sample, and why its ruler is the prior's

The HMC model puts $\ln a$ on 60 nodes, spaced 0.15 in $\ln k$, with a smoothness prior
$\lambda = 200$. A BAO oscillation spans $2\pi/(r_d k)$ in $\ln k$ — 1.2 at $k = 0.05$, 0.6 at 0.1,
0.3 at $0.2\ h/$Mpc — so there are about four nodes per wiggle and the basis can reproduce a
wiggle *shift* almost perfectly.

Concretely: write $T = T_{\rm nw}(1+O)$, with $T_{\rm nw}$ the smooth part and $O$ the wiggles.
Sliding the wiggle pattern by $\delta$ in $\ln k$ changes the spectrum by
$$\Delta \ln P = \delta\, v(k), \qquad v \equiv \frac{{\rm d}\ln(1+O)}{{\rm d}\ln k}.$$
If $\ln a$ can contain $v$, the fit cannot tell a template slide from a change in distance. The
figure shows $v$ and how much of it each basis can represent: **96% for the 60-node basis we
sample, 16% for a basis with one node per BAO period**.
"""),
code("%run meeting_bao_pb.py"),
show('fig_phase_prior_explained.png', width=860),
md(r"""
## 3. Stage C — the phase prior, and how it is implemented

The assumption a standard BAO analysis makes is: *the broadband is free, the wiggle phase is not*.
Here that is one Gaussian prior, on the component of $\ln a$ along $v$:

$$\ln a = \delta(k)\,v(k) + (\text{the rest}), \qquad
  \delta(k) = \sum_{j=0}^{5} c_j P_j(\ln k), \qquad
  c_j \sim \mathcal{N}(0, \sigma_\delta^2).$$

$\sigma_\delta$ is literally *how far the template is allowed to move the ruler*, in per cent. One
global slide ($j = 0$ alone) is **not** enough — the template can drift the phase slowly across the
band and take most of the freedom back — so the prior covers $v$ times Legendre envelopes of degree
0–5 over $0.01 < k < 0.4\ h/$Mpc: the phase is pinned *everywhere*, not just on average. The wiggle
**amplitude is never touched**: it is the damping nuisance of a BAO fit and stays free, as does the
entire broadband.

For reference, the smoothness prior alone allows a 3.2% slide — which is why stage B's "2%"
distance was really the prior's ruler.
"""),
code(r"""
import inspect, mi_prior_fisher as M
print(inspect.getsource(M.phase_modes))
print(inspect.getsource(M.phase_precision))
"""),
md(r"""
### Tightening the prior

Left and centre: as $\sigma_\delta$ shrinks, the distances sharpen and then saturate — once the
template can no longer move the ruler, the data's own BAO feature sets the error. Right: pinning
the phase with more envelope modes converges toward the dashed line, which is a basis too coarse
to slide anything at all.

The saturated value is what this method delivers: **$D_M/r_d$ to 0.88% and $D_H/r_d$ to 1.35% at
$z = 0.71$ from P(k) alone**, against 0.65% and 1.53% for a standard pre-reconstruction BAO fit on
the same mock (Seo & Eisenstein 2007, the formula DESI's design forecasts used). The
model-independent analysis is slightly worse on $D_M$ and slightly *better* on $D_H$, with a
completely free broadband.
"""),
show('fig_phase_scan.png', width=1150),
md(r"""
## 4. P(k) vs P(k)+B(k), at every redshift

Solid: the model we sample (no phase prior). Dashed: with the phase prior. Grey: standard BAO,
pre- and post-reconstruction, on the same bins.
"""),
show('fig_bao_per_z.png', width=1150),
md(r"""
### All twelve BAO parameters at once

$D_M/r_d$ and $D_H/r_d$ at every redshift, with the phase prior, everything else marginalized.
The off-diagonal panels are what a single-redshift plot cannot show: the redshift bins are
correlated, because they share one template, one set of EFT priors and one $h_{\rm conv}$.
"""),
show('fig_bao_triangle_all_z.png', width=1250),
md(r"""
### One redshift in detail

The same at $z = 0.71$ only, with the growth rate, and with the no-phase-prior case for contrast.
"""),
show('fig_bao_corner.png', width=760),
md(r"""
## Summary

At $z = 0.71$, fractional $1\sigma$:

| | $D_M/r_d$ | $D_H/r_d$ | $F_{\rm AP}$ | $f$ |
|---|---|---|---|---|
| free template, no prior | not measurable | not measurable | 5.1% → 3.2% | — → 19% |
| + smoothness prior (what we sample) | 2.13% → 1.94% | 2.37% → 2.17% | 1.62% → 1.59% | 18.5% → 9.7% |
| **+ phase prior 0.2%** | **0.88% → 0.85%** | **1.35% → 1.32%** | 1.59% → 1.57% | 18.3% → 9.4% |
| coarse 16-node basis (rigid by construction) | 0.71% → 0.69% | 1.30% → 1.26% | 1.58% → 1.55% | 18.5% → 9.6% |
| standard BAO, pre-reconstruction | 0.65% | 1.53% | — | — |
| standard BAO, post-reconstruction | 0.41% | 0.73% | — | — |

*(each cell: P(k) → P(k)+B(k))*

**Reading.**

* With a genuinely free template there is no distance scale — the template carries its own ruler.
  This is not a numerical problem; it is the exact statement that the data cannot separate a
  wiggle shift from a distance.
* One explicit prior — the wiggles may slide by at most $\sigma_\delta$ — recovers
  pre-reconstruction BAO precision while leaving the broadband entirely free. This is the
  *minimal* form of the assumption every BAO analysis already makes, written down rather than
  implied by a fixed template.
* The bispectrum adds a few per cent on the geometry and a factor two on $f$. Its value here is
  the growth rate, not the ruler.
* The gap to post-reconstruction BAO (0.41% / 0.73%) is reconstruction, not the
  model-independent approach.

**Still to do.** A bias test (generate the mock at a shifted cosmology — $\omega_b$, $\omega_{cdm}$
or $N_{\rm eff}$ — fit with the phase prior, and check the shift in $D_M/r_d$ against $\sigma$);
an HMC run with the phase prior; and combining with post-reconstruction BAO. The honest next step
beyond a fixed phase is to marginalize over the *physical* ways the wiggle shape can change
(derivatives of $O$ with respect to $\omega_b$, $\omega_{\rm cdm}$, $N_{\rm eff}$, $n_s$ at fixed
$r_d$), which removes the fiducial-template dependence while still excluding the pure dilation
that is being measured.
"""),
]

for name, cells in [('08_growth_pk_bk.ipynb', nb08), ('09_bao_pk_bk.ipynb', nb09)]:
    for i, c in enumerate(cells):
        c['id'] = f'c{i:02d}'
    nb = {'cells': cells,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                      'name': 'python3'},
                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    json.dump(nb, open(name, 'w'), indent=1)
    print(f"wrote {name}: {len(cells)} cells")
