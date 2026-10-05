"""Sync 01_fisher_v3.ipynb from run_fisher_v3.py: code cells are replaced verbatim from
the `# %% [cell N]` sections, outputs cleared. Markdown edits are applied here too."""
import io, json, re

src = io.open('run_fisher_v3.py', encoding='utf-8').read()
parts = re.split(r'^# %% \[cell (\d+)\][^\n]*\n', src, flags=re.M)
cells = {int(parts[i]): parts[i + 1].strip('\n') for i in range(1, len(parts), 2)}

nb = json.load(io.open('01_fisher_v3.ipynb', encoding='utf-8'))
n_code = 0
for i, c in enumerate(nb['cells']):
    if c['cell_type'] != 'code':
        continue
    assert i in cells, f"notebook code cell {i} has no matching script section"
    c['source'] = [l + '\n' for l in cells[i].split('\n')]
    c['source'][-1] = c['source'][-1].rstrip('\n')
    c['outputs'] = []
    c['execution_count'] = None
    n_code += 1

MD = {}
MD[16] = """## 8. Marginalization with clean linear algebra

Every sector "marginal" below is a **Schur complement**, i.e. it *divides* by the block being
marginalized, so the conditioning of that block is what decides whether the answer means
anything.

- `psd_clip` symmetrizes a Fisher and clips its negative eigenvalues to zero **before** it is
  marginalized or plotted. A Fisher is PSD by construction, so the negative eigenvalues of the
  autodiff Hessian (here $\\sim10^{-7}$ of the largest) are numerical noise. v3.0 left them in
  and inverted them with weight $\\sim10^{7}$: its growth marginal came out with eigenvalues
  $-134$ and $-1.2$, and its $(\\omega_{\\rm cdm}, h)$ projection $-24.5$ — not a probability
  distribution, which is what made the purple contour and $\\sigma(h_{\\rm conv}) = 8\\times10^4$
  nonsense.
- `pos_pinv` (zero weight for null directions) is used **only** to invert nuisance blocks inside
  Schur complements, now with a threshold safely above the measured Hessian noise (Gate 6 scans
  it over four decades).
- `fisher_to_cov` floors unconstrained eigenvalues to a **large** variance — never zero — when
  converting a Fisher to a covariance (this was v1's plotting bug).
- Gate 2b compares the Hessian with the Gauss–Newton Fisher $J^T P J$, which is PSD by
  construction, to confirm the clipped part is numerical."""

MD[18] = """## 9. Jacobians and the six pieces; Gates 3 & 4

All six objects in the figure are 3×3 in $(\\omega_{\\rm cdm}, \\ln 10^{10}A_s, h)$ and are built
from the same three blocks $A$, $G$, $C$ of $F_{\\rm phys}^{\\rm marg}$, so that **both** chain
orderings are exactly additive:

$$F_{\\rm comb} \;=\; \\underbrace{F_{P(k)}^{\\rm marg} + F_{g|P(k)}}_{p(a)\\,p(g\\mid a)}
\;=\; \\underbrace{F_{g}^{\\rm marg} + F_{P(k)|g}}_{p(g)\\,p(a\\mid g)}.$$

The two orderings are alternative *partitions* of the same information: each assigns all of the
cross-sector information to its own conditional, so a marginal from one ordering must never be
added to a conditional from the other.

Gate 4 is the hard correctness check: **no marginal or conditional sub-piece may beat the full
Direct constraint on any parameter.**"""

MD[20] = """## 10. Where the $h$ information lives

The answer is fixed by two degeneracies of the MI physical block, both checked numerically in
Gate 5a.

1. **$a\\,D^2$ invariant** (exact). Raising every template amplitude by $d\\ln a$ and lowering every
   $D(z)/D(z_{\\rm ref})$ by $d\\ln a/2$ leaves every model spectrum unchanged. The absolute
   normalization — i.e. $A_s$ — is therefore not measurable from the template alone, and
   $\\ln A_s$ is **flat** in the P(k) marginal.
2. **Units bridge** (near-exact). $h_{\\rm conv}$ only rescales the grid on which the template
   is handed to the emulator, $k[h/{\\rm Mpc}] = k[1/{\\rm Mpc}]/h_{\\rm conv}$ with
   $P \\to P h_{\\rm conv}^3$, which with all 80 knot amplitudes free is a rigid dilation of the
   template, $d\\ln a_j = (3 + d\\ln P/d\\ln k|_j)\\, d\\ln h_{\\rm conv}$, and the amplitudes can
   follow it. Only $\\sim10^{-5}$ of the $h_{\\rm conv}$ information survives marginalizing the
   template ($\\sigma(h_{\\rm conv}): 0.0019 \\to 0.52$), so $h$ is unconstrained in the growth
   marginal for any practical purpose and that marginal collapses to an $\\Omega_m$ band. The
   $10^{-5}$ residual is not physics: the emulator sees the template only after re-interpolating
   it onto its own fixed knots and normalizing by $\\max(P)$, and those two steps are what stop
   the family from closing exactly. (`mi_model.py` makes the same statement for `units='mpc'`.)

So **neither sector marginal contains $h$**, and this is not a numerical accident: the
standard-ruler calibration is the product of a template scale and a dilation and lives entirely
in the cross term. It shows up in whichever conditional the chain ordering puts second — which
is why *growth | P(k)* carries all of it. Conditioning is a different question from
marginalizing, and a conditional can be tighter than the full fit, because fixing the other
sector removes uncertainty; it is not an independent, addable information sector."""

MD[22] = """## 11. The triangle plots

`make_fisher_fig.py` writes three files: `triangle_v3.png` (the four curves of the paper
caption), `triangle_v3_split.png` (all six, at the scale of the direct fit) and
`triangle_v3_wide.png` (the same six, on a window wide enough that the two sector marginals are
visible). It is also runnable standalone against the saved `.npz`, so the plotting can be
changed without re-executing the notebook.

The six curves, in the two chain orderings:

- **Gray filled — Direct**: the full three-parameter fit.
- **Blue — Combined**: the complete projected MI Fisher, lying on top of Direct (Gate 3).
- **Red — P(k) marginal**: template alone, growth marginalized. A pure $\\omega_{\\rm cdm}$ slab;
  $\\ln A_s$ and $h$ are exactly flat.
- **Purple dotted — Growth+AP marginal**: growth alone, all 80 amplitudes marginalized. A pure
  $\\Omega_m$ band; $h$ is exactly flat along it. Wider than the plot window at the scale of the
  direct fit, which is why the second, zoomed-out figure exists.
- **Green — Growth+AP | P(k)**: the second factor of $p(a,g) = p(a)\\,p(g\\mid a)$. It carries
  **all** of the $h$ and the $A_s$ calibration, including the shift of the conditional growth
  mean induced when the template changes — which is why it has $A_s$ information even though the
  growth Jacobian has no $A_s$ column.
- **Yellow dashed — P(k) | Growth+AP**: the second factor of the reverse ordering
  $p(a,g) = p(g)\\,p(a\\mid g)$. Because the growth marginal is nearly empty, this conditional
  takes almost everything and sits essentially on top of Direct. That is the correct answer, not
  a duplicated curve.

Red + green reproduces Combined; purple + yellow reproduces Combined in the reverse ordering
(Gate 5b). Green and yellow are alternatives and must not be added to each other."""

for i, txt in MD.items():
    assert nb['cells'][i]['cell_type'] == 'markdown', i
    nb['cells'][i]['source'] = [l + '\n' for l in txt.split('\n')]
    nb['cells'][i]['source'][-1] = nb['cells'][i]['source'][-1].rstrip('\n')

# header cell: record the v3.1 fixes
h = ''.join(nb['cells'][0]['source'])
h = h.split('\n### v3.1')[0].rstrip()   # idempotent: drop a previously appended block
add = """

### v3.1 (2026-09-04): two defects fixed, both hitting the growth sector

1. **Cusp at the expansion point.** pybird's emulator interpolates its $P_{\\rm lin}$ input onto
   the 80 knots with a piecewise-**linear** interpolant, and our template sits exactly *on* those
   knots at $h_{\\rm conv} = h_{\\rm fid}$. An infinitesimal grid shift therefore picks up a
   one-sided slope, and the log-likelihood curvature along $h_{\\rm conv}$ jumps by $\\sim20\\%$
   at the expansion point, so `jax.hessian` returned a one-sided second derivative for every
   $h_{\\rm conv}$ entry. Patched to cubic interpolation, as `mi_model.py` already does, so the
   Fisher and the sampled analysis share exactly one model.
2. **Noise inversion in the marginals.** The sector marginals are Schur complements; the autodiff
   Hessian is PSD only to $\\sim10^{-7}$ of its largest eigenvalue, and `pos_pinv(rtol=1e-12)`
   inverted that noise with weight $\\sim10^{7}$. v3.0's growth marginal had eigenvalues $-134$
   and $-1.2$ (its $(\\omega_{\\rm cdm}, h)$ projection $-24.5$). Every Fisher is now PSD-clipped
   before being marginalized or plotted, and the pseudo-inverse threshold sits above the measured
   noise (Gate 6 scans it).

Gates 5a/5b and Gate 6 are new: the two two degeneracies of the MI block, the additivity of
both chain orderings, and the threshold stability of every quoted number."""
nb['cells'][0]['source'] = [l + '\n' for l in (h + add).split('\n')]
nb['cells'][0]['source'][-1] = nb['cells'][0]['source'][-1].rstrip('\n')

json.dump(nb, io.open('01_fisher_v3.ipynb', 'w', encoding='utf-8'), indent=1)
print(f"synced {n_code} code cells + {len(MD)+1} markdown cells")
