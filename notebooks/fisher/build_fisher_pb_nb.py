"""Build 06_fisher_pk_bk.ipynb from run_fisher_pb.py.

Code cells are copied verbatim from the `# %% [cell N]` sections of the script; the markdown
cells live here, so the notebook can be rebuilt at any time and the script stays the single
source of the code. Idempotent: rerun after editing either file.

    python3 build_fisher_pb_nb.py        # writes the notebook with cleared outputs
    sbatch exec_fisher_pb.sbatch         # executes it in place
"""
import io, json, re

SCRIPT = 'run_fisher_pb.py'
NOTEBOOK = '06_fisher_pk_bk.ipynb'

src = io.open(SCRIPT, encoding='utf-8').read()
parts = re.split(r'^# %% \[cell (\d+)\][^\n]*\n', src, flags=re.M)
preamble = parts[0]
code = {int(parts[i]): parts[i + 1].strip('\n') for i in range(1, len(parts), 2)}

MD = {}

MD[0] = r"""# Model-Independent Fisher: P(k) alone vs P(k) + B(k)
## DESI Y6 configuration (7 skies, 6 unique redshifts), tree-level bispectrum

How much does the galaxy **bispectrum** tighten the model-independent growth/AP sector?
This notebook answers that with the *same* model-independent (MI) parametrization as
`01_fisher_v3.ipynb`, so the two are directly comparable, and it computes **every** Fisher
object twice: once for a P(k)-only data vector and once for P(k)+B(k).

### The model (unchanged from v3.1)

- **Template**: $P_{\rm lin}$ in Mpc³ at $z_{\rm ref}=5$ on the EFT emulator's **80 native
  knots** (a fixed grid in 1/Mpc), with a free amplitude $a_j$ per knot. Parametrizing there
  makes the emulator input interpolation-free.
- **Growth/AP block**: $[f,\ H/H_0,\ D_A H_0]$ per unique redshift, the ratios
  $D(z)/D(z_{\rm ref})$, and a free unit-conversion parameter $h_{\rm conv}$ carrying
  Mpc³ → (Mpc/$h$)³.
- **Direct ΛCDM model** = the MI model composed *exactly* with the cosmology mappings, so the
  chain rule $F^{\rm direct} = J^T F^{\rm MI} J$ is an identity (Gate 3) and the information
  split is unambiguous.
- **Data** = this notebook's own direct model at the fiducial ⇒ $\chi^2(\rm fid) = 0$ and the
  expansion point is exactly the likelihood maximum (Gate 1).

### What is new here

1. **The likelihood is pybird-dev's.** The bispectrum JAX code (`pybird/bk_*.py`) exists only
   in the `pybird-dev` checkout, so that tree is put first on `sys.path`. The plan for making
   this permanent is in `PLAN_merge_bispectrum.md`.
2. **The EFT basis is `eth`** (2211.17130, App. D.4) — the only basis pybird accepts with
   `bBk` in the output. For the *power spectrum* this is not a different model: `Bird.setBias`
   maps it onto the familiar parameters,
   $b_1 = Bb_1$, $b_2 = Bb_2$, $b_3 = Bb_3 + 15\,Bb_8$, $b_4 = Bb_5$, $c_{ct} = -Bc_1$,
   $c_{r1} = f\,Bc_2 - \tfrac{f^2}{2} Bc_4$, $c_{r2} = -\tfrac{f^2}{2} Bc_3$,
   $c_{e0} = Be_1$, $c_{e1} = Be_2 + \tfrac{1}{2} c_{e2}$,
   and the loop contraction is the same 35-term layout the 80-knot emulator produces. So the
   **P(k) loops still come from the emulator** (`with_emu: True`), exactly as in v3; the free
   non-marginalized parameters $Bb_1, Bb_2, Bb_5$ per sky are v3's $b_1, b_2, b_4$ renamed.
   The tree-level bispectrum adds $Bd_1$ and $Be_5$ to the analytically marginalized set.
3. **The bispectrum is at tree level** (`with_bk_tree_level: True`), which needs no external
   loop matrices. Switching to one loop is one configuration change: set it to `False` and
   give each sky a `bk_loop_matrix_path`. Nothing else in the script depends on it.
4. **Two likelihoods read the same data file** — `'bPk'` and `'bPk,bBk'` — so the P-only
   baseline and the P+B result differ *only* by the presence of the bispectrum block.

### Caveats worth stating up front

- The power spectrum is $P_0+P_2+P_4$ to $k = 0.2\,h/$Mpc (`MULTIPOLE = 3`), the same
  multipoles as the v3 P(k)-only Fisher, so the P(k) column here is the v3-comparable
  baseline. **Both** likelihoods carry it, not the P(k)-only one alone: the comparison rests
  on the P(k) data being a strict subset of the P(k)+B(k) data, which is what makes Gate 7
  ($F_{P+B} - F_P$ PSD) true and what lets every width ratio be read as "what $B(k)$ adds".
  Giving P(k) a multipole that P(k)+B(k) lacks would break the nesting and could make the
  P(k)-only column tighter on some directions, which would not be a statement about the
  bispectrum at all. Nothing in the bispectrum code reads the P(k) multipole count
  (`bk_Common` passes `Nl` through to `Common`; no `bk_*` module touches `self.co.Nl`), so
  `multipole: 3` is safe alongside `bBk` even though every bispectrum configuration shipped
  with pybird uses 2. Set `MULTIPOLE=2` to recover that convention.
- $B_0$ on closed ordered triangles with $0.02 \le k_i \le 0.10\,h/$Mpc, analytic PPP
  covariance, **zero** P–B cross-covariance, no binning or window — the same idealization the
  v3 P(k) Fisher makes.
- The computation is pinned to the **CPU**: autodiff Hessian reductions are non-deterministic
  on GPU, and every marginal below is a Schur complement that divides by the block being
  marginalized, so that noise is amplified into the answers."""

MD[2] = r"""## 1. Setup: fiducial cosmology, DESI Y6 survey, CosmoPower-JAX

The survey is the same seven-sky DESI-Y6-like configuration as v3 (six unique effective
redshifts; sky 4 and 5 share $z=0.930$). `SMOKE=1` cuts it to two skies for a pipeline check."""

MD[4] = r"""## 2. The P(k) parametrization: the emulator's own 80 knots

The EFT emulator compresses whatever `pk_lin`/`kk` it is handed onto 80 fixed knots spanning
$[10^{-4}, 0.7]\,h/$Mpc (`pybird/emu_data/knots.npy`). Parametrizing the template *exactly
there* means the emulator input is interpolation-free at the expansion point.

The template is stored in **Mpc³ at fixed knots in 1/Mpc at $z_{\rm ref} = 5$**, where the
cell below verifies it is $h$-independent to well below a percent. That is what lets the
amplitudes carry $(\omega_{\rm cdm}, A_s)$ and **no $h$**: the entire $h$ response is routed
through the growth/AP map and $h_{\rm conv}$."""

MD[6] = r"""## 3. Cosmology → observable mappings

Verbatim from v3, so that the two notebooks share one model:

- `cosmo_to_amps`: $(\omega_{\rm cdm}, \ln 10^{10}A_s, h) \to$ amplitude ratios at the fixed
  physical knots (the $h$ column is $\approx 0$ by construction).
- `cosmo_to_growth`: $(\omega_{\rm cdm}, h) \to [f, H/H_0, D_A H_0, D\text{-ratios},
  h_{\rm conv}{=}h]$; $h$ enters through $\Omega_m = \omega_m/h^2$ *and* through the unit
  conversion.

The fiducial growth vector is defined through the same function used for the Jacobians, so
the Fisher expansion point and the projection agree by construction."""

MD[8] = r"""## 4. Shared model builder

One function maps (amps, growth) → pybird inputs, and the **direct model is its exact
composition** with the cosmology mappings. Unit conversion via $h_{\rm conv}$ moves the same
physical points, so no interpolation is involved:
$k\,[h/{\rm Mpc}] = k\,[1/{\rm Mpc}]/h_{\rm conv}$ and
$P\,[({\rm Mpc}/h)^3] = P\,[{\rm Mpc}^3]\,h_{\rm conv}^3$.

pybird-dev's emulator path sets `with_exact_time` and then requires the four time
coefficients explicitly, so the dicts carry the Einstein-de Sitter values
$G_1 = 1$, $Y_1 = 0$, $\tilde G_1 = 3/7$, $\tilde V_{12} = 1/7$ — the same values the
model-independent tree hard-codes. The model is therefore identical to v3's."""

MD[10] = r"""## 5. Fake DESI Y6 $P_0+P_2+P_4+B_0$ data = the direct model at the fiducial

`Fake` builds the survey, the analytic covariances and the likelihood configuration, but its
own `set()` would evaluate the model through a Boltzmann solver. `fake_set_from_cosmo_dicts`
replaces that one step with the notebook's own direct model, evaluated from explicit
`cosmo_dict`s, and keeps every other part of pybird's code path: the Gaussian P(k)
covariance, the PPP bispectrum covariance, a zero P–B cross-covariance block, EFT priors
recentred on the injected truth, and the written YAML.

Both likelihoods therefore peak exactly at the fiducial, for both data vectors."""

MD[12] = r"""## 6. The two likelihoods and the four model functions; Gate 1

`L['P']` and `L['P+B']` read the **same** HDF5 file; the P-only configuration simply drops
`bBk` from the output and removes the two bispectrum-only EFT priors ($Bd_1$, $Be_5$).

**Gate 1** requires $\chi^2(\rm fid) = 0$ for the direct *and* the model-independent
likelihood, for both data vectors: the Fisher is only a Fisher at the likelihood maximum."""

MD[14] = r"""## 7. The Fisher matrices, and Gate 7

Four Hessians: model-independent and direct, for P and for P+B.

**Gate 7** is the new consistency check that the bispectrum brings: the P(k) data are a
subset of the P(k)+B(k) data with a zero cross-covariance block, so
$F_{P+B} - F_{P}$ must be positive semi-definite. Adding data cannot remove information."""

MD[16] = r"""## 8. The sector split, with clean linear algebra

Every "marginal" below is a **Schur complement**, i.e. it *divides* by the block being
marginalized, so the conditioning of that block decides whether the answer means anything.
`fisher_utils.py` holds the v3.1 routines: `psd_clip` (a Fisher is PSD by construction, so
the negative eigenvalues of an autodiff Hessian are noise and are clipped before anything is
marginalized or plotted), `pos_pinv` (only ever used to invert nuisance blocks inside Schur
complements, with a threshold above the measured noise) and `fisher_to_cov` (unconstrained
directions floored to a *large* variance, never zero).

`sector_split` returns the EFT-marginalized physical block and its three pieces: $A$ (the 80
template amplitudes), $G$ (growth/AP + $h_{\rm conv}$) and the cross block $C$."""

MD[18] = r"""## 9. Projection onto ΛCDM; Gates 3, 4 and 5b

All six objects are $3\times3$ in $(\omega_{\rm cdm}, \ln 10^{10}A_s, h)$ and are built from
the same three blocks, so **both** chain orderings are exactly additive:

$$F_{\rm comb} \;=\; \underbrace{F_{P(k)}^{\rm marg} + F_{g|P(k)}}_{p(a)\,p(g\mid a)}
\;=\; \underbrace{F_{g}^{\rm marg} + F_{P(k)|g}}_{p(g)\,p(a\mid g)}.$$

The two orderings are alternative *partitions* of the same information: each assigns all of
the cross-sector information to its own conditional, so a marginal from one ordering must
never be added to a conditional from the other.

- **Gate 3**: the chain rule $F^{\rm direct} = J^T F^{\rm MI} J$.
- **Gate 4**: no marginal or conditional sub-piece may beat the full direct constraint.
- **Gate 5b**: both orderings reproduce the combined Fisher."""

MD[20] = r"""## 10. The deliverable: the growth/AP sector, P(k) vs P(k)+B(k)

The growth block is reported in **two conditionings**, both with the EFT parameters
marginalized, because they answer two different questions and they behave very differently.

**Template fixed** ($G$): the shape of $P_{\rm lin}$ is held at the fiducial. This is what a
conventional fixed-template AP/RSD measurement delivers. It is a well-conditioned block, the
Gate 7 ordering is verified for every plotted sub-block, and the corner plots and the
$\sigma$ table are meaningful as they stand.

**Template free** ($G^{\rm marg}$): the 80 amplitudes are marginalized. This is the honest
model-independent statement, and the two degeneracies of v3 survive the bispectrum:
(i) $a\,D^2$ is an exact invariant — and $B_{\rm tree} \propto P_{\rm lin}^2$ is invariant
under it too, so $\ln A_s$ stays flat in the template marginal; (ii) $h_{\rm conv}$ remains a
rigid dilation that the free amplitudes can follow.

**That degeneracy is not a nuisance to be plotted around — it is the result, and it forces a
change of instrument.** A covariance for a degenerate block only exists once the flat
directions are truncated at some $\sigma_{\rm max}$, and the truncation is applied along
*each data set's own* eigenvectors, so it can destroy the very ordering Gate 7 guarantees,
drawing a P(k)+B(k) contour *wider* than the P(k) one. That is exactly what happens here: no
truncation in $\sigma_{\rm max} \in [0.5, 20]$ preserves the ordering of the template-free
$(H, D_A)$ block, and the P(k)-alone widths drift by a factor of six across that range while
the P(k)+B(k) widths move by 50%.

So the template-free comparison is made with objects that need no truncation:

- every width in the table that moves by more than 10% when the cap is relaxed five-fold is
  prefixed `~` and its ratio is suppressed — it is a statement about the truncation, not the
  data;
- the **eigen-spectrum** of the growth Fisher (`growth_spectrum.png`) and the **count of
  directions measured better than $\sigma_{\rm max}$** are computed from eigenvalues, which
  need no cap at all;
- the eigen-structure of the AP sub-block is printed so the surviving combinations are named
  rather than inferred from per-parameter widths.

Everything is quoted in **fractional** units ($F_{\ln} = {\rm diag}(g)\,F\,{\rm diag}(g)$)
so that every redshift and both AP parameters sit on the same footing. Gate 5a re-measures
the two degeneracies for both data vectors."""

MD[22] = r"""## 11. Figures

`make_fisher_pb_fig.py` writes them and is runnable standalone against the saved `.npz`, so
the plotting can be changed without re-executing the notebook:

- `corner_growth_z<z>.png` — one 3×3 corner per redshift in fractional
  $(\Delta f/f, \Delta H/H, \Delta D_A/D_A)$, **template held fixed**.
- `corner_ap_all_z.png` — every redshift's AP pair at once, template held fixed.
- `corner_cosmo_pb.png` — $(\omega_{\rm cdm}, \ln 10^{10}A_s, h)$: the direct fit and the
  growth conditional, P vs P+B.
- `ap_summary.png` — fractional $\sigma$ against redshift, with the P+B / P ratio underneath.
- `growth_spectrum.png` — the **template-free** comparison, done with eigenvalues rather than
  contours for the reason given in §10: how many directions of the growth/AP block each data
  set actually measures, and how well.

**Color is the data set** (orange = P(k), blue = P(k)+B(k)); **line style is the conditioning
of the template** (solid = held fixed, dashed = marginalized); and exactly one curve per
figure is **filled** — P(k)+B(k) with the template fixed, the headline result — because two
overlapping semi-transparent fills blend into a third colour and destroy the identity
encoding. `_check_ordering` verifies, for every block that is actually drawn, that each
P(k)+B(k) contour lies inside its P(k) counterpart as Gate 7 requires."""

def _lines(text):
    parts = text.split('\n')
    return [l + '\n' for l in parts[:-1]] + [parts[-1]]

SUPERSEDED_NOTE = '> **Superseded in part (2026-09-11).** The per-redshift $H$, $D_A$ constraints, the growth/AP corners, the all-$z$ AP corner, the AP summary and the growth eigen-spectrum of this notebook were the discretization breaking of an exact symmetry (rescaling $h_{\\rm conv}$, every $D_A$ and $1/H$ together) read as information, in a prior-free 80-knot model the HMC never samples. Use **`07_robust_pk_bk.ipynb`** for the validated results; the old figures are in `output/fisher_pb/superseded/` with a README. $F_{\\rm AP}$, $f$ and the direct-ΛCDM results here are unaffected.'

cells = []
for idx in sorted(set(list(MD) + list(code))):
    if idx in MD:
        cells.append({'cell_type': 'markdown', 'id': f'md{idx}', 'metadata': {},
                      'source': _lines(MD[idx])})
        if idx == 0:
            cells.append({'cell_type': 'markdown', 'id': 'superseded-note',
                          'metadata': {}, 'source': SUPERSEDED_NOTE})
    if idx in code:
        cells.append({'cell_type': 'code', 'id': f'code{idx}', 'metadata': {},
                      'execution_count': None, 'outputs': [], 'source': _lines(code[idx])})

nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                   'language_info': {'name': 'python', 'version': '3.12'}},
      'nbformat': 4, 'nbformat_minor': 5}
with io.open(NOTEBOOK, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write('\n')
print(f"wrote {NOTEBOOK}: {len(cells)} cells "
      f"({sum(c['cell_type'] == 'code' for c in cells)} code, "
      f"{sum(c['cell_type'] == 'markdown' for c in cells)} markdown)")
# markdown headers are the even indices, code sections the odd ones directly after them
missing = sorted(n for n in code if (n - 1) not in MD)
if missing:
    print(f"note: code sections without a markdown header: {missing}")
