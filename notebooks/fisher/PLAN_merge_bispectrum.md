# Merging the bispectrum code into the model-independent pipeline

*Written 2026-09-09 alongside `06_fisher_pk_bk.ipynb`, the first model-independent (MI)
analysis that uses the bispectrum.*

## The situation

The bispectrum lives in **`pybird-dev`** (`pierrexyz/pybird-dev`, commit `6042880`):
`pybird/bk_bird.py`, `bk_common.py`, `bk_functions.py`, `bk_nonlinear.py`, `bk_projection.py`,
`bk_treelevel.py`, `bk_utils.py`, `bk_uv.py`, plus the restructured IO (`_io_format.py`,
`_io_hdf5.py`, `_io_output.py`, `_io_plot.py`, `_io_schema.py`), `utils_cov.py`,
`exact_time.py`, `baccoemu.py`, `cobaya.py`, `fnl.py`, `uvsub.py`.

Our repo carries a **fork** of pybird in `pybird_model_independent/pybird/`, registered as the
editable `pybird-lss 0.3.1` install in `~/pybird/jax_env`. It has diverged substantially:

| file | changed lines (dev vs ours) |
|---|---|
| `correlator.py` | 1094 |
| `io_pb.py` | 725 |
| `inference.py` | 618 |
| `cosmo.py` | 520 |
| `greenfunction.py` | 514 |
| `likelihood.py` | 503 |
| `module.py` | 431 |
| everything else | < 250 each |

`06_fisher_pk_bk.ipynb` sidesteps the problem for now: it puts the `pybird-dev` checkout
first on `sys.path` (`PYBIRD_DEV`), so that one notebook runs entirely against dev while the
rest of the pipeline keeps using our fork. That is fine for one Fisher forecast and wrong as
a permanent arrangement — two pybirds in one repository will drift.

## Recommendation: rebase our package onto pybird-dev, do not port `bk_*.py` into the fork

Porting the bispectrum into our fork means reconciling ~5000 lines of divergent
correlator/IO/inference code by hand, and then doing it again at every upstream release.

The rebase is far smaller than it looks, because **pybird-dev already contains almost every
hook the MI analysis needs** — they were upstreamed along the way:

| MI hook | status in pybird-dev |
|---|---|
| `Likelihood.loglkl(..., cosmo_module=None, cosmo_dict=[per-sky dict])` | present (`likelihood.py:437`) |
| `Correlator.compute(cosmo_dict=...)` direct-input path | present (`correlator.py:562`) |
| `get_alpha_bao_rec` with the `cosmo_module=None` DH/DM/rd branch | present (`likelihood.py:308`) — this *is* the MI BAO hook |
| `set_bao_rec` + `Symbolic.get_bao_distance` | present (`likelihood.py:143`, `symbolic.py`) |
| `with_loop_prior` | present |
| 80-knot emulator + `knots.npy` | present, byte-identical knots |

**Importantly, the repository we work from stays `pybird_model_independent`.** Only the
*package* directory `pybird_model_independent/pybird/` is replaced. `notebooks/`, `paper/`,
`output/`, `data/`, `scripts/` and `configs/` do not move, and no launcher path changes; only
`import pybird` resolves to newer code.

## What is genuinely ours, and what to do with it

| item | verdict |
|---|---|
| `cpj_pk_on_knots`, `cpj_z_ref` (correlator/cosmo/io options) | **obsolete.** They existed to make `Fake`'s internal CosmoPower path reproduce the notebook's direct model. `run_fisher_pb.py` instead writes the fake data *from the direct model itself* (`fake_set_from_cosmo_dicts`), which is simpler and exact by construction. |
| `Emulator(emu_path, knots_path)` | **redundant.** Dev reads `emu_data/knots.h5`; the knot values are identical. |
| in-tree `pybird/fftlog.py` | **obsolete.** Dev uses the `fftlog-lss` package, already installed in `jax_env`. |
| `with_rs_marg`, `with_boss_correlated_skies_prior` | **needed only if used.** Neither is active in the four production setups (`setups.py`). Port on demand. |
| EdS defaulting of `G1, Y1, G1t, V12t` under `with_emu` | **needed** (small). Our `bird.py` hard-codes them; dev raises unless the cosmo dict supplies them. Either add the default to `Cosmo.add_exact_time` upstream, or have `MIModel.build_cosmo_dicts` pass the four numbers, as `run_fisher_pb.py` does. |
| cubic `Emulator.make_params` interpolation (the v3.1 cusp fix) | **needed.** Keep as the monkeypatch (`mi_model.py`, `run_fisher_v3.py`, `run_fisher_pb.py` all carry it) or upstream it as an `EMU_INTERP_KIND` option. |
| `sympy` dependency | **environment.** `bk_projection.py` imports `sympy.physics.wigner`; installed into `jax_env` on 2026-09-09 (sympy 1.14, mpmath 1.3). Add it to the recipe. |

## Steps

1. **Audit** — `scripts/audit_mi_vs_dev.py`: walk `git diff` of our fork against dev, classify
   every hunk as already-in-dev / needed / obsolete, and print anything unaccounted for. The
   table above is the expected output; the script exists to catch what the manual read missed.
2. **Branch `mi-hooks` on pybird-dev** with the two or three genuinely needed items plus our
   tests (`robustness/bao_composition_test.py`, and a direct-model fake-data writer test that
   replaces `fake_cpj_options_test.py`). This is PR-able upstream.
3. **Gate the switch before touching the environment.** Run each of these against the dev tree
   via `PYTHONPATH`, with our fork still installed:
   - `01_fisher_v3` Gates 1 and 3, and `fisher_v3_results.npz` reproduced to floating-point
     noise (this is the strongest single check: emulator, eftoflss basis, the whole MI model);
   - `mi_model.py` smoke: χ²(fid), `gn_fisher`, `chain_split` additivity;
   - `setups.build_model` for `mock_full`, `boss_dr12`, `desi_dr1`: χ² at the stored best fits
     must reproduce the values in each `sampling_meta.json`;
   - the BAO composition test (direct χ² == MI-composed χ² to 1e-15 relative).
4. **Switch**: `pip uninstall pybird-lss && pip install --no-deps -e <pybird-dev>@mi-hooks`;
   add `sympy` to the environment recipe; delete the `sys.path` shim from `run_fisher_pb.py`;
   pin the dev commit in `notebooks/README.md`; move our `pybird/` to `archive/`.
5. **Then build the MI bispectrum model**: `MIModel` gains a `with_bk` path (the same
   `build_cosmo_dicts`, plus the Bk block appended in `model_vector` so `gn_fisher` sees it),
   `setups.py` gains a `mock_full_pb` setup, and the notebook flips `with_bk_tree_level: False`
   with a per-sky `bk_loop_matrix_path` when the one-loop matrices arrive.

## Note on the EFT basis

The bispectrum requires `eft_basis: 'eth'` (2211.17130 App. D.4); pybird refuses any other
basis with `bBk` in the output. This is **not** a constraint on the power spectrum: `setBias`
maps eth onto the usual parameters (b1=Bb1, b2=Bb2, b3=Bb3+15·Bb8, b4=Bb5, cct=−Bc1,
cr1=f·Bc2−f²/2·Bc4, cr2=−f²/2·Bc3, ce0=Be1, ce1=Be2+ce2/2) and the loop contraction is the
same 35-term layout the 80-knot emulator produces. So **the emulator stays on for P(k) in the
P+B analysis**, and the P(k) side of the model is bit-for-bit the v3 model with renamed
parameters. When the MI HMC pipeline gains a bispectrum setup it should use eth throughout,
including for the P-only runs, so that the two are directly comparable.
