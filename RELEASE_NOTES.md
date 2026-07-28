# Release Notes

Mirrors MATLAB `InstanceSpace`'s release-notes convention: entries are grouped into
*New functionality*, *Better engineering*, *Bug fixes*, and *Licence*. Every PR that
changes behaviour gets an entry here before merge.

## 0.2.1 (baseline)

Seeded as the starting point for this convention — describes the current state of the
toolkit rather than a single change.

### New functionality

- Stage architecture (`preprocessing → prelim → sifted → pilot → pythia → cloister →
  trace`) with an `InstanceSpace` class exposing `build()`, `explore()`, and
  `explore_iter()`.
- `explore()`/`explore_iter()` apply a previously trained model to unseen instances,
  mirroring MATLAB's `exploreIS.m`, by calling the `build()`-trained scikit-learn
  objects (e.g. `SVC.predict`/`predict_proba`) directly. (This originally went through
  `build_explore_adapter.py`, converting the trained model into the flattened form
  MATLAB-exported models use; that module has since been deleted — see the *Better
  engineering* entry below.)

### Better engineering

- Frozen dataclasses for options and metadata give real immutability MATLAB structs
  don't have.
- A self-validating DAG scheduler (`stage_builder.py`/`stage_runner.py`) infers stage
  execution order from each stage's declared inputs/outputs, with its own cycle and
  ambiguity detection, rather than a hand-maintained dependency map.
- `tests/matlab_reference/` provides a cross-implementation golden-reference harness:
  MATLAB-trained artifacts checked in, validated against Python's output stage by stage
  under documented tolerance thresholds.

### Bug fixes

(None catalogued retroactively — the list starts from here forward.)

### Licence

PolyForm Noncommercial 1.0.0, matching the MATLAB `InstanceSpace` toolkit.

---

## Unreleased

### New functionality

- `explore()` now logs a warning when more than 5% of test instances have a feature
  outside the training PRELIM bounds and get clipped to them, matching MATLAB's
  equivalent out-of-distribution check (#250).
- Added `InstanceSpace.plot_sources()`/`plot_portfolio()`/`plot_good()`/
  `plot_footprint()` convenience methods (`instancespace/plotting.py`), thin
  matplotlib wrappers mirroring MATLAB's `InstanceSpace.plot()` (#255).

### Bug fixes

- Graph labels, exported CSV headers, and filenames no longer leak the metadata.csv
  `feature_`/`algo_` column-naming prefix (e.g. `algo_CART` is now `CART`), and the
  `z_1`/`z_2` axis labels on generated plots now actually render as subscripts instead
  of literal text `z_{1}`/`z_{2}` (#222).
- `PythiaStage` no longer mutates the caller's `y_raw` array in place while generating
  its summary table; it now copies before mutating, matching the pattern already used
  for the same array's other derived copies (#229).
- `SiftedStage` no longer returns a stale, un-narrowed `idx` (computed once from the
  pre-selection feature count) alongside the correctly-narrowed `selvars`. Previously
  this caused `Model.save_to_csv`/`save_instance_space_for_web` to crash or misalign
  columns whenever SIFTED reduced the feature set; the redundant `idx` field has since
  been removed from `SiftedOutput`/`SiftedOut` entirely (`FeatSel.idx` is unaffected and
  now reads `selvars` directly).
- `build_explore_adapter._svc_to_artifact` no longer raises `NotImplementedError` for
  polynomial-kernel PYTHIA SVMs — `explore()` after a `build()` trained with
  `opts.pythia.is_poly_krnl = True` now works. The support vectors are pre-scaled by the
  trained `gamma` so `_explore_pythia`'s existing polynomial-kernel formula (which
  assumes `gamma=1`, matching its hardcoded `coef0=1`) reproduces the actual trained
  decision function. (`build_explore_adapter.py` itself has since been deleted — see the
  *Better engineering* entry below; `explore()` now calls `SVC.predict_proba` directly
  for every kernel type, so this specific formula no longer exists either.)

### Better engineering

- Dependency security bumps: `pillow` (12.2.0 → 12.3.0), `tornado` (6.5.5 → 6.5.7),
  `click` (8.1.7 → 8.4.2), `jupyter-core` (5.7.2 → 5.9.1) — resolves all known CVEs in
  the locked dependency set at the time of writing.
- Added `.github/dependabot.yml` (weekly, pip ecosystem) so future dependency drift is
  caught automatically.
- Added a non-blocking `pip-audit` CI step to `validation-tests.yml`.
- Added `CITATION.cff` and resolved the README's citation placeholder.
- README: Contact section now points at this repository's own issue tracker rather than
  the MATLAB repository's; PYTHIA options section rewritten to describe the actual
  scikit-learn-backed implementation instead of MATLAB's Statistics and Machine Learning
  Toolbox/LIBSVM; added *Repository layout* and *Working with the code* sections.
- `instance_space_from_files`' options listing now recurses into nested option
  dataclasses (`instancespace/utils/print_options.py`), printing one line per leaf
  field instead of one line per top-level group with a raw nested-dataclass repr (#252).
- Documented and added a regression test for the already-decided permissive
  feature-order behaviour in `explore()`: test metadata's feature columns are matched
  by name, not position (#253).
- Added `SECURITY.md` and `CONTRIBUTING.md` (#258).
- `explore()` now calls the `build()`-trained scikit-learn objects (`SVC.predict`/
  `predict_proba`) directly instead of converting them into a flattened, MATLAB-artifact
  shaped representation first. `instance_space.py`'s model-shape detection branch (which
  handled only that flattened shape as a second, never-actually-reachable case) has been
  removed along with it (S1, #282). `build_explore_adapter.py` and its test file, which
  existed solely to produce that flattened shape, have been deleted in full — nothing
  called them once S1 landed (S3, #284).
- Removed a dead, commented-out CI lint/format-check step from `validation-tests.yml`
  rather than leaving it neither enabled nor deleted (#259).
