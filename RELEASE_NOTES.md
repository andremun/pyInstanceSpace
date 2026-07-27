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
  mirroring MATLAB's `exploreIS.m`, via `build_explore_adapter.py`'s conversion of a
  `build()`-trained model (fitted scikit-learn `SVC` objects) into the flattened form
  MATLAB-exported models already use.

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

### Bug fixes

- Graph labels, exported CSV headers, and filenames no longer leak the metadata.csv
  `feature_`/`algo_` column-naming prefix (e.g. `algo_CART` is now `CART`), and the
  `z_1`/`z_2` axis labels on generated plots now actually render as subscripts instead
  of literal text `z_{1}`/`z_{2}` (#222).
- `PythiaStage` no longer mutates the caller's `y_raw` array in place while generating
  its summary table; it now copies before mutating, matching the pattern already used
  for the same array's other derived copies (#229).
- `SiftedStage` no longer returns a stale, un-narrowed `idx` (computed once from the
  pre-selection feature count) alongside the correctly-narrowed `selvars`; `idx` now
  tracks the actual selected feature indices. Previously this caused
  `Model.save_to_csv`/`save_instance_space_for_web` to crash or misalign columns
  whenever SIFTED reduced the feature set.

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
