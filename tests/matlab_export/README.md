# MATLAB reference-data export script

Design and implementation for T5 / roadmap §8.3's proposal 1: "a MATLAB export script
that dumps training artifacts + explore outputs in exactly the CSV interchange format
`tests/matlab_reference/` already documents." This directory holds that script.

**This is a MATLAB script meant to run inside a checkout of
[`andremun/InstanceSpace`](https://github.com/andremun/InstanceSpace), not inside this
repo.** It lives here (not there) because this repo is the consumer of the data it
produces, and because — per direct instruction — nothing gets pushed to the MATLAB repo
as part of this work. Copy `pyis_export_reference_data.m` into the MATLAB checkout's root
(alongside `buildIS.m`/`InstanceSpace.m`) to run it.

**Not executed against real MATLAB.** No MATLAB installation is available in the
environment this was written in. The script was written from direct inspection of
`andremun/InstanceSpace`'s actual source (`InstanceSpace.m`, `core/*.m`,
`output/scriptcsv.m`/`scriptfcn.m`, `test_integration.m`, commit `a0197ee3`) rather than
guessed, but it has not been run. Review it — and run it once on a small case before
trusting its output — before using it to regenerate any committed fixture.

## The problem this solves

`tests/matlab_reference/` and `tests/test_data/<stage>/` were built up over many sessions
by ad hoc, undocumented MATLAB runs — inconsistent file naming (`model-data-x.csv` next to
`input_X.csv` next to `Xraw_split.txt`), no recorded MATLAB commit, no repeatable process.
As MATLAB keeps changing (this repo's own roadmap has logged a dozen-plus MATLAB-side
behavior differences), that fixture set can only drift further out of sync, silently. This
script turns "regenerate the reference data" from bespoke copy-paste into one function
call, with its provenance recorded alongside the output.

## Data required for coverage

Two kinds of fixture already exist in this repo, and the script produces both:

1. **Full-pipeline / `explore()` fixtures** (`tests/matlab_reference/` today) — training
   artifacts plus stage-by-stage outputs on a held-out test set. Consumed by
   `tests/test_explore_<stage>.py`.
2. **Isolated per-stage fixtures** (`tests/test_data/<stage>/` today) — one stage's raw
   inputs and outputs, used to check that stage's Python port bit-for-bit (within
   tolerance) against MATLAB. Consumed by `tests/test_build_<stage>.py`.

Both are really the same underlying data — every stage's real input/output pair, from a
real pipeline run — sliced differently. The design below produces the raw material for
both from a single run, rather than two independent processes.

### Coverage matrix

Per stage, cross-referenced against the MATLAB output-struct fields actually returned
(verified directly against `core/*.m`, not assumed) and the existing Python fixture
consumers:

| Stage | MATLAB inputs | MATLAB `out.*` fields (verified in `core/<STAGE>.m`) | Existing Python consumer |
|---|---|---|---|
| PRELIM | `X, Y, opts` | `Ybest, Ybin, P, numGoodAlgos, beta, medval, iqrange, hibound, lobound, minX, minY, lambdaX, lambdaY, muX, muY, sigmaX, sigmaY` (+ returned `X, Y`) | `test_build_prelim.py`, `test_data/prelim/` |
| SIFTED | `X, Y, Ybin, featlabels, opts` | `selvars, rho, p, clust, eva, Ksuggested` (+ returned `X`) | `test_build_sifted.py`, `test_data/sifted/` |
| PILOT | `X, Y, featlabels, opts` | `A, B, C, Z, X0, alpha, eoptim, error, perf, R2, summary` | `test_build_pilot.py`, `test_data/pilot/` |
| CLOISTER | `X, A, opts` | `Zedge, Zecorr` (see **Known gap** below — `rho`/`pval`/`xEdge`/`remove` are internal, not returned) | `test_build_cloister.py`, `test_data/cloister/` |
| PYTHIA | `Z, Y, Ybin, Ybest, algolabels, opts` | `Yhat, Ysub, Pr0hat, Pr0sub, W, cp, classifiers, accuracy, precision, recall, cvcmat, selection0, selection1, summary, mu, sigma, param1, param2, param2Label, classifierType` | `test_build_pythia.py`, `test_build_pilot_pythia.py`, `test_data/pythia/` |
| TRACE | `Z, Ybin, Yhat, P, beta, algolabels, opts` | `good, best, hard, space, summary` (`good`/`best` are per-algorithm cell arrays of polygon structs; `hard` is a single one, the all-instances footprint) | `test_build_trace.py`, `test_data/trace_csvs/` |

Full-pipeline coverage (`InstanceSpace.build()` + `.explore()`) reuses the exact same
per-stage fields, just captured from `obj.model.<stage>` / `obj.getResults(1).<stage>`
instead of a stage function's direct return value — see **Two ways to get the data**
below for why that distinction matters less than it sounds.

**A verification note, since this table is asserted as fact rather than a guess:** a
naive `grep -oE "out\.[A-Za-z_]+" core/<STAGE>.m` over-reports for two of these six files.
SIFTED.m's `costfcn` (its GA fitness function) calls `PILOT(...)` internally and names
*that* result `out` too (`Z = out.Z;`) — a same-named local variable in a nested function,
not a field SIFTED itself ever returns; an earlier draft of this table wrongly listed `Z`
as a SIFTED output field because of this. PYTHIA.m's `buildSummary`/`emptyPYTHIAout` helper
functions also parameterise on a variable named `out`, and one stray match came from a
*comment* (`% out.good/out.best for algorithms beyond its trained count.`), not code — an
earlier draft wrongly listed `best`/`good` as PYTHIA fields; they belong to TRACE, not
PYTHIA. The table above was re-derived by grepping each file's *own* top-level function
body only (up to the line where the next `function` keyword starts), and separately
confirming any field set via a helper (e.g. PYTHIA's `computeSelection`) is actually
reassigned back at the call site (`out = computeSelection(out, ...)`), not just read.

### Known gap: CLOISTER's and SIFTED's non-returned internals

`tests/test_data/cloister/output/` today has `rho.csv`, `index.csv`, `remove.csv`,
`x_edge.csv` — none of which `CLOISTER.m` actually returns (only `Zedge`/`Zecorr` are
`out.*` fields; `rho`, `pval`, `xEdge`, `remove` are local variables inside the function).
Whoever produced those fixtures originally must have either patched a local MATLAB copy to
expose them, or copy-pasted CLOISTER's own internal computation out by hand. This script
does **not** attempt to shadow-reimplement CLOISTER's internal correlation/masking logic to
recover them (duplicating pipeline logic in a test-data exporter is exactly the kind of
thing that silently drifts out of sync with the real implementation — the same problem
this whole exercise exists to fix). If that level of granularity is still needed going
forward, the honest fix is a tiny, additive MATLAB-side change (extra `out.rho`/`out.pval`
fields, non-breaking for every existing caller) proposed as its own MATLAB-repo issue —
out of scope here per "do not push it to MATLAB." Recorded as a follow-up, not silently
dropped.

SIFTED's `out.eva` is a MATLAB `evalclusters` **object**, not a plain struct — not directly
CSV-able. The script exports its two numeric fields that Python's `evaluate_cluster()`
actually checks (`InspectedK`, `CriterionValues`), not the object itself.

Similarly, PYTHIA's `out.classifiers` (one fitted classifier object per algorithm —
`fitcsvm`/`fitcknn`/etc.) is listed in the coverage matrix above for completeness, but the
script does **not** export it either. Python no longer replicates MATLAB's classifier math
directly (S1 rewrote `explore()` to call scikit-learn's own `predict`/`predict_proba`
rather than reimplement SVM scoring by hand), so the raw support-vector/kernel internals
the older `svm_<algo>.csv` fixtures captured have no current consumer — what Python's tests
actually check is the reported hyperparameters (`param1`/`param2`) and the CV
accuracy/precision/recall/predictions, all of which the script does export (`summary.csv`,
`hyperparameters.csv`, `ysub.csv`/`yhat.csv`/`pr0sub.csv`/`pr0hat.csv`).

## Available transfer options, and why CSV wins by default

| Format | Used today | Verdict for new fixtures |
|---|---|---|
| **CSV** | Dominant already (`tests/matlab_reference/`, most of `tests/test_data/`) | **Default choice.** Every MATLAB numeric array/table maps to one via `writetable`/`array2table` (already the pattern in `output/scriptcsv.m`); every Python side reads it with `pandas.read_csv`/`np.genfromtxt`, no extra dependency. Human-diffable in a PR — a fixture regeneration's actual numeric delta is visible in the diff, not hidden in a binary blob. |
| **.mat** | A handful of fixtures (`tests/test_data/pilot/*.mat`) | Keep using it **only** where a fixture already is one and changing it isn't part of this task (e.g. `test_numerical.mat`/`matlab_results_num.mat`) — `scipy.io.loadmat` handles it fine — but don't default to it for anything new. It hides deeply-nested MATLAB struct/object quirks (`evalclusters`, `polyshape`) that need decomposing to be Python-usable anyway, and a `.mat` diff in a PR review is opaque. |
| **JSON** | `options.json` (opts only) | Right tool for scalar/nested option structs specifically (already how both repos exchange `opts`) — not proposed for numeric arrays. |
| **`.txt` "split" files** | `tests/test_data/prelim/*/[...]_split.txt` | Legacy leftover from whatever manual process produced them; not proposed for new output — CSV supersedes it with no loss. |

Net: **CSV is the primary interchange format**, one file per logical array/table field,
column/row-labelled exactly like `output/scriptcsv.m` already does it (reusing that file's
`writeArray2CSV`/`writeCell2CSV` pattern rather than inventing a second convention).
Per-algorithm cell-array fields (SVM classifiers, TRACE footprints) get one file per
algorithm, matching the existing `svm_<algo>.csv`/`good_<algo>.csv` convention. Polygon
objects (`polyshape`/`alphaShape`) export as an `(x, y)` vertex list with a blank/NaN row
delimiting separate regions — the convention `tests/matlab_reference/README.md` already
documents, reusing `output/scriptcsv.m`'s own `footprintBoundary` extraction logic for the
`alphaShape` (TRACE3, MATLAB's current default) case.

## Two ways to get the data — and why the script uses both

The task named two options: run the pipeline by stages, or extract data from the class
object. These aren't actually alternatives — they compose:

- **`InstanceSpace`'s own staged `build()`** (`obj.build('stages', {'pythia'})`) is what
  makes re-running just the downstream stages affordable. PYTHIA alone needs multiple
  fixture variants — Sobol vs. Bayes tuning, Gaussian vs. polynomial kernel, `svm` vs. the
  other five registered classifiers (`tests/test_data/pythia/output/BO_gaussian`,
  `BO_poly`, `GS_gaussian`, `GS_poly` all already exist as separate cases) — and PRELIM
  through PILOT is the expensive, option-invariant part. The script builds
  prelim→sifted→pilot **once**, then loops over a small set of PYTHIA/TRACE option
  variants, re-running only `{'pythia','trace'}` per variant via `obj.opts.pythia = ...;
  obj = obj.build('stages', {'pythia','trace'});` — mirroring the variant-case pattern
  `test_integration.m` already uses for its own regression suite, just for fixture
  generation instead of pass/fail checking.
- **Extracting from the class object** is how the per-stage fixtures actually get written:
  after each `build()` call, `obj.model.<stage>` holds exactly that stage's real output
  struct — the same shape `PRELIM(...)`/`SIFTED(...)`/etc. would return if called directly.
  The export helper functions below operate on that struct's fields, not on how it was
  produced, so the same `exportXxxArtifacts(stageOutput, destDir)` function works whether
  `stageOutput` came from `obj.model.prelim` (staged class build) or a direct
  `[~, ~, stageOutput] = PRELIM(X, Y, opts)` call (useful later for hand-crafted
  edge-case fixtures — degenerate/all-NaN input, single-instance data, etc. — that a real
  dataset run won't exercise). The script itself only uses the staged-class path; the
  direct-call path is documented as the pattern to follow for anyone adding a
  hand-crafted-edge-case fixture later, not implemented as its own dataset here.

## Build path and explore path — one layout, same names on both sides

An earlier draft of this script only called `.explore()` once, for a separate
default-options-only pass, leaving the three `svm`/Bayes/kernel PYTHIA/TRACE variants
build-only. Fixed: every variant in the loop above now gets **both**, and both sides use
the exact same `<top>/<stage>/<variant>/` shape — `build_data/` and `explore_data/` are
structurally identical, not two conventions that happen to sit side by side:

- **Build path** (training): `obj.build('stages', {'pythia','trace'})`, exported to
  `build_data/pythia/<variant>/` and `build_data/trace/<variant>/` —
  `PythiaOutput`/`TraceOutput`'s training-mode shape (hyperparameters included).
- **Explore path** (test-set inference on the model *that variant* just trained):
  `obj.explore(datasetRoot)`, exported to `explore_data/pythia/<variant>/` and
  `explore_data/trace/<variant>/` — MATLAB's own distinct eval-mode code path
  (`PYTHIAevalMode` in `core/PYTHIA.m`; TRACE's `trainedTrace`-argument branch), which has
  a different output shape (no hyperparameter columns in PYTHIA's eval-mode summary,
  since eval mode doesn't retrain) exported under its own name (`eval_summary.csv`) rather
  than conflated with the training-mode `summary.csv` sitting next to it in
  `build_data/pythia/<variant>/`.

Stages whose output doesn't depend on `opts.pythia`/`opts.trace` (prelim, sifted, pilot,
cloister, build-side only — see **Known gap** above for why they have no explore-path
counterpart yet) still get a `<variant>/` level, always named `default`, so every stage
sits at the same path depth — a consumer walking `build_data/` never has to special-case
"this stage has no variant folder."

The `default` variant additionally gets the flat, backward-compatible
`step1_after_prelim.csv`-style layout `tests/matlab_reference/explore_outputs/` already
documents (`exportLegacyExploreLayout`), written to its own `legacy_explore_outputs/`
root — kept structurally and *nominally* separate from `explore_data/` so the two can't be
confused with each other — so this script's `default`-variant output is a byte-for-byte
drop-in replacement for the existing fixture set, not a same-data, different-name
reshuffle of it. `step1`–`step3` (PRELIM/SIFTED/PILOT's test-set transform) don't depend
on `opts.pythia`/`opts.trace`, so they're only written once (at the `default` variant, not
duplicated per variant) rather than four times over with identical content.

## Provenance (T5's actual ask)

Every run writes `provenance.json` alongside the exported fixtures:

```json
{
  "matlab_commit": "<git rev-parse HEAD, from the toolkit checkout>",
  "matlab_repo": "https://github.com/andremun/InstanceSpace",
  "toolkit_version": "<Contents.m's version line, if present>",
  "dataset": "test/data/metadata.csv + metadata_test.csv (Munoz et al. 2018 study)",
  "generated_at": "<ISO-8601 UTC timestamp>",
  "generator_script": "pyis_export_reference_data.m",
  "matlab_version": "<version() string>"
}
```

This is the "we know exactly which MATLAB version this validates against" record §8.3
proposed — commit it alongside any regenerated fixture set, and diff it on review the same
way the numeric CSVs get diffed.

## Directory layout this script writes

`build_data/` and `explore_data/` are the same shape, `<stage>/<variant>/`, on purpose —
no stage-only folder on one side and a flat per-variant folder on the other. The one
exception, `legacy_explore_outputs/`, is deliberately named and structured differently
*because* it isn't part of this layout: it's a byte-for-byte reproduction of an existing,
already-committed fixture convention (`tests/matlab_reference/explore_outputs/`), kept
around only for backward compatibility until T10e's migration lands.

```
<outputRoot>/
├── provenance.json
├── build_data/
│   ├── prelim/default/                                (option-invariant, written once)
│   ├── sifted/default/
│   ├── pilot/default/
│   ├── cloister/default/
│   ├── pythia/<variant>/                              (one per variant, build-time)
│   └── trace/<variant>/                                (one per variant, build-time)
├── explore_data/
│   ├── pythia/<variant>/
│   │   └── predictions.csv, probabilities.csv, eval_summary.csv
│   └── trace/<variant>/
│       └── eval_summary.csv, membership.csv
├── legacy_explore_outputs/
│   └── step1_after_prelim.csv ... step5_trace_membership.csv   (default variant only,
│                                                        same flat names/layout as the
│                                                        existing tests/matlab_reference/
│                                                        explore_outputs/ fixture)
└── input/
    ├── metadata.csv           (copied from the toolkit's test/data/, unchanged)
    └── metadata_test.csv
```

`<variant>` is one of `default` (MATLAB's own untouched options), `sobol_svm`,
`bayes_svm_gaussian`, `bayes_svm_poly` — matching the option cases already named in
`test_integration.m` / `tests/test_data/pythia/output/`'s existing `BO_gaussian`/`BO_poly`
naming, so a future migration of the ad hoc `tests/test_data/` fixtures onto this script's
output can reuse the same directory names rather than inventing new ones. Every variant
gets both `build_data/{pythia,trace}/<variant>/` (build path) and
`explore_data/{pythia,trace}/<variant>/` (explore path) — see **Build path and explore
path** above. `prelim`/`sifted`/`pilot`/`cloister` only ever have a `default/` variant
today (their build output doesn't depend on `opts.pythia`/`opts.trace`, and they have no
explore-path export yet at all — see **Known gap**) — the `default/` level is still
written for them so `build_data/<stage>/<variant>/` is a reliable path shape regardless of
stage, not a shortcut some stages skip.

Migrating the *existing* committed fixtures (`tests/matlab_reference/`,
`tests/test_data/<stage>/`) onto this layout is a separate decision for a future session
(tracked as GitHub issue T10e, blocked on this script actually being run once), not done
here — this script only defines what a *regeneration* would look like.

## Running it

```matlab
% From inside a MATLAB session, cd'd anywhere:
cd /path/to/andremun/InstanceSpace   % the MATLAB toolkit checkout
copyfile('/path/to/pyInstanceSpace/tests/matlab_export/pyis_export_reference_data.m', '.');
pyis_export_reference_data('.', '/path/to/output/dir');
```

Then inspect `/path/to/output/dir`, diff it against the existing `tests/matlab_reference/`
and `tests/test_data/` fixtures, and copy over whichever files have genuinely changed
(plus the new `provenance.json`) — don't blind-overwrite, since a real numeric diff here
is exactly the "did the reference data go stale" signal §8.3 wants surfaced, not hidden.

## What's deliberately not in scope here

- Migrating the existing `tests/test_data/`/`tests/matlab_reference/` fixtures onto this
  script's output layout — a real behavior-relevant decision (which stale ad hoc fixtures
  get replaced) that belongs in its own reviewed pass, not bundled into writing the
  exporter.
- §8.3's proposal 3 (CI-triggered export + automated staleness diff) — needs MATLAB CI to
  exist first, which the roadmap already notes it doesn't.
- Any change to MATLAB source (`core/*.m`, `InstanceSpace.m`) — including the small
  CLOISTER accessor addition that would close the "Known gap" above.
