# Test data audit and remediation proposal

**Status:** Audit complete. §7's decision is made: single unified layout (Option A).
The hardened exporter and independent verifier completed an R2024a diagnostic run on
2026-08-17. A verified R2025a+ run is still required before fixtures move. Steps 1
(delete confirmed-dead data), 2 (resolve the `prelim/run/output/` partial-orphan), 3
(fix the `serialisers/actual_output/` scratch leak), and 4 (relocate `test_data/demo/`)
are done. Step 5 (the unified-layout migration itself) remains gated on #278 and is
tracked as GitHub issue T10e (#310); full status in §9. **§7.1 (added on request,
2026-08-03) extends the target layout with a cross-stage/shared-input rule**,
documented now so T10e's migration only has to move fixtures once, not twice.
**Scope:** every file under `tests/` in this repository, not only
`tests/matlab_reference/`. Companion to `docs/pyIS_docs_quality_roadmap.md` §8.3 and
`tests/matlab_export/README.md`.

## 1. Why this audit

A question about the MATLAB export script's directory layout exposed a larger problem.
Test data lives in several places under `tests/`, in several incompatible layouts, and
some of it has no reader at all. Nobody had checked, file by file, which fixture feeds
which test, or which files are dead weight. This document is that check.

## 2. Method

For every `tests/*.py` file, this audit extracted each literal file path the code opens.
It then compared that list against every file that actually exists under `tests/`. A
file counts as used when a test opens it directly, or when a test opens its parent
directory as a base path. Every path claim below comes from a direct `grep` across the
whole repository, not from a single test file in isolation. Where a first pass looked
like an orphan, a second, repository-wide search confirmed it before this document
calls it dead.

## 3. Findings: dead data

Three tiers, by how confident the evidence is.

### 3.1 Exact duplicates (byte-identical to data a test already reads)

| Path | Size | Duplicate of | Confirmed reader of the duplicate |
|---|---|---|---|
| `tests/fileidx/` | 22 files | `tests/test_data/prelim/fileidx/` | `test_build_prelim_filter.py` |
| `tests/fractional/` | 21 files | `tests/test_data/prelim/fractional/` | `test_build_prelim.py`, `test_build_prelim_filter.py` |
| `tests/split/` | 21 files | `tests/test_data/prelim/split/` | `test_build_prelim_filter.py` |

`diff -rq` between each pair returns no differences. No test anywhere opens the
top-level copy. These three directories are pure leftover copies, most likely left
behind when the data moved under `test_data/` during an earlier reorganization.

### 3.2 Orphaned and stale (no reader, and the content itself is out of date)

`tests/Prelim_out/` has no reader anywhere in the repository. Its file
`model-data-x.csv` differs in content from the file of the same name at
`tests/test_data/prelim/output/model-data-x.csv`, which `test_build_prelim.py` does
read. `tests/Prelim_out/` is also missing most of the per-feature parameter files
(`model-prelim-hibound.csv`, `model-prelim-lambdaX.csv`, and eight more) that the
active fixture set has.

This is the riskiest kind of dead file. A reader could open `tests/Prelim_out/` by
hand, assume it holds current PRELIM output, and draw a wrong conclusion from stale
numbers. The likely origin: a snapshot taken before a PRELIM fix landed, never deleted
after `tests/test_data/prelim/output/` replaced it.

### 3.3 Orphaned, no active duplicate found

| Path | Size | Note |
|---|---|---|
| `tests/process_data/` | 8 files | No reader anywhere. |
| `tests/test_integration/` | 6 files | No reader anywhere. Not the same thing as `tests/test_data/preprocessing/`, which `test_build_integration.py` actually reads. |
| `tests/test_data/cloister/pythia/` | 36 files | Two full input+output cases (`fitlibsvm_gaussian`, `fitlibsvm_polynomial`). No reader in any `.py` file. The only repository-wide hit for `fitlibsvm` is a one-line, unrelated mention in the roadmap's prose. |
| `tests/test_data/prelim/input/filter/`, `tests/test_data/prelim/output/filter/` | 11 files | Superseded by `tests/test_data/filter/`, which `test_build_filter.py` reads instead. |
| `tests/test_data/prelim/input/model-data-x.csv` | 1 file | A same-purpose file, `model-data-x-input.csv`, is the one `test_build_prelim.py` actually reads. |
| `tests/test_data/prelim/input/options.json` | 1 file | No reader found. |
| `tests/test_data/preprocessing/X.mat`, `Y.mat` | 2 files | The CSV siblings `X.csv`/`Y.csv` are the ones in use. |

### 3.4 Partially orphaned — a possible test-coverage gap, not just dead data

`tests/test_data/prelim/run/output/output_Xraw.csv`, `output_Yraw.csv`, and
`output_instlabels.csv` sit next to files that `test_build_prepro_n_prelim.py` reads
(`output_X.csv`, `output_Y.csv`, `output_P.csv`, and others), but that test never opens
these three. Either the fixture generator wrote three extra files nobody asked for, or
the test is missing an assertion it was meant to have. This needs a look at
`test_build_prepro_n_prelim.py` itself before deciding which.

## 4. Findings: data that is not dead, but is not what its location implies

### 4.1 Two incompatible layouts for MATLAB-comparison data

`tests/matlab_reference/` groups by pipeline phase first: `input/`,
`training_artifacts/<stage>/`, `explore_outputs/step<N>_<name>.csv`. `tests/test_data/
<stage>/` groups by stage first: `<stage>/input/`, `<stage>/output/`, with no separate
concept of build time versus explore time at all.

The practical effect: build-time (training) fixtures live under `test_data/<stage>/`.
Explore-time (test-set inference) fixtures live only under `matlab_reference/`, and
only for one option configuration (MATLAB's plain defaults). PRELIM, SIFTED, PILOT,
CLOISTER, and PYTHIA each have build-time coverage under `test_data/` but no
stage-specific explore-time fixture of their own. A reader who wants to know "does this
directory hold build data or explore data" has to already know which of the two
top-level conventions they are standing in. Nothing in the data says so.

### 4.2 Data that legitimately is not MATLAB-derived

Three groups exist for reasons that have nothing to do with numerical parity against
MATLAB. Their location under `tests/test_data/` next to the MATLAB-comparison fixtures
obscures that difference.

- **`tests/test_data/demo/`** — a multi-dataset collection (BBO, JSS, KP, MFP, and more)
  plus 20 `options_*.json` files. `integration_demo.py` and `liveDemoIS.ipynb` use it as
  worked examples. No pytest file reads it. It is not test data at all in the sense the
  rest of this document uses that term.
- **`tests/test_data/load_file/`** — hand-built JSON and CSV, including deliberately
  broken files (`illegal.json`, `options_invalid.json`). This checks Python's own
  options-loading and validation code. MATLAB has no equivalent concept to compare
  against, so this data is correctly Python-only, and should stay that way.
- **`tests/test_data/serialisers/actual_output/`** — not a fixture. It is a test's own
  write target, regenerated on every run of `test_serialisers.py`. A `.gitignore` file
  sits inside each of its subdirectories, but git still tracks `output.zip` inside it —
  the same file that kept showing up as "modified" in this session's `git status`
  output after running the test suite. `expected_output/` (the golden comparison
  target) is the fixture that actually belongs in version control here.

## 5. Category framework

Every file under `tests/` that holds data falls into one of five categories. Naming the
category is the fix for the "hard to know what is consumed by what" problem — once each
directory declares its category, the reader does not have to guess.

| Category | Origin | Purpose | Example |
|---|---|---|---|
| MATLAB build-path parity | MATLAB, training run | Check one stage's Python output against MATLAB, same inputs | `test_data/pythia/output/BO_gaussian/gaussian.csv` |
| MATLAB explore-path parity | MATLAB, `explore()` run | Check test-set inference against MATLAB | `matlab_reference/explore_outputs/step4_pythia_predictions.csv` |
| Python-only synthetic | Hand-built or generated in Python | Exercise a code path no real MATLAB dataset reaches (degenerate input, invalid options, edge cases) | `test_data/load_file/illegal.json` |
| Example / demo | Real-world, not a golden-value comparison | Show a working usage pattern, not verify numeric correctness | `test_data/demo/metadata/metadata_BBO.csv` |
| Test-run scratch output | Written by the test itself, not read by anything | Debugging artifact only, should not need version control | `test_data/serialisers/actual_output/` |

Answering the question in your message directly: MATLAB-parity data (the first two
rows) should keep coming from MATLAB, and only from MATLAB — hand-editing a
MATLAB-comparison fixture to make a test pass would defeat the point of the comparison.
Python-only synthetic data (row three) should keep coming from Python, since it tests
Python-side logic that has no MATLAB counterpart to derive from. The problem this audit
found is not that both kinds of origin exist. It is that nothing in the current
directory layout tells you which kind you are looking at.

## 6. Remediation proposal

Ordered by risk, cheapest and safest first. Each step names its compatibility tag per
this repository's convention.

### Step 1 — Delete confirmed-dead data
**[Additive]** (removes files nothing reads; no behavior change for any passing test).
**Done.**

Before deleting, this step re-ran every check from §3 across the whole repository
(not only `tests/*.py` — also `.ipynb`, `.md`, `.toml`, `.yml`), to catch a reader this
audit's first pass might have missed. All candidates still came back with zero real
matches. Deleted via `git rm -r`, one commit, with the §3 reasoning in the commit
message: `tests/fileidx/`, `tests/fractional/`, `tests/split/`, `tests/Prelim_out/`,
`tests/process_data/`, `tests/test_integration/`, `tests/test_data/cloister/pythia/`,
`tests/test_data/prelim/input/filter/`, `tests/test_data/prelim/output/filter/`,
`tests/test_data/prelim/input/model-data-x.csv`, `tests/test_data/prelim/input/
options.json`, `tests/test_data/preprocessing/X.mat`, `tests/test_data/preprocessing/
Y.mat`. 135 files removed. Full suite run before and after — pass count unchanged, per
CLAUDE.md's rule against treating "should be safe" as a substitute for a checked
result.

While resolving §3.4 (see Step 2 below), reading `test_build_prepro_n_prelim.py`
directly showed the partial-orphan list was an undercount: the test checks only
`output_P/X/Y/Ybin/Ybest.csv` and never opens `output_Xraw.csv`, `output_Yraw.csv`,
`output_beta.csv`, `output_instlabels.csv`, or `output_numGoodAlgos.csv` — five files
unread, not three. None of the five were deleted in this step, since §3.4's own
finding still holds: this looks like an incomplete test, not dead data, and deleting
would foreclose the "add the missing assertions" fix before anyone decided against it.

### Step 2 — Resolve the partial-orphan question (§3.4)
**[Additive] if the fix is a new assertion, [Behavior-changing] if the fix changes what
the fixture generator writes.**

**Done (T10b, #307).** Resolved as "missing assertions," not "dead fixture" — read
`test_build_prepro_n_prelim.py` directly and confirmed `PrelimOutput` has real
`x_raw`/`y_raw`/`beta`/`num_good_algos`/`instlabels` fields, all five verified to match
their corresponding MATLAB CSV exactly before writing any assertion. 5 more `assert`
calls added to `test_integrated_prepro_n_prelim`, mirroring the existing pattern.

### Step 3 — Fix the serialisers scratch-output leak (§4.1, third bullet)
**[Additive]** — tooling only.

**Done (T10c, #308).** Root cause confirmed: `output.zip` was committed before
`actual_output/.gitignore`'s `*.zip` pattern existed, so the pattern never actually
applied to it (`.gitignore` has no retroactive effect on already-tracked files).
`git rm --cached` untracked it; verified `git status` stays clean across a fresh
`test_serialisers.py` run, and confirmed `expected_output/` is the one under normal
version control as the real fixture.

### Step 4 — Rename or relocate the non-parity categories (§4.2)
**[Additive]** — path changes to non-MATLAB-comparison data only, with every reader
updated in the same commit.

**Done for `test_data/demo/` (T10d, #309).** Moved to `examples/data/` via `git mv`
(content unchanged, git history preserved); `integration_demo.py` and
`example_plugin.py` — the two real readers, not `liveDemoIS.ipynb` as this document's
first pass assumed; that notebook actually reads `tests/matlab_reference/input/`, not
`test_data/demo/`, confirmed by grep before relying on the earlier claim — both updated
in the same commit. Verified the move itself works (both scripts resolve and read the
new path successfully); running them further surfaced a genuine, pre-existing bug
unrelated to the move — `examples/data/options.json`'s `selvars.type` is `"Ftr&&Good"`
(double ampersand), present at the same value before the move too — filed separately as
#311, not fixed here since this step is relocation-only, no fixture content changes.
`tests/test_data/load_file/` left where it is, since "test data for loading tests" is an
accurate name for exactly what it is.

### Step 5 — Decide the target layout for MATLAB-parity data (open decision, see §7)
**[Behavior-changing] if it moves any fixture a currently-passing test reads** — needs
the full `tests/matlab_reference`-style verification pass before and after, same as any
other change to existing test data.

This is the big one, and it is a decision, not a mechanical step. §7 lays out the
options. Do not start moving `test_data/<stage>/` fixtures until one option is chosen.

## 7. Decision: one layout, not two

**Decided: Option A, single unified layout.** The exporter now produces and verifies
this layout, and `tools.fixture_provenance install` can atomically install only a
verified bundle. No historical fixture has moved: the available R2024a diagnostic
bundle is deliberately rejected by that installer.

Target shape: `tests/matlab_export/pyis_export_reference_data.m`'s own output —
`build_data/<stage>/<variant>/` and `explore_data/<stage>/<variant>/`, the same shape and
the same naming convention on both sides (revised from an earlier draft's
`training_artifacts/`/`explore_outputs/` split, which used two different names for the
build and explore roots and, worse, dropped the `<stage>/` level entirely on the explore
side — exactly the kind of asymmetry this decision exists to rule out) — applied to every
MATLAB-comparison fixture, including the ones now under `test_data/<stage>/`. Every stage
gains explore-path coverage it does not have today, not only PYTHIA and TRACE. One
convention, one place to look, for both build-path and explore-path data.

Cost, recorded so it is not a surprise when Step 5 starts: every
`test_build_*.py`/`test_explore_*.py` file's fixture paths need updating to point at
the new layout. This is real work across a large number of test files, not a single
mechanical rename, since the old and new layouts group the same data differently
(stage-first versus phase-first) rather than just renaming directories in place.

## 7.1 Addendum: cross-stage and shared-input data (added 2026-08-03)

§7's target shape fixes the build-vs-explore naming split, but says nothing about a
second, independent duplication pattern this section found by hashing every file
under `tests/test_data/` and `tests/matlab_reference/` and grouping by identical
content rather than by name. Two distinct patterns, two distinct fixes — folded into
the target layout now, before Step 5 (T10e) executes, so fixtures move once, not
twice.

### Pattern A — pipeline-chained data (a downstream stage's input is an upstream
stage's output)

PRELIM's output (`X`, `Y`, `Ybin`, `Ybest`, `beta`, `instlabels`, `num_good_algos`) is
byte-identical across as many as nine separate files, each held as its own private
copy: `prelim/fileidx|fractional|split/{before,after}/*_split.txt`,
`prelim/run/{input,output}/*.csv`, `sifted/input/input_dense_*.csv`,
`pythia/input/{y,ybin,ybest}.csv`, and `trace_csvs/{beta,dataP}.csv`. Confirmed each
copy has its own distinct reader — `test_build_prelim.py`/
`test_build_prepro_n_prelim.py` (PRELIM), `test_build_sifted.py` (SIFTED),
`test_build_pythia.py`/`test_build_pilot_pythia.py` (PYTHIA),
`test_build_trace.py` (TRACE) — so this isn't dead-data duplication in the §3 sense;
every copy is read by something. It's the same data reaching four independent test
files because each stage's fixtures were built by hand-copying the previous stage's
output rather than pointing at it.

**Fix, folded into the target layout:** under `build_data/<stage>/<variant>/`, a
downstream stage's tests read the upstream stage's own `build_data/<upstream-stage>/
<variant>/output/` directly — no separate copy under the downstream stage's own
`input/`. This needs no new top-level category; it only requires each
`test_build_*.py` file's fixture-loading code (or a shared `conftest.py` fixture, per
T3) to point at the producing stage's output path instead of a private copy. This is
squarely inside T10e's existing scope (it already touches every `test_build_*.py`
fixture path), not separate work.

### Pattern B — genuinely shared input data (no single producing stage)

The same `metadata.csv` is byte-identical across `test_data/load_file/`,
`test_data/preprocessing/`, and `matlab_reference/input/`, each read by an
independent test suite (`test_load_file.py`, `test_build_preprocessing.py`,
`test_build_prepro_n_prelim.py`, `test_instance_space_checkpoint.py`) exercising
different code paths against the *same* raw input. Unlike Pattern A, there's no
upstream stage whose output this is — it's the pipeline's own entry point, needed
identically by several unrelated test suites.

**Fix, folded into the target layout:** a new top-level `shared_inputs/<name>/`
directory, sibling to `build_data/` and `explore_data/`, holding raw input data with
more than one independent consumer and no single owning stage. `metadata.csv` moves
there; every reader points at the one copy. Naming is not locked in — `shared_inputs/`
is this document's proposal, favoured over "global" (collides with Python's own
`global` keyword) and "class" (already overloaded as "class label" in this codebase's
own ML vocabulary) — open to revision before T10e executes.

**Migration gate:** move files and update readers only after a clean R2025a+ export
passes manifest review and the Python verifier. This keeps unknown-provenance data out
of the new oracle tree.

## 8. What this audit did not do

The exporter has now run against the current MATLAB checkout under R2024a. Its 196-file
diagnostic bundle passed hash, path, shape, toolbox, and exact file-set validation.
Because the toolkit declares R2025a+, this run validates tooling rather than scientific
parity. A clean R2025a+ run and review remain external gates for #278 and #310.

## 9. Tracking issues

Remediation is tracked under Phase T, as sub-issues of the Phase T parent (#273):

| Issue | Title | Maps to | Status |
|---|---|---|---|
| #305 | T10 — Test data audit remediation | This document (parent tracking issue) | Open |
| #306 | T10a — Delete confirmed-dead test data | Step 1 | Done — see commit history |
| #307 | T10b — Resolve `prelim/run/output/` partial-orphan | Step 2 | Done — see commit history |
| #308 | T10c — Fix `serialisers/actual_output/` scratch-output leak | Step 3 | Done — see commit history |
| #309 | T10d — Relocate `test_data/demo/` out of `test_data/` | Step 4 | Done — see commit history |
| #310 | T10e — Migrate onto the unified layout (§7, §7.1) | Step 5 | Installer ready; fixture move blocked on verified #278 bundle |
| #311 | `examples/data/options.json`'s `selvars.type` invalid value | Found verifying Step 4 | Done — see commit history |

Pick up a step by reading its GitHub issue first, then the corresponding section above —
the issue records scope and compat tags, this document records the evidence.
