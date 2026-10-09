# MATLAB parity assessment and proposed work plan

Reviewed 5 October 2026. The supplied 2 October note was treated as evidence and proposals, not as authorization to implement its instructions. This review changes no package implementation or fixtures.

Baseline clarification from the user: treat MATLAB `master` and Python `main` as the authoritative current versions. Unmerged branches are proposals, not current functionality. The implementation target is parity with current MATLAB master; record its resolved commit when generating fixtures so each bundle remains reproducible. No separate baseline-choice decision is required.

## Evidence and limits

- Python checkout: clean `main` at `66e65be`; inspected local remote-tracking refresh branch at `5582b50`.
- MATLAB checkout: clean `codex/octave-compatibility-foundation` at `73570e6`; separately inspected local `master` at `aac7d7d`, which contains the merge of PR #64.
- The GitHub page retrieved for Python PR #346 reports it open. Direct API access failed, and remote refs were not refreshed; these local SHAs are the scope of source conclusions, not a claim that every remote branch is unchanged.
- Full Python tests were not run: available Python is 3.14, the package requires 3.12, and pytest is absent. No dependency installation was attempted. MATLAB tests were inspected but not rerun. Existing MATLAB review reports are historical verification evidence.
- An isolated execution of the actual Python `PythiaStage.evaluate` method, extracted with AST to avoid unavailable package dependencies, reproduced accuracy 2/3 instead of 1 for a three-row example whose final observation is missing. Its counts were `[1,0,1,1]` instead of `[1,0,0,1]`. A separate calculation of the current selector formula gives recall 1/2 for one successful selection when both algorithms are good; the corrected recall is 1.

## Assessment

The note's direction is suitable: port specific correctness fixes, distinguish candidate validation from approved-fixture verification, and prioritize controlled stage references alongside integration snapshots. Its detailed target semantics need updating to current MATLAB source. Neither a large fixture replacement nor tolerance increases should substitute for diagnosing discrepancies.

### Confirmed Python gaps

1. **Missing-observation scoring:** `instancespace/stages/pythia.py`, `evaluate`, uses all truth rows and the total row count. PR #346 introduces an appropriate per-instance/per-trained-algorithm mask and threads it from raw test outcomes through `InstanceSpace._explore_evaluate`. It also covers partial missingness and missing classifier slots. This is a useful patch to retain.
2. **Incomplete no-data contract in that patch:** its unscored confusion rows remain zero, while MATLAB master initializes them to NaN. The patch's statement that zero rows match MATLAB should be corrected. Review the exposed Python compatibility implications explicitly rather than silently changing counts.
3. **Summary probabilities:** `_generate_summary` still hard-codes Oracle probability to 1 and averages binary labels for algorithm rates. Current MATLAB divides algorithm successes by observed outcomes, Oracle successes by rows with any observed outcome, and selector successes by rows with an observed selected outcome. Therefore the note's `mean(any(y_bin, axis=1))` replacement alone is insufficient for wholly unobserved rows. Preserve the raw observation mask before masking arrays for selection summaries.
4. **Selector recall:** Python counts any unselected good algorithm as a miss even when the selected algorithm is good. MATLAB now uses disjoint successful and missed-opportunity rows. Port this with summary fixes.
5. **Cost-sensitive weights:** Python uses `abs(y - global_mean(y))`; MATLAB uses `abs(Y - Ybest)` with zero/degenerate handling. This is an additional confirmed source discrepancy beyond the note; test it as its own behavior-changing patch.

### Geometry and reproducibility

- Python TRACE already handles absent predictions. Do not introduce a speculative fallback fix.
- Existing TRACE parity tests reconstruct polygon parts and holes and check hole counts. Extend these with deliberately nonempty holes, disconnected components, and round-trip/export cases; do not build duplicate infrastructure or assume a parser proves the entire rendering path.
- Python CLOISTER can invoke SciPy's hull in all projected dimensions, so it does not simply reproduce MATLAB's old two-coordinate mistake. However, `CloisterOutput` has only vertex arrays, no face triangulation; coplanar 3D Qhull errors return an empty array, whereas MATLAB supports a triangulated flat polygon. Audit computation, persistence, plotting, and export together before defining the public face contract. Retain the explicit `hull_dims` option's intended semantics.
- MATLAB's posterior reseeding fix does not establish a sklearn bug. Test repeated fits at fixed inputs/hyperparameters and distinguish probability reproducibility from label stability.

## MATLAB fixes already present

Local master includes the earlier evaluation mask (#58), Oracle rate (#59), 3D CLOISTER (#50), boundary-cycle/hole (#52), absent TRACE predictions, axes error, and posterior RNG (#61) corrections. Relevant history includes `c62408f`, `354375f`, `5c24dc5`, and `f726291`; regression tests cover these behaviors.

PR #64's merge (`aac7d7d`) adds substantial further corrections: SIFTED cache isolation; preprocessing after portfolio pruning; training-only recommendation policy; fitted PLS centering; constant-column scales; out-of-fold summary selections; stage-option provenance and partial persistence; recoverable feature selection; small-batch TRACE evaluation; missing-data guards; selector recall; regret weights; and complete geometry exports. These are candidates for a wider Python audit, not proof that every analogous Python path is defective.

The historical `review/2026-09-30/FINAL_REPORT.md` records 198/198 integration tests and subsequent targeted reruns; `COPILOT_FOLLOWUP.md` records 12/12 targeted tests. Those are not fresh results for the current Octave branch. The latter branch adds a substantial separate compatibility layer and geometry work beyond master and should not accidentally become the MATLAB oracle simply because it is checked out.

Fixture instability remains unresolved. `review/2026-09-30/ISSUE63_INVESTIGATION.md` explicitly does not attribute its resolution to PR #64. In its controlled PILOT experiment, repeated runs and matched-thread source comparisons were equal; thread-count changes perturbed restart scores by about 6.55e-15 without changing the winning restart. It did not reproduce the original full-export differences. The default fixture input bypasses SIFTED's GA, so the cache fix is not an explanation for that case.

## Fixture and CI assessment

Python main pins `98a01ac` in the provenance verifier. PR #346 advances it to `fdad7a43`, still earlier than the posterior correction and PR #64. Its verifier and content-root pins change together, but it retains the conceptual coupling between approved identity and candidate acceptance.

The existing validation workflow clones moving MATLAB `master` and requires that HEAD equal the committed manifest SHA. This makes an unrelated MATLAB merge capable of breaking Python's release gate. Merely updating the SHA once will recur as a problem.

Use an immutable approved SHA for reproducible release validation. Check upstream freshness separately and produce a reviewable refresh candidate when it changes. Candidate validation should check the requested source identity, inputs, options, exporter, schema and hashes without requiring the old approved content root. Promotion should atomically update manifest, identity/content pins, inventory where needed, and reference assertions. Preserve hashes for transport integrity; use numerical and semantic comparisons for independently generated outputs.

## Proposed implementation sequence

1. **Record the current-master contract.** Use MATLAB `master` and Python `main` as the authoritative versions. Resolve and record MATLAB master's exact SHA for fixture generation, and document intentional Python differences. Inventory existing stage tests and consumers before changing output schemas. Provision the supported Python 3.12 environment.
2. **Focused PYTHIA evaluation patch.** Reuse/rebase the useful code from PR #346 separately from its bulk fixture update. Cover complete and partial observations, missing columns, all-unobserved rows, missing classifiers, and test-only algorithms. Decide and test NaN confusion rows. Acceptance: exact counts/masks and MATLAB-consistent undefined rates through both direct stage calls and exploration.
3. **Summary and cost corrections.** Correct observed-only algorithm/Oracle/selector rates and disjoint selector recall. Include all-good, no-good, multiple-good, no-selection and zero-denominator cases. Port regret weighting in a separate small change with nonuniform and all-zero regret tests. Acceptance: hand-computable cases agree with pinned MATLAB stage outputs.
4. **Candidate/provenance workflow.** Separate candidate validation, committed-bundle verification and promotion; replace the moving-master release gate with approved-SHA validation plus an independent freshness check. Acceptance: a valid new source can be reviewed without accepting tampered inputs, exporter identities or hashes.
5. **Controlled geometry references.** Add TRACE ring/component, boundary-membership and export round trips; add full-rank, coplanar and degenerate 3D CLOISTER cases and face-schema tests. Compare membership/topology and area or volume; normalize ordering rather than comparing arbitrary vertex order.
6. **Wider current-master audit.** Prioritize PLS train/explore centering, test-label independence of recommendations, pruning and feature replay, then stage provenance and persistence. For each, first demonstrate whether Python is affected. Build fixed-input SIFTED and PILOT cases using explicit initial conditions/replay where supported. Avoid rotating classifier coordinates independently of their fitted models when testing projection equivalence.
7. **Reproducibility experiment and promotion.** Repeat complete exports at the same SHA and runtime, recording each stage's input hashes, solver/restart diagnostics, MATLAB update, computation threads and workers. Locate the first divergence and measure discrete, numerical and semantic effects. Choose tolerances from evidence, not failing-test accommodation. Promote only after the discrepancy report and focused tests pass; retain a small end-to-end suite.

Immediate next implementation target: the evaluation and summary contracts, with small synthetic regressions, before any bulk fixture regeneration.

## Step 2 implementation update

The focused evaluation change is implemented locally. It reuses the evaluation,
observation-mask plumbing and regression tests from PR #346's local branch,
then updates unscored confusion rows to NaN to match current MATLAB master.
Additional regressions cover independently missing observations across reordered
algorithm columns, wholly missing rows, empty batches, mask shape validation and
input preservation. Direct `PythiaEvaluateInput` callers must supply the new mask;
exploration supplies it automatically. The release notes record this API change.

Validation: 134 tests passed with warnings treated as errors across evaluation,
exploration orchestration, predictive contracts, current MATLAB stage references,
and PYTHIA build/explore tests, using Python 3.12 and dependencies constrained by
`poetry.lock`. No fixture contents or provenance pins changed. The historical
skip-mode reference test explicitly checks the corrected no-classifier contract
instead of comparing to stale evaluation rates. Summary and regret-weight changes
remain step 3.

GitHub status checked during implementation: #345 is an open issue, with fixes
proposed in open PRs #343 and #346. The existing work is reused, not counted as
already merged into main.

## Step 3 implementation update

Summary probabilities now use observed-only denominators for each algorithm,
the Oracle, and the fallback selector. The observation mask is captured before
selection-specific copies are masked. Selector recall counts successes and missed
opportunities on disjoint rows. Cost-sensitive training now uses absolute
per-instance regret; zero-regret substitution and uniform degenerate fallback are
preserved. These changes match the inspected MATLAB master formulas.

New hand-computable tests cover partial/all missingness, stale true labels at
missing outcomes, no-good and multiple-good rows, abstention versus fallback,
empty input, minimization/maximization regret, tied algorithms across nonconstant
rows, and input preservation. The original uniform-weight regression now supplies
best outcomes consistent with its constant raw outcomes. Numeric 0/1 label arrays
remain supported. The focused run passed 89 tests with warnings treated as errors;
lint and strict type checks passed. Full-suite validation is recorded below when
complete. No fixture or baseline-pin changes are part of this step.

Final validation: the full suite completed with 1,066 passes and one failure in
`test_matlab_source_invariants_when_reference_repo_is_available`, which reads the
adjacent MATLAB working tree rather than master. That tree is on the Octave branch
and no longer contains the test's literal optimizer-option spelling. The unchanged
test was copied into a temporary sibling layout pointing at an export of MATLAB
master (`aac7d7d`) and passed there. Neither the test nor the MATLAB checkout was
changed to accommodate this environment difference. Separately, real MATLAB
executed the mixed-observation summary and nonuniform regret examples against that
master export; both numerical assertions passed. Black formatting, Ruff, strict
mypy checks, and `git diff --check` passed for the changed step-3 code.

The next provenance change should preserve the current verifier's structural,
canonical-input, exporter, effective-option, and numerical-lineage checks. Only
the approved source/content identity should differ for candidate validation;
using the existing diagnostic mode would weaken the trust contract. Candidate
validation must not make a bundle eligible for installation automatically.

## Step 4 implementation update

The provenance tool now has separate `candidate` and `prepare-promotion` commands.
Candidate validation requires independently supplied MATLAB/generator commits and
the exporter script hash, retains the v2 structural/environment/input/option and
numerical-lineage checks, and reports `matlab-candidate` with `approved: false`.
It does not require the old approved source or content root. Approved verification
and installation keep their existing identity checks.

Promotion preparation revalidates a copied candidate and atomically publishes a
new review directory with the bundle, manifest hash, previous/proposed identities
and review instructions. It refuses an existing destination or a destination
inside the source, cleans up failed staging, and does not edit the source or
approved files. Actual promotion remains a reviewed Git commit updating the
bundle, verifier pins, exporter and any inventory/reference-test changes together.
No new bundle was promoted in this step.

Release CI now checks out the committed approved SHA, checks that checkout against
the manifest, and verifies the approval pins. An independent non-blocking job
reports whether MATLAB master has moved and requests a reviewed refresh candidate.
MATLAB master remains the current implementation target; the old snapshot is
explicitly an approved fixture revision, not a claim that it is current master.

Validation: 99 provenance tests passed, then three additional CLI/failed-staging
tests passed (102 total). Current-bundle verification passed for all 423 artifacts;
inventory validation passed for 764 files. Ruff, strict mypy, Black formatting and
diff whitespace checks passed. Workflow YAML and embedded Python parse correctly;
the pinned source exists locally and satisfies the existing MATLAB source-invariant
test. GitHub Actions itself has not been dispatched. Exported fixtures and approval
pins remain unchanged. Usage and promotion instructions are in
`tests/matlab_export/README.md`.

## Step 5 and documentation integration (9 October 2026)

Remote MATLAB master has advanced to `929acfd` and includes PR #65's Octave work.
The earlier warning about using an unmerged Octave branch is historical. Six
controlled geometry cases were generated with local MATLAB R2026a Update 5 from
a clean checkout of current master, with hashed provenance in
`tests/fixtures/matlab/geometry`. The approved full fixture bundle is unchanged.

The references exposed and now cover planar 3D CLOISTER boundaries, which Python
previously returned empty. Derived triangle faces preserve the two-array result
API and are available after model persistence; CSV exports declare zero-based
face indices in a mesh manifest. Collinear 3D input raises a clear error. TRACE
hole/component area, membership and topology match MATLAB, including boundary
CSV round trips and plotting cycles. No new CLOISTER graph view is introduced.

PR #343's curated site, contributor documentation and docs build are integrated;
the weight description was corrected to the new per-instance regret definition.
Superseded runtime/CI changes and unreviewed full fixture replacements were not
imported. See `branch-consolidation-2026-10-09.md` for all development branch
relationships and the local integration branch.

Final consolidation validation: 1,090 tests passed in the broad suite, which
excluded the pre-existing local MATLAB source check while its current-master
adapter update was being verified. That check and the eight controlled geometry
tests then passed separately (nine passes). The focused geometry/serialization/
provenance run passed 196 tests. Strict mypy passed all 88 source files; Ruff,
Black checks on changed Python files, and `git diff --check` passed. The docs site
build and every generated local link target passed. Approved-bundle verification
passed and fixture inventory validation covered 771 files. GitHub Actions has not
been run as part of these local checks.

The MATLAB source-invariant test now follows the current master's
`isacompat.minimize` call to the optimizer helper and still asserts the same
`FunctionTolerance` value. Its original inline-option assertion remains for older
source checkouts. This updates source-location knowledge rather than changing
the expected numerical contract.


## Step 6 audit in progress (9 October 2026)

Current source identities remain MATLAB master `929acfd` and Python main
`66e65be`; work continues on PR #348 with unchanged dependency versions.

Confirmed and corrected after failing regression tests:

- PLS training centred features but discarded the fitted mean, while exploration
  always used uncentred `X @ A.T`. `PilotOutput.pilot_x_mean` now reaches persisted
  `PilotOut.x_mean`, exploration and CLOISTER boundary generation. Legacy models
  without the field retain their prior projection. Four regressions cover 2D/3D
  train/query consistency, single-row inference, persistence and CLOISTER's regular
  and feature-cap paths. All four failed before the fix. The focused PILOT and
  CLOISTER run passed 88 tests. Local MATLAB R2026a Update 5 at clean `929acfd`
  independently verified stored-mean reconstruction and matched Python train/query
  pairwise distances in both dimensions, using deterministic trigonometric inputs.
- PYTHIA constant projection columns produced NaNs from zscore. Training now stores
  unit scale for constant columns, and inference guards zero scales in older models.
- PYTHIA selector summary rows used fitted training predictions despite reporting
  CV metrics. The summary now derives recommendations from held-out `y_sub`, while
  exposed fitted selections remain unchanged. A controlled classifier with perfect
  fitted predictions and 50% held-out precision reproduced the optimism before the
  fix; the summary now reports the held-out result. Both new PYTHIA tests failed
  before the changes and passed afterward.

Source inspection also confirms that Python already selects the algorithm
portfolio before missing-value row filtering, and its prediction path has no test
performance input. Existing SIFTED tests cover replay order; checkpoint tests cover
partial persistence and resumption. These are evidence to examine, not new fixes.
Stage-option provenance now records the option inputs consumed by each completed
stage. Model construction reconstructs effective options, including the aggregate
PRELIM flags and performance settings; preprocessing and evaluation during
exploration use these fitted settings. Records survive model/checkpoint persistence
and roll back when their stages are invalidated. Legacy payloads without stage
records retain their constructor defaults.
The wider audit and complete-export reproducibility/promotion step are not yet
complete. No full fixture bundle or dependency version has been changed.

Issue housekeeping: the original audit parent #297 was closed after verifying its
six closed sub-issues and merged fix history. PR #348 now closes #344 and #345 on
merge. #347 stays open for reproducibility and promotion; #304 and #341 are not
claimed resolved.

Validation of this first step-6 patch: the full suite passed all 1,097 tests,
strict mypy passed 90 files, Ruff passed, and whitespace checks passed. This is
additional tested work on the draft, not completion of steps 6–7.

## Stage provenance and repeated full exports (9 October 2026)

The fitted-option regression is fixed: preprocessing and evaluation during explore
now use `Model.opts`, reconstructed from the option inputs consumed by completed
stages. `Model.stage_options` retains those records through persistence. Checkpoint
rollback removes invalidated records. Tests exercise different constructor/fitted
settings, persistence and rollback. The full suite passed 1,104 tests after this
change and the exporter/context changes below. Three additional tests cover the
new MATLAB `sifted.diagnostics` Boolean option (including invalid-type rejection).

The unchanged exporter initially failed against current MATLAB because its 3D
PILOT cases retained SIFTED fitted under 2D options. The exporter now rebuilds SIFTED
through the public API for each variant. The canonical dataset still selects the
same inputs, which the validator checks. Stage-context v2 records the rebuild,
actual dimensions and fitted feature mean, and checks centred PLS exploration.
Historical context v1 remains supported for the existing approved bundle. The
approved exporter hash is tied to that bundle's manifest, rather than requiring
future candidate generators to keep the old source bytes.

Two preliminary diagnostic runs were followed by two clean verified-mode exports:

- MATLAB master: `929acfd889e7a17c7ee40c4004ad1bc18639b0e6`.
- Published generator: `9bdc37aabfafa31041256813b1045e282007d799`.
- Exporter SHA-256: `e9b5887538aa1eec7757918474e6a1641995315e290bd3a8fe8a04c7161e5e8b`.
- MATLAB: `26.1.0.3346908 (R2026a) Update 5`, GLNXA64, one computation
  thread, parallel execution disabled, seed 42.
- Both exports pass candidate validation. All 423 artifact files are byte-identical
  between runs. Manifest differences are the timestamp and MATLAB adding the
  now-loaded Instance Space Analysis Toolkit to its `ver` listing on the second
  pass in the same process. No artifact divergence was found.
- Shared artifact content root:
  `c18c467c55ce46b19785503c4d88ce8acf64e94542de02e8da5fa8c177acd27d`.

The local review package is `.cache/matlab-review-2026-10-09/promotion/`, with the
repeat export, driver, runtime log, reproducibility record and changed-path report
beside it. These are local candidate evidence, excluded from Git; source changes
and this report are in PR #348. No approved fixture, inventory or numerical
tolerance has been changed.

Compared with the approved bundle, 218 of 423 artifact hashes differ, with no
added or missing paths. The earliest observed numerical differences are PRELIM
`y_best` values (14 cells, maximum absolute difference `2.22e-16`), followed by
SIFTED correlation outputs (maximum `1.03e-15`). Downstream differences include
changed legacy SVM training selections (two cells each in `selection0/1`) and
TRACE explore membership changes: 73 legacy SVM cells, 172 standard-analytic 3D
cells, 44 default TRACE3 cells and 198 skip-PYTHIA TRACE3 cells. These are
cross-revision/environment comparisons, not evidence of repeat-run instability
or proof of a particular causal chain.

Step 7's controlled repeatability experiment is complete for this pinned local
runtime. Numerical/semantic review against Python and the previous oracle remains
before fixture promotion and closing #347. The wider audit is not claimed
exhaustive; Bayesian-search comparison (#304), cross-platform legacy tolerance
(#341), dependency upgrades and repository-history changes remain outside these
completed checks.

The subsequent [TRACE investigation](trace-fixture-investigation-2026-10-09.md)
classifies 466 of 487 membership changes as numerical boundary cases. The other
21 legacy changes are reproduced by changing only the frozen projection under
one runtime: contradiction-removal decisions amplify the small coordinate
changes. This narrows the remaining review to explicit boundary semantics and
legacy trimming stability; it does not promote the candidate.
