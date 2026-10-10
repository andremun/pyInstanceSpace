# Wider Python parity audit — 10 October 2026

Baseline: Python main `3e3bd57e49ab769f2cb6955986298fa9dde94924` and MATLAB master
`ed4bbc26b3381fd97e45d2b32ff93513f7d675c4`, after PRs #348 and #66 merged.
This completes the four targeted checks in the remaining wider-audit task. It is
not an exhaustive claim about every MATLAB/Python path.

| Area | Finding | Result |
| --- | --- | --- |
| Test-label independence | Recommendations are made from fitted PYTHIA state before test-outcome evaluation. | Controlled real-model exploration passes with outcomes changed, absent, wholly missing, or extended by a new algorithm. Coordinates, trained predictions/probabilities, recommendations and membership are unchanged. |
| Manual portfolio selection | Python selects algorithms before washing rows and calculating labels. | A selected-portfolio run matches a dataset containing only that portfolio, including a row observed solely for a discarded algorithm. No runtime fix needed. |
| Automatic portfolio pruning | MATLAB removes algorithms with no good training instances; Python retained them. The user explicitly chose MATLAB's policy. | Pipeline PRELIM now prunes and refits on raw retained data. Controlled minimization/maximization builds exactly match independently reduced-portfolio builds, including normalization. Empty portfolios fail clearly. |
| Feature replay | Exploration used original metadata feature names even after manual exclusion or data washing. Both controlled cases failed before the change. | PREPROCESSING records its retained column order; Model persists it; exploration aligns incoming columns to that fitted order before PRELIM/SIFTED. Omitted excluded columns, reordered inputs, complete metadata and model save/load are tested. |
| SIFTED cache and index replay | Cache belongs to each GA instance; decoded indices map back through the correlation-selected feature indices. | Fixed chromosomes exercise real PILOT/KNN fitness. A repeated mask hits cache within a run; changing labels in a second run changes fitness; a third original-data run reproduces it. Non-prefix selected columns replay correctly. No runtime fix needed. |

## Pruning evidence

The adversarial dataset has a third algorithm which is never good but is best on
one row where all algorithms are bad. Another row ties the retained algorithms
and changes beta when the denominator drops from three to two. Simply slicing
previous outputs would retain the wrong winner and beta. Both directions use
nonnegative raw performance, as required by MATLAB.

Clean current-master MATLAB R2026a Update 5 executed the same inputs and retained
portfolio procedure. Python and MATLAB agree exactly on good/bad labels, winner
indices, good-algorithm counts and beta for both minimization and maximization.
Raw best performance agrees within the CSV serialization allowance of `1e-14`.
The full Python normalization/refit outputs are additionally compared exactly
against an independent reduced-portfolio build.

Local MATLAB driver, inputs and outputs are under
`.cache/wider-parity-2026-10-10/`; execution log is `/tmp/wider-matlab-pruning.log`.
No approved fixture or provenance pin changes are needed for these fixes.

## Compatibility and limits

See the unreleased entry in `RELEASE_NOTES.md`: direct stage callers must supply
algorithm labels and account for the final retained-label output field. Fixed
PYTHIA parameter rows describe the retained stage portfolio. Older persisted
models lacking the fitted feature-name record use the previous metadata fallback;
rebuild those models if training removed features.

The cache check fixes the GA candidate sequence, rather than claiming Python and
MATLAB stochastic optimizers follow identical search trajectories. The earlier
PLS centering, held-out summaries and fitted stage-option persistence fixes are
already merged and remain covered by their regression tests.

Remaining separate work: #341 legacy probability test portability, #304 opt-in
Bayesian tuning, legacy TRACE trimming sensitivity, dependency updates and the
optional repository-history restart. These are not resolved by this audit.

## Validation

The full Python suite passed 1,128 tests; the final eight controlled audit cases
also pass with valid nonnegative minimization/maximization inputs. Strict mypy
passes 93 files, changed-file Ruff/Black and whitespace checks pass, and the
documentation site builds. MATLAB source and approved reference pins are unchanged.
