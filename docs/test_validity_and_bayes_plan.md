# Test validity and PYTHIA Bayesian plan

## Authority

MATLAB InstanceSpace v0.9.1 at `98a01ac`, executed with R2026a Update 4, is the behavioral authority.
GitHub issues and reviews are audit leads. Only the installed, verified 423-file
`reference-export/v2` bundle is a numerical oracle; `legacy-unknown` CSVs are regression
snapshots only.

## Test-audit rules

- Compare exact discrete outputs, counts, labels, connectivity, and rounded summaries
  exactly.
- Use documented combined tolerances only for floating geometry or optimizer-invariant
  quantities.
- A broad optimizer tolerance must be paired with an exact replay or formula-level test and
  a discrimination probe showing that arbitrary outputs normally fail.
- Read full-precision fixture numbers with round-trip parsing.
- Promote relevant numerical warnings to errors in denominator-zero and optimizer tests.
- A Bayesian budget test must exceed the four seed evaluations so at least one guided
  acquisition step executes.
- No test may describe historical or diagnostic data as MATLAB ground truth.

## Issue #304 boundary

The issue's historical 24/30 and 28/30 comparisons do not establish convergence: they use
rounded final metrics from fixtures without a recorded MATLAB commit, release, options,
folds, or candidate trace. They are not a basis for changing defaults.

Current MATLAB explicitly uses:

- `MaxObjectiveEvaluations = nTuningIter`;
- four default seed points; and
- `expected-improvement-plus` without parallel evaluation.

Python uses the same total budget and four seed points. `skopt` plain EI is the closest base
acquisition but does not implement MATLAB's anti-overexploitation “plus” loop. Matching the
per-algorithm seed boundary does not make MATLAB and sklearn fold partitions identical.

## Correction plan

1. Strengthen constructor/budget tests to execute guided Bayesian acquisition and prove the
   exact number of evaluations.
2. Pin per-algorithm estimator, splitter, and optimizer seeds and warning-free candidate
   execution.
3. Keep the production budget/default unchanged until a verified equal-budget trace exists.
4. Add a scoped current-MATLAB diagnostic that records fold identities, evaluated candidates,
   observed CV error, running best, selected parameters, seed count, and acquisition strategy.
5. Compare repeated seeds and cross-evaluate candidate points before deciding whether a custom
   EI-plus adapter is scientifically justified.

Completion may correct unsupported claims and test coverage without claiming that plain EI is
numerically identical to MATLAB EI-plus.
