# MATLAB Compatibility

Understand shared scientific contracts and current implementation differences

Both packages implement Instance Space Analysis, but they use independently engineered pipelines and numerical libraries. Matching scientific contracts does not promise identical optimizers, fitted classifiers or arbitrary geometry.

## Shared Contracts

- Observed-only scoring, Oracle/selector probabilities and per-instance regret weights.
- Training-portfolio pruning with preprocessing recomputed from the retained raw data.
- Fitted feature order, PLS centering, constant-column scales and stage options reused during exploration.
- 2D/3D standard and PLS projection, 3D viewpoints and footprint exports.
- Closed-boundary membership with an explicit persisted Euclidean distance tolerance, default zero.

## Configuration Differences

Python uses immutable dataclasses and snake_case fields. The [Options Reference](options-reference.html) lists accepted JSON aliases. Python worker settings are `parallel.flag` and `parallel.n_cores`; MATLAB uses `general.parallel` and `general.ncores`.

Choose performance direction, threshold mode and epsilon explicitly. Python defaults to absolute, lower-is-better, epsilon 0.20; MATLAB defaults to relative, lower-is-better, epsilon 0.05. Python's general seed is 0; MATLAB's is 42. Python defaults to an SVM classifier; MATLAB defaults to KNN.

## Footprints

Python defaults to legacy TRACE; MATLAB defaults to TRACE3. Select `trace.method="trace3"` explicitly for the corresponding footprint construction. TRACE3's omitted purity is 0.60; Python legacy's is 0.55.

MATLAB records TRACE3 acceptance and search-termination fields. Python reports footprint geometry and metrics but does not currently persist those same status fields. Inspect the reported purity; reaching the end of a search does not prove its target was achieved.

A shared membership policy does not make sequential legacy contradiction trimming continuous. Small coordinate changes can change overlap counts and the selected trimming branch. Holes, boundary precision and serialization should be considered when comparing geometry.

## Optimization and Classification

MATLAB and Python use different SVM training/calibration implementations and Bayesian acquisition behavior. See [Classifiers and Tuning](tuning.html). Seeds and equal budgets do not guarantee the same folds, optimizer trajectories or classifier outputs.

Python provides optional canonical 2D PILOT rotation through `adjust_rotation`; MATLAB does not currently implement that optional rotation. Compare projections with their fitted dependent artifacts, rather than rotating only the coordinate array.

## Persistence and Output

Native fitted-model archives are runtime-specific. Use CSV and declared mesh schemas for data interchange; use each package's native persistence for later inference in the same ecosystem. Python exports files through explicit save methods, while MATLAB build/explore can write configured outputs automatically.

## See Also

[Options Reference](options-reference.html) · [TRACE](trace.html) · [Classifiers and Tuning](tuning.html)
