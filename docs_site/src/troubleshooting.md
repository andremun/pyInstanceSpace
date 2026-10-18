# Troubleshooting

Resolve common input, configuration and workflow errors

## Not Enough Features or Instances

Training requires at least three viable input features. Check manual column selection and the missing-feature threshold. The configured folds also need enough examples of each class. A small synthetic example does not establish model quality on a real domain.

## No Retained Algorithms

Algorithms never good in training are pruned. If none remain, review the performance direction, absolute/relative mode, epsilon and data units. Do not change the threshold solely to avoid the error without confirming its scientific meaning.

## Missing or Reordered Test Features

Exploration uses the fitted retained feature order. Supply those columns with the same meanings and units; ordering is reconciled. Excluded training columns are not required. Missing required features are an error.

## Undefined Evaluation Metrics

An algorithm without observed test outcomes cannot be scored. `NaN` is the appropriate undefined rate, not zero accuracy. Predictions can still be produced without outcome labels.

## Unexpected Files or Missing Exports

Use explicit model save methods for CSV, PNG, web and MAT output. Output preference flags do not trigger automatic file creation. See [Outputs](outputs.html).

## Model Unavailable After a Partial Run

A partial stage run produces raw outputs and does not imply a complete `Model`. Consume `run_iter()` to completion or finish the required stages. Checkpoints and standalone fitted-model saves serve different purposes.

## Different MATLAB and Python Results

Check the performance settings, retained portfolio, projection method/dimensions, TRACE method/purity, fitted tolerance, search budget, seed, folds and parameter units first. Numerical libraries, classifiers and optimizers can still differ under the same nominal configuration. See [MATLAB Compatibility](matlab-compatibility.html).

## See Also

[Metadata Format](metadata-format.html) · [Build and Explore](workflows.html) · [Options Reference](options-reference.html)
