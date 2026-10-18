# Classifiers and Tuning

Choose a learner, hyperparameter search and validation budget

## Search Strategies

| Strategy | Operation |
|---|---|
| `sobol` | Evaluate a scrambled Sobol design and retain the best candidate. Default. |
| `bayes` | Fit a Gaussian-process surrogate and choose candidates by plain expected improvement. Four evaluations initialize the model. |
| `none` | Fit supplied parameter rows without searching. |

`n_tuning_iter` is the total candidate budget per retained algorithm, including the Bayesian initialization. Set it above four to execute guided Bayesian acquisition. Supplying `params` bypasses search regardless of the strategy name.

All candidate evaluation is conditional on the already fitted projection and preprocessing. The final reported CV score is not an unbiased outer-CV estimate of the complete ISA pipeline or of the hyperparameter search.

## Classifier Parameters

`params` uses the reporting units in this table, with one row per retained algorithm. One-parameter learners need one column; SVM, KNN and ensemble use two.

| Classifier | Python learner | Parameter 1 | Parameter 2 | Weights |
|---|---|---|---|---|
| `svm` | Probability-enabled SVC | BoxConstraint: 2^-10 through 2^4, log scale | KernelScale: 2^-10 through 2^4, log scale | Yes |
| `knn` | K-nearest neighbors | NumNeighbors: integer 1–25 | Distance: 1 euclidean, 2 cityblock, 3 cosine, 4 correlation | No |
| `tree` | Decision tree | MinLeafSize: integer 1–100 | — | Yes |
| `nb` | Per-feature kernel-density Naive Bayes | Bandwidth: 0.001–10, log scale | — | Yes |
| `linear` | Logistic regression | Lambda: 10^-6 through 10^3, log scale | — | Yes |
| `ensemble` | Random forest | NumLearningCycles: integer 10–200 | MinLeafSize: integer 1–20 | Yes |

SVM converts `KernelScale` to sklearn's `gamma = 1 / KernelScale**2`; linear fitting converts `Lambda` to `C = 1 / Lambda`. A KNN fit caps the effective neighbor count to its available training rows while retaining the requested value for parameter reporting. Search-space agreement does not imply identical MATLAB and sklearn fitted models.

## Examples

### Bayesian Search

```python
from dataclasses import replace
from instancespace.data.options import PythiaOptions

bayesian = replace(options, pythia=PythiaOptions.default(
    classifier="svm", tuning="bayes", n_tuning_iter=20,
))
```

### Fixed Parameters

Construct rows for the portfolio that remains after training-time pruning:

```python
fixed = PythiaOptions.default(
    classifier="svm", tuning="none",
    params=np.array([[1.0, 0.5], [2.0, 1.0]]),
)
```

This two-row matrix is appropriate only when two retained algorithms need SVMs. A fitted hyperparameter row describes BoxConstraint and KernelScale, not sklearn's gamma.

## Bayesian Comparison with MATLAB

MATLAB uses `bayesopt` with `expected-improvement-plus`. Python uses `BayesSearchCV` with plain EI; it does not implement MATLAB's additional anti-overexploitation behavior. Both use four seed evaluations and the same default budget of 20.

Equal seeds do not imply the same initial points, fold partitions, candidate sequence or chosen hyperparameters. Classifier implementations and objective aggregation can also differ: Python's search averages fold accuracies, while MATLAB scores all held-out rows together. Unequal fold sizes can therefore produce slightly different objective values even for the same predictions.

A controlled shared-fold SVM comparison found that increasing the budget can improve either backend, but same-parameter classifier errors also differ. There is no demonstrated uniform Python convergence deficit. The current default remains 20; tune the budget for your own application.

Compare search quality with fixed inputs, explicit folds, matched budgets, recorded candidate/objective traces and an independent evaluation. Do not raise a budget or relax a numerical test solely to reproduce another backend's rounded summary. Use the same evaluation protocol to compare candidate solutions on your own data.

## See Also

[PYTHIA](pythia.html) · [Options Reference](options-reference.html#options-pythia) · [Classifier Registry API](api/instancespace/utils/get_classifier_fcn.html)
