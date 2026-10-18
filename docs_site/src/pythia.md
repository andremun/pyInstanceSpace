# PYTHIA

Train algorithm selectors and score observed outcomes

## Syntax

```python
from instancespace.stages.pythia import PythiaStage

outputs = space.run_until_stage(PythiaStage)
```

## Description

PYTHIA standardizes projected coordinates and trains a binary classifier for each retained algorithm. A prediction means that the algorithm is expected to be good. A saved training policy combines those predictions into selections; exploration does not choose recommendations from test outcomes.

Supported classifiers are SVM, KNN, decision tree, kernel Naive Bayes, logistic linear and random-forest ensemble. Hyperparameters use Sobol search, Bayesian search or supplied parameter rows. Single-class targets have a constant classifier instead of an invalid stratified search.

## Parameters

`classifier`, `tuning`, `n_tuning_iter`, `cv_folds`, `params`, `is_poly_krnl`, `use_weights` and `skip` are documented in [PYTHIA options](options-reference.html#options-pythia). Fixed parameter rows must follow the retained algorithm order and reporting-unit contract.

Sample weighting uses `abs(y - y_best)` on the performance arrays supplied to PYTHIA. Zero weights are replaced with the smallest finite nonzero weight; nonfinite weights use the largest. An entirely zero/nonfinite weight matrix falls back to uniform weights. Unsupported learners warn and fit without weights.

## Returns

Classifiers, fitted normalization, held-out and fitted predictions, bad-class probabilities, hyperparameters, selections and a summary. `selection0` can abstain; `selection1` uses the saved fallback. Selectors use zero-based algorithm indices, with `-1` for no choice.

The training summary uses held-out classifier predictions. These CV statistics remain conditional on the fitted preprocessing, feature selection, projection and tuning; they are not outer cross-validation of the complete ISA workflow. Use independent test data to evaluate the full fitted analysis.

During exploration, metrics use only observed outcomes for each algorithm. Oracle and selector probabilities likewise exclude unobserved outcomes. No observations means an undefined rate, reported as `NaN`. A fitted classifier for a missing test algorithm still predicts, but cannot be scored there.

## Examples

```python
from dataclasses import replace
from instancespace.data.options import PythiaOptions

classifier_options = replace(options, pythia=PythiaOptions.default(
    classifier="svm", tuning="bayes", n_tuning_iter=20,
))
```

## See Also

[Classifiers and Tuning](tuning.html) · [TRACE](trace.html) · [PYTHIA API](api/instancespace/stages/pythia.html)
