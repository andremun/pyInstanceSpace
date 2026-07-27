# Instance Space Analysis: A toolkit for the assessment of algorithmic power

![Tests](https://github.com/andremun/pyInstanceSpace/actions/workflows/validation-tests.yml/badge.svg)
[![DOI](https://zenodo.org/badge/770130753.svg)](https://doi.org/10.5281/zenodo.15562567)

API docs aren't built or hosted by CI yet — run `pdoc instancespace` locally (see *Documentation Instructions* below).

Instance Space Analysis is a methodology for assessing the strengths and weaknesses of an algorithm, and an approach to objectively compare algorithmic power without bias introduced by a restricted choice of test instances. At its core is modelling the relationship between an instance's structural properties and the performance of a group of algorithms. Instance Space Analysis allows the construction of **footprints** for each algorithm, defined as regions in the instance space where we statistically infer good performance. Other insights that can be gathered from Instance Space Analysis include:

-	Objective metrics of each algorithm’s footprint across the instance space as a measure of algorithmic power;
-	Explanation through visualisation of how instance features correlate with algorithm performance in various regions of the instance space;
-	Visualisation of the distribution and diversity of existing benchmark and real-world instances;
-	Assessment of the adequacy of the features used to characterise an instance;
-	Partitioning of the instance space into recommended regions for automated algorithm selection;
-	Distinguishing areas of the instance space where it may be useful to generate additional instances to gain further insights.

The unique advantage of visualizing algorithm performance in the instance space, rather than as a small set of summary statistics averaged across a selected collection of instances, is the nuanced analysis that becomes possible to explain strengths and weaknesses and examine interesting variations in performance that tables of summary statistics may hide.

This repository provides a set of Python tools to conduct a comprehensive Instance Space Analysis in an automated pipeline. We expect it to become the computational engine that powers the Melbourne Algorithm Test Instance Library with Data Analytics ([MATILDA](http://matilda.unimelb.edu.au/matilda/)) web tools for online analysis. If you would like more information on the Instance Space Analysis methodology, you can refer to [here](http://matilda.unimelb.edu.au/matilda/our-methodology).

If you follow the Instance Space Analysis methodology, please cite as follows:

> K. Smith-Miles and M.A. Muñoz. *Instance Space Analysis for Algorithm Testing: Methodology and Software Tools*. ACM Comput. Surv. 55(12:255),1-31 [DOI:10.1145/3572895](https://doi.org/10.1145/3572895), 2023.

Also, if you specifically use this code, please cite as follows (see `CITATION.cff` for the full, machine-readable record):

> M.A. Muñoz and K. Smith-Miles. *Instance Space Analysis: A Python toolkit for the assessment of algorithmic power*. andremun/pyInstanceSpace on GitHub. Zenodo, [DOI:10.5281/zenodo.15562567](https://doi.org/10.5281/zenodo.15562567), 2025.

> Y.B. Güzel, K. Khare, N. Harvey, K. Dsouza, D.H. Jang, J. Chen, C.Z. Lam, and M.A. Muñoz. *instancespace: A Python package for insightful algorithm testing through Instance Space Analysis*. SoftwareX, 31:102246, [DOI:10.1016/j.softx.2025.102246](https://doi.org/10.1016/j.softx.2025.102246), 2025.

**DISCLAIMER: This repository contains research code. On occasion, new features will be added or changes made that may result in crashes. Although we have made every effort to minimise bugs, this code comes with NO GUARANTEES. If you encounter any issues, please let us know as soon as possible through the contact methods outlined at the end of this document.**

## Installation Instructions

run `pip install instancespace`

An example of running can be found in integration_demo.py

An example of a plugin can be found in example_plugin.py

## Documentation Instructions

run `pdoc instancespace`

Please take a look at the pdoc documentation for instructions on exporting static HTML files for hosting on GitHub Pages.

## Repository layout

- `instancespace/` — the package itself: `instance_space.py` (the `InstanceSpace` class — `build()`/`explore()`/`explore_iter()`), `stage_builder.py`/`stage_runner.py` (the DAG scheduler that infers stage order from each stage's declared inputs/outputs), `stages/` (one module per pipeline stage — `preprocessing`, `prelim`, `sifted`, `pilot`, `pythia`, `cloister`, `trace`), `data/` (option and metadata dataclasses), `model.py` (the trained `Model` and its `save_to_csv`/`save_for_web`/`save_graphs`/`save_to_mat`/`save_zip` methods), `build_explore_adapter.py` (converts a `build()`-trained model into the form `explore()` consumes), and `scripting/` (CSV/plot output helpers).
- `tests/` — the test suite; `tests/matlab_reference/` holds MATLAB-trained golden-reference artifacts used to validate the Python port stage by stage, and `tests/exploreIS/` holds `explore()`/`explore_iter()`-specific validation and unit tests. Most other test files are named `test_<stage>.py` per stage.
- `integration_demo.py` — the minimal runnable example: load metadata + options from `tests/test_data/demo/`, construct an `InstanceSpace` with the full stage list, and `build()` it.
- `example_plugin.py` — demonstrates writing a custom `Stage` and slotting it into the pipeline alongside the built-in stages.
- `liveDemoExploreIS.ipynb` — the operation manual: a stage-by-stage walkthrough of `build()` and `explore()`/`explore_iter()`, meant to be read as a usage guide.
- `docs/` — `explore_validation.ipynb` (how the MATLAB-reference validation numbers were obtained) and the project roadmap/implementation-pathway documents used to plan ongoing work.
- `CLIDocs.txt` — notes on the (not yet built) command-line interface.

## Working with the code

The basic flow is: construct an `InstanceSpace` from metadata and options, `build()` it, then save the results.

```python
from instancespace import InstanceSpace
from instancespace.data import metadata, options

metadata_object = metadata.from_csv_file("metadata.csv")
options_object = options.from_json_file("options.json")

instance_space = InstanceSpace(metadata_object, options_object)
instance_space.build()

instance_space.model.save_to_csv("output/")
instance_space.model.save_graphs("output/")
```

See `integration_demo.py` for a complete, runnable version of this (including the explicit stage list), and `example_plugin.py` for how to add a custom `Stage` to the pipeline.

### Applying a trained model to new data: `explore()`

`InstanceSpace.explore()` applies a previously trained model to unseen instances, mirroring the MATLAB toolkit's `exploreIS.m`: the test metadata is bounded and scaled with the stored PRELIM parameters, reduced to the selected SIFTED features, projected with the trained PILOT matrix, and evaluated by the trained PYTHIA selectors and TRACE footprints. No stage is re-fitted.

`explore()` identifies the shape of the trained model itself and logs what it detected. A model trained by the Python `build()` — which stores the PYTHIA SVMs as fitted scikit-learn `SVC` objects — is converted internally, once, into the flattened structure `explore()` consumes; the scikit-learn model is not modified and remains available via the `model` property for anyone who prefers to keep working with scikit-learn objects. A model already in the flattened form (assembled from MATLAB-exported artifacts) is consumed as-is.

The conversion lives in `instancespace/build_explore_adapter.py` and only rewrites the PYTHIA SVMs into the flattened form; PRELIM, SIFTED, PILOT and TRACE pass through unchanged. Its module docstring explains each choice. The normal flow is therefore direct:

```python
space = InstanceSpace(train_metadata, options)
space.build()
result = space.explore(test_metadata)
```

`explore()` returns the full result in one call; `explore_iter()` runs the same stages but yields each one's output in turn (`prelim`, `sifted`, `pilot`, `pythia`, `trace`), for inspecting the pipeline one stage at a time. The operation manual `liveDemoExploreIS.ipynb` — the Python counterpart of the MATLAB live demo (`liveDemoIS.m`) — walks through both `build()` and `explore()`/`explore_iter()` stage by stage and is meant to be read as a usage guide; run it from the repository root.

The port is validated stage by stage against the MATLAB implementation: `tests/matlab_reference/` holds the MATLAB-trained artifacts and reference outputs, `tests/exploreIS/` holds the validation and unit tests (run `pytest tests/exploreIS/`), and `docs/explore_validation.ipynb` documents how the validation numbers were obtained and how a from-scratch Python build behaves. The test folders document their contents in their own README files.

## Development Environment Setup Guide

REQUIREMENTS:
- Python 3.12 installed
- Be inside the repository directory

### Step 1: Install poetry

*Linux, Mac, WSL*

`curl -sSL https://install.python-poetry.org | python3 -`

*Windows*

`(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -`

### Step 2: Setup virtual environment
`poetry shell`

### Step 3: Install Python dependencies into a virtual environment
`poetry install`

## The metadata file

The ```metadata.csv``` file should contain a table where each row corresponds to a problem instance, and each column must strictly follow the naming convention mentioned below:

-	**instances** instance identifier - We expect the instance identifier to be of type "String". This column is mandatory.
-	**source** instance source - This column is optional
-	**feature_name** The keyword "feature_" concatenated with feature name. For instance, if the feature name is "density", the header name should be mentioned as "feature_density". If the name consists of more than one word, each word should be separated by "_" (spaces are not allowed). There must be more than two features for the software to work. We expect the features to be of the type "Double".
-	**algo_name** The keyword "algo_" concatenated with algorithm name. For instance, if the algorithm name is "Greedy", the column header should be "algo_greedy". If the name consists of more than one word, each word should be separated by "_" (spaces are not allowed). You can add the performance of more than one algorithm in the same ```.csv```. We expect the algorithm performance to be of the type "Double".

Moreover, empty cells, NaN or null values are allowed but **not recommended**. We'd like for you to handle missing values in your data before processing. You may use [this file](https://matilda.unimelb.edu.au/matilda/matildadata/graph_coloring_problem/metadata/metadata.csv) as a reference.

## Options

The ```options.json``` contains a structure that contains all the settings used by the code. Broadly, there are settings required for the analysis itself, settings for data pre-processing, and output settings. These are divided into general, dimensionality reduction, bound estimation, algorithm selection and footprint construction settings. Additionally, the toolkit includes routines for bounding outliers, scaling the data, and selecting features.

### General settings

-	```opts.perf.MaxPerf``` determines whether the algorithm performance values provided are **efficiency** measures that should be maximised (set as ```TRUE```), or **cost** measures that should be minimised (set as ```FALSE```).
-	```opts.perf.AbsPerf``` determines whether good performance is defined absolutely, e.g., misclassification error is lower than a 20%, (set as ```TRUE```), or if it is defined relatively to the best performing algorithm, e.g., misclassification error is within at least 5% of the best algorithm, (set as ```FALSE```).
-	```opts.perf.epsilon``` corresponds to the threshold used to calculate good performance. It must be of the type "Double".
-	```opts.general.betaThreshold``` corresponds to the fraction of algorithms in the portfolio that must have good performance in the instance, for it to be considered an **easy** instance. It must be a value between 0 and 1.
- ```opts.parallel.flag``` determines whether parallel processing will be available (set as ```TRUE```), or not (set as ```FALSE```). The toolkit uses Python's ```multiprocessing``` (in TRACE) and scikit-learn's ```n_jobs``` (in PYTHIA) to distribute work across local cores.
- ```opts.parallel.ncores``` number of available cores for parallel procesing.
-	```opts.selvars.smallscaleflag``` by setting this flag as ```TRUE```, you can carry out a small scale experiment using a randomly selected fraction of the original data. This is useful if you have a large dataset with more than 1000 instances, and you want to explore the parameters of the model.
-	```opts.selvars.smallscale``` fraction taken from the original data on the small scale experiment.
-	```opts.selvars.fileidxflag``` by setting this flag as ```TRUE```, you can carry out a small scale experiment. This time you must provide a ```.csv``` file that contains in one column the indices of the instances to be taken. This may be useful if you want to make a more controlled experiment than just randomly selecting instances.
-	```opts.selvars.fileidx``` name of the file containing the indexes of the instances.

### Dimensionality reduction settings

The toolkit uses PILOT as a dimensionality reduction method, with [BFGS](https://en.wikipedia.org/wiki/Broyden-Fletcher-Goldfarb-Shanno_algorithm) as numerical solver. Technical details about it can be found [here](https://doi.org/10.1007/s10994-017-5629-5).

-	```opts.pilot.analytic``` determines whether the analytic (set as ```TRUE```) or the numerical (set as ```FALSE```) solution to the dimensionality reduction problem should be used. We recommend to leave this setting as ```FALSE```, due to the instability of the analytical solution due to possible poor-conditioning.
-	```opts.pilot.ntries``` number of iterations that the numerical solution is attempted.

### Empirical bound estimation settings.

The toolkit uses CLOISTER, an algorithm based on correlation to detect the empirical bounds of the Instance Space.

- ```opts.cloister.cthres``` Determines the maximum [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) that would indicate non-correlated variables. The lower this value is, the more stringent is the algorithm; hence, it would be less likely to produce a good bound.
- ```opts.cloister.pval``` Determines the p-value of the Pearson correlation coefficient that indicates no correlation.

###  Algorithm selection settings

The toolkit trains one [scikit-learn](https://scikit-learn.org/) `SVC` per algorithm as the algorithm selection model.

- ```opts.pythia.cv_folds``` number of folds of the stratified cross-validation (CV) experiment used during hyper-parameter tuning.
- ```opts.pythia.is_poly_krnl``` determines whether to use a polynomial (set as ```TRUE```) or Gaussian/RBF (set as ```FALSE```, the default) kernel. The RBF kernel is usually significantly faster to calculate and more accurate; however, it also has the disadvantage of producing discontinuous areas of good performance which may look overfitted. We tend to recommend a polynomial kernel if the dataset is higher than 1000 instances.
- ```opts.pythia.use_grid_search``` selects the hyper-parameter tuning strategy: a Sobol-sampled grid search (set as ```TRUE```) or Bayesian optimization via [scikit-optimize](https://scikit-optimize.github.io/)'s `BayesSearchCV` (set as ```FALSE```, the default), both tuning the SVM's box constraint and kernel scale with stratified CV.
- ```opts.pythia.use_weights``` determines whether weighted (set as ```TRUE```) or unweighted (set as ```FALSE```, the default) classification is performed. The weights are calculated as <img src="https://render.githubusercontent.com/render/math?math=\left|y-\bar{y}\right|">, i.e. each instance's absolute deviation from the algorithm's mean performance.
- ```opts.pythia.uselibsvm``` **(legacy)** accepted for backward compatibility with option files from the MATLAB toolkit; treated as an alias for ```opts.pythia.use_grid_search``` and does not select LIBSVM, which this implementation does not use.

### Footprint construction settings

The toolkit uses TRACE, an algorithm based on [```shapely```](https://shapely.readthedocs.io/) polygons to define the regions in the space where we statistically infer good algorithm performance. The polygons are then pruned to remove those sections for which the evidence, as defined by a minimum purity value, is poor or non-existing.

-	```opts.trace.usesim``` makes use of the actual (set as ```FALSE```) or simulated data from the SVM results (set as ```TRUE```) to produce the footprints.
-	```opts.trace.PI``` minimum purity required for a section of a footprint.

### Automatic data bounding and scaling

The toolkit implements simple routines to bound outliers and scale the data. **These routines are by no means perfect, and users should pre-process their data independently if preferred**. However, the automatic bounding and scaling routines should give some idea of the kind of results may be achieved. In general, we recommend that the data is transformed to become **close to normally distributed** due to the linear nature of PILOT's optimal projection algorithm.

- ```opts.auto.preproc``` turns on (set as ```TRUE```) the automatic pre-processing.
- ```opts.bound.flag``` turns on (set as ```TRUE```) data bounding. This sub-routine calculates the median and the interquartile range ([IQR](https://en.wikipedia.org/wiki/Interquartile_range)) of each feature and performance measure, and bounds the data to the median plus or minus five times the IQR.
- ```opts.norm.flag``` turns on (set as ```TRUE```) scalling. This sub-routine scales into a positive range each feature and performance measure. Then it calculates a [box-cox transformation](https://en.wikipedia.org/wiki/Power_transform#Box%E2%80%93Cox_transformation) to stabilise the variance, and a [Z-transformation](https://en.wikipedia.org/wiki/Standard_score) to standarise the data. The result are features and performance measures that are close to normally distributed.

### Automatic feature selection

The toolkit implements SIFTED, a routine to select features, given their cross-correlation and correlation to performance. Ideally, we want the smallest number of orthogonal and predictive features. **This routine are by no means perfect, and users should pre-process their data independently if preferred**.  In general, we recommend **using no more than 10 features** as input to PILOT's optimal projection algorithm, due to the numerical nature of its solution and issues in identifying meaningful linear trends.

- ```opts.sifted.flag``` turns on (set as ```TRUE```) the automatic feature selection. SIFTED is composed of two sub-processes. On the first one, SIFTED calculates the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) between the features and the performance. Then it takes its absolute value, and sorts them from largest to smallest. Then, it takes all features that have a correlation above the threshold. It automatically bounds itself to a minimum of 3 features. Then, SIFTED uses the [Pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) as a dissimilarity metric between features. Then, [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) is used to identify groups of similar features. To select one feature per group, the algorithm first projects the subset of selected featurs into two dimensions using Principal Components Analysis ([PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)) and then [Random Forests](https://en.wikipedia.org/wiki/Random_forest) to predict whether an instance is easy or not for a given algorithm. Then, the subset of features that gives the most accurate models is selected. This section of the routine is **potentially computationally very expensive** due to the multiple-layer training process. However, it is our current recommended approach to select the most relevant features. This routine tests all possible combinations if they are less than 1000, or uses the combination of a [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) and a Look-up table otherwise.
- ```opts.sifted.rho``` correlation threshold indicating the lowest acceptable absolute correlation between a feature and performance. It should be a value between 0 and 1.
- ```opts.sifted.K``` number of clusters which corresponds to the final number of features returned. The routine assumes at least 3 clusters and no more than the number of features. Ideally, it **should not** be a value larger than 10.
- ```opts.sifted.NTREES``` number of threes used by the Random Forest models. Typically, this setting does not require adjustment.
- ```opts.sifted.MaxIter``` number of iterations used to converge the k-means algorithm. Typically, this setting does not require adjustment.
- ```opts.sifted.Replicates``` number of repeats carried out of the k-means algorithm. Typically, this setting does not require adjustment.

### Output settings

These settings result in more information being stored in files or presented in the console output.

- ```opts.outputs.csv``` This flag produces the output CSV files for post-processing and analysis. It is recommended to leave this setting as ```TRUE```.
- ```opts.outputs.png``` This flag produces the output figures files for post-processing and analysis. It is recommended to leave this setting as ```TRUE```.
- ```opts.outputs.web``` This flag produces the output files employed to draw the figures in MATILDA's web tools (click [here](https://matilda.unimelb.edu.au/matilda/newuser) to open an account). It is recommended to leave this setting as ```FALSE```.

## Contact

If you have any suggestions or ideas (e.g. for new features), or if you encounter any problems while running the code, please use this repository's own [issue tracker](https://github.com/andremun/pyInstanceSpace/issues) or contact us through the MATILDA's [Queries and Feedback](http://matilda.unimelb.edu.au/matilda/contact-us) page.

## Acknowledgements

Funding for the development of this code was provided by:

- The University of Melbourne, through grant 2025DYA013.
- The Australian Research Council, through the ARC Industrial Transformation Training Centre in Optimisation Technologies, Integrated Methodologies, and Applications (OPTIMA; grant No. IC200100009).

This code was partly developed as part of the subject SWEN90017-18 by students Junheng Chen, Yusuf Berdan Guzel, Kushagra Khare, Dong Hyeog Jang, Kian Dsouza, Nathan Harvey, Tao Yu, Xin Xiang, Jiaying Yi, and Cheng Ze Lam. Ben Golding mentored the team, and Mansooreh Zahedi coordinated the subject.
