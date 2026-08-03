# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Contains default values for options as constants."""

from typing import Literal

DEFAULT_GENERAL_VERBOSE = True
DEFAULT_GENERAL_SEED = 0

DEFAULT_PARALLEL_FLAG = False
DEFAULT_PARALLEL_N_CORES = 2

DEFAULT_PERFORMANCE_MAX_PERF = False
DEFAULT_PERFORMANCE_ABS_PERF = True
DEFAULT_PERFORMANCE_EPSILON = 0.20
DEFAULT_PERFORMANCE_BETA_THRESHOLD = 0.55

DEFAULT_AUTO_PREPROC = True

DEFAULT_BOUND_FLAG = True

DEFAULT_NORM_FLAG = True

DEFAULT_SELVARS_SMALL_SCALE_FLAG = False
DEFAULT_SELVARS_SMALL_SCALE = 0.50
DEFAULT_SELVARS_FILE_IDX_FLAG = False
DEFAULT_SELVARS_FILE_IDX = ""
DEFAULT_SELVARS_DENSITY_FLAG = False
DEFAULT_SELVARS_MIN_DISTANCE = 0.1
DEFAULT_SELVARS_TYPE = "Ftr&Good"

DEFAULT_SIFTED_FLAG = True
DEFAULT_SIFTED_RHO = 0.1
DEFAULT_SIFTED_K = 6
# Significance threshold for the correlation filter, matching MATLAB's
# opts.pval (core/SIFTED.m). Was a hardcoded SiftedStage.PVAL_THRESHOLD
# class constant; now a real option (#300 audit finding, issue 2).
DEFAULT_SIFTED_PVAL = 0.05
# Projection dimensionality the GA fitness function's internal PILOT call
# uses for its KNN neighbour count (kneighbours = dims + 1), matching
# MATLAB's opts.dims (core/SIFTED.m, restricted to {2, 3}). PILOT itself is
# 2D-only in this port (3D projection is F2's unshipped future work), so
# dims=3 is accepted for forward API compatibility but has no effect on
# PILOT's actual output yet - same caveat as CloisterOptions.hull_dims.
DEFAULT_SIFTED_DIMS = 2
DEFAULT_SIFTED_NTREES = 50
DEFAULT_SIFTED_MAX_ITER = 1000
DEFAULT_SIFTED_REPLICATES = 100
DEFAULT_SIFTED_NUM_GENERATION = 100
DEFAULT_SIFTED_NUM_PARENTS_MATING = 2
DEFAULT_SIFTED_SOL_PER_POP = 50
DEFAULT_SIFTED_PARENT_SELECTION_TYPE = "tournament"
DEFAULT_SIFTED_K_TOURNAMENT = 4
DEFAULT_SIFTED_KEEP_ELITISM = 2
DEFAULT_SIFTED_CROSSOVER_TYPE = "scattered"
DEFAULT_SIFTED_CROSSOVER_PROBABILITY = 0.8
DEFAULT_SIFTED_MUTATION_TYPE = "random"
DEFAULT_SIFTED_MUTATION_PROBABILITY = 0.05
DEFAULT_SIFTED_STOP_CRITERIA = "saturate_5"

DEFAULT_PILOT_ANALYTICS = False
DEFAULT_PILOT_N_TRIES = 5
DEFAULT_PILOT_ADJUST_ROTATION = False
# Scalar performance-reconstruction weight (MATLAB's opts.costWeight). 1.0
# weights the performance block the same as the feature block - a no-op that
# reproduces the pre-cost_weight behaviour exactly.
DEFAULT_PILOT_COST_WEIGHT = 1.0

DEFAULT_CLOISTER_P_VAL = 0.05
DEFAULT_CLOISTER_C_THRES = 0.7
DEFAULT_CLOISTER_MAX_FEATURES = 20
# "all" uses every projected dimension for the convex hull (this port's own
# default; scipy.spatial.ConvexHull handles n-D natively). MATLAB always
# builds a 2D hull on the first two projected columns regardless of how
# many columns A has (core/CLOISTER.m) - set to 2 for that behaviour.
# #299 audit finding, issue 5. PILOT's projection is 2D-only in this port
# (3D is F2's unshipped future work), so "all" and 2 are currently
# equivalent in practice - documented, not silently assumed obvious.
DEFAULT_CLOISTER_HULL_DIMS: Literal["all"] = "all"

DEFAULT_PYTHIA_CV_FOLDS = 5
DEFAULT_PYTHIA_IS_POLY_KRNL = False
DEFAULT_PYTHIA_USE_WEIGHTS = False
DEFAULT_PYTHIA_CLASSIFIER = "svm"
DEFAULT_PYTHIA_TUNING = "sobol"
DEFAULT_PYTHIA_N_TUNING_ITER = 20

DEFAULT_TRACE_USE_SIM = True
DEFAULT_TRACE_PURITY = 0.55
DEFAULT_TRACE_METHOD = "legacy"
DEFAULT_TRACE_CONTRA = True

DEFAULT_OUTPUTS_CSV = True
DEFAULT_OUTPUTS_WEB = False
DEFAULT_OUTPUTS_PNG = True
