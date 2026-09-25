# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Contains modules for instance space analysis.

The package builds an instance space from instance meta-data (features and algorithm
performance) and uses it to predict algorithm performance and to find algorithm
footprints. The stages run in this order:

- preprocessing: Filter the meta-data and remove instances or features with too many
  missing values.
- prelim: Preparation for Learning of Instance Meta-Data. Set a binary measure of
  "good" performance, then bound and scale the meta-data.
- sifted: Selection of Instance Features to Explain Difficulty. Select a subset of
  features that correlate with algorithm performance and are not redundant.
- pilot: Projecting Instances with Linearly Observable Trends. Project the instances
  from the feature space to a 2D (or 3D) instance space with linear trends in the
  features and in algorithm performance.
- pythia: Train one classifier per algorithm on the instance space to predict good
  performance, and recommend an algorithm for each instance.
- cloister: Correlated Limits of the Instance Space's Theoretical or Experimental
  Regions. Estimate the boundary of the instance space from the feature bounds and
  the correlations between features.
- trace: Triangulation with Removal of Areas with Contradicting Evidence. Find the
  regions of the instance space (footprints) where each algorithm performs well.

`InstanceSpace.build()` trains all stages on a data set. `InstanceSpace.explore()`
projects new instances into a trained instance space.

Reference: K. Smith-Miles and M. A. Muñoz, "Instance Space Analysis for Algorithm
Testing: Methodology and Software Tools", ACM Comput. Surv. 55(12), 2023.
"""

from . import data, instance_space, progress_reporter, stages
from .data import metadata, options
from .data.metadata import Metadata
from .data.model import ExploreResult
from .data.options import InstanceSpaceOptions
from .instance_space import InstanceSpace
from .model import Model
from .progress_reporter import (
    CompositeProgressReporter,
    FileProgressReporter,
    HttpProgressReporter,
    NullProgressReporter,
    ProgressReporter,
)

__all__ = [
    "CompositeProgressReporter",
    "ExploreResult",
    "FileProgressReporter",
    "HttpProgressReporter",
    "InstanceSpace",
    "InstanceSpaceOptions",
    "Metadata",
    "Model",
    "NullProgressReporter",
    "ProgressReporter",
    "options",
    "metadata",
    "data",
    "stages",
    "instance_space",
    "stage_runner",
    "progress_reporter",
]
