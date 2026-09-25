# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Contains modules for instance space analysis.

The package builds an instance space from instance meta-data. The meta-data are the
instance features and the algorithm performance. The instance space predicts
algorithm performance and shows the algorithm footprints. The stages run in this
order:

- preprocessing: Filters the meta-data. Removes the instances and features that have
  too many missing values.
- prelim: Preparation for Learning of Instance Meta-Data. Sets a binary measure of
  "good" performance. Then it bounds and scales the meta-data.
- sifted: Selection of Instance Features to Explain Difficulty. Selects the features
  that correlate with algorithm performance and are not redundant.
- pilot: Projecting Instances with Linearly Observable Trends. Projects the instances
  from the feature space to a 2D or 3D instance space. The projection shows linear
  trends in the features and in the algorithm performance.
- pythia: Trains one classifier for each algorithm on the instance space. Each
  classifier predicts good performance. PYTHIA then recommends an algorithm for
  each instance.
- cloister: Correlated Limits of the Instance Space's Theoretical or Experimental
  Regions. Estimates the boundary of the instance space from the feature bounds and
  the correlations between the features.
- trace: Triangulation with Removal of Areas with Contradicting Evidence. Finds the
  regions of the instance space where each algorithm performs well. These regions
  are the footprints.

`InstanceSpace.build()` trains all the stages on a data set. `InstanceSpace.explore()`
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
