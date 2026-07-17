"""Validation test for TRACE stage against MATLAB reference data.

Loads MATLAB-trained footprint polygons (training_artifacts/trace/good_<algo>.csv,
best_<algo>.csv) together with the MATLAB-projected 2D test coordinates
(explore_outputs/step3_after_pilot.csv) and verifies that _explore_trace
reproduces the boolean membership matrix in step5_trace_membership.csv.

Scope note. step5's ``in_space`` column is sourced from CLOISTER's
``model.cloist.Zecorr`` via the extract script's ``inpolygon`` call, not
from exploreIS. exploreIS does not compute per-instance in_space, and
CLOISTER training is out of scope for this port. The validation here covers
the ``in_good_*`` and ``in_best_*`` columns only.

Threshold: per-column boolean agreement >= 99%. MATLAB ``inpolygon``
treats boundary points as inside; ``shapely.Polygon.covers`` matches that
semantics. The 1% tolerance allows for floating-point boundary edge cases
after CSV round-trip.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon

from instancespace.data.model import TraceOut
from instancespace.instance_space import InstanceSpace

REFERENCE_DIR = Path("tests/matlab_reference")
ARTIFACTS_DIR = REFERENCE_DIR / "training_artifacts" / "trace"
OUTPUTS_DIR = REFERENCE_DIR / "explore_outputs"

ALGO_ORDER = [
    "NB", "LDA", "QDA", "CART", "J48",
    "KNN", "L_SVM", "poly_SVM", "RBF_SVM", "RandF",
]


def load_polygon(path: Path):
    """Reconstruct a shapely (Multi)Polygon from a MATLAB polyshape CSV.

    MATLAB exports multi-region polyshapes as a single CSV with NaN rows
    delimiting the regions; rebuild each region as a Polygon and combine.
    """
    if not path.exists():
        return None
    df = pd.read_csv(path)
    vertices = df[["x", "y"]].to_numpy(dtype=np.double)
    regions = []
    current = []
    for row in vertices:
        if np.isnan(row).any():
            if len(current) >= 3:
                regions.append(Polygon(current))
            current = []
        else:
            current.append(tuple(row))
    if len(current) >= 3:
        regions.append(Polygon(current))
    if not regions:
        return None
    return regions[0] if len(regions) == 1 else MultiPolygon(regions)


def build_trace_from_artifacts():
    good_polys = [load_polygon(ARTIFACTS_DIR / f"good_{a}.csv") for a in ALGO_ORDER]
    best_polys = [load_polygon(ARTIFACTS_DIR / f"best_{a}.csv") for a in ALGO_ORDER]
    trace = Mock(spec=TraceOut)
    trace.good = [SimpleNamespace(polygon=p) for p in good_polys]
    trace.best = [SimpleNamespace(polygon=p) for p in best_polys]
    return trace


def test_trace_matches_matlab():
    """Per-column boolean agreement >= 99% on all in_good_*/in_best_* columns."""
    trace = build_trace_from_artifacts()

    instance_space = Mock(spec=InstanceSpace)
    instance_space._explore_model = Mock()
    instance_space._explore_model.trace = trace

    z = pd.read_csv(OUTPUTS_DIR / "step3_after_pilot.csv", index_col=0)
    in_good, in_best = InstanceSpace._explore_trace(instance_space, z.to_numpy())

    ref = pd.read_csv(OUTPUTS_DIR / "step5_trace_membership.csv", index_col=0)

    per_col_agreement = {}
    for j, algo in enumerate(ALGO_ORDER):
        ref_good = ref[f"in_good_{algo}"].to_numpy(dtype=np.bool_)
        ref_best = ref[f"in_best_{algo}"].to_numpy(dtype=np.bool_)
        per_col_agreement[f"in_good_{algo}"] = (in_good[:, j] == ref_good).mean()
        per_col_agreement[f"in_best_{algo}"] = (in_best[:, j] == ref_best).mean()

    overall = float(np.mean(list(per_col_agreement.values())))

    print(f"\nInput:    {z.shape[0]} instances x 2 coordinates")
    print(f"Algorithms: {len(ALGO_ORDER)}")
    for col, agr in per_col_agreement.items():
        print(f"  {col:20s}: {agr * 100:6.2f}%")
    print(f"Overall mean agreement: {overall * 100:.2f}%")

    for col, agr in per_col_agreement.items():
        assert agr >= 0.99, f"{col} agreement {agr * 100:.2f}% < 99%"
    print(f"[PASS] TRACE validation: overall {overall * 100:.2f}% across "
          f"{len(per_col_agreement)} columns")
