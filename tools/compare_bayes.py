# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Mario Andrés Muñoz
# ruff: noqa: SLF001
# The benchmark intentionally exercises the production private search boundaries.
"""Record equal-budget, shared-fold Python/MATLAB Bayesian SVM experiments.

This is maintainer tooling. It never changes runtime defaults or approved fixtures.
MATLAB evaluators are copied verbatim from the independently pinned source tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import PredefinedSplit
from skopt import BayesSearchCV
from skopt.sampler import InitialPointGenerator

from instancespace.stages.pythia import _BAYES_OPTIMIZER_KWARGS, PythiaStage
from instancespace.utils.get_classifier_fcn import get_classifier_fcn

ROOT = Path(__file__).resolve().parents[1]
MATLAB_COMMIT = "5f8cb25682202a8243bd2a2481ab8feb40c8c4ce"
SEEDS = (7, 42, 101)
BUDGETS = (20, 40)
FOLDS = 5
PARTITION_SHA256 = "8df5b86c2691f8451719e7590451183e319bb658345994a32c2de8a7d3a1aff4"
MATLAB_FILES = {
    "core/PYTHIA.m": "88d5388d962a13a5d7f86e3598ca8d5b8a87fe5bc5ff2cc22ba204454ad29b3e",
    "utils/+isacompat/Partition.m": PARTITION_SHA256,
}


def read_json(path: Path) -> Any:  # noqa: ANN401
    """Read one experiment record."""
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    """Write finite, reproducible JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class FixedStarts(InitialPointGenerator):  # type: ignore[misc]
    """Use the same four physical SVM starting points in both optimizers."""

    def __init__(self, points: list[list[float]]) -> None:
        """Keep estimator-unit points in BayesSearchCV's sorted key order."""
        self.points = points

    def generate(
        self,
        dimensions: object,  # noqa: ARG002
        n_samples: int,
        random_state: object = None,  # noqa: ARG002
    ) -> list[list[float]]:
        """Supply exactly the declared seed design."""
        if n_samples != len(self.points):
            raise ValueError("The comparison requires the declared four seed points.")
        return self.points


def prepare(output: Path, matlab: Path) -> None:
    """Freeze current-reference coordinates, labels, folds and common starts."""
    from sklearn.model_selection import StratifiedKFold

    if (output / "inputs.json").exists():
        raise FileExistsError("Use a new output directory to preserve frozen evidence.")
    for name, expected in MATLAB_FILES.items():
        actual = hashlib.sha256((matlab / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"MATLAB source does not match {MATLAB_COMMIT}: {name}")
    output.mkdir(parents=True, exist_ok=True)
    source = matlab / "core/PYTHIA.m"
    text = source.read_text()
    suffix = text[text.index("function err = bayesObjective(") :]
    inputs = ROOT / "tests/fixtures/matlab/current/build_data/pythia/legacy_svm/inputs"
    z = pd.read_csv(inputs / "z.csv", index_col=0).to_numpy(dtype=float)
    labels = pd.read_csv(inputs / "y_bin.csv", index_col=0)
    truth = labels.to_numpy(dtype=bool)
    _, _, normalized = PythiaStage._compute_znorm(z)
    selected = np.argsort(np.abs(truth.mean(axis=0) - 0.5))[:2]
    cases: list[dict[str, object]] = []
    for algorithm_index in selected:
        y = truth[:, algorithm_index]
        for seed in SEEDS:
            fold_ids = np.zeros(len(y), dtype=int)
            splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
            for fold, (_, test) in enumerate(splitter.split(z, y)):
                fold_ids[test] = fold
            starts = np.exp2(np.random.RandomState(seed).uniform(-10, 4, size=(4, 2)))
            for budget in BUDGETS:
                cases.append(
                    {
                        "id": f"a{int(algorithm_index)}-s{seed}-b{budget}",
                        "algorithm": str(labels.columns[algorithm_index]),
                        "algorithm_index": int(algorithm_index),
                        "seed": seed,
                        "budget": budget,
                        "folds": fold_ids.tolist(),
                        "initial": starts.tolist(),
                    },
                )
    record = {
        "schema": "pyinstancespace.bayes-comparison/v1",
        "matlab_commit": MATLAB_COMMIT,
        "matlab_pythia_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "matlab_helpers_sha256": hashlib.sha256(suffix.encode()).hexdigest(),
        "reference_manifest_sha256": hashlib.sha256(
            (ROOT / "tests/fixtures/matlab/current/manifest.json").read_bytes(),
        ).hexdigest(),
        "source_files": {
            name: hashlib.sha256((inputs / name).read_bytes()).hexdigest()
            for name in ("z.csv", "y_bin.csv")
        },
        "normalization": (
            "Python production PYTHIA normalization, frozen for both backends"
        ),
        "kernel": "gaussian/rbf",
        "weights": "uniform",
        "seed_points": 4,
        "fold_count": FOLDS,
        "x": normalized.tolist(),
        "y": truth.astype(int).tolist(),
        "cases": cases,
    }
    write_json(output / "inputs.json", record)
    driver = (ROOT / "tools/matlab_bayes_comparison.m").read_text()
    (output / "pyis_bayes_comparison.m").write_text(driver + "\n" + suffix)
    print(
        f"Prepared {len(cases)} searches; {len(z)} rows; "
        f"algorithms {selected.tolist()}.",
    )


def python_runs(output: Path) -> None:
    """Run the real Python search implementation with common initial points."""
    import platform
    import warnings

    import scipy
    import sklearn
    import skopt

    record = read_json(output / "inputs.json")
    x = np.asarray(record["x"], dtype=float)
    all_y = np.asarray(record["y"], dtype=bool)
    spec = get_classifier_fcn("svm")
    space = PythiaStage._bayes_param_space(spec)
    for case in record["cases"]:
        destination = output / "python" / f"{case['id']}.json"
        if destination.exists():
            raise FileExistsError(f"Preserving existing evidence: {destination}")
        y = all_y[:, case["algorithm_index"]]
        initial = [[point[0], 1 / point[1] ** 2] for point in case["initial"]]
        search = BayesSearchCV(
            estimator=spec.build(case["seed"], False),
            search_spaces=space,
            n_iter=case["budget"],
            optimizer_kwargs={
                **_BAYES_OPTIMIZER_KWARGS,
                "initial_point_generator": FixedStarts(initial),
            },
            cv=PredefinedSplit(case["folds"]),
            random_state=case["seed"],
            n_jobs=1,
            error_score="raise",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", FitFailedWarning)
            search.fit(x, y)
        params = search.cv_results_["params"]
        values = 1 - np.asarray(search.cv_results_["mean_test_score"])
        assert len(params) == case["budget"]
        trace = [
            {
                "iteration": i + 1,
                "box_constraint": float(p["C"]),
                "kernel_scale": float(1 / np.sqrt(p["gamma"])),
                "cv_error": float(values[i]),
                "running_best": float(np.min(values[: i + 1])),
            }
            for i, p in enumerate(params)
        ]
        np.testing.assert_allclose(
            [[t["box_constraint"], t["kernel_scale"]] for t in trace[:4]],
            case["initial"],
            rtol=1e-14,
            atol=0,
        )
        write_json(
            destination,
            {
                "case": case["id"],
                "backend": "Python BayesSearchCV, plain EI",
                "objective": "unweighted mean of fold classification error",
                "trace": trace,
                "versions": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scipy": scipy.__version__,
                    "sklearn": sklearn.__version__,
                    "skopt": skopt.__version__,
                },
            },
        )
        print(f"Python {case['id']}: best error {min(values):.6f}", flush=True)


def evaluate_python(
    record: dict[str, Any],
    case: dict[str, Any],
    box: float,
    scale: float,
) -> dict[str, object]:
    """Cross-score one physical parameter pair on the frozen Python folds."""
    spec = get_classifier_fcn("svm")
    estimator = spec.build(case["seed"], False)
    estimator.set_params(C=box, gamma=1 / scale**2)
    x = np.asarray(record["x"], dtype=float)
    y = np.asarray(record["y"], dtype=bool)[:, case["algorithm_index"]]
    fold_ids = np.asarray(case["folds"])
    mistakes = []
    sizes = []
    for fold in range(FOLDS):
        test = fold_ids == fold
        fitted = clone(estimator).fit(x[~test], y[~test])
        mistakes.append(int(np.sum(fitted.predict(x[test]) != y[test])))
        sizes.append(int(np.sum(test)))
    return {
        "mistakes": mistakes,
        "sizes": sizes,
        "global_error": sum(mistakes) / sum(sizes),
        "fold_mean_error": float(np.mean(np.asarray(mistakes) / sizes)),
    }


def prepare_cross(output: Path) -> None:
    """Cross-evaluate both optimizers' winners and an independent fixed grid."""
    if (output / "cross-inputs.json").exists():
        raise FileExistsError("Preserving existing cross-evaluation inputs.")
    record = read_json(output / "inputs.json")
    points: list[dict[str, object]] = []
    for case in record["cases"]:
        for backend in ("python", "matlab"):
            run = read_json(output / backend / f"{case['id']}.json")
            best = min(run["trace"], key=lambda t: t["cv_error"])
            points.append(
                {
                    "case": case["id"],
                    "origin": backend,
                    "box_constraint": best["box_constraint"],
                    "kernel_scale": best["kernel_scale"],
                },
            )
        if case["budget"] == BUDGETS[0]:
            for box in (2**-7, 2**-1, 2**3):
                for scale in (2**-7, 2**-1, 2**3):
                    points.append(
                        {
                            "case": case["id"],
                            "origin": "fixed-grid",
                            "box_constraint": box,
                            "kernel_scale": scale,
                        },
                    )
    by_id = {c["id"]: c for c in record["cases"]}
    for i, point in enumerate(points):
        point["id"] = i
        point["python"] = evaluate_python(
            record,
            by_id[point["case"]],
            cast(float, point["box_constraint"]),
            cast(float, point["kernel_scale"]),
        )
    write_json(output / "cross-inputs.json", points)
    print(f"Prepared {len(points)} cross-evaluations.")


def main() -> None:
    """Prepare, run or cross-evaluate the controlled experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "python", "cross"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--matlab-source", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.matlab_source is None:
            parser.error("prepare requires --matlab-source")
        prepare(args.output, args.matlab_source)
    elif args.action == "python":
        python_runs(args.output)
    else:
        prepare_cross(args.output)


if __name__ == "__main__":
    main()
