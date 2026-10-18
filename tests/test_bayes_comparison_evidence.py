# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2026 Mario Andrés Muñoz
"""Verify research evidence integrity without rerunning stochastic optimization."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

EVIDENCE = Path(__file__).parents[1] / "docs/research/bayes-304"


def test_evidence_checksums_and_recorded_traces() -> None:
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == expected
    inputs = json.loads((EVIDENCE / "inputs.json").read_text())
    assert len(inputs["cases"]) == 12
    assert len(inputs["x"]) == 212
    assert inputs["seed_points"] == 4
    for backend in ("python", "matlab"):
        total = 0
        for case in inputs["cases"]:
            trace = json.loads(
                (EVIDENCE / backend / f"{case['id']}.json").read_text(),
            )["trace"]
            assert len(trace) == case["budget"]
            assert sorted(set(case["folds"])) == list(range(5))
            objectives = np.asarray([point["cv_error"] for point in trace])
            assert np.all(np.isfinite(objectives))
            assert np.all((objectives >= 0) & (objectives <= 1))
            np.testing.assert_array_equal(
                np.minimum.accumulate(objectives),
                [p["running_best"] for p in trace],
            )
            np.testing.assert_allclose(
                [[p["box_constraint"], p["kernel_scale"]] for p in trace[:4]],
                case["initial"],
                rtol=1e-14,
                atol=0,
            )
            total += len(trace)
            if case["budget"] == 20:
                name = case["id"].replace("-b20", "-b40")
                longer = EVIDENCE / backend / f"{name}.json"
                assert trace == json.loads(longer.read_text())["trace"][:20]
        assert total == 360


def test_cross_scores_recompute_from_fold_counts() -> None:
    records = json.loads((EVIDENCE / "cross-results.json").read_text())
    manifest = json.loads((EVIDENCE / "manifest.json").read_text())
    assert len(records) == 78
    differences = []
    for record in records:
        for backend in ("python", "matlab"):
            score = record[backend]
            assert len(score["mistakes"]) == len(score["sizes"]) == 5
            assert sum(score["sizes"]) == 212
            assert score["global_error"] == pytest.approx(
                sum(score["mistakes"]) / sum(score["sizes"]),
            )
            assert score["fold_mean_error"] == pytest.approx(
                np.mean(np.asarray(score["mistakes"]) / score["sizes"]),
            )
        differences.append(
            abs(
                record["python"]["global_error"] - record["matlab"]["global_error"],
            ),
        )
    assert max(differences) == manifest["max_global_error_difference"]
    assert np.mean(differences) == pytest.approx(
        manifest["mean_global_error_difference"],
    )
