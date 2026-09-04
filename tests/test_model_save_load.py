# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
# Copyright (c) 2024-2026 Mario Andrés Muñoz
"""Tests for Model.save()/Model.load() (F7): signed/unsigned pickle round-trip.

Builds a small but real Model (real Data/PrelimOut/.../PythiaOut/TraceOut, a
genuinely fitted SVC, a genuine shapely Polygon) rather than mocking, since
F7's whole point is that these objects round-trip natively through
`joblib`/`pickle` - a Mock wouldn't exercise that at all (and mocks generally
aren't picklable in the first place).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Polygon
from sklearn.svm import SVC

from instancespace.data.model import (
    CloisterOut,
    Data,
    DataDense,
    FeatSel,
    Footprint,
    PilotOut,
    PrelimOut,
    PythiaOut,
    SiftedOut,
    TraceOut,
)
from instancespace.data.options import InstanceSpaceOptions
from instancespace.model import Model, ModelSignatureError


def _build_minimal_model() -> Model:
    n_inst, n_feat, n_algo = 6, 2, 1

    x = np.random.default_rng(0).normal(size=(n_inst, n_feat))
    y_class = np.array([0, 1, 0, 1, 0, 1])
    svc = SVC(probability=True, random_state=0)
    svc.fit(x, y_class)

    data = Data(
        inst_labels=pd.Series([f"i{i}" for i in range(n_inst)]),
        feat_labels=[f"feature_{i}" for i in range(n_feat)],
        algo_labels=[f"algo_{i}" for i in range(n_algo)],
        x=x,
        y=np.random.default_rng(1).normal(size=(n_inst, n_algo)),
        x_raw=x.copy(),
        y_raw=np.random.default_rng(2).normal(size=(n_inst, n_algo)),
        y_bin=np.zeros((n_inst, n_algo), dtype=np.bool_),
        y_best=np.zeros(n_inst, dtype=np.double),
        p=np.zeros(n_inst, dtype=np.int_),
        num_good_algos=np.ones(n_inst, dtype=np.double),
        beta=np.zeros(n_inst, dtype=np.bool_),
        s=None,
    )
    data_dense = DataDense(
        inst_labels=data.inst_labels.iloc[:4],
        x=data.x[:4] + 10.0,
        y=data.y[:4],
        x_raw=data.x_raw[:4],
        y_raw=data.y_raw[:4],
        y_bin=data.y_bin[:4],
        y_best=data.y_best[:4],
        p=data.p[:4],
        num_good_algos=data.num_good_algos[:4],
        beta=data.beta[:4],
        s=None,
    )

    prelim = PrelimOut(
        med_val=np.zeros(n_feat),
        iq_range=np.ones(n_feat),
        hi_bound=np.ones(n_feat) * 10,
        lo_bound=np.zeros(n_feat),
        min_x=np.zeros(n_feat),
        lambda_x=np.ones(n_feat),
        mu_x=np.zeros(n_feat),
        sigma_x=np.ones(n_feat),
        min_y=0.0,
        lambda_y=np.ones(n_algo),
        sigma_y=np.ones(n_algo),
        mu_y=np.zeros(n_algo),
    )

    sifted = SiftedOut(
        selvars=np.array([0, 1], dtype=np.intc),
        rho=None,
        pval=None,
        silhouette_scores=None,
        clust=None,
    )

    pilot = PilotOut(
        X0=None,
        alpha=None,
        eoptim=None,
        perf=None,
        a=np.eye(2, n_feat),
        z=x[:, :2],
        c=np.zeros((n_feat, 2)),
        b=np.zeros((n_feat, 2)),
        error=np.zeros(1),
        r2=np.ones(1),
        summary=pd.DataFrame({"a": [1]}),
    )

    cloister = CloisterOut(
        z_edge=np.zeros((4, 2)),
        z_ecorr=np.zeros((4, 2)),
    )

    pythia = PythiaOut(
        mu=[0.0, 0.0],
        sigma=[1.0, 1.0],
        cp=None,
        svm=[svc],
        cvcmat=np.zeros((2, 2)),
        y_sub=np.zeros((n_inst, n_algo), dtype=np.bool_),
        y_hat=np.zeros((n_inst, n_algo), dtype=np.bool_),
        pr0_sub=np.zeros((n_inst, n_algo)),
        pr0_hat=np.zeros((n_inst, n_algo)),
        box_consnt=[1.0],
        k_scale=[1.0],
        precision=[1.0],
        recall=[1.0],
        accuracy=[1.0],
        selection0=np.zeros(n_inst, dtype=np.int_),
        selection1=None,
        summary=pd.DataFrame({"algo": ["algo_0"]}),
    )

    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    footprint = Footprint(polygon, 1.0, 4, 2, 0.5, 0.5)
    trace = TraceOut(
        space=footprint,
        good=[footprint],
        best=[footprint],
        hard=footprint,
        summary=pd.DataFrame({"algo": ["algo_0"]}),
    )

    opts = InstanceSpaceOptions.default(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    return Model(
        data=data,
        data_dense=data_dense,
        feat_sel=FeatSel(idx=np.array([0, 1], dtype=np.intc)),
        prelim=prelim,
        sifted=sifted,
        pilot=pilot,
        cloister=cloister,
        pythia=pythia,
        trace=trace,
        opts=opts,
    )


def _assert_models_equal(original: Model, loaded: Model) -> None:
    assert np.array_equal(original.data.x, loaded.data.x)
    assert original.data.inst_labels.equals(loaded.data.inst_labels)
    assert original.data.feat_labels == loaded.data.feat_labels
    assert np.array_equal(original.feat_sel.idx, loaded.feat_sel.idx)
    assert np.array_equal(original.sifted.selvars, loaded.sifted.selvars)
    assert original.opts == loaded.opts
    assert isinstance(original.data_dense, DataDense)
    assert isinstance(loaded.data_dense, DataDense)
    np.testing.assert_array_equal(original.data_dense.x, loaded.data_dense.x)

    original_svc = original.pythia.svm[0]
    loaded_svc = loaded.pythia.svm[0]
    assert type(loaded_svc) is type(original_svc)
    probe = np.random.default_rng(42).normal(size=(5, 2))
    assert np.array_equal(original_svc.predict(probe), loaded_svc.predict(probe))
    assert np.allclose(
        original_svc.predict_proba(probe),
        loaded_svc.predict_proba(probe),
    )

    original_polygon = original.trace.space.polygon
    loaded_polygon = loaded.trace.space.polygon
    assert original_polygon is not None
    assert loaded_polygon is not None
    assert original_polygon.equals(loaded_polygon)


def _stage_output_for_model(
    model: Model,
    data_dense: DataDense | None,
) -> dict[str, object]:
    """Recreate the flat StageRunner payload consumed by Model."""
    output: dict[str, object] = dict(vars(model.data))
    output["data_dense"] = data_dense
    output.update(vars(model.prelim))
    output.update(vars(model.sifted))
    output.update({k: v for k, v in vars(model.pilot).items() if k != "summary"})
    output["pilot_summary"] = model.pilot.summary
    output.update(vars(model.cloister))
    output.update({k: v for k, v in vars(model.pythia).items() if k != "summary"})
    output["pythia_summary"] = model.pythia.summary
    output.update({k: v for k, v in vars(model.trace).items() if k != "summary"})
    output["trace_summary"] = model.trace.summary
    return output


def test_from_stage_output_preserves_data_dense() -> None:
    """Model construction keeps PRELIM's actual dense dataset, not final Data."""
    source = _build_minimal_model()
    dense = source.data_dense
    assert isinstance(dense, DataDense)

    built = Model.from_stage_runner_output(
        _stage_output_for_model(source, dense),
        source.opts,
    )

    assert built.data_dense is dense
    assert not np.array_equal(built.data_dense.x, built.data.x)


def test_from_stage_output_preserves_absent_data_dense() -> None:
    """No density subset remains None in the public Model."""
    source = _build_minimal_model()

    built = Model.from_stage_runner_output(
        _stage_output_for_model(source, None),
        source.opts,
    )

    assert built.data_dense is None


def test_round_trip_unsigned(tmp_path: Path) -> None:
    """An unsigned model round-trips without creating a signature."""
    model = _build_minimal_model()
    path = tmp_path / "model.joblib"

    model.save(path)
    loaded = Model.load(path)

    _assert_models_equal(model, loaded)
    assert not path.with_name(path.name + ".sig").exists()


def test_round_trip_signed(tmp_path: Path) -> None:
    """A signed model round-trips when the verification key matches."""
    model = _build_minimal_model()
    path = tmp_path / "model.joblib"
    secret_key = b"a-server-managed-secret"

    model.save(path, secret_key=secret_key)
    loaded = Model.load(path, secret_key=secret_key)

    _assert_models_equal(model, loaded)
    assert path.with_name(path.name + ".sig").exists()


def test_load_with_wrong_key_raises(tmp_path: Path) -> None:
    """Loading a signed model with another key fails verification."""
    model = _build_minimal_model()
    path = tmp_path / "model.joblib"

    model.save(path, secret_key=b"correct-key")

    with pytest.raises(ModelSignatureError):
        Model.load(path, secret_key=b"wrong-key")


def test_signature_tampering_refuses_before_deserialising(tmp_path: Path) -> None:
    """Payload tampering is detected before joblib deserialisation."""
    model = _build_minimal_model()
    path = tmp_path / "model.joblib"
    secret_key = b"a-server-managed-secret"
    model.save(path, secret_key=secret_key)

    # Flip one byte in the serialised payload itself.
    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF
    path.write_bytes(bytes(data))

    with pytest.raises(ModelSignatureError):
        Model.load(path, secret_key=secret_key)


def test_downgrade_attack_is_refused(tmp_path: Path) -> None:
    """A signed save() must not become loadable-unverified via secret_key=None."""
    model = _build_minimal_model()
    path = tmp_path / "model.joblib"
    model.save(path, secret_key=b"a-server-managed-secret")

    with pytest.raises(ModelSignatureError):
        Model.load(path, secret_key=None)


def test_signed_key_but_no_signature_file_is_refused(tmp_path: Path) -> None:
    """A verification key requires an accompanying signature file."""
    model = _build_minimal_model()
    path = tmp_path / "model.joblib"
    model.save(path)  # unsigned - writes no .sig

    with pytest.raises(ModelSignatureError):
        Model.load(path, secret_key=b"some-key")


def test_resaving_unsigned_removes_a_stale_signature(tmp_path: Path) -> None:
    """An unsigned resave removes a signature that no longer applies."""
    model = _build_minimal_model()
    path = tmp_path / "model.joblib"
    model.save(path, secret_key=b"a-server-managed-secret")
    assert path.with_name(path.name + ".sig").exists()

    model.save(path)  # re-save unsigned to the same path

    assert not path.with_name(path.name + ".sig").exists()
    loaded = Model.load(path)  # must not raise the downgrade-attack guard
    _assert_models_equal(model, loaded)
