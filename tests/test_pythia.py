"""Test input parameters, particularly metrics, are accurately parsed and stored."""

from pathlib import Path

import numpy as np
import pandas as pd

from matilda.data.options import PythiaOptions
from matilda.stages.pythia import Pythia

script_dir = Path(__file__).parent
output_dir = script_dir / "test_data/pythia/output"

csv_path_z_input = script_dir / "test_data/pythia/input/Z.csv"
csv_path_y_input = script_dir / "test_data/pythia/input/y.csv"
csv_path_algo_input = script_dir / "test_data/pythia/input/algolabels.csv"
csv_path_y_best_input = script_dir / "tests/pythia/input/ybest.csv"
csv_path_y_bin_input = script_dir / "tests/pythia/input/ybin.csv"


def test_fitlibsvm_gaussian() -> None:
    """Test that the output of the `fitlibsvm_gaussian` function is as expected."""
    pythia_option = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=False,
        use_weights=False,
        use_lib_svm=True,
    )
    atol = 1e-2
    z_input = pd.read_csv(csv_path_z_input)
    y_input = pd.read_csv(csv_path_y_input)
    algo_input = pd.read_csv(csv_path_algo_input)
    y_best_input = pd.read_csv(csv_path_y_best_input)
    y_bin_input = pd.read_csv(csv_path_y_bin_input)
    [pythiaDataChanged, pythiaOut] = Pythia.run(  # noqa: N806
        z_input,
        y_input,
        y_bin_input,
        y_best_input,
        algo_input,
        pythia_option,
    )
    accuracy_output = pd.read_csv(output_dir / "accuracy.csv", header=None).values
    precision_output = pd.read_csv(
        output_dir / "fitlibsvm_gaussian/precision.csv", header=None,
    ).values
    recall_output = pd.read_csv(
        output_dir / "fitlibsvm_gaussian/recall.csv", header=None,
    ).values

    boxcosnt_output = pd.read_csv(
        output_dir / "fitlibsvm_gaussian/boxconst.csv", header=None,
    )
    cvcmat_output = pd.read_csv(
        output_dir / "itlibsvm_gaussian/cvcmat.csv", header=None,
    )
    kscale_output = pd.read_csv(
        output_dir / "fitlibsvm_gaussian/kscale.csv", header=None,
    )
    mu_output = pd.read_csv(output_dir / "fitlibsvm_gaussian/mu.csv", header=None)
    sigma_output = pd.read_csv(output_dir / "fitlibsvm_gaussian/sigma.csv", header=None)
    pr0hat_output = pd.read_csv(
        output_dir / "fitlibsvm_gaussian/pr0hat.csv", header=None,
    )
    pr0sub_output = pd.read_csv(
        output_dir / "fitlibsvm_gaussian/pr0sub.csv", header=None,
    )
    selection0_output = pd.read_csv(
        output_dir / "fitlibsvm_gaussian/selection0.csv", header=None,
    )
    selection1_output = pd.read_csv(
        output_dir / "fitlibsvm_gaussian/selection1.csv", header=None,
    )
    yhat_output = pd.read_csv(output_dir / "fitlibsvm_gaussian/yhat.csv", header=None)
    ysub_output = pd.read_csv(output_dir / "fitlibsvm_gaussian/ysub.csv", header=None)
    assert np.allclose(accuracy_output, pythiaOut.accuracy, atol=atol)
    assert np.allclose(precision_output, pythiaOut.precision, atol=atol)
    assert np.allclose(recall_output, pythiaOut.recall, atol=atol)

    assert np.allclose(boxcosnt_output, pythiaOut.boxconst, atol=atol)
    assert np.allclose(cvcmat_output, pythiaOut.cvcmat, atol=atol)
    assert np.allclose(kscale_output, pythiaOut.kscale, atol=atol)
    assert np.allclose(mu_output, pythiaOut.mu, atol=atol)
    assert np.allclose(sigma_output, pythiaOut.sigma, atol=atol)
    assert np.allclose(pr0hat_output, pythiaOut.pr0hat, atol=atol)
    assert np.allclose(pr0sub_output, pythiaOut.pr0sub, atol=atol)
    assert np.allclose(selection0_output, pythiaOut.selection0, atol=atol)
    assert np.allclose(selection1_output, pythiaOut.selection1, atol=atol)
    assert np.allclose(yhat_output, pythiaOut.yhat, atol=atol)
    assert np.allclose(ysub_output, pythiaOut.ysub, atol=atol)


def test_fitlibsvm_poly() -> None:
    """Test that the output of the `fitlibsvm_poly` function is as expected."""
    atol = 1e-2
    pythia_option = PythiaOptions(
        cv_folds=5,
        is_poly_krnl=True,
        use_weights=False,
        use_lib_svm=True,
    )
    z_input = pd.read_csv(csv_path_z_input)
    y_input = pd.read_csv(csv_path_y_input)
    algo_input = pd.read_csv(csv_path_algo_input)
    y_best_input = pd.read_csv(csv_path_y_best_input)
    y_bin_input = pd.read_csv(csv_path_y_bin_input)
    [pythiaDataChanged, pythiaOut] = Pythia.run(  # noqa: N806
        z_input,
        y_input,
        y_bin_input,
        y_best_input,
        algo_input,
        pythia_option,
    )

    accuracy_output = pd.read_csv(
        output_dir / "fitlibsvm_polynomial/accuracy.csv", header=None,
    )
    precision_output = pd.read_csv(
        output_dir / "fitlibsvm_polynomial/precision.csv", header=None,
    )
    recall_output = pd.read_csv(
        output_dir / "fitlibsvm_polynomial/recall.csv", header=None,
    )
    boxcosnt_output = pd.read_csv(
        output_dir / "fitlibsvm_polynomial/boxconst.csv", header=None,
    )
    cvcmat_output = pd.read_csv(
        output_dir / "fitlibsvm_polynomial/cvcmat.csv", header=None,
    )
    kscale_output = pd.read_csv(
        output_dir / "fitlibsvm_polynomial/kscale.csv", header=None,
    )
    mu_output = pd.read_csv(output_dir / "fitlibsvm_polynomial/mu.csv", header=None)
    sigma_output = pd.read_csv(
        output_dir / "fitlibsvm_polynomial/sigma.csv", header=None,
    )
    pr0hat_output = pd.read_csv(
        output_dir / "fitlibsvm_polynomial/pr0hat.csv", header=None,
    )
    pr0sub_output = pd.read_csv(
        output_dir / "fitlibsvm_polynomial/pr0sub.csv", header=None,
    )
    selection0_output = pd.read_csv(
        output_dir / "fitlibsvm_polynomial/selection0.csv", header=None,
    )
    selection1_output = pd.read_csv(
        output_dir / "fitlibsvm_polynomial/selection1.csv", header=None,
    )
    yhat_output = pd.read_csv(output_dir / "fitlibsvm_polynomial/yhat.csv", header=None)
    ysub_output = pd.read_csv(output_dir / "fitlibsvm_polynomial/ysub.csv", header=None)
    assert np.allclose(accuracy_output, pythiaOut.accuracy, atol=atol)
    assert np.allclose(precision_output, pythiaOut.precision, atol=atol)
    assert np.allclose(recall_output, pythiaOut.recall, atol=atol)

    assert np.allclose(boxcosnt_output, pythiaOut.boxconst, atol=atol)
    assert np.allclose(cvcmat_output, pythiaOut.cvcmat, atol=atol)
    assert np.allclose(kscale_output, pythiaOut.kscale, atol=atol)
    assert np.allclose(mu_output, pythiaOut.mu, atol=atol)
    assert np.allclose(sigma_output, pythiaOut.sigma, atol=atol)
    assert np.allclose(pr0hat_output, pythiaOut.pr0hat, atol=atol)
    assert np.allclose(pr0sub_output, pythiaOut.pr0sub, atol=atol)
    assert np.allclose(selection0_output, pythiaOut.selection0, atol=atol)
    assert np.allclose(selection1_output, pythiaOut.selection1, atol=atol)
    assert np.allclose(yhat_output, pythiaOut.yhat, atol=atol)
    assert np.allclose(ysub_output, pythiaOut.ysub, atol=atol)
