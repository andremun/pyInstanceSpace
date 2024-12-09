"""Test module for Filter class to verify its functionality.

The file contains multiple unit tests to ensure that the `Filter` class correctly
perform its tasks. The basic mechanism of the test is to compare its output against
output from MATLAB and check if the outputs are the same or reasonable similar.

Tests include:
- Verifying ouput against MATLAB's output with 'Ftr' option type
- Verifying ouput against MATLAB's output with 'Ftr&AP' option type
- Verifying ouput against MATLAB's output with 'Ftr&AP&Good' option type
- Verifying ouput against MATLAB's output with 'Ftr&Good' option type
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from instancespace.data.options import SelvarsOptions
from instancespace.utils.filter import compute_uniformity, do_filter, filter_instance

script_dir = Path(__file__).parent

csv_path_x = script_dir / "test_data/filter/input/input_X.csv"
csv_path_y = script_dir / "test_data/filter/input/input_Y.csv"
csv_path_y_bin = script_dir / "test_data/filter/input/input_Ybin.csv"

input_x = pd.read_csv(csv_path_x, header=None).to_numpy()
input_y = pd.read_csv(csv_path_y, header=None).to_numpy()
input_y_bin = pd.read_csv(csv_path_y_bin, header=None).to_numpy()


class TestFtr:
    """Test output with Ftr option type."""

    @pytest.fixture()
    def ftr_option(self) -> SelvarsOptions:
        """Fixture for creating selvars option with Ftr type.

        Returns
        -------
            SelvarsOption: A selvars option object with Ftr for type.
        """
        return SelvarsOptions(
            small_scale_flag=False,
            small_scale=0.5,
            file_idx_flag=False,
            file_idx="",
            selvars_type="Ftr",
            density_flag=True,
            min_distance=0.1,
            algos=list("abc"),
            feats=list("abc"),
        )

    def test_ftr_filter(self, ftr_option: SelvarsOptions) -> None:
        """Test output from filtering against MATLAB's output.

        Compare subset_index, is_dissimilar, is_visa obtained from filter_instance
        method against each corresponding output from MATLAB.

        Args
        ----
            ftr_options (SelvarsOptions): SelvarsOption with type equals to "Ftr"
        """
        csv_path_subset_index = (
            script_dir / "test_data/filter/output/ftr/subsetIndex.csv"
        )
        csv_path_is_dissimilar = (
            script_dir / "test_data/filter/output/ftr/isDissimilar.csv"
        )
        csv_path_is_visa = script_dir / "test_data/filter/output/ftr/isVISA.csv"

        subset_index_ml = pd.read_csv(
            csv_path_subset_index,
            header=None,
            dtype=bool,
        ).to_numpy()
        is_dissimilar_ml = pd.read_csv(
            csv_path_is_dissimilar,
            header=None,
            dtype=bool,
        ).to_numpy()
        is_visa_ml = pd.read_csv(csv_path_is_visa, header=None, dtype=bool).to_numpy()

        subset_index, is_dissimilar, is_visa, _ = do_filter(
            input_x,
            input_y,
            input_y_bin,
            ftr_option.selvars_type,
            ftr_option.min_distance,
        )

        assert np.all(subset_index == subset_index_ml[:, 0])
        assert np.all(is_dissimilar == is_dissimilar_ml[:, 0])
        assert np.all(is_visa == is_visa_ml[:, 0])

    def test_ftr_uniformity(self, ftr_option: SelvarsOptions) -> None:
        """Test output from comuting uniformity against MATLAB's output.

        Compare computed uniformity value from Filter against the uniformity value
        obtained from MATLAB.

        Args
        ----
            ftr_options (SelvarsOptions): SelvarsOption with type equals to "Ftr"
        """
        csv_path_uniformity = script_dir / "test_data/filter/output/ftr/uniformity.csv"

        uniformity_ml = (
            pd.read_csv(csv_path_uniformity, header=None, dtype=float).to_numpy().item()
        )

        subset_index, _, _ = filter_instance(
            input_x,
            input_y,
            input_y_bin,
            ftr_option.selvars_type,
            ftr_option.min_distance,
        )
        uniformity = compute_uniformity(
            input_x,
            subset_index,
        )

        assert np.allclose(uniformity, uniformity_ml)


class TestFtrAp:
    """Test output with Ftr&AP option type."""

    @pytest.fixture()
    def ftr_ap_option(self) -> SelvarsOptions:
        """Fixture for creating selvars option with Ftr&AP type.

        Returns
        -------
            SelvarsOption: A selvars option object with Ftr&AP for type.
        """
        return SelvarsOptions(
            small_scale_flag=False,
            small_scale=0.5,
            file_idx_flag=False,
            file_idx="",
            selvars_type="Ftr&AP",
            density_flag=True,
            min_distance=0.1,
            algos=list("abc"),
            feats=list("abc"),
        )

    def test_ftr_ap_filter(self, ftr_ap_option: SelvarsOptions) -> None:
        """Test output from filtering against MATLAB's output.

        Compare subset_index, is_dissimilar, is_visa obtained from filter_instance
        method against each corresponding output from MATLAB.

        Args
        ----
            ftr_ap_options (SelvarsOptions): SelvarsOption with type equals to "Ftr&AP"
        """
        csv_path_subset_index = (
            script_dir / "test_data/filter/output/ftr_ap/subsetIndex.csv"
        )
        csv_path_is_dissimilar = (
            script_dir / "test_data/filter/output/ftr_ap/isDissimilar.csv"
        )
        csv_path_is_visa = script_dir / "test_data/filter/output/ftr_ap/isVISA.csv"

        subset_index_ml = pd.read_csv(
            csv_path_subset_index,
            header=None,
            dtype=bool,
        ).to_numpy()
        is_dissimilar_ml = pd.read_csv(
            csv_path_is_dissimilar,
            header=None,
            dtype=bool,
        ).to_numpy()
        is_visa_ml = pd.read_csv(csv_path_is_visa, header=None, dtype=bool).to_numpy()

        subset_index, is_dissimilar, is_visa, _ = do_filter(
            input_x,
            input_y,
            input_y_bin,
            ftr_ap_option.selvars_type,
            ftr_ap_option.min_distance,
        )

        assert np.all(subset_index == subset_index_ml[:, 0])
        assert np.all(is_dissimilar == is_dissimilar_ml[:, 0])
        assert np.all(is_visa == is_visa_ml[:, 0])

    def test_ftr_ap_uniformity(self, ftr_ap_option: SelvarsOptions) -> None:
        """Test output from comuting uniformity against MATLAB's output.

        Compare computed uniformity value from Filter against the uniformity value
        obtained from MATLAB.

        Args
        ----
            ftr_ap_options (SelvarsOptions): SelvarsOption with type equals to "Ftr&AP"
        """
        csv_path_uniformity = (
            script_dir / "test_data/filter/output/ftr_ap/uniformity.csv"
        )

        uniformity_ml = (
            pd.read_csv(csv_path_uniformity, header=None, dtype=float).to_numpy().item()
        )

        subset_index, _, _ = filter_instance(
            input_x,
            input_y,
            input_y_bin,
            ftr_ap_option.selvars_type,
            ftr_ap_option.min_distance,
        )
        uniformity = compute_uniformity(input_x, subset_index)

        assert np.allclose(uniformity, uniformity_ml)


class TestFtrApGood:
    """Test output with Ftr&AP&Good option type."""

    @pytest.fixture()
    def ftr_ap_good_option(self) -> SelvarsOptions:
        """Fixture for creating selvars option with Ftr&AP&Good type.

        Returns
        -------
            SelvarsOption: A selvars option object with Ftr&AP&Good for type.
        """
        return SelvarsOptions(
            small_scale_flag=False,
            small_scale=0.5,
            file_idx_flag=False,
            file_idx="",
            selvars_type="Ftr&AP&Good",
            density_flag=True,
            min_distance=0.1,
            algos=list("abc"),
            feats=list("abc"),
        )

    def test_ftr_ap_good_filter(self, ftr_ap_good_option: SelvarsOptions) -> None:
        """Test output from filtering against MATLAB's output.

        Compare subset_index, is_dissimilar, is_visa obtained from filter_instance
        method against each corresponding output from MATLAB.

        Args
        ----
            ftr_options (SelvarsOptions): SelvarsOption with type equals to
                "Ftr&AP&Good"
        """
        csv_path_subset_index = (
            script_dir / "test_data/filter/output/ftr_ap_good/subsetIndex.csv"
        )
        csv_path_is_dissimilar = (
            script_dir / "test_data/filter/output/ftr_ap_good/isDissimilar.csv"
        )
        csv_path_is_visa = script_dir / "test_data/filter/output/ftr_ap_good/isVISA.csv"

        subset_index_ml = pd.read_csv(
            csv_path_subset_index,
            header=None,
            dtype=bool,
        ).to_numpy()
        is_dissimilar_ml = pd.read_csv(
            csv_path_is_dissimilar,
            header=None,
            dtype=bool,
        ).to_numpy()
        is_visa_ml = pd.read_csv(csv_path_is_visa, header=None, dtype=bool).to_numpy()

        subset_index, is_dissimilar, is_visa, _ = do_filter(
            input_x,
            input_y,
            input_y_bin,
            ftr_ap_good_option.selvars_type,
            ftr_ap_good_option.min_distance,
        )

        assert np.all(subset_index == subset_index_ml[:, 0])
        assert np.all(is_dissimilar == is_dissimilar_ml[:, 0])
        assert np.all(is_visa == is_visa_ml[:, 0])

    def test_ftr_ap_good_uniformity(self, ftr_ap_good_option: SelvarsOptions) -> None:
        """Test output from comuting uniformity against MATLAB's output.

        Compare computed uniformity value from Filter against the uniformity value
        obtained from MATLAB.

        Args
        ----
            ftr_options (SelvarsOptions): SelvarsOption with type equals to
                "Ftr&AP&Good"
        """
        csv_path_uniformity = (
            script_dir / "test_data/filter/output/ftr_ap_good/uniformity.csv"
        )

        uniformity_ml = (
            pd.read_csv(csv_path_uniformity, header=None, dtype=float).to_numpy().item()
        )

        subset_index, _, _ = filter_instance(
            input_x,
            input_y,
            input_y_bin,
            ftr_ap_good_option.selvars_type,
            ftr_ap_good_option.min_distance,
        )
        uniformity = compute_uniformity(input_x, subset_index)

        assert np.allclose(uniformity, uniformity_ml)


class TestFtrGood:
    """Test output with Ftr&Good option type."""

    @pytest.fixture()
    def ftr_good_option(self) -> SelvarsOptions:
        """Fixture for creating selvars option with Ftr&Good type.

        Returns
        -------
            SelvarsOption: A selvars option object with Ftr&Good for type.
        """
        return SelvarsOptions(
            small_scale_flag=False,
            small_scale=0.5,
            file_idx_flag=False,
            file_idx="",
            selvars_type="Ftr&Good",
            density_flag=True,
            min_distance=0.1,
            algos=list("abc"),
            feats=list("abc"),
        )

    def test_ftr_good_filter(self, ftr_good_option: SelvarsOptions) -> None:
        """Test output from filtering against MATLAB's output.

        Compare subset_index, is_dissimilar, is_visa obtained from filter_instance
        method against each corresponding output from MATLAB.

        Args
        ----
            ftr_options (SelvarsOptions): SelvarsOption with type equals to "Ftr&Good"
        """
        csv_path_subset_index = (
            script_dir / "test_data/filter/output/ftr_good/subsetIndex.csv"
        )
        csv_path_is_dissimilar = (
            script_dir / "test_data/filter/output/ftr_good/isDissimilar.csv"
        )
        csv_path_is_visa = script_dir / "test_data/filter/output/ftr_good/isVISA.csv"

        subset_index_ml = pd.read_csv(
            csv_path_subset_index,
            header=None,
            dtype=bool,
        ).to_numpy()
        is_dissimilar_ml = pd.read_csv(
            csv_path_is_dissimilar,
            header=None,
            dtype=bool,
        ).to_numpy()
        is_visa_ml = pd.read_csv(csv_path_is_visa, header=None, dtype=bool).to_numpy()

        subset_index, is_dissimilar, is_visa, _ = do_filter(
            input_x,
            input_y,
            input_y_bin,
            ftr_good_option.selvars_type,
            ftr_good_option.min_distance,
        )

        assert np.all(subset_index == subset_index_ml[:, 0])
        assert np.all(is_dissimilar == is_dissimilar_ml[:, 0])
        assert np.all(is_visa == is_visa_ml[:, 0])

    def test_ftr_good_uniformity(self, ftr_good_option: SelvarsOptions) -> None:
        """Test output from comuting uniformity against MATLAB's output.

        Compare computed uniformity value from Filter against the uniformity value
        obtained from MATLAB.

        Args
        ----
            ftr_options (SelvarsOptions): SelvarsOption with type equals to "Ftr&Good"
        """
        csv_path_uniformity = (
            script_dir / "test_data/filter/output/ftr_good/uniformity.csv"
        )

        uniformity_ml = (
            pd.read_csv(csv_path_uniformity, header=None, dtype=float).to_numpy().item()
        )

        subset_index, _, _ = filter_instance(
            input_x,
            input_y,
            input_y_bin,
            ftr_good_option.selvars_type,
            ftr_good_option.min_distance,
        )
        uniformity = compute_uniformity(input_x, subset_index)

        assert np.allclose(uniformity, uniformity_ml)
