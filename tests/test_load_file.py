"""
Test module to verify the functionality to load file from source.

The file contains multiple unit test to ensure that metadata is correctly loaded from
csv file and option is also correctly loaded from json file.
"""

from pathlib import Path
from typing import Self

import pytest

from matilda.data.metadata import Metadata
from matilda.data.option import Options

script_dir = Path(__file__).parent


class TestMetadata:
    """
    Test loading metadata from csv.

    Class containing suit of unit tests to test functionality of loading metadata
    from csv files.
    """

    expected_instances = 212
    expected_features = 10
    expected_algorithms = 10
    expected_source = 1088

    @pytest.fixture()
    def valid_metadata(self: Self) -> Metadata:
        """
        Fixture to load metadata from a standard CSV file.

        Returns:
        -------
            Metadata: An instance loaded from a predefined CSV file without source.

        """
        metadata_path = script_dir / "test_data/load_file/metadata.csv"
        return Metadata.from_file(metadata_path)

    @pytest.fixture()
    def valid_metadata_with_source(self: Self) -> Metadata:
        """
        Fixture to load metadata from a CSV file that includes a 'source' field.

        Returns:
        -------
            Metadata: An instance loaded from a predefined CSV file with a source field.

        """
        metadata_path = script_dir / "test_data/load_file/metadata_with_source.csv"
        return Metadata.from_file(metadata_path)

    def test_instance_labels_count(self: Self, valid_metadata: Metadata) -> None:
        """Check label count of metadata."""
        assert valid_metadata.inst_labels.count() == self.expected_instances

    def test_feature_names_length(self: Self, valid_metadata: Metadata) -> None:
        """Check number of features in metadata."""
        assert len(valid_metadata.feature_names) == self.expected_features

    def test_algorithm_names_length(self: Self, valid_metadata: Metadata) -> None:
        """Check number of algorithms in metadata."""
        assert len(valid_metadata.algorithm_names) == self.expected_algorithms

    def test_features_dimensions(self: Self, valid_metadata: Metadata) -> None:
        """Check dimension of feature data from metadata."""
        assert valid_metadata.features.shape == (
            self.expected_instances,
            self.expected_features,
        )

    def test_algorithms_dimensions(self: Self, valid_metadata: Metadata) -> None:
        """Check dimension of algorithm data from metadata."""
        assert valid_metadata.algorithms.shape == (
            self.expected_instances,
            self.expected_algorithms,
        )

    def test_s_is_none(self: Self, valid_metadata: Metadata) -> None:
        """Check source from metadata."""
        assert valid_metadata.s is None

    def test_s_not_none(self: Self, valid_metadata_with_source: Metadata) -> None:
        """Check source is not none from metadata."""
        source = valid_metadata_with_source.s
        assert source is not None, "Expected 's' to be not None"
        assert source.count() == self.expected_source

    def test_metadata_invalid_path(self: Self) -> None:
        """Test FileNotFound exception is thrown with invalid path."""
        invalid_path = script_dir / "invalid_path"
        with pytest.raises(FileNotFoundError):
            Metadata.from_file(invalid_path)


class TestOption:
    """Test loading option from json."""

    @pytest.fixture()
    def valid_options(self: Self) -> Options:
        """Load option json file from path."""
        option_path = script_dir / "test_data/load_file/options.json"
        return Options.from_file(option_path)

    @pytest.mark.parametrize(
        ("option_key", "subkey", "expected_value"),
        [
            ("parallel", "flag", False),
            ("parallel", "n_cores", 2),
            ("perf", "max_perf", False),
            ("perf", "abs_perf", True),
            ("perf", "epsilon", 0.2),
            ("perf", "beta_threshold", 0.55),
            ("auto", "preproc", True),
            ("bound", "flag", True),
            ("norm", "flag", True),
            ("selvars", "small_scale_flag", False),
            ("selvars", "small_scale", 0.5),
            ("selvars", "file_idx_flag", False),
            ("selvars", "file_idx", ""),
            ("selvars", "density_flag", False),
            ("selvars", "min_distance", 0.1),
            ("selvars", "type", "Ftr&Good"),
            ("sifted", "flag", True),
            ("sifted", "rho", 0.1),
            ("sifted", "k", 10),
            ("sifted", "n_trees", 50),
            ("sifted", "max_iter", 1000),
            ("sifted", "replicates", 100),
            ("pilot", "analytic", False),
            ("pilot", "n_tries", 5),
            ("cloister", "c_thres", 0.7),
            ("cloister", "p_val", 0.05),
            ("pythia", "cv_folds", 5),
            ("pythia", "is_poly_krnl", False),
            ("pythia", "use_weights", False),
            ("pythia", "use_lib_svm", False),
            ("trace", "use_sim", True),
            ("trace", "PI", 0.55),
            ("outputs", "csv", True),
            ("outputs", "png", True),
            ("outputs", "web", False),
        ],
    )
    def test_option_loading(
        self: Self,
        valid_options: Options,
        option_key: str,
        subkey: str,
        expected_value: bool | float | int,
    ) -> None:
        """
        Test attributes for each options is loaded.

        The test will iterate over all attributes defined in pytest's mark parametrize
        to verify that the attributes are correctly loaded.
        """
        assert getattr(getattr(valid_options, option_key), subkey) == expected_value

    def test_option_value_error(self: Self) -> None:
        """Test loading option with invalid attribute name will raise value error."""
        invalid_option_path = script_dir / "test_data/load_file/options_invalid.json"
        error_msg = "Field 'MaxPerf' in JSON is not defined"
        with pytest.raises(ValueError, match=error_msg):
            Options.from_file(invalid_option_path)

    def test_option_invalid_path(self: Self) -> None:
        """Test FileNotFound exception is thrown with invalid path."""
        invalid_path = script_dir / "invalid_path"
        with pytest.raises(FileNotFoundError):
            Options.from_file(invalid_path)
