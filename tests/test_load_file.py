"""
Test module to verify the functionality to load file from source.

The file contains multiple unit test to ensure that metadata is correctly loaded from
csv file and option is also correctly loaded from json file.
"""
import json
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
        metadata_path = "test_data/load_file/metadata.csv"
        return Metadata.from_file(metadata_path)

    @pytest.fixture()
    def valid_metadata_with_source(self: Self) -> Metadata:
        """
        Fixture to load metadata from a CSV file that includes a 'source' field.

        Returns:
        -------
            Metadata: An instance loaded from a predefined CSV file with a source field.

        """
        metadata_path = "test_data/load_file/metadata_with_source.csv"
        return Metadata.from_file(metadata_path)

    def test_instance_labels_count(self: Self, valid_metadata: Metadata) -> None:
        """Check label count of metadata."""
        assert valid_metadata.instance_labels.count() == self.expected_instances

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
        assert valid_metadata.instance_sources is None

    def test_s_not_none(self: Self, valid_metadata_with_source: Metadata) -> None:
        """Check source is not none from metadata."""
        source = valid_metadata_with_source.instance_sources
        assert source is not None, "Expected 's' to be not None"
        assert source.count() == self.expected_source

    def test_metadata_invalid_path(self: Self) -> None:
        """Test FileNotFound exception is thrown with invalid path."""
        invalid_path = "invalid_path"
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
        error_msg = "Field 'MaxPerf_invalid' in JSON is " \
                    "not defined in the dataclass 'PerformanceOptions'"
        with pytest.raises(ValueError, match=error_msg):
            Options.from_file(invalid_option_path)

    def test_option_invalid_path(self: Self) -> None:
        """Test FileNotFound exception is thrown with invalid path."""
        path = script_dir / "invalid_path"
        with pytest.raises(FileNotFoundError):
            Options.from_file(path)

    def test_missing_field(self: Self) -> None:
        """Loading from json, and any top field and sub fields are missing."""
        invalid_path = script_dir / "test_data/load_file/options_dropped.json"
        with Path.open(invalid_path) as file:
            options_dict = json.load(file)
            loaded_options = Options.from_file(invalid_path)

        if loaded_options.auto is not None:
            assert loaded_options.auto.preproc == options_dict["auto"]["preproc"], \
                "Auto preproc value mismatch."
        else:
            pytest.fail("auto should not be None")

            # Bound Options
        if loaded_options.bound is not None:
            assert loaded_options.bound.flag == options_dict["bound"]["flag"], \
                "Bound flag value mismatch."
        else:
            pytest.fail("bound should not be None")

            # Norm Options
        if loaded_options.norm is not None:
            assert loaded_options.norm.flag == options_dict["norm"]["flag"], \
                "Norm flag value mismatch."
        else:
            pytest.fail("norm should not be None")

        # Selvars subfields
        if loaded_options.selvars is not None:
            assert loaded_options.selvars.small_scale_flag == options_dict["selvars"][
                "small_scale_flag"], "Selvars small_scale_flag mismatch."
            assert loaded_options.selvars.small_scale == options_dict["selvars"][
                "small_scale"], "Selvars small_scale mismatch."
            assert loaded_options.selvars.file_idx_flag == options_dict["selvars"][
                "file_idx_flag"], "Selvars file_idx_flag mismatch."
            assert loaded_options.selvars.file_idx \
                   == options_dict["selvars"]["file_idx"], "Selvars file_idx mismatch."
            assert loaded_options.selvars.density_flag == options_dict["selvars"][
                "density_flag"], "Selvars density_flag mismatch."
            assert loaded_options.selvars.min_distance == options_dict["selvars"][
                "min_distance"], "Selvars min_distance mismatch."
            assert loaded_options.selvars.type == options_dict["selvars"]["type"], \
                "Selvars type mismatch."
        else:
            pytest.fail("Selvars should not be None")

        # Sifted subfields
        if loaded_options.sifted is not None:
            assert loaded_options.sifted.flag == options_dict["sifted"]["flag"], \
                "Sifted flag mismatch."
            assert loaded_options.sifted.rho == options_dict["sifted"]["rho"], \
                "Sifted rho mismatch."

            assert loaded_options.sifted.k is None, "Sifted k mismatch."
            assert loaded_options.sifted.n_trees is None, "Sifted n_trees mismatch."
            assert loaded_options.sifted.max_iter == \
                   options_dict["sifted"]["max_iter"], "Sifted max_iter mismatch."
            assert loaded_options.sifted.replicates == options_dict["sifted"][
                "replicates"], "Sifted replicates mismatch."
        else:
            pytest.fail("Sifted  should not be None")

        # Other fields
        assert loaded_options.pilot.analytic == options_dict["pilot"]["analytic"], \
            "Pilot analytic mismatch."
        assert loaded_options.pilot.n_tries == options_dict["pilot"]["n_tries"], \
            "Pilot n_tries mismatch."
        assert loaded_options.cloister.c_thres == options_dict["cloister"]["c_thres"], \
            "Cloister c_thres mismatch."
        assert loaded_options.cloister.p_val == options_dict["cloister"]["p_val"], \
            "Cloister p_val mismatch."
        assert loaded_options.pythia.cv_folds == options_dict["pythia"]["cv_folds"], \
            "Pythia cv_folds mismatch."
        assert loaded_options.pythia.is_poly_krnl == options_dict["pythia"][
            "is_poly_krnl"], "Pythia is_poly_krnl mismatch."
        assert loaded_options.pythia.use_weights == options_dict["pythia"][
            "use_weights"], "Pythia use_weights mismatch."
        assert loaded_options.pythia.use_lib_svm == options_dict["pythia"][
            "use_lib_svm"], "Pythia use_lib_svm mismatch."
        assert loaded_options.trace.use_sim == options_dict["trace"]["use_sim"], \
            "Trace use_sim mismatch."

        assert loaded_options.parallel is None, "Parallel should be None"
        assert loaded_options.perf is None, "Perf should be None"

    def test_extra_top_fields(self: Self) -> None:
        """Any top field are not defined in the class."""
        path = script_dir / "test_data/load_file/options_extra_topfield.json"
        error_msg = "Extra fields in JSON not defined in Options: {'top'}"
        with pytest.raises(ValueError, match=error_msg):
            Options.from_file(path)
