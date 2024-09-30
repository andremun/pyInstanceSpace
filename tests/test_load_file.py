"""
Test module to verify the functionality to load file from source.

The file contains multiple unit test to ensure that metadata is correctly loaded from
csv file and option is also correctly loaded from json file.
"""

from pathlib import Path
from typing import Self

import pytest

from matilda.data.default_options import (
    DEFAULT_AUTO_PREPROC,
    DEFAULT_BOUND_FLAG,
    DEFAULT_CLOISTER_C_THRES,
    DEFAULT_CLOISTER_P_VAL,
    DEFAULT_NORM_FLAG,
    DEFAULT_OUTPUTS_CSV,
    DEFAULT_OUTPUTS_PNG,
    DEFAULT_OUTPUTS_WEB,
    DEFAULT_PARALLEL_FLAG,
    DEFAULT_PARALLEL_N_CORES,
    DEFAULT_PERFORMANCE_ABS_PERF,
    DEFAULT_PERFORMANCE_BETA_THRESHOLD,
    DEFAULT_PERFORMANCE_EPSILON,
    DEFAULT_PERFORMANCE_MAX_PERF,
    DEFAULT_PILOT_ANALYTICS,
    DEFAULT_PILOT_N_TRIES,
    DEFAULT_PYTHIA_CV_FOLDS,
    DEFAULT_PYTHIA_IS_POLY_KRNL,
    DEFAULT_PYTHIA_USE_LIB_SVM,
    DEFAULT_PYTHIA_USE_WEIGHTS,
    DEFAULT_SELVARS_DENSITY_FLAG,
    DEFAULT_SELVARS_FILE_IDX,
    DEFAULT_SELVARS_FILE_IDX_FLAG,
    DEFAULT_SELVARS_MIN_DISTANCE,
    DEFAULT_SELVARS_SMALL_SCALE,
    DEFAULT_SELVARS_SMALL_SCALE_FLAG,
    DEFAULT_SELVARS_TYPE,
    DEFAULT_SIFTED_FLAG,
    DEFAULT_SIFTED_K,
    DEFAULT_SIFTED_MAX_ITER,
    DEFAULT_SIFTED_NTREES,
    DEFAULT_SIFTED_REPLICATES,
    DEFAULT_SIFTED_RHO,
    DEFAULT_TRACE_PURITY,
    DEFAULT_TRACE_USE_SIM,
)
from matilda.data.metadata import Metadata
from matilda.data.options import InstanceSpaceOptions
from matilda.instance_space import (
    instance_space_from_directory,
    instance_space_from_files,
)

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
    def test_valid_metadata(self: Self) -> Metadata:
        """
        Fixture to load metadata from a standard CSV file.

        Returns:
        -------
            Metadata: An instance loaded from a predefined CSV file without source.

        """
        metadata_path = script_dir / "test_data/load_file/metadata.csv"
        option_path = script_dir / "test_data/load_file/options.json"

        returned = instance_space_from_files(metadata_path, option_path)
        assert returned is not None
        return returned.metadata

    @pytest.fixture()
    def test_directory_metadata(self: Self) -> Metadata:
        """
        Fixture to load metadata from a directory.

        Returns:
        -------
            Metadata: An instance loaded from a predefined CSV file without source.

        """
        directory_path = script_dir / "test_data/load_file"
        returned = instance_space_from_directory(directory_path)
        assert returned is not None, "Expected instance space to be returned"
        return returned.metadata

    @pytest.fixture()
    def test_valid_metadata_with_source(self: Self) -> Metadata:
        """
        Fixture to load metadata from a CSV file that includes a 'source' field.

        Returns:
        -------
            Metadata: An instance loaded from a predefined CSV file with a source field.

        """
        metadata_path = script_dir / "test_data/load_file/metadata_with_source.csv"
        option_path = script_dir / "test_data/load_file/options.json"
        returned = instance_space_from_files(metadata_path, option_path)
        assert returned is not None
        return returned.metadata

    def test_instance_labels_count(
        self: Self,
        test_valid_metadata: Metadata,
        test_directory_metadata: Metadata,
    ) -> None:
        """Check label count of metadata."""
        assert (
            test_directory_metadata.instance_labels.count()
            == test_valid_metadata.instance_labels.count()
        )
        assert test_valid_metadata.instance_labels.count() == self.expected_instances

    def test_feature_names_length(
        self: Self,
        test_valid_metadata: Metadata,
        test_directory_metadata: Metadata,
    ) -> None:
        """Check number of features in metadata."""
        assert len(test_valid_metadata.feature_names) == self.expected_features
        assert len(test_valid_metadata.feature_names) == len(
            test_directory_metadata.feature_names,
        )

    def test_algorithm_names_length(
        self: Self,
        test_valid_metadata: Metadata,
        test_directory_metadata: Metadata,
    ) -> None:
        """Check number of algorithms in metadata."""
        assert len(test_valid_metadata.algorithm_names) == self.expected_algorithms
        assert len(test_valid_metadata.algorithm_names) == len(
            test_directory_metadata.algorithm_names,
        )

    def test_features_dimensions(
        self: Self,
        test_valid_metadata: Metadata,
        test_directory_metadata: Metadata,
    ) -> None:
        """Check dimension of feature data from metadata."""
        assert test_valid_metadata.features.shape == (
            self.expected_instances,
            self.expected_features,
        )

        assert (
            test_valid_metadata.features.shape == test_directory_metadata.features.shape
        )

    def test_algorithms_dimensions(
        self: Self,
        test_valid_metadata: Metadata,
        test_directory_metadata: Metadata,
    ) -> None:
        """Check dimension of algorithm data from metadata."""
        assert test_valid_metadata.algorithms.shape == (
            self.expected_instances,
            self.expected_algorithms,
        )

        assert (
            test_valid_metadata.algorithms.shape
            == test_directory_metadata.algorithms.shape
        )

    def test_s_is_none(
        self: Self,
        test_valid_metadata: Metadata,
        test_directory_metadata: Metadata,
    ) -> None:
        """Check source from metadata."""
        assert test_valid_metadata.instance_sources is None
        assert test_directory_metadata.instance_sources is None

    def test_s_not_none(self: Self, test_valid_metadata_with_source: Metadata) -> None:
        """Check source is not none from metadata."""
        source = test_valid_metadata_with_source.instance_sources
        assert source is not None, "Expected 's' to be not None"
        assert source.count() == self.expected_source

    def test_metadata_invalid_path(
        self: Self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test FileNotFound exception is thrown with invalid path."""
        invalid_path = script_dir / "invalid_path"
        option_path = script_dir / "test_data/load_file/options.json"

        returned = instance_space_from_files(invalid_path, option_path)

        assert returned is None

        captured = capsys.readouterr()
        output = captured.out

        expected_error_msg = "[Errno 2] No such file or directory:"
        assert expected_error_msg in output

    def test_data_empty(self: Self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test dummy exception is thrown with invalid path."""
        data_path = script_dir / "test_data/load_file/dummydata.csv"
        option_path = script_dir / "test_data/load_file/options.json"

        returned = instance_space_from_files(data_path, option_path)
        assert returned is None

        captured = capsys.readouterr()
        expected_error_msg = "is empty."
        assert expected_error_msg in captured.out

    def test_illegal_csv(self: Self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test dummy exception is thrown with invalid path."""
        data_path = script_dir / "test_data/load_file/illegal.csv"
        option_path = script_dir / "test_data/load_file/options.json"

        returned = instance_space_from_files(data_path, option_path)
        assert returned is None

        captured = capsys.readouterr()
        expected_error_msg = "Error tokenizing data"
        assert expected_error_msg in captured.out


class TestOption:
    """Test loading option from json."""

    @pytest.fixture()
    def directory_options(self: Self) -> InstanceSpaceOptions:
        """Load option json file from directory."""
        directory_path = script_dir / "test_data/load_file"
        returned = instance_space_from_directory(directory_path)
        assert returned is not None, "Expected instance space to be returned"
        return returned.options

    @pytest.fixture()
    def test_valid_options(self: Self) -> InstanceSpaceOptions:
        """Load option json file from path."""
        option_path = script_dir / "test_data/load_file/options.json"
        metadata_path = script_dir / "test_data/load_file/metadata.csv"
        returned = instance_space_from_files(metadata_path, option_path)
        assert returned is not None
        return returned.options

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
            ("selvars", "selvars_type", "Ftr&Good"),
            (
                "selvars",
                "feats",
                [
                    "feature_Max_Normalized_Entropy_attributes",
                    "feature_Normalized_Entropy_Class_Attribute",
                    "feature_Nonlinearity_Nearest_Neighbor_Classifier_N4",
                ],
            ),
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
            ("trace", "purity", 0.55),
            ("outputs", "csv", True),
            ("outputs", "png", True),
            ("outputs", "web", False),
        ],
    )
    def test_option_loading(
        self: Self,
        test_valid_options: InstanceSpaceOptions,
        option_key: str,
        subkey: str,
        expected_value: bool | float | int,
    ) -> None:
        """
        Test attributes for each options is loaded.

        The test will iterate over all attributes defined in pytest's mark parametrize
        to verify that the attributes are correctly loaded.
        """
        assert (
            getattr(getattr(test_valid_options, option_key), subkey) == expected_value
        )

    def test_dir_loading(
        self: Self,
        directory_options: InstanceSpaceOptions,
        test_valid_options: InstanceSpaceOptions,
    ) -> None:
        """
        Test attributes for each options is loaded.

        The test will iterate over all attributes defined in pytest's mark parametrize
        to verify that the attributes are correctly loaded.
        """
        assert directory_options == test_valid_options

    def test_option_value_error(self: Self, capsys: pytest.CaptureFixture[str]) -> None:
        """Test loading option with invalid attribute name will raise value error."""
        invalid_option_path = script_dir / "test_data/load_file/options_invalid.json"
        metadata_path = script_dir / "test_data/load_file/metadata.csv"

        returned = instance_space_from_files(metadata_path, invalid_option_path)
        assert returned is None
        captured = capsys.readouterr()
        expected_error_msg = (
            "Error details: Field(s) '{'MaxPerf_invalid'}' "
            "in JSON are not defined in the data class "
            "'PerformanceOptions'"
        )

        assert expected_error_msg in captured.out

    def test_option_value_unexpected(
        self: Self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """
        Another invalid attribute case in the JSON file.

        Test loading option with attribute name that is definded by us
        but exists in the JSON, will raise value error.
        """
        invalid_option_path = script_dir / "test_data/load_file/options_name_by_us.json"
        metadata_path = script_dir / "test_data/load_file/metadata.csv"

        returned = instance_space_from_files(metadata_path, invalid_option_path)
        assert returned is None
        captured = capsys.readouterr()
        expected_error_msg = (
            "Error details: Field(s) '{'purity'}' in JSON are not defined in the field "
            "mapping for the data class 'TraceOptions'.\n"
            "Failed to initialize options\n"
        )

        assert expected_error_msg in captured.out

    def test_option_invalid_path(
        self: Self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test FileNotFound exception is thrown with invalid path."""
        invalid_options_path = script_dir / "invalid_path"
        metadata_path = script_dir / "test_data/load_file/metadata.csv"

        returned = instance_space_from_files(metadata_path, invalid_options_path)
        assert returned is None

        captured = capsys.readouterr()
        expected_error_msg = " [Errno 2] No such file or directory: "
        assert expected_error_msg in captured.out

    def test_missing_field(self: Self) -> None:
        """Loading from json, and any top field and sub fields are missing."""
        missing_field = script_dir / "test_data/load_file/options_dropped.json"
        metadata_path = script_dir / "test_data/load_file/metadata.csv"

        returned = instance_space_from_files(metadata_path, missing_field)
        assert returned is not None
        loaded_options = returned.options

        # check the dropped pythia is filled with default values
        assert loaded_options.pythia.cv_folds == DEFAULT_PYTHIA_CV_FOLDS
        assert loaded_options.pythia.is_poly_krnl == DEFAULT_PYTHIA_IS_POLY_KRNL
        assert loaded_options.pythia.use_weights == DEFAULT_PYTHIA_USE_WEIGHTS
        assert loaded_options.pythia.use_lib_svm == DEFAULT_PYTHIA_USE_LIB_SVM

        # check the dropped selvars.feats is filled with default value
        assert loaded_options.selvars.feats is None
        wanted_value = 0.8
        assert loaded_options.selvars.small_scale == wanted_value
        assert loaded_options.selvars.file_idx_flag is True

    def test_extra_top_fields(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Any top field are not defined in the class."""
        path = script_dir / "test_data/load_file/options_extra_topfield.json"
        metadata_path = script_dir / "test_data/load_file/metadata.csv"

        returned = instance_space_from_files(metadata_path, path)
        assert returned is None
        captured = capsys.readouterr()
        expected_error_msg = (
            "Extra fields in JSON are not defined in InstanceSpaceOptions: "
            "{'INTENDED_EXTRA_FIELD_IN_JSON'}"
        )

        assert expected_error_msg in captured.out

    def test_json_with_invalid_content(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Any top field are not defined in the class."""
        path = script_dir / "test_data/load_file/illegal.json"
        metadata_path = script_dir / "test_data/load_file/metadata.csv"

        returned = instance_space_from_files(metadata_path, path)
        assert returned is None
        captured = capsys.readouterr()
        expected_error_msg = "Expecting value: line 2 column 24 (char 25)"

        assert expected_error_msg in captured.out

    @pytest.fixture()
    def test_dummy_options(self: Self) -> InstanceSpaceOptions:
        """Load dummy option json file from path."""
        option_path = script_dir / "test_data/load_file/dummy.json"
        metadata_path = script_dir / "test_data/load_file/metadata.csv"

        returned = instance_space_from_files(metadata_path, option_path)
        assert returned is not None
        return returned.options

    @pytest.mark.parametrize(
        ("option_key_dummy", "subkey_dummy", "expected_value_dummy"),
        [
            ("parallel", "flag", DEFAULT_PARALLEL_FLAG),
            ("parallel", "n_cores", DEFAULT_PARALLEL_N_CORES),
            ("perf", "max_perf", DEFAULT_PERFORMANCE_MAX_PERF),
            ("perf", "abs_perf", DEFAULT_PERFORMANCE_ABS_PERF),
            ("perf", "epsilon", DEFAULT_PERFORMANCE_EPSILON),
            ("perf", "beta_threshold", DEFAULT_PERFORMANCE_BETA_THRESHOLD),
            ("auto", "preproc", DEFAULT_AUTO_PREPROC),
            ("bound", "flag", DEFAULT_BOUND_FLAG),
            ("norm", "flag", DEFAULT_NORM_FLAG),
            ("selvars", "small_scale_flag", DEFAULT_SELVARS_SMALL_SCALE_FLAG),
            ("selvars", "small_scale", DEFAULT_SELVARS_SMALL_SCALE),
            ("selvars", "file_idx_flag", DEFAULT_SELVARS_FILE_IDX_FLAG),
            ("selvars", "file_idx", DEFAULT_SELVARS_FILE_IDX),
            ("selvars", "density_flag", DEFAULT_SELVARS_DENSITY_FLAG),
            ("selvars", "min_distance", DEFAULT_SELVARS_MIN_DISTANCE),
            ("selvars", "selvars_type", DEFAULT_SELVARS_TYPE),
            ("selvars", "feats", None),
            ("selvars", "algos", None),
            ("sifted", "flag", DEFAULT_SIFTED_FLAG),
            ("sifted", "rho", DEFAULT_SIFTED_RHO),
            ("sifted", "k", DEFAULT_SIFTED_K),
            ("sifted", "n_trees", DEFAULT_SIFTED_NTREES),
            ("sifted", "max_iter", DEFAULT_SIFTED_MAX_ITER),
            ("sifted", "replicates", DEFAULT_SIFTED_REPLICATES),
            ("pilot", "analytic", DEFAULT_PILOT_ANALYTICS),
            ("pilot", "n_tries", DEFAULT_PILOT_N_TRIES),
            ("cloister", "c_thres", DEFAULT_CLOISTER_C_THRES),
            ("cloister", "p_val", DEFAULT_CLOISTER_P_VAL),
            ("pythia", "cv_folds", DEFAULT_PYTHIA_CV_FOLDS),
            ("pythia", "is_poly_krnl", DEFAULT_PYTHIA_IS_POLY_KRNL),
            ("pythia", "use_weights", DEFAULT_PYTHIA_USE_WEIGHTS),
            ("pythia", "use_lib_svm", DEFAULT_PYTHIA_USE_LIB_SVM),
            ("trace", "use_sim", DEFAULT_TRACE_USE_SIM),
            ("trace", "purity", DEFAULT_TRACE_PURITY),
            ("outputs", "csv", DEFAULT_OUTPUTS_CSV),
            ("outputs", "png", DEFAULT_OUTPUTS_PNG),
            ("outputs", "web", DEFAULT_OUTPUTS_WEB),
        ],
    )
    def test_dummy_option_loading(
        self: Self,
        test_dummy_options: InstanceSpaceOptions,
        option_key_dummy: str,
        subkey_dummy: str,
        expected_value_dummy: bool | float | int,
    ) -> None:
        """
        Test attributes for each options is loaded.

        The test will iterate over all attributes defined in pytest's mark parametrize
        to verify that the attributes are correctly loaded.
        """
        assert (
            getattr(getattr(test_dummy_options, option_key_dummy), subkey_dummy)
            == expected_value_dummy
        )
