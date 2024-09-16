"""Process the input data before running the main analysis."""

from pathlib import Path

import numpy as np
import pandas as pd
from numpy._typing import NDArray
from sklearn.model_selection import train_test_split

from matilda.data.metadata import Metadata
from matilda.data.model import (
    Data,
    PreprocessingDataChanged,
    PreprocessingOut,
)
from matilda.data.options import InstanceSpaceOptions, PrelimOptions
from matilda.stages.filter import Filter


class Preprocessing:
    """See file docstring."""

    def __init__(
        self,
        matadata: Metadata,
        opts: InstanceSpaceOptions,
    ) -> None:
        """Initialize the preprocessing stage.

        Args
        ----
            matadata: A Metadata object constructed from the parsed CSV data
            opts (Options): Configuration options.
        """
        self.matadata = matadata
        self.opts = opts

    @staticmethod
    def run(
        matadata: Metadata,
        opts: InstanceSpaceOptions,
    ) -> tuple[PreprocessingDataChanged, PreprocessingOut]:
        """Perform preliminary processing on the input data 'x' and 'y'.

        Args
            matadata: An object of data class that contains data
                from CSV file.
            opts: An object of type Options containing options for
                processing.

        Returns
        -------
            A tuple containing the processed data (as 'PreprocessingDataChanged' object)
            and dummy object for further implementation (as 'PreprocessingOut' object).
        """

        new_x, new_y, new_feat_labels, new_algo_labels = Preprocessing.select_features_and_algorithms(matadata.features,
            matadata.algorithms, matadata.feature_names, matadata.algorithm_names, opts)

        after_washing = Preprocessing.remove_instances_with_many_missing_values(new_x, new_y,
        matadata.instance_sources, new_feat_labels, matadata.instance_labels)

        # From here return the tuple[PreprocessingDataChanged, PreprocessingOut(dummy)]

        pre_data_changed = PreprocessingDataChanged(
            inst_labels=after_washing.inst_labels,
            feat_labels=after_washing.feat_labels,
            algo_labels=after_washing.algo_labels,
            x=after_washing.x,
            y=after_washing.y,
            s=after_washing.s,
        )

        preprocess_out = PreprocessingOut()

        # these stuff will be moved into PRILIM
        """after_process, prelim_opts = Preprocessing.process_data(after_washing, opts)
        prelim_data, prelim_out = Prelim.run(
            after_process.x,
            after_process.y,
            prelim_opts,
        )

        # These should be a part of FILTRER, leave it not delete

        bad_instances_removed = Preprocessing.remove_bad_instances(
            prelim_data.merge_with(data),
        )
        # Preprocessing.split_data(bad_instances_removed, opts, model)
        """

        return pre_data_changed, preprocess_out

    @staticmethod
    def select_features_and_algorithms(x: NDArray[np.double],y: NDArray[np.double],feat_labels: list[str],
                                       algo_labels: list[str], opts: InstanceSpaceOptions) -> \
            tuple[NDArray[np.double],NDArray[np.double],list[str],list[str]]:
        """Select features and algorithms based on options provided in opts.

        Remove instances with too many missing values.

        Parameters
        ----------
        data
            the Data class that contains the content of instances, with
            algorithm and feature labels
        opts
            the Option class that contains setting for analysis.

        :return Data: the Data class that has been modified based on the settings
        """
        print("---------------------------------------------------")
        new_x = x
        new_feat_labels = feat_labels
        new_y = y
        new_algo_labels = algo_labels
        if opts.selvars.feats is not None:
            selected_features = [
                feat for feat in feat_labels if feat in opts.selvars.feats
            ]

            # if something were chosen, based on the logic index,
            # rather than the name string
            if selected_features:
                print(
                    f"-> Using the following features: "
                    f"{' '.join(selected_features)}",
                )

                # based on manually selected feature to update the data.x
                is_selected_feature = [
                    feat_labels.index(feat) for feat in selected_features
                ]
                new_x = x[:, is_selected_feature]
                new_feat_labels = selected_features
            else:
                print(
                    "No features were specified in opts.selvars."
                    "feats or it was an empty list.",
                )

        print("---------------------------------------------------")
        if opts.selvars.algos is not None:
            selected_algorithms = [
                algo for algo in algo_labels if algo in opts.selvars.algos
            ]

            if selected_algorithms:
                print(
                    f"-> Using the following algorithms: "
                    f"{' '.join(selected_algorithms)}",
                )

                is_selected_algo = [
                    algo_labels.index(algo) for algo in selected_algorithms
                ]
                new_y = y[:, is_selected_algo]
                new_algo_labels = selected_algorithms
            else:
                print(
                    "No algorithms were specified in opts.selvars."
                    "algos or it was an empty list.",
                )
        return new_x, new_y, new_feat_labels, new_algo_labels

    @staticmethod
    def remove_instances_with_many_missing_values(x: NDArray[np.double],y: NDArray[np.double],
                                                  s: pd.Series | None, feat_labels: list[str],inst_labels: pd.Series) \
            -> tuple[NDArray[np.double], NDArray[np.double], pd.Series, list[str], pd.Series]:
        """Remove rows (instances) and features (X columns).

        Parameters
        ----------
        data
            the Data class that contains the content of instances, with
            algorithm and feature labels

        :return Data: Data class that has been updated based on the Washing criterion

         Washing criterion:
            1. For any row, if that row in both X and Y are NaN, remove
            2. For X columns, if that column's 20% grids are filled with NaN, remove
        """
        new_x = x
        new_y = y
        new_inst_labels = inst_labels
        new_s = s
        new_feat_labels = feat_labels
        # Identify rows where all elements are NaN in X or Y
        idx = np.all(np.isnan(x), axis=1) | np.all(np.isnan(y), axis=1)
        if np.any(idx):
            print(
                "-> There are instances with too many missing values. "
                "They are being removed to increase speed.",
            )
            # Remove instances (rows) where all values are NaN
            new_x = x[~idx]
            new_y = y[~idx]

            new_inst_labels = inst_labels[~idx]

            if s is not None:
                new_s = s[~idx]

        # Check for features(column) with more than 20% missing values
        threshold = 0.20
        idx = np.mean(np.isnan(new_x), axis=0) >= threshold

        if np.any(idx):
            print(
                "-> There are features with too many missing values. "
                "They are being removed to increase speed.",
            )
            new_x = new_x[:, ~idx]
            new_feat_labels = [
                label for label, keep in zip(feat_labels, ~idx) if keep
            ]

        ninst = new_x.shape[0]
        nuinst = len(np.unique(new_x, axis=0))
        # check if there are too many repeated instances
        max_duplic_ratio = 0.5
        if nuinst / ninst < max_duplic_ratio:
            print(
                "-> There are too many repeated instances. "
                "It is unlikely that this run will produce good results.",
            )
        return new_x, new_y, new_inst_labels, new_feat_labels, new_s

    @staticmethod
    def process_data(
        data: Data,
        opts: InstanceSpaceOptions,
    ) -> tuple[Data, PrelimOptions]:
        """Store the raw data for further processing and remove the template data.

        Parameters
        ----------
        data : Data
            The data object containing the data to be processed.
        opts : Options
            The options object containing the options for processing.

        Returns
        -------
        tuple[Data, PrelimOptions]
            A tuple containing the processed data and preliminary options.
        """
        # Storing the raw data for further processing
        x_raw = data.x.copy()
        y_raw = data.y.copy()

        # Removing the template data such that it can be used in the labels of graphs
        # and figures
        feat_labels = [label.replace("feature_", "") for label in data.feat_labels]
        algo_labels = [label.replace("algo_", "") for label in data.algo_labels]

        # Creating a new Data object with the processed data
        return_data = Data(
            inst_labels=data.inst_labels,
            feat_labels=feat_labels,
            algo_labels=algo_labels,
            x=data.x,
            y=data.y,
            x_raw=x_raw,
            y_raw=y_raw,
            y_bin=data.y_bin,
            y_best=data.y_best,
            p=data.p,
            num_good_algos=data.num_good_algos,
            beta=data.beta,
            s=data.s,
            uniformity=data.uniformity,
        )

        # Creating a PrelimOptions object for further processing
        prelim_opts = PrelimOptions(
            max_perf=opts.perf.max_perf,
            abs_perf=opts.perf.abs_perf,
            epsilon=opts.perf.epsilon,
            beta_threshold=opts.perf.beta_threshold,
            bound=opts.bound.flag,
            norm=opts.norm.flag,
        )
        return return_data, prelim_opts

    @staticmethod
    def remove_bad_instances(data: Data) -> Data:
        """Remove algorithms with no "good" instances from the model.

        Parameters
        ----------
        data : Data
            The data object containing the data to be processed.

        Returns
        -------
        data : Data
            The model object containing the data being processed.
        """
        idx = np.all(~data.y_bin, axis=0)
        if np.any(idx):
            print(
                '-> There are algorithms with no "good" instances. They are being\
    removed to increase speed.',
            )
            y_raw = data.y_raw[:, ~idx]
            y = data.y[:, ~idx]
            y_bin = data.y_bin[:, ~idx]

            algo_labels_array = np.array(data.algo_labels)
            filtered_algo_labels = algo_labels_array[~idx]
            algo_labels = filtered_algo_labels.tolist()
            nalgos = y.shape[1]
            if nalgos == 0:
                raise ValueError(
                    "'-> There are no ''good'' algorithms. Please verify\
    the binary performance measure. STOPPING!'",
                )

            # Creating a new Data object with the processed data
            return Data(
                inst_labels=data.inst_labels,
                feat_labels=data.feat_labels,
                algo_labels=algo_labels,
                x=data.x,
                y=y,
                x_raw=data.x_raw,
                y_raw=y_raw,
                y_bin=y_bin,
                y_best=data.y_best,
                p=data.p,
                num_good_algos=data.num_good_algos,
                beta=data.beta,
                s=data.s,
                uniformity=data.uniformity,
            )

        return data

    @staticmethod
    def split_data(data: Data, opts: InstanceSpaceOptions, model: Model) -> Model:
        # return type used to be the model
        # model: Model

        """Split the data into training and testing sets.

        Parameters
        ----------
        data : Data
            The data object containing the data to be processed.
        opts : Options
            The options object containing the options for processing.
        model : Model
            The model object containing the data to be processed.

        Returns
        -------
        Model
            The model object containing the data has been processed.
        """
        # If we are only meant to take some observations
        print("-------------------------------------------------------------------")
        ninst = data.x.shape[0]
        fractional = (
            hasattr(opts, "selvars")
            and hasattr(opts.selvars, "small_scale_flag")
            and opts.selvars.small_scale_flag
            and hasattr(opts.selvars, "small_scale")
            and isinstance(opts.selvars.small_scale, float)
        )
        path = Path(opts.selvars.file_idx)
        fileindexed = (
            hasattr(opts, "selvars")
            and hasattr(opts.selvars, "file_idx_flag")
            and opts.selvars.file_idx_flag
            and hasattr(opts.selvars, "file_idx")
            and Path(opts.selvars.file_idx).is_file()
            and Path.is_file(path)
        )
        bydensity = (
            hasattr(opts, "selvars")
            and hasattr(opts.selvars, "density_flag")
            and opts.selvars.density_flag
            and hasattr(opts.selvars, "min_distance")
            and isinstance(opts.selvars.min_distance, float)
            and hasattr(opts.selvars, "type")
            and isinstance(opts.selvars.type, str)
        )

        if fractional:
            print(
                f"-> Creating a small scale experiment for validation. \
                Percentage of subset: \
                {round(100 * opts.selvars.small_scale, 2)}%",
            )
            _, subset_idx = train_test_split(
                np.arange(ninst),
                test_size=opts.selvars.small_scale,
                random_state=0,
            )
            subset_index = np.zeros(ninst, dtype=bool)
            subset_index[subset_idx] = True
        elif fileindexed:
            print("-> Using a subset of instances.")
            subset_index = np.zeros(ninst, dtype=bool)
            aux = np.genfromtxt(opts.selvars.file_idx, delimiter=",", dtype=int)
            aux = aux[aux < ninst]
            # for some reason, this makes the indices perform correctly.
            for i in range(len(aux)):
                aux[i] = aux[i] - 1
            subset_index[aux] = True
        elif bydensity:
            print(
                "-> Creating a small scale experiment for validation based on density.",
            )
            subset_index, _, _, _ = Filter.run(
                data.x,
                data.y,
                data.y_bin,
                opts.selvars,
            )
            subset_index = ~subset_index
            print(
                f"-> Percentage of instances retained: \
                {round(100 * np.mean(subset_index), 2)}%",
            )
        else:
            print("-> Using the complete set of the instances.")
            subset_index = np.ones(ninst, dtype=bool)

        # Get the data from the model
        inst_labels = data.inst_labels
        feat_labels = data.feat_labels
        algo_labels = data.algo_labels
        x = data.x
        y = data.y
        x_raw = data.x_raw
        y_raw = data.y_raw
        y_bin = data.y_bin
        y_best = data.y_best
        p = data.p
        num_good_algos = data.num_good_algos
        beta = data.beta
        s = data.s
        uniformity = data.uniformity
        data_dense = model.data_dense

        if fileindexed or fractional or bydensity:
            if bydensity:
                data_dense = model.data_dense

            x = x[subset_index, :]
            y = y[subset_index, :]
            x_raw = x_raw[subset_index, :]
            y_raw = y_raw[subset_index, :]
            y_bin = y_bin[subset_index, :]
            beta = beta[subset_index]
            num_good_algos = num_good_algos[subset_index]
            y_best = y_best[subset_index]
            p = p[subset_index]
            inst_labels = inst_labels[subset_index]

            if s is not None:
                s = s[subset_index]

        # create a new data object with the processed data
        processed_data = Data(
            inst_labels=inst_labels,
            feat_labels=feat_labels,
            algo_labels=algo_labels,
            x=x,
            y=y,
            x_raw=x_raw,
            y_raw=y_raw,
            y_bin=y_bin,
            y_best=y_best,
            p=p,
            num_good_algos=num_good_algos,
            beta=beta,
            s=s,
            uniformity=uniformity,
        )

        # create a new model object with the processed data
        return Model(
            data=processed_data,
            data_dense=data_dense,
            feat_sel=model.feat_sel,
            prelim=model.prelim,
            sifted=model.sifted,
            pilot=model.pilot,
            cloist=model.cloist,
            pythia=model.pythia,
            trace=model.trace,
            opts=model.opts,
        )

