import os

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

from matilda.data.model import Data, Model, PrelimOut
from matilda.data.option import Options, PrelimOptions
from matilda.stages.filter import Filter


class Preprocessing:
    """See file docstring."""

    @staticmethod
    def run(
        x: NDArray[np.double],
        y: NDArray[np.double],
        opts: Options,
    ) -> tuple[Data, PrelimOut]:
        """Perform preliminary processing on the input data 'x' and 'y'.

        Args
            x: The feature matrix (instances x features) to process.
            y: The performance matrix (instances x algorithms) to
                process.
            opts: An object of type Options containing options for
                processing.

        Returns
        -------
            A tuple containing the processed data (as 'Data' object) and
            preliminary output information (as 'PrelimOut' object).
        """
        raise NotImplementedError

    # don't understand
    @staticmethod
    def process_data(data: Data, opts: Options) -> tuple[Data, PrelimOptions]:
        """
        Store the raw data for further processing and remove the template data.

        :param model: The model object containing the data to be processed.
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
        """
        Remove algorithms with no "good" instances from the model.

        :param data: The model object containing the data to be processed.
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
            nalgos = data.y.shape[1]
            if nalgos == 0:
                raise Exception(
                    "'-> There are no ''good'' algorithms. Please verify\
    the binary performance measure. STOPPING!'",
                )

            # Creating a new Data object with the processed data
            return_data = Data(
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
            )
        return return_data

    @staticmethod
    def split_data(data: Data, opts: Options, model: Model) -> Model:
        """
        Split the data into training and testing sets.

        :param idx: The indices of the data to split.
        :param data: The model object containing the data to be processed.
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
        fileindexed = (
            hasattr(opts, "selvars")
            and hasattr(opts.selvars, "file_idx_flag")
            and opts.selvars.file_idx_flag
            and hasattr(opts.selvars, "file_idx")
            and os.path.isfile(opts.selvars.file_idx)
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
            print(f"-> Creating a small scale experiment for validation. \
                Percentage of subset: \
                {round(100 * opts.selvars.small_scale, 2)}%")
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
                "-> Creating a small scale experiment for validation based on density."
            )
            subset_index, _, _ = Filter.run(
                data.x,
                data.y,
                opts.selvars,
            )
            subset_index = ~subset_index
            print(f"-> Percentage of instances retained: \
                {round(100 * np.mean(subset_index), 2)}%")
        else:
            print("-> Using the complete set of the instances.")
            subset_index = np.ones(ninst, dtype=bool)

        if fileindexed or fractional or bydensity:
            if bydensity:
                data_dense = data

            x = data.x[subset_index, :]
            y = data.y[subset_index, :]
            x_raw = data.x_raw[subset_index, :]
            y_raw = data.y_raw[subset_index, :]
            y_bin = data.y_bin[subset_index, :]
            beta = data.beta[subset_index]
            num_good_algos = data.num_good_algos[subset_index]
            y_best = data.y_best[subset_index]
            p = data.p[subset_index]
            inst_labels = data.inst_labels[subset_index]

            if hasattr(data, "S"):
                s = data.S[subset_index, :]
        # create a new data object with the processed data
        data = Data(
            inst_labels=inst_labels,
            feat_labels=data.feat_labels,
            algo_labels=data.algo_labels,
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
        )

        # create a new model object with the processed data
        return Model(
            data=data,
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
