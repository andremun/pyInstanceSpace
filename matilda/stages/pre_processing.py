import numpy as np

from matilda.data.model import Data, PrelimOut
from matilda.data.option import Options


class PrePro:
    """See file docstring."""

    @staticmethod
    def run(
        data: Data,
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

    @staticmethod
    def select_features_and_algorithms(data: Data, opts: Options) -> Data:
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
        new_x = data.x
        new_feat_labels = data.feat_labels
        new_y = data.y
        new_algo_labels = data.algo_labels
        if opts.selvars.feats is not None:

            selected_features = [feat for feat in data.feat_labels
                                 if feat in opts.selvars.feats]

            # if something were chosen, based on the logic index,
            # rather than the name string
            if selected_features:
                print(f"-> Using the following features: "
                      f"{' '.join(selected_features)}")

                # based on manually selected feature to update the data.x
                is_selected_feature = [data.feat_labels.index(feat)
                                       for feat in selected_features]
                new_x = data.x[:, is_selected_feature]
                new_feat_labels = selected_features
            else:
                print("No features were specified in opts.selvars."
                      "feats or it was an empty list.")

        print("---------------------------------------------------")
        if (opts.selvars is not None) and \
                (opts.selvars.algos is not None):
            selected_algorithms = [algo for algo in data.algo_labels
                                   if algo in opts.selvars.algos]

            if selected_algorithms:
                print(f"-> Using the following algorithms: "
                      f"{' '.join(selected_algorithms)}")

                is_selected_algo = [data.algo_labels.index(algo)
                                    for algo in selected_algorithms]
                new_y = data.y[:, is_selected_algo]
                new_algo_labels = selected_algorithms
            else:
                print("No algorithms were specified in opts.selvars."
                      "algos or it was an empty list.")
        return Data(
            x=new_x,
            y=new_y,
            inst_labels=data.inst_labels,
            feat_labels=new_feat_labels,
            algo_labels=new_algo_labels,
            x_raw=data.x_raw,
            y_raw=data.y_raw,
            y_bin=data.y_bin,
            y_best=data.y_best,
            p=data.p,
            num_good_algos=data.num_good_algos,
            beta=data.beta,
            s=data.s,
            uniformity = data.uniformity,
        )

    @staticmethod
    def remove_instances_with_many_missing_values(data: Data) -> Data:
        """Remove rows (instances) and features (X columns).

        Parameters
        ----------
        data
            the Data class that contains the content of instances, with
            algorithm and feature labels

        :return Data: the Data class that has been updated based on the Washing criterion

         Washing criterion:
            1. For any row, if that row in both X and Y are NaN, remove
            2. For X columns, if that column's 20% grids are filled with NaN, remove
        """
        new_x = data.x
        new_y = data.y
        new_inst_labels = data.inst_labels
        new_s = data.s
        new_feat_labels = data.feat_labels
        # Identify rows where all elements are NaN in X or Y
        idx = np.all(np.isnan(data.x), axis=1) | \
              np.all(np.isnan(data.y), axis=1)
        if np.any(idx):
            print("-> There are instances with too many missing values. "
                  "They are being removed to increase speed.")
            # Remove instances (rows) where all values are NaN
            new_x = data.x[~idx]
            new_y = data.y[~idx]

            new_inst_labels = data.inst_labels[~idx]

            if data.s is not None:
                new_s = data.s[~idx]

        # Check for features(column) with more than 20% missing values
        threshold = 0.20
        idx = np.mean(np.isnan(new_x), axis=0) >= threshold

        if np.any(idx):
            print("-> There are features with too many missing values. "
                  "They are being removed to increase speed.")
            new_x = new_x[:, ~idx]
            new_feat_labels = [label for label,
            keep in zip(data.feat_labels, ~idx) if keep]

        ninst = new_x.shape[0]
        nuinst = len(np.unique(new_x, axis=0))
        # check if there are too many repeated instances
        max_duplic_ratio = 0.5
        if nuinst / ninst < max_duplic_ratio:
            print("-> There are too many repeated instances. "
                  "It is unlikely that this run will produce good results.",
                  )
        return Data(
            x=new_x,
            y=new_y,
            inst_labels=new_inst_labels,
            feat_labels=new_feat_labels,
            algo_labels=data.algo_labels,
            x_raw=data.x_raw,
            y_raw=data.y_raw,
            y_bin=data.y_bin,
            y_best=data.y_best,
            p=data.p,
            num_good_algos=data.num_good_algos,
            beta=data.beta,
            s=new_s,
            uniformity=None,
        )

