"""Test cases are here."""

import sys
from pathlib import Path

import numpy as np

from matilda.build import select_features_and_algorithms
from matilda.data.model import Data, Model
from matilda.data.option import Opts

path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))


def test_assertions() -> None:
    """
    The test case for demonstration.

    Returns
    -------
        None

    """
    # string = True
    assert True, "Something is wrong with the Github Workflow - " \
                 "please contact kharek@student.unimelb.edu.au"


def test_manual_selection() -> None:
    """
    The test case for select_features_and_algorithms

    Returns
    -------
        None
    """
    data = Data(
        feat_labels=['feature1', 'feature2', 'feature3'],
        algo_labels=['algo1', 'algo2'],
        x=np.array([[1, 2, 3], [4, 5, 6]]),
        y=np.array([[0.1, 0.2], [0.3, 0.4]]),

    )

    opts = Opts(
        selvars={
            'feats': ['feature1', 'feature3'],
            'algos': ['algo2']
        }
    )

    model = Model(data=data, opts=opts)

    select_features_and_algorithms(model, model.opts)

    assert model.data.feat_labels == ['feature1', 'feature3'], "Feature selection failed"
    assert model.data.algo_labels == ['algo2'], "Algorithm selection failed"

    assert model.data.x.shape == (2, 2), "Feature matrix shape is incorrect after selection"
    assert model.data.y.shape == (2, 1), "Algorithm matrix shape is incorrect after selection"
