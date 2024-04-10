"""Test cases are here."""

import sys
<<<<<<< HEAD
path_root = Path(__file__).parents[0].parents[0].parents[0]
sys.path.append(str(path_root))

def test_assertions():
    string = True
    print("this is as test case")
    assert True, "Something is wrong with the Github Workflow - please contact kharek@student.unimelb.edu.au"
=======
from pathlib import Path

path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))

def test_assertions() -> None:
    """
    The test case for demonstration.

    Returns
    -------
        None

    """
    #string = True
    assert True, "Something is wrong with the Github Workflow - " \
                 "please contact kharek@student.unimelb.edu.au"

>>>>>>> eec783226b664cb2cf2633de3cfc9bfe89466b50
