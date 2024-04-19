"""Test cases are here."""

import sys
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
    assert False, "Something is wrong with the Github Workflow - " \
                 "please contact kharek@student.unimelb.edu.au"

