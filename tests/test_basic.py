import sys
from pathlib import Path

path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))

def test_assertions():
    string = True
    assert True, "Something is wrong with the Github Workflow - please contact kharek@student.unimelb.edu.au"
