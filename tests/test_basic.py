from pathlib import Path
import sys
path_root = Path(__file__).parents[0].parents[0].parents[0]
sys.path.append(str(path_root))

def test_assertions():
    string = True
    print("this is as test case")
    assert True, "Something is wrong with the Github Workflow - please contact kharek@student.unimelb.edu.au"
