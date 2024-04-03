from pathlib import Path
import sys
import subprocess
path_root = Path(__file__).parents[0].parents[0].parents[0]
sys.path.append(str(path_root))

def test_assertions():
    matlab_script = str(path_root) + '/InstanceSpace/example.m'
    process = subprocess.Popen(["matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r", f"run('{matlab_script}');exit;"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        print("MATLAB code executed successfully.")
        print("Output:")
        print(stdout.decode())
        assert True
    else:
        print("Error executing MATLAB code:")
        print(stderr.decode())
        assert False
    # string = True
    # print("this is as test case")
    # assert True, "Something is wrong with the Github Workflow - please contact kharek@student.unimelb.edu.au"

def import_matlab():
    matlab_script = str(path_root) + '/InstanceSpace/example.m'
    process = subprocess.Popen(["matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r", f"run('{matlab_script}');exit;"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        print("MATLAB code executed successfully.")
        print("Output:")
        print(stdout.decode())
        assert True
    else:
        print("Error executing MATLAB code:")
        print(stderr.decode())
        assert False