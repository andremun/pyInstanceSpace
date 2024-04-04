import sys

from matilda.data.model import Model


def build_instance_space(rootdir: str) -> Model:
    """
    Construct and return a Model object after instance space analysis.

    :param rootdir: The root directory containing the data and configuration files
    :return: A Model object representing the built instance space.
    """
    # TODO: Rewrite buildIS logic in Python
    raise NotImplementedError


if __name__ == "__main__":
    rootdir = sys.argv[1]
    build_instance_space(rootdir)
