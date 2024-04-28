"""Generate eb-compatible output files.

As the part of the Matilda package and is responsible for generating and
saving web-compatible output files from a Model object.

"""

from matilda.data.model import Model


def script_web(container: Model, rootdir: str) -> None:
    """Generate and save web-compatible output files to the specified directory.

    Args:
        container: The Model object containing the data to be visualized
            and saved as web-compatible files.
        rootdir: The root directory where the web output files will be
            saved.
    """
    # TODO: Rewrite scriptweb logic in python
    raise NotImplementedError
