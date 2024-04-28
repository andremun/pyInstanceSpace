"""Provides functionality for generating and saving PNG images.

The main functionality is encapsulated in the `script_png` function, which takes a
Model object and a root directory as inputs. The function then processes the data from
the Model object to create visual representations in the form of PNG images, which are
saved to the specified root directory.
"""

from matilda.data.model import Model


def script_png(contianer: Model, rootdir: str) -> None:
    """Generate and save PNG images.

    The image represent the data in the Model object.

    Args:
        contianer: The Model object containing the data to be visualized
            and saved as PNG images.
        rootdir: The root directory where the PNG images will be saved.
    """
    # TODO: Rewrite PRELIM logic in python
    raise NotImplementedError
