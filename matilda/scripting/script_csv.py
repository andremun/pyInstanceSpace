"""Provides functionalities for exporting data from Model objects into CSV files.

The primary function in this module, `script_csv`, allows users to export the data
contained in a Model object into CSV files. These files are saved in a specified
directory, making it easier to manage and use the exported data for further analysis,
reporting, or processing.
"""

from matilda.data.model import Model


def script_csv(model: Model, rootdir: str) -> None:
    """Export and save the data from a Model object into CSV files.

    Args
        model: The Model object containing the data to be exported.
        rootdir: The root directory where the CSV files will be saved.
    """
    # TODO: Transcripte scriptcsv into python
    raise NotImplementedError
