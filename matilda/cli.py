"""The command line interface for instance space."""

from pathlib import Path

import click

from matilda.instance_space import instance_space_from_files


@click.group()
@click.version_option()
def cli() -> None:
    """Entry point for command line interface."""
    pass


@click.option("--metadata-in", type=click.Path(exists=True), required=True)
@click.option("--options-in", type=click.Path(exists=True), required=True)
@click.option("--out-csv", type=click.Path())
@click.option("--out-png", type=click.Path())
@click.option("--out-web", type=click.Path())
@cli.command()
def build(
    metadata_in: Path,
    options_in: Path,
    out_csv: Path|None,
    out_png: Path|None,
    out_web: Path|None,
) -> None:
    """TODO: documentation.

    Work how we want to do docstrings for commands. I don't think we want to include
    the cli in the code documentation. It might be confusing to document it as
    functions.
    """
    instance_space = instance_space_from_files(metadata_in, options_in)
    model = instance_space.build()

    if out_csv is not None:
        out_csv_file = out_csv.open("w")
        out_csv_file.write(model.to_csv())

        if out_web is not None:
            out_web_file = out_web.open("w")
            out_web_file.write(model.to_web())

    if out_png is not None:
        _out_png_file = out_png.open("w")
        # TODO: Work out image library we want to use.

