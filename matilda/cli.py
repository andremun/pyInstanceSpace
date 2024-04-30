"""The command line interface for instance space."""

import click


@click.group()
@click.version_option()
def cli() -> None:
    """Entry point for command line interface."""
    pass

@click.option("--metadata-in", type=click.Path(exists=True))
@click.option("--options-in", type=click.Path(exists=True))
@click.option("--out", default="out.csv", show_default=True)
@cli.command()
def build() -> None:
    """TODO: documentation.

    Work out if we want to do docstrings for commands. I don't think we want to include
    the cli in the code documentation. It might be confusing to document it as
    functions.
    """
    click.echo("build")
