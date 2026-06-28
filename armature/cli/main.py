"""CLI entry point for Armature.

Provides main Click group with global flags and command registration.
"""

from __future__ import annotations

import sys

import click

from armature.cli.commands.classify import classify
from armature.cli.commands.discover import discover_cmd
from armature.cli.commands.inspect import inspect
from armature.cli.commands.weights import weights_cmd


@click.group()
@click.option("--verbose", is_flag=True, help="Show detailed output")
@click.option("--quiet", is_flag=True, help="Suppress all non-essential output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool) -> None:
    """Process mining workbench for Activity Relationship Matrix (ARM) analysis."""
    # Store global flags in context for subcommands
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet

    # Validate mutually exclusive flags
    if verbose and quiet:
        click.echo("Error: --verbose and --quiet are mutually exclusive", err=True)
        sys.exit(2)


# Register commands
cli.add_command(classify, name="classify")
cli.add_command(discover_cmd, name="discover")
cli.add_command(inspect, name="inspect")
cli.add_command(weights_cmd, name="weights")


if __name__ == "__main__":
    cli()
