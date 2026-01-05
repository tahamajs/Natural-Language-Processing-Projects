"""Main CLI interface for the coding agent."""

import asyncio
from pathlib import Path

import click
from rich.console import Console

from coding_agent.cli.interactive import run_interactive_session

console = Console()

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Professional CLI-based coding agent with LangGraph.

    Run 'coding-agent chat' to start an interactive session.
    """
    pass


@cli.command()
@click.option(
    "--project-root",
    "-p",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path.cwd(),
    help="Root directory of the project to work on",
)
def chat(
    project_root: Path,
):
    """Start an interactive multi-turn coding session.

    Example:
        coding-agent chat --project-root ./my-project
    """
    try:
        asyncio.run(
            run_interactive_session(
                project_root=str(project_root),
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Session interrupted[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        raise click.Abort()


@cli.command()
def version():
    """Display version information."""
    console.print("[bold cyan]Coding Agent[/bold cyan] version 0.1.0")
    console.print("\nA professional CLI-based coding agent.")

if __name__ == "__main__":
    cli()
