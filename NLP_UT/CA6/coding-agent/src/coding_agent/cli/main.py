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
@click.option("--no-aggregate", is_flag=True, default=False, help="Do not aggregate generated logs into a single summary")
@click.option("--output", type=click.Path(file_okay=True, dir_okay=False, path_type=Path), default=None, help="Output path for aggregated summary (JSONL)")
@click.option("--format", "fmt", type=click.Choice(["jsonl", "csv", "md"], case_sensitive=False), default="jsonl", help="Export format for aggregated summary")
@click.option("--no-full", is_flag=True, default=False, help="Do not include full file contents in generated logs (use truncation)")
def generate_logs(no_aggregate: bool, output: Path | None, fmt: str, no_full: bool):
    """Generate simulated 'good' logs for test projects and optionally aggregate them."""
    try:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        from scripts.generate_good_logs import main as gen_main
        gen_main(aggregate=not no_aggregate, output=str(output) if output else None, output_format=fmt, full=not no_full)
    except Exception as e:
        console.print(f"\n[bold red]Error generating logs:[/bold red] {e}")
        raise click.Abort()


@cli.command()
def version():
    """Display version information."""
    console.print("[bold cyan]Coding Agent[/bold cyan] version 0.1.0")
    console.print("\nA professional CLI-based coding agent.")

if __name__ == "__main__":
    cli()
