"""`eda` CLI entry point.

Verbs:
    new-project   create a project under projects/
    list          list known projects
    status        show one project's status
    library       inspect or query knowledge/
    replay        replay a project's experiment_log deterministically

Implementations live in their respective `lib/*` modules; this file is
purely the click surface.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from lib import __version__

_log = logging.getLogger("eda.cli")
_console = Console()


@click.group(invoke_without_command=False)
@click.version_option(version=__version__, prog_name="eda")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging.")
def main(verbose: bool) -> None:
    """EDA Framework command-line interface."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )


# --- new-project --------------------------------------------------------


@main.command(name="new-project")
@click.argument("name", type=str)
@click.option(
    "--domain",
    type=str,
    required=True,
    help="Domain module key (e.g., 'manufacturing', 'forecasting_demand', 'general').",
)
@click.option(
    "--recipe",
    type=str,
    default=None,
    help="Recipe key from recipes/ (e.g., 'manufacturing_defect_classification').",
)
@click.option(
    "--budget",
    type=int,
    required=True,
    help="Token budget cap (in thousands by convention; e.g., 30 = 30k).",
)
@click.option(
    "--branch",
    type=str,
    default=None,
    help="Git branch name (defaults to project/<user>/<name>).",
)
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Workspace root (default: current repo root).",
)
def new_project(
    name: str,
    domain: str,
    recipe: Optional[str],
    budget: int,
    branch: Optional[str],
    workspace: Optional[Path],
) -> None:
    """Create a new project under projects/<name>/ from the template."""
    try:
        from lib.project import create_project
    except ImportError as e:
        _console.print(f"[red]project module not available: {e}[/red]")
        sys.exit(2)
    proj_path = create_project(
        name=name,
        domain=domain,
        recipe=recipe,
        token_budget=budget * 1000,
        branch=branch,
        workspace=workspace,
    )
    _console.print(f"[green]Created project at {proj_path}[/green]")


# --- list ---------------------------------------------------------------


@main.command(name="list")
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
)
def list_projects(workspace: Optional[Path]) -> None:
    """List known projects in projects/."""
    from lib.workspace import resolve_workspace
    from lib.project import list_projects as do_list

    ws = resolve_workspace(workspace)
    rows = do_list(ws)

    table = Table(title=f"Projects in {ws}")
    table.add_column("Name", style="cyan")
    table.add_column("Domain")
    table.add_column("Status")
    table.add_column("Confidence")
    table.add_column("Branch")
    for r in rows:
        table.add_row(
            r.get("project_name", "?"),
            r.get("domain", "?"),
            r.get("status", "?"),
            r.get("confidence_tier", "?"),
            r.get("branch", "?"),
        )
    _console.print(table)


# --- status -------------------------------------------------------------


@main.command(name="status")
@click.argument("project_name", type=str)
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
)
def status(project_name: str, workspace: Optional[Path]) -> None:
    """Show status of one project."""
    from lib.workspace import resolve_workspace
    from lib.project import project_status

    ws = resolve_workspace(workspace)
    info = project_status(ws, project_name)
    for k, v in info.items():
        _console.print(f"[bold]{k}[/bold]: {v}")


# --- library ------------------------------------------------------------


@main.command(name="library")
@click.option("--domain", type=str, default=None, help="Filter by domain.")
@click.option(
    "--capability",
    type=str,
    default=None,
    help="Filter by capability composition signature.",
)
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
)
def library(domain: Optional[str], capability: Optional[str], workspace: Optional[Path]) -> None:
    """Inspect cross-project knowledge in knowledge/."""
    try:
        from lib.retrieval import summarize_library
    except ImportError as e:
        _console.print(f"[yellow]library not available yet: {e}[/yellow]")
        return
    summary = summarize_library(workspace=workspace, domain=domain, capability=capability)
    _console.print_json(data=summary)


# --- replay -------------------------------------------------------------


@main.command(name="replay")
@click.argument("project_name", type=str)
@click.option(
    "--up-to-iteration",
    type=int,
    default=None,
    help="Replay only up to and including this iteration (default: all).",
)
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
)
def replay_cmd(
    project_name: str,
    up_to_iteration: Optional[int],
    workspace: Optional[Path],
) -> None:
    """Deterministically replay a project's experiment log."""
    from lib.replay import replay_project

    out = replay_project(
        project_name=project_name,
        workspace=workspace,
        up_to_iteration=up_to_iteration,
    )
    _console.print_json(data=out)


if __name__ == "__main__":
    main()
