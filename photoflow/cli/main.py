"""Main CLI interface for PhotoFlow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from photoflow.cli.metadata import metadata
from photoflow.core.actions import default_registry
from photoflow.core.duplicate_detection import DuplicateDetector
from photoflow.core.image_loader import load_image_asset
from photoflow.core.image_processor import ImageProcessingError, default_processor
from photoflow.core.models import ActionSpec, Pipeline
from photoflow.core.serialization import load_pipeline, save_pipeline

# Constants
MAX_DESCRIPTION_LENGTH = 50


console = Console()


@click.group()
@click.version_option()
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, config: str | None, verbose: bool) -> None:
    """PhotoFlow - Python batch photo manipulation tool."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = config
    ctx.obj["verbose"] = verbose
    if verbose:
        console.print("[dim]PhotoFlow CLI - Verbose mode enabled[/dim]")


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--action",
    "-a",
    multiple=True,
    help="Action to apply (format: action_name:param=value,param2=value2)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (default: <input>_processed.<ext>)",
)
@click.option(
    "--pipeline",
    "-p",
    type=click.Path(exists=True),
    help="Pipeline file to use",
)
@click.option("--dry-run", is_flag=True, help="Show what would be done without processing")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def process(  # noqa: PLR0915
    ctx: click.Context,
    input_path: str,
    action: tuple[str, ...],
    output: str | None,
    pipeline: str | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Process a single image with actions or pipeline."""
    verbose = verbose or ctx.obj.get("verbose", False)
    try:
        # Convert string paths to Path objects
        input_path_obj = Path(input_path)
        output_path_obj = Path(output) if output else None
        pipeline_path_obj = Path(pipeline) if pipeline else None

        if verbose:
            console.print(f"[dim]Loading image: {input_path}[/dim]")
        image = load_image_asset(input_path_obj)
        console.print(f"📸 Loaded: {image.width}x{image.height} {image.format}")
        if pipeline:
            if verbose:
                console.print(f"[dim]Loading pipeline: {pipeline}[/dim]")
            pipeline_obj = load_pipeline(pipeline_path_obj)  # type: ignore[arg-type]
            actions_to_apply = pipeline_obj.enabled_actions
            console.print(f"🔧 Using pipeline: {pipeline_obj.name}")
        elif action:
            actions_to_apply = []  # type: ignore[assignment]
            for action_str in action:
                action_spec = _parse_action_string(action_str)
                actions_to_apply.append(action_spec)  # type: ignore[attr-defined]
            actions_to_apply = tuple(actions_to_apply)
            console.print(f"🔧 Using {len(actions_to_apply)} action(s)")
        else:
            console.print("❌ No actions or pipeline specified")
            raise click.ClickException("Specify either --action or --pipeline")
        if verbose or dry_run:
            console.print("\n[bold]Processing plan:[/bold]")
            for i, action_spec in enumerate(actions_to_apply, 1):
                console.print(f"  {i}. {action_spec.name}")
                for param, value in action_spec.parameters.items():
                    console.print(f"     • {param}: {value}")
        if dry_run:
            console.print("\n[yellow]Dry run - no processing performed[/yellow]")
            return
        processed_image = image
        with console.status("[bold green]Processing...") as status:
            for i, action_spec in enumerate(actions_to_apply, 1):
                status.update(f"[bold green]Step {i}/{len(actions_to_apply)}: {action_spec.name}")
                action_instance = default_registry.get_action(action_spec.name)
                processed_image = action_instance.execute(processed_image, **action_spec.parameters)
                if verbose:
                    console.print(f"✅ {action_spec.name}: {processed_image.width}x{processed_image.height}")
        if not output_path_obj:
            output_path_obj = input_path_obj.parent / f"{input_path_obj.stem}_processed{input_path_obj.suffix}"
        if not default_processor:
            raise ImageProcessingError("PIL/Pillow not available for image saving")
        saved_path = default_processor.save_image(processed_image, output_path_obj)
        console.print(f"💾 Saved: {saved_path}")
        console.print(f"📐 Result: {processed_image.width}x{processed_image.height}")
        if saved_path.exists():  # type: ignore[attr-defined]
            file_size_mb = saved_path.stat().st_size / (1024 * 1024)  # type: ignore[attr-defined]
            console.print(f"📏 Size: {file_size_mb:.1f} MB")
    except Exception as e:
        console.print(f"❌ Error: {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command("list-actions")
@click.option("--tag", help="Filter actions by tag")
@click.option("--detailed", is_flag=True, help="Show detailed information")
def list_actions(tag: str | None, detailed: bool) -> None:
    """List available actions."""
    if tag:
        action_names = default_registry.get_actions_by_tag(tag)
        console.print(f"🏷️  Actions tagged '{tag}':")
    else:
        action_names = default_registry.list_actions()
        console.print("🎯 Available actions:")
    if not action_names:
        console.print("No actions found")
        return
    if detailed:
        table = Table(title="PhotoFlow Actions")
        table.add_column("Name", style="bold")
        table.add_column("Label")
        table.add_column("Description")
        table.add_column("Version")
        table.add_column("Tags")
        for name in action_names:
            action_info = default_registry.get_action_info(name)
            table.add_row(
                name,
                action_info["label"],
                (
                    action_info["description"][:MAX_DESCRIPTION_LENGTH] + "..."
                    if len(action_info["description"]) > MAX_DESCRIPTION_LENGTH
                    else action_info["description"]
                ),
                action_info["version"],
                ", ".join(action_info["tags"]),
            )
        console.print(table)
    else:
        for name in action_names:
            action_info = default_registry.get_action_info(name)
            console.print(f"  • {name} - {action_info['label']}")


@cli.command()
@click.argument("action_name")
def info(action_name: str) -> None:
    """Show detailed information about an action."""
    try:
        action_info = default_registry.get_action_info(action_name)
        console.print(f"[bold]{action_info['label']}[/bold] ({action_info['name']})")
        console.print(f"📝 {action_info['description']}")
        console.print(f"🔖 Version: {action_info['version']}")
        if action_info["author"]:
            console.print(f"👤 Author: {action_info['author']}")
        if action_info["tags"]:
            console.print(f"🏷️  Tags: {', '.join(action_info['tags'])}")
        if action_info["fields"]:
            console.print("\n[bold]Parameters:[/bold]")
            table = Table()
            table.add_column("Name", style="bold")
            table.add_column("Type")
            table.add_column("Default")
            table.add_column("Required")
            table.add_column("Description")
            for field_name, field_info in action_info["fields"].items():
                table.add_row(
                    field_name,
                    field_info["type"],
                    (str(field_info["default"]) if field_info["default"] is not None else "—"),
                    "✓" if field_info["required"] else "—",
                    field_info.get("help_text", ""),
                )
            console.print(table)
    except KeyError:
        console.print(f"❌ Action '{action_name}' not found")
        console.print("Use 'photoflow list-actions' to see available actions")
        sys.exit(1)


@cli.group()
def pipeline() -> None:
    """Pipeline management commands."""
    pass


@pipeline.command("create")
@click.argument("name")
@click.option(
    "--action",
    "-a",
    multiple=True,
    help="Action to add (format: action_name:param=value,param2=value2)",
)
@click.option("--output", "-o", type=click.Path(), help="Output pipeline file")
@click.option("--description", help="Pipeline description")
def create_pipeline(name: str, action: tuple[str, ...], output: str | None, description: str | None) -> None:
    """Create a new pipeline."""
    if not action:
        console.print("❌ No actions specified")
        raise click.ClickException("Specify at least one action with --action")
    actions = []
    for action_str in action:
        action_spec = _parse_action_string(action_str)
        actions.append(action_spec)
    pipeline_obj = Pipeline(
        name=name,
        actions=tuple(actions),
        description=description or f"Pipeline with {len(actions)} actions",  # type: ignore[call-arg]
    )
    output_path = Path(f"{name}.json") if not output else Path(output)
    save_pipeline(pipeline_obj, output_path)
    console.print(f"✅ Created pipeline: {output_path}")
    console.print(f"🔧 Actions: {len(actions)}")
    for i, action_spec in enumerate(actions, 1):
        console.print(f"  {i}. {action_spec.name}")


@pipeline.command("list")
@click.option("--directory", "-d", type=click.Path(exists=True), default=".")
def list_pipelines(directory: str) -> None:
    """List pipeline files in directory."""
    directory_path = Path(directory)
    pipeline_files = list(directory_path.glob("*.json"))
    if not pipeline_files:
        console.print("No pipeline files found")
        return
    console.print(f"🔧 Pipeline files in {directory}:")
    for pipeline_file in pipeline_files:
        try:
            pipeline_obj = load_pipeline(pipeline_file)
            console.print(f"  • {pipeline_file.name} - {pipeline_obj.name} ({pipeline_obj.action_count} actions)")
        except Exception as e:
            console.print(f"  • {pipeline_file.name} - [red]Error loading: {e}[/red]")


@pipeline.command("run")
@click.argument("pipeline_file", type=click.Path(exists=True))
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output file path")
@click.option("--dry-run", is_flag=True, help="Show what would be done without processing")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def run_pipeline(
    ctx: click.Context,
    pipeline_file: str,
    input_path: str,
    output: str | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Run a pipeline on an image."""
    ctx.invoke(
        process,
        input_path=input_path,
        action=(),
        output=output,
        pipeline=pipeline_file,
        dry_run=dry_run,
        verbose=verbose,
    )


@cli.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False),
    help="Output directory (default: <input_dir>_processed)",
)
@click.option("--pattern", default="*.jpg", help="File pattern to match (default: *.jpg)")
@click.option(
    "--action",
    "-a",
    multiple=True,
    help="Action to apply (format: action_name:param=value,param2=value2)",
)
@click.option(
    "--pipeline",
    "-p",
    type=click.Path(exists=True),
    help="Pipeline file to use",
)
@click.option("--dry-run", is_flag=True, help="Show what would be done without processing")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--max-workers",
    type=int,
    default=4,
    help="Maximum number of worker threads (default: 4)",
)
@click.pass_context
def batch(  # noqa: PLR0915
    # TODO: Refactor this function to reduce complexity (60 statements > 50)
    ctx: click.Context,
    input_dir: str,
    output_dir: str | None,
    pattern: str,
    action: tuple[str, ...],
    pipeline: str | None,
    dry_run: bool,
    verbose: bool,
    max_workers: int,  # noqa: ARG001
) -> None:
    """Process multiple images in a directory."""
    verbose = verbose or ctx.obj.get("verbose", False)

    # Convert string paths to Path objects
    input_dir_path = Path(input_dir)
    output_dir_path = Path(output_dir) if output_dir else None
    pipeline_path_obj = Path(pipeline) if pipeline else None

    input_files = list(input_dir_path.glob(pattern))
    if not input_files:
        console.print(f"❌ No files found matching pattern '{pattern}' in {input_dir_path}")
        return
    if not output_dir_path:
        output_dir_path = input_dir_path.parent / f"{input_dir_path.name}_processed"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    if pipeline:
        pipeline_obj = load_pipeline(pipeline_path_obj)  # type: ignore[arg-type]
        actions_to_apply = pipeline_obj.enabled_actions
        console.print(f"🔧 Using pipeline: {pipeline_obj.name}")
    elif action:
        actions_to_apply = []  # type: ignore[assignment]
        for action_str in action:
            action_spec = _parse_action_string(action_str)
            actions_to_apply.append(action_spec)  # type: ignore[attr-defined]
        console.print(f"🔧 Using {len(actions_to_apply)} action(s)")
    else:
        console.print("❌ No actions or pipeline specified")
        raise click.ClickException("Specify either --action or --pipeline")
    console.print(f"📁 Input directory: {input_dir}")
    console.print(f"📁 Output directory: {output_dir}")
    console.print(f"📊 Found {len(input_files)} files to process")
    if verbose or dry_run:
        console.print("\n[bold]Processing plan:[/bold]")
        for i, action_spec in enumerate(actions_to_apply, 1):
            console.print(f"  {i}. {action_spec.name}")
            for param, value in action_spec.parameters.items():
                console.print(f"     • {param}: {value}")
    if dry_run:
        console.print("\n[yellow]Dry run - no processing performed[/yellow]")
        console.print("Files that would be processed:")
        for file_path in input_files:
            output_file = output_dir_path / file_path.name
            console.print(f"  • {file_path.name} → {output_file}")
        return
    failed_files = []
    successful_files = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing images...", total=len(input_files))
        for input_file in input_files:
            try:
                image = load_image_asset(input_file)
                processed_image = image
                for action_spec in actions_to_apply:
                    action_instance = default_registry.get_action(action_spec.name)
                    processed_image = action_instance.execute(processed_image, **action_spec.parameters)
                output_file = output_dir_path / input_file.name
                if not default_processor:
                    raise ImageProcessingError("PIL/Pillow not available for image saving")
                saved_path = default_processor.save_image(processed_image, output_file)
                successful_files.append((input_file, saved_path))
                if verbose:
                    file_size_mb = saved_path.stat().st_size / (1024 * 1024)  # type: ignore[attr-defined]
                    console.print(
                        f"✅ {input_file.name} → {processed_image.width}x{processed_image.height} "
                        f"({file_size_mb:.1f}MB)"
                    )
            except Exception as e:
                failed_files.append((input_file, str(e)))
                if verbose:
                    console.print(f"❌ {input_file.name}: {e}")
            progress.update(task, advance=1)
    console.print("\n📊 Processing complete:")
    console.print(f"  ✅ Successful: {len(successful_files)}")
    console.print(f"  ❌ Failed: {len(failed_files)}")
    if failed_files and verbose:
        console.print("\n[red]Failed files:[/red]")
        for file_path, error in failed_files:
            console.print(f"  • {file_path.name}: {error}")


def _parse_action_string(action_str: str) -> ActionSpec:
    """Parse action string format: action_name:param=value,param2=value2"""
    if ":" not in action_str:
        return ActionSpec(name=action_str, parameters={})
    name, params_str = action_str.split(":", 1)
    parameters = {}
    if params_str:
        for param_pair in params_str.split(","):
            if "=" in param_pair:
                key, value = param_pair.split("=", 1)
                if value.lower() in ("true", "false"):
                    parameters[key] = value.lower() == "true"
                elif value.isdigit():
                    parameters[key] = int(value)  # type: ignore[assignment]
                elif value.replace(".", "").isdigit():
                    parameters[key] = float(value)  # type: ignore[assignment]
                else:
                    parameters[key] = value  # type: ignore[assignment]
    return ActionSpec(name=name, parameters=parameters)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--threshold",
    "-t",
    type=int,
    default=5,
    help="Hamming distance threshold for duplicate detection (lower = more strict)",
)
@click.option(
    "--output-format",
    "-f",
    type=click.Choice(["table", "json", "paths"]),
    default="table",
    help="Output format for duplicate groups",
)
@click.pass_context
def duplicates(ctx: click.Context, input_path: str, threshold: int, output_format: str) -> None:
    """Find duplicate and near-duplicate images using perceptual hashing.

    INPUT_PATH can be a single image file or directory containing images.
    """
    verbose = ctx.obj.get("verbose", False)
    if verbose:
        console.print(f"🔍 Scanning for duplicates with threshold={threshold}")

    # Convert string path to Path object
    input_path_obj = Path(input_path)

    if input_path_obj.is_file():
        image_files = [input_path_obj]
    else:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        image_files = [f for f in input_path_obj.rglob("*") if f.is_file() and f.suffix.lower() in extensions]
    if not image_files:
        console.print("[red]No image files found![/red]")
        return
    if verbose:
        console.print(f"📁 Found {len(image_files)} image files")
    detector = DuplicateDetector(threshold=threshold)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing images...", total=1)
        try:
            duplicate_groups = detector.find_duplicates(image_files)  # type: ignore[arg-type]
            progress.update(task, completed=1)
        except Exception as e:
            console.print(f"[red]Error during duplicate detection: {e}[/red]")
            return
    if not duplicate_groups:
        console.print("✅ No duplicates found!")
        return
    console.print(f"🔍 Found {len(duplicate_groups)} duplicate groups:")
    if output_format == "table":
        _display_duplicates_table(duplicate_groups, verbose)
    elif output_format == "json":
        _display_duplicates_json(duplicate_groups)
    elif output_format == "paths":
        _display_duplicates_paths(duplicate_groups)


def _display_duplicates_table(duplicate_groups: list[list[Path]], verbose: bool) -> None:
    """Display duplicate groups in a table format."""
    for i, group in enumerate(duplicate_groups, 1):
        table = Table(title=f"Duplicate Group {i}")
        table.add_column("File", style="cyan")
        table.add_column("Size", style="magenta")
        if verbose:
            table.add_column("Path", style="dim")
        for file_path in group:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            row = [file_path.name, f"{size_mb:.1f}MB"]
            if verbose:
                row.append(str(file_path.parent))
            table.add_row(*row)
        console.print(table)
        console.print()


def _display_duplicates_json(duplicate_groups: list[list[Path]]) -> None:
    """Display duplicate groups in JSON format."""
    groups_data = [[str(path) for path in group] for group in duplicate_groups]
    console.print(json.dumps(groups_data, indent=2))


def _display_duplicates_paths(duplicate_groups: list[list[Path]]) -> None:
    """Display duplicate groups as simple paths."""
    for i, group in enumerate(duplicate_groups, 1):
        console.print(f"Group {i}:")
        for file_path in group:
            console.print(f"  {file_path}")
        console.print()


def main() -> None:
    """Main entry point for CLI."""
    cli.add_command(metadata)
    cli()


if __name__ == "__main__":
    main()
