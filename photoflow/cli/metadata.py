"""Metadata commands for PhotoFlow CLI."""

import json
from pathlib import Path
from typing import Any, Optional, Union

import click
import yaml
from PIL import Image
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..core.metadata import (
    ExifData,
    IPTCData,
    MetadataError,
    read_metadata,
    write_metadata,
)

console = Console()


@click.group()
def metadata() -> None:
    """Metadata inspection and manipulation commands."""
    pass


@metadata.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
@click.option("--exif-only", is_flag=True, help="Show only EXIF data")
@click.option("--iptc-only", is_flag=True, help="Show only IPTC data")
def inspect(image_path: str, format: str, exif_only: bool, iptc_only: bool) -> None:
    """Inspect metadata in an image file."""
    try:
        image_path_obj = Path(image_path)
        exif_data, iptc_data = read_metadata(image_path_obj)
        if format == "json":
            result = {}
            if not iptc_only:
                result["exif"] = exif_data.to_dict()
            if not exif_only:
                result["iptc"] = iptc_data.to_dict()
            console.print(json.dumps(result, indent=2))
        elif format == "yaml":
            result = {}
            if not iptc_only:
                result["exif"] = exif_data.to_dict()
            if not exif_only:
                result["iptc"] = iptc_data.to_dict()
            console.print(yaml.dump(result, default_flow_style=False))
        else:
            _display_metadata_table(image_path_obj, exif_data, iptc_data, exif_only, iptc_only)
    except MetadataError as e:
        console.print(f"[red]Error reading metadata: {e}[/red]")
        raise click.Abort() from e
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise click.Abort() from e


@metadata.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (overwrites input if not specified)",
)
@click.option("--title", help="Set image title")
@click.option("--caption", help="Set image caption")
@click.option("--creator", help="Set image creator/photographer")
@click.option("--copyright", help="Set copyright notice")
@click.option("--keywords", help="Set keywords (comma-separated)")
@click.option("--location", help="Set location")
@click.option("--city", help="Set city")
@click.option("--state", help="Set state/province")
@click.option("--country", help="Set country")
@click.option("--credit", help="Set credit line")
@click.option("--source", help="Set source")
def write(image_path: str, output: Optional[str], **kwargs: Any) -> None:
    """Write metadata to an image file."""
    try:
        image_path_obj = Path(image_path)
        output_path_obj = Path(output) if output else None
        exif_data, iptc_data = read_metadata(image_path_obj)
        if kwargs.get("title"):
            iptc_data.title = kwargs["title"]
        if kwargs.get("caption"):
            iptc_data.description = kwargs["caption"]
        if kwargs.get("creator"):
            iptc_data.byline = kwargs["creator"]
            exif_data.creator = kwargs["creator"]
        if kwargs.get("copyright"):
            iptc_data.copyright_notice = kwargs["copyright"]
            exif_data.copyright = kwargs["copyright"]
        if kwargs.get("keywords"):
            iptc_data.keywords = [kw.strip() for kw in kwargs["keywords"].split(",")]
        if kwargs.get("location"):
            iptc_data.location = kwargs["location"]
        if kwargs.get("city"):
            iptc_data.city = kwargs["city"]
        if kwargs.get("state"):
            iptc_data.state = kwargs["state"]
        if kwargs.get("country"):
            iptc_data.country = kwargs["country"]
        if kwargs.get("credit"):
            iptc_data.credit = kwargs["credit"]
        if kwargs.get("source"):
            iptc_data.source = kwargs["source"]
        write_metadata(image_path_obj, exif_data, iptc_data, output_path_obj)
        output_path = output or image_path
        console.print(f"[green]Metadata written to {output_path}[/green]")
    except MetadataError as e:
        console.print(f"[red]Error writing metadata: {e}[/red]")
        raise click.Abort() from e
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise click.Abort() from e


@metadata.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (overwrites input if not specified)",
)
@click.option("--exif", is_flag=True, help="Remove EXIF data")
@click.option("--iptc", is_flag=True, help="Remove IPTC data")
@click.option("--all", "remove_all", is_flag=True, help="Remove all metadata")
def remove(image_path: str, output: Optional[str], exif: bool, iptc: bool, remove_all: bool) -> None:
    """Remove metadata from an image file."""
    if not (exif or iptc or remove_all):
        console.print("[yellow]No metadata types specified. Use --exif, --iptc, or --all[/yellow]")
        return
    try:
        image_path_obj = Path(image_path)
        output_path_obj = Path(output) if output else image_path_obj
        with Image.open(image_path_obj) as img:
            clean_img = Image.new(img.mode, img.size)
            clean_img.putdata(list(img.getdata()))
            if hasattr(img, "info"):
                clean_info: dict[Union[str, tuple[int, int]], Any] = {}
                if not remove_all:
                    for key in ["dpi", "format"]:
                        if key in img.info:
                            clean_info[key] = img.info[key]
                clean_img.info = clean_info
            save_kwargs: dict[str, Any] = {}
            if img.format:
                save_kwargs["format"] = img.format
            clean_img.save(output_path_obj, **save_kwargs)
        console.print(f"[green]Metadata removed from {output_path_obj}[/green]")
    except Exception as e:
        console.print(f"[red]Error removing metadata: {e}[/red]")
        raise click.Abort() from e


@metadata.command()
@click.argument("source_path", type=click.Path(exists=True))
@click.argument("target_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (overwrites target if not specified)",
)
@click.option("--exif-only", is_flag=True, help="Copy only EXIF data")
@click.option("--iptc-only", is_flag=True, help="Copy only IPTC data")
def copy(
    source_path: str,
    target_path: str,
    output: Optional[str],
    exif_only: bool,
    iptc_only: bool,
) -> None:
    """Copy metadata from one image to another."""
    try:
        source_path_obj = Path(source_path)
        target_path_obj = Path(target_path)
        output_path_obj = Path(output) if output else target_path_obj
        source_exif, source_iptc = read_metadata(source_path_obj)
        target_exif, target_iptc = read_metadata(target_path_obj)
        copy_exif = source_exif if not iptc_only else target_exif
        copy_iptc = source_iptc if not exif_only else target_iptc
        write_metadata(target_path_obj, copy_exif, copy_iptc, output_path_obj)
        console.print(f"[green]Metadata copied from {source_path_obj} to {output_path_obj}[/green]")
    except MetadataError as e:
        console.print(f"[red]Error copying metadata: {e}[/red]")
        raise click.Abort() from e
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise click.Abort() from e


def _display_metadata_table(  # noqa: PLR0915
    # TODO: Refactor this function to reduce complexity (89 statements > 50)
    image_path: Path,
    exif_data: ExifData,
    iptc_data: IPTCData,
    exif_only: bool,
    iptc_only: bool,
) -> None:
    """Display metadata in rich table format."""
    console.print(
        Panel(
            f"[bold]Metadata for: {image_path.name}[/bold]",
            title="Image Metadata Inspector",
        )
    )
    panels = []
    if not iptc_only and any(
        getattr(exif_data, field) is not None for field in exif_data.__dataclass_fields__ if field != "raw_exif"
    ):
        exif_table = Table(title="EXIF Data", show_header=True, header_style="bold blue")
        exif_table.add_column("Field", style="cyan", width=20)
        exif_table.add_column("Value", style="white")
        if exif_data.make or exif_data.model:
            camera = f"{exif_data.make or ''} {exif_data.model or ''}".strip()
            if camera:
                exif_table.add_row("Camera", camera)
        if exif_data.lens_make or exif_data.lens_model:
            lens = f"{exif_data.lens_make or ''} {exif_data.lens_model or ''}".strip()
            if lens:
                exif_table.add_row("Lens", lens)
        if exif_data.focal_length:
            exif_table.add_row("Focal Length", exif_data.focal_length)
        if exif_data.f_number:
            exif_table.add_row("Aperture", exif_data.f_number)
        if exif_data.exposure_time:
            exif_table.add_row("Shutter Speed", exif_data.exposure_time)
        if exif_data.iso_speed:
            exif_table.add_row("ISO", str(exif_data.iso_speed))
        if exif_data.width and exif_data.height:
            exif_table.add_row("Dimensions", f"{exif_data.width} x {exif_data.height}")
        if exif_data.color_space:
            exif_table.add_row("Color Space", exif_data.color_space)
        if exif_data.datetime_original:
            if hasattr(exif_data.datetime_original, "strftime"):
                exif_table.add_row("Date Taken", exif_data.datetime_original.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                exif_table.add_row("Date Taken", str(exif_data.datetime_original))
        if exif_data.datetime_digitized:
            if hasattr(exif_data.datetime_digitized, "strftime"):
                exif_table.add_row(
                    "Date Digitized",
                    exif_data.datetime_digitized.strftime("%Y-%m-%d %H:%M:%S"),
                )
            else:
                exif_table.add_row("Date Digitized", str(exif_data.datetime_digitized))
        if exif_data.gps_latitude and exif_data.gps_longitude:
            exif_table.add_row(
                "GPS Coordinates",
                f"{exif_data.gps_latitude:.6f}, {exif_data.gps_longitude:.6f}",
            )
        if exif_data.gps_altitude:
            exif_table.add_row("GPS Altitude", f"{exif_data.gps_altitude:.1f}m")
        if exif_data.software:
            exif_table.add_row("Software", exif_data.software)
        if exif_data.creator:
            exif_table.add_row("Creator", exif_data.creator)
        if exif_data.copyright:
            exif_table.add_row("Copyright", exif_data.copyright)
        panels.append(Panel(exif_table, title="EXIF"))
    if not exif_only and any(
        getattr(iptc_data, field) is not None for field in iptc_data.__dataclass_fields__ if field != "raw_iptc"
    ):
        iptc_table = Table(title="IPTC Data", show_header=True, header_style="bold green")
        iptc_table.add_column("Field", style="cyan", width=20)
        iptc_table.add_column("Value", style="white")
        if iptc_data.title:
            iptc_table.add_row("Title", iptc_data.title)
        if iptc_data.description:
            iptc_table.add_row("Caption", iptc_data.description)
        if iptc_data.keywords:
            iptc_table.add_row("Keywords", ", ".join(iptc_data.keywords))
        if iptc_data.category:
            iptc_table.add_row("Category", iptc_data.category)
        if iptc_data.byline:
            iptc_table.add_row("Photographer", iptc_data.byline)
        if iptc_data.byline_title:
            iptc_table.add_row("Job Title", iptc_data.byline_title)
        if iptc_data.credit:
            iptc_table.add_row("Credit", iptc_data.credit)
        if iptc_data.source:
            iptc_table.add_row("Source", iptc_data.source)
        if iptc_data.copyright_notice:
            iptc_table.add_row("Copyright", iptc_data.copyright_notice)
        location_parts = []
        if iptc_data.location:
            location_parts.append(iptc_data.location)
        if iptc_data.city:
            location_parts.append(iptc_data.city)
        if iptc_data.state:
            location_parts.append(iptc_data.state)
        if iptc_data.country:
            location_parts.append(iptc_data.country)
        if location_parts:
            iptc_table.add_row("Location", ", ".join(location_parts))
        if iptc_data.date_created:
            if hasattr(iptc_data.date_created, "strftime"):
                iptc_table.add_row("Date Created", iptc_data.date_created.strftime("%Y-%m-%d"))
            else:
                iptc_table.add_row("Date Created", str(iptc_data.date_created))
        if iptc_data.urgency:
            iptc_table.add_row("Urgency", f"{iptc_data.urgency}/8")
        if iptc_data.instructions:
            iptc_table.add_row("Instructions", iptc_data.instructions)
        panels.append(Panel(iptc_table, title="IPTC"))
    if panels:
        if len(panels) == 1:
            console.print(panels[0])
        else:
            console.print(Columns(panels, equal=True))
    else:
        console.print("[yellow]No metadata found in image[/yellow]")
