"""Metadata manipulation actions for PhotoFlow."""

import csv
import json
from pathlib import Path
from typing import Any, ClassVar, Union

import yaml
from PIL import Image as PILImage

from ..core.actions import BaseAction
from ..core.fields import BooleanField, ChoiceField, StringField
from ..core.metadata import (
    ExifData,
    IPTCData,
    MetadataError,
    read_metadata,
    write_metadata,
)
from ..core.models import ImageAsset


class WriteMetadataAction(BaseAction):
    """Write metadata tags to image."""

    name = "write_metadata"
    label = "Write Metadata"
    description = "Add or update EXIF and IPTC metadata in image"
    version = "1.0.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["metadata", "exif", "iptc"]

    def setup_fields(self) -> None:
        """Set up action fields."""
        self.fields = {
            "title": StringField(label="Title", help_text="Image title", required=False),
            "caption": StringField(label="Caption", help_text="Image caption/description", required=False),
            "creator": StringField(label="Creator", help_text="Photographer/creator name", required=False),
            "copyright": StringField(label="Copyright", help_text="Copyright notice", required=False),
            "keywords": StringField(label="Keywords", help_text="Comma-separated keywords", required=False),
            "location": StringField(
                label="Location",
                help_text="Specific location/sublocation",
                required=False,
            ),
            "city": StringField(label="City", help_text="City name", required=False),
            "state": StringField(label="State/Province", help_text="State or province", required=False),
            "country": StringField(label="Country", help_text="Country name", required=False),
            "credit": StringField(label="Credit", help_text="Credit line", required=False),
            "source": StringField(label="Source", help_text="Image source", required=False),
            "overwrite": BooleanField(
                label="Overwrite Existing",
                help_text="Overwrite existing metadata values",
                default=True,
            ),
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply metadata writing to image."""
        try:
            exif_data, iptc_data = read_metadata(image.path)
            overwrite = params.get("overwrite", True)
            if params.get("creator") and (overwrite or not exif_data.creator):
                exif_data.creator = params["creator"]
            if params.get("copyright") and (overwrite or not exif_data.copyright):
                exif_data.copyright = params["copyright"]
            if params.get("title") and (overwrite or not iptc_data.title):
                iptc_data.title = params["title"]
            if params.get("caption") and (overwrite or not iptc_data.description):
                iptc_data.description = params["caption"]
            if params.get("creator") and (overwrite or not iptc_data.byline):
                iptc_data.byline = params["creator"]
            if params.get("copyright") and (overwrite or not iptc_data.copyright_notice):
                iptc_data.copyright_notice = params["copyright"]
            if params.get("keywords") and (overwrite or not iptc_data.keywords):
                if isinstance(params["keywords"], str):
                    iptc_data.keywords = [kw.strip() for kw in params["keywords"].split(",") if kw.strip()]
                elif isinstance(params["keywords"], list):
                    iptc_data.keywords = params["keywords"]
            if params.get("location") and (overwrite or not iptc_data.location):
                iptc_data.location = params["location"]
            if params.get("city") and (overwrite or not iptc_data.city):
                iptc_data.city = params["city"]
            if params.get("state") and (overwrite or not iptc_data.state):
                iptc_data.state = params["state"]
            if params.get("country") and (overwrite or not iptc_data.country):
                iptc_data.country = params["country"]
            if params.get("credit") and (overwrite or not iptc_data.credit):
                iptc_data.credit = params["credit"]
            if params.get("source") and (overwrite or not iptc_data.source):
                iptc_data.source = params["source"]
            write_metadata(image.path, exif_data, iptc_data)
            updated_metadata = image.metadata.copy()
            updated_metadata.update(
                {
                    "metadata_written": True,
                    "last_metadata_update": "write_metadata_action",
                }
            )
            return image.with_updates(metadata=updated_metadata)
        except MetadataError as e:
            raise ValueError(f"Failed to write metadata: {e}") from e


class RemoveMetadataAction(BaseAction):
    """Remove metadata from image."""

    name = "remove_metadata"
    label = "Remove Metadata"
    description = "Remove EXIF and/or IPTC metadata from image"
    version = "1.0.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["metadata", "privacy", "cleanup"]

    def setup_fields(self) -> None:
        """Set up action fields."""
        self.fields = {
            "metadata_type": ChoiceField(
                label="Metadata Type",
                help_text="Type of metadata to remove",
                choices=[
                    ("all", "All metadata"),
                    ("exif", "EXIF data only"),
                    ("iptc", "IPTC data only"),
                    ("gps", "GPS data only"),
                    ("personal", "Personal info (creator, copyright, location)"),
                ],
                default="all",
            ),
            "preserve_basic": BooleanField(
                label="Preserve Basic Info",
                help_text="Keep essential image properties (dimensions, format)",
                default=True,
            ),
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply metadata removal to image."""
        metadata_type = params.get("metadata_type", "all")
        preserve_basic = params.get("preserve_basic", True)
        try:
            if metadata_type == "all":
                with PILImage.open(image.path) as pil_img:
                    clean_img = PILImage.new(pil_img.mode, pil_img.size)
                    clean_img.putdata(list(pil_img.getdata()))
                    if preserve_basic and hasattr(pil_img, "info"):
                        clean_info: dict[Union[str, tuple[int, int]], Any] = {}
                        for key in ["dpi", "format"]:
                            if key in pil_img.info:
                                clean_info[key] = pil_img.info[key]
                        clean_img.info = clean_info
                    save_kwargs: dict[str, Any] = {}
                    if pil_img.format:
                        save_kwargs["format"] = pil_img.format
                    clean_img.save(image.path, **save_kwargs)
            else:
                exif_data, iptc_data = read_metadata(image.path)
                if metadata_type == "exif":
                    exif_data = ExifData()
                elif metadata_type == "iptc":
                    iptc_data = IPTCData()
                elif metadata_type == "gps":
                    exif_data.gps_latitude = None
                    exif_data.gps_longitude = None
                    exif_data.gps_altitude = None
                    exif_data.gps_timestamp = None
                elif metadata_type == "personal":
                    exif_data.creator = None
                    exif_data.copyright = None
                    exif_data.gps_latitude = None
                    exif_data.gps_longitude = None
                    exif_data.gps_altitude = None
                    iptc_data.byline = None
                    iptc_data.byline_title = None
                    iptc_data.copyright_notice = None
                    iptc_data.location = None
                    iptc_data.city = None
                    iptc_data.state = None
                    iptc_data.country = None
                write_metadata(image.path, exif_data, iptc_data)
            updated_metadata = image.metadata.copy()
            updated_metadata.update(
                {
                    "metadata_removed": metadata_type,
                    "last_metadata_update": "remove_metadata_action",
                }
            )
            return image.with_updates(metadata=updated_metadata)
        except Exception as e:
            raise ValueError(f"Failed to remove metadata: {e}") from e


class CopyMetadataAction(BaseAction):
    """Copy metadata from another image."""

    name = "copy_metadata"
    label = "Copy Metadata"
    description = "Copy metadata from another image file"
    version = "1.0.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["metadata", "copy", "transfer"]

    def setup_fields(self) -> None:
        """Set up action fields."""
        self.fields = {
            "source_image": StringField(
                label="Source Image",
                help_text="Path to image to copy metadata from",
                required=True,
            ),
            "metadata_type": ChoiceField(
                label="Metadata Type",
                help_text="Type of metadata to copy",
                choices=[
                    ("all", "All metadata"),
                    ("exif", "EXIF data only"),
                    ("iptc", "IPTC data only"),
                    ("camera", "Camera settings only"),
                    ("location", "Location data only"),
                ],
                default="all",
            ),
            "overwrite": BooleanField(
                label="Overwrite Existing",
                help_text="Overwrite existing metadata values",
                default=True,
            ),
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:  # noqa: PLR0915
        """Apply metadata copying to image."""
        # TODO: Refactor this method to reduce complexity (58 statements > 50)
        source_path = Path(params["source_image"])
        metadata_type = params.get("metadata_type", "all")
        overwrite = params.get("overwrite", True)
        if not source_path.exists():
            raise ValueError(f"Source image not found: {source_path}")
        try:
            source_exif, source_iptc = read_metadata(source_path)
            target_exif, target_iptc = read_metadata(image.path)
            if metadata_type == "all":
                copy_exif = source_exif
                copy_iptc = source_iptc
            elif metadata_type == "exif":
                copy_exif = source_exif
                copy_iptc = target_iptc
            elif metadata_type == "iptc":
                copy_exif = target_exif
                copy_iptc = source_iptc
            elif metadata_type == "camera":
                copy_exif = target_exif
                if overwrite or not copy_exif.make:
                    copy_exif.make = source_exif.make
                if overwrite or not copy_exif.model:
                    copy_exif.model = source_exif.model
                if overwrite or not copy_exif.lens_make:
                    copy_exif.lens_make = source_exif.lens_make
                if overwrite or not copy_exif.lens_model:
                    copy_exif.lens_model = source_exif.lens_model
                if overwrite or not copy_exif.focal_length:
                    copy_exif.focal_length = source_exif.focal_length
                if overwrite or not copy_exif.f_number:
                    copy_exif.f_number = source_exif.f_number
                if overwrite or not copy_exif.exposure_time:
                    copy_exif.exposure_time = source_exif.exposure_time
                if overwrite or not copy_exif.iso_speed:
                    copy_exif.iso_speed = source_exif.iso_speed
                copy_iptc = target_iptc
            elif metadata_type == "location":
                copy_exif = target_exif
                if overwrite or not copy_exif.gps_latitude:
                    copy_exif.gps_latitude = source_exif.gps_latitude
                    copy_exif.gps_longitude = source_exif.gps_longitude
                    copy_exif.gps_altitude = source_exif.gps_altitude
                    copy_exif.gps_timestamp = source_exif.gps_timestamp
                copy_iptc = target_iptc
                if overwrite or not copy_iptc.location:
                    copy_iptc.location = source_iptc.location
                if overwrite or not copy_iptc.city:
                    copy_iptc.city = source_iptc.city
                if overwrite or not copy_iptc.state:
                    copy_iptc.state = source_iptc.state
                if overwrite or not copy_iptc.country:
                    copy_iptc.country = source_iptc.country
            write_metadata(image.path, copy_exif, copy_iptc)
            updated_metadata = image.metadata.copy()
            updated_metadata.update(
                {
                    "metadata_copied_from": str(source_path),
                    "metadata_copy_type": metadata_type,
                    "last_metadata_update": "copy_metadata_action",
                }
            )
            return image.with_updates(metadata=updated_metadata)
        except MetadataError as e:
            raise ValueError(f"Failed to copy metadata: {e}") from e


class ExtractMetadataAction(BaseAction):
    """Extract metadata to a file."""

    name = "extract_metadata"
    label = "Extract Metadata"
    description = "Extract metadata to JSON or YAML file"
    version = "1.0.0"
    author = "PhotoFlow"
    tags: ClassVar[list[str]] = ["metadata", "export", "extract"]

    def setup_fields(self) -> None:
        """Set up action fields."""
        self.fields = {
            "output_format": ChoiceField(
                label="Output Format",
                help_text="Format for extracted metadata",
                choices=[
                    ("json", "JSON format"),
                    ("yaml", "YAML format"),
                    ("csv", "CSV format (tabular)"),
                    ("txt", "Text format (human readable)"),
                ],
                default="json",
            ),
            "output_path": StringField(
                label="Output Path",
                help_text="Path for metadata file (uses image name if empty)",
                required=False,
            ),
            "include_raw": BooleanField(
                label="Include Raw Data",
                help_text="Include raw EXIF/IPTC data",
                default=False,
            ),
        }

    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply metadata extraction."""
        output_format = params.get("output_format", "json")
        output_path = params.get("output_path")
        include_raw = params.get("include_raw", False)
        try:
            exif_data, iptc_data = read_metadata(image.path)
            metadata_dict = {
                "image_path": str(image.path),
                "exif": exif_data.to_dict(),
                "iptc": iptc_data.to_dict(),
            }
            if include_raw:
                # Raw metadata not implemented yet
                metadata_dict["raw_exif"] = {}
                metadata_dict["raw_iptc"] = {}
            if not output_path:
                base_name = image.path.stem
                ext_map = {
                    "json": ".json",
                    "yaml": ".yaml",
                    "csv": ".csv",
                    "txt": ".txt",
                }
                output_path = image.path.parent / f"{base_name}_metadata{ext_map[output_format]}"
            else:
                output_path = Path(output_path)
            if output_format == "json":
                with Path(output_path).open("w", encoding="utf-8") as f:
                    json.dump(metadata_dict, f, indent=2, default=str)
            elif output_format == "yaml":
                with Path(output_path).open("w", encoding="utf-8") as f:
                    yaml.dump(metadata_dict, f, default_flow_style=False)
            elif output_format == "csv":
                flat_data = {}
                for section, data in metadata_dict.items():
                    if isinstance(data, dict):
                        for key, value in data.items():
                            flat_data[f"{section}_{key}"] = value
                    else:
                        flat_data[section] = data
                with Path(output_path).open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(flat_data.keys())
                    writer.writerow([str(v) for v in flat_data.values()])
            elif output_format == "txt":
                with Path(output_path).open("w", encoding="utf-8") as f:
                    f.write(f"Metadata for: {image.path}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write("EXIF Data:\n")
                    f.write("-" * 20 + "\n")
                    for key, value in exif_data.to_dict().items():
                        f.write(f"{key}: {value}\n")
                    f.write("\nIPTC Data:\n")
                    f.write("-" * 20 + "\n")
                    for key, value in iptc_data.to_dict().items():
                        f.write(f"{key}: {value}\n")
            updated_metadata = image.metadata.copy()
            updated_metadata.update(
                {
                    "metadata_extracted_to": str(output_path),
                    "metadata_extract_format": output_format,
                    "last_metadata_update": "extract_metadata_action",
                }
            )
            return image.with_updates(metadata=updated_metadata)
        except Exception as e:
            raise ValueError(f"Failed to extract metadata: {e}") from e


__all__ = [
    "CopyMetadataAction",
    "ExtractMetadataAction",
    "RemoveMetadataAction",
    "WriteMetadataAction",
]
