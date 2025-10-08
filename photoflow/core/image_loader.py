"""Image loading utilities for PhotoFlow.

This module provides functions to load images from files and convert them
to ImageAsset objects with proper metadata extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .models import ImageAsset

# Constants
RATIONAL_TUPLE_LENGTH = 2

try:
    import piexif

    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False

try:
    from iptcinfo3 import IPTCInfo

    IPTCINFO_AVAILABLE = True
except ImportError:
    IPTCInfo = None
    IPTCINFO_AVAILABLE = False

try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    Image = None  # type: ignore[assignment]
    PIL_AVAILABLE = False


class ImageLoadError(Exception):
    """Raised when image loading fails."""

    pass


def load_image_asset(image_path: Path | str) -> ImageAsset:
    """Load an image file and create an ImageAsset with extracted metadata.

    Args:
        image_path: Path to the image file

    Returns:
        ImageAsset object with metadata extracted from the file

    Raises:
        ImageLoadError: If the image cannot be loaded
        ImportError: If required dependencies are not available
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL (Pillow) is required for image loading")
    image_path = Path(image_path)
    if not image_path.exists():
        raise ImageLoadError(f"Image file not found: {image_path}")
    try:
        with Image.open(image_path) as pil_image:
            width, height = pil_image.size
            mode = pil_image.mode
            format_name = pil_image.format or "UNKNOWN"
            exif_data = {}
            if pil_image.info.get("exif"):
                exif_data = _extract_exif_data(pil_image.info.get("exif", b""))
            if not exif_data and hasattr(pil_image, "info"):
                exif_data = _extract_pil_info(pil_image.info)
            iptc_data = _extract_iptc_data(image_path)
            dpi = getattr(pil_image, "info", {}).get("dpi", (72, 72))
            if isinstance(dpi, int | float):
                dpi = int(dpi), int(dpi)
    except Exception as e:
        raise ImageLoadError(f"Failed to load image {image_path}: {e}") from e
    return ImageAsset(
        path=image_path,
        format=format_name,
        size=(width, height),
        mode=mode,
        exif=exif_data,
        iptc=iptc_data,
        dpi=dpi,
        color_profile=None,
    )


def create_image_asset_from_pil(pil_image: Image.Image, path: Path | str, format_name: str | None = None) -> ImageAsset:
    """Create an ImageAsset from a PIL Image object.

    Args:
        pil_image: PIL Image object
        path: Path to associate with the image
        format_name: Image format name (detected if None)

    Returns:
        ImageAsset object

    Raises:
        ImportError: If PIL is not available
    """
    if not PIL_AVAILABLE:
        raise ImportError("PIL (Pillow) is required for PIL image conversion")
    path = Path(path)
    width, height = pil_image.size
    mode = pil_image.mode
    format_name = format_name or pil_image.format or "RGB"
    exif_data = {}
    if hasattr(pil_image, "info") and pil_image.info:
        exif_data = _extract_pil_info(pil_image.info)
    dpi = getattr(pil_image, "info", {}).get("dpi", (72, 72))
    if isinstance(dpi, int | float):
        dpi = int(dpi), int(dpi)
    return ImageAsset(
        path=path,
        format=format_name,
        size=(width, height),
        mode=mode,
        exif=exif_data,
        iptc={},
        dpi=dpi,
        color_profile=None,
    )


def _parse_exif_data(exif_dict: dict[str, Any]) -> dict[str, Any]:
    """Parse EXIF dictionary into a more usable format.

    Args:
        exif_dict: Raw EXIF dictionary from piexif

    Returns:
        Cleaned EXIF data dictionary
    """
    if not PIEXIF_AVAILABLE:
        return {}
    parsed_exif = {}
    if "0th" in exif_dict:
        for tag, value in exif_dict["0th"].items():
            tag_name = piexif.TAGS["0th"].get(tag, {}).get("name", f"Tag{tag}")
            parsed_exif[tag_name] = _clean_exif_value(value)
    if "Exif" in exif_dict:
        for tag, value in exif_dict["Exif"].items():
            tag_name = piexif.TAGS["Exif"].get(tag, {}).get("name", f"ExifTag{tag}")
            parsed_exif[tag_name] = _clean_exif_value(value)
    return parsed_exif


def _clean_exif_value(value: Any) -> Any:
    """Clean and convert EXIF values to Python types.

    Args:
        value: Raw EXIF value

    Returns:
        Cleaned value
    """
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8").rstrip("\x00")
        except UnicodeDecodeError:
            return value.hex()
    elif isinstance(value, tuple) and len(value) == RATIONAL_TUPLE_LENGTH:
        if value[1] != 0:
            return value[0], value[1]
        else:
            return value[0]
    else:
        return value


def _extract_pil_info(info_dict: dict[str | tuple[int, int], Any] | None) -> dict[str, Any]:
    """Extract metadata from PIL image info dictionary.

    Args:
        info_dict: PIL image info dictionary

    Returns:
        Extracted metadata
    """
    if not info_dict:
        return {}
    extracted = {}
    key_mapping = {
        "dpi": "DPI",
        "quality": "Quality",
        "optimize": "Optimize",
        "progressive": "Progressive",
        "transparency": "Transparency",
    }
    for pil_key, our_key in key_mapping.items():
        if pil_key in info_dict:
            extracted[our_key] = info_dict[pil_key]
    return extracted


def _extract_exif_data(exif_bytes: bytes) -> dict[str, Any]:
    """Extract EXIF data from bytes.

    Args:
        exif_bytes: Raw EXIF data bytes

    Returns:
        Parsed EXIF data dictionary
    """
    if not PIEXIF_AVAILABLE or not exif_bytes:
        return {}
    try:
        exif_dict = piexif.load(exif_bytes)
        return _parse_exif_data(exif_dict)
    except Exception:
        return {}


def _extract_iptc_data(image_path: Path) -> dict[str, Any]:
    """Extract IPTC data from image file.

    Args:
        image_path: Path to image file

    Returns:
        IPTC data dictionary
    """
    if not IPTCINFO_AVAILABLE:
        return {}
    try:
        info = IPTCInfo(str(image_path))
    except Exception:
        return {}
    data = getattr(info, "data", None)
    if isinstance(data, dict):
        return dict(data)
    return {}


def load_test_image(image_path: Path | str) -> ImageAsset:
    """Load a test image - alias for load_image_asset.

    Args:
        image_path: Path to the image file

    Returns:
        ImageAsset object
    """
    return load_image_asset(image_path)
