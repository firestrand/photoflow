"""Advanced metadata handling for PhotoFlow.

This module provides comprehensive EXIF and IPTC metadata support including
reading, writing, and manipulation of image metadata.
"""

from __future__ import annotations

import contextlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003
from typing import Any

# Constants
IPTC_DATE_LENGTH = 8
RATIONAL_TUPLE_LENGTH = 2
GPS_COORDINATE_PARTS = 3
TIME_STAMP_PARTS = 3
ADOBE_RGB_CODE = 2
UNCALIBRATED_CODE = 65535
ASCII_PRINTABLE_MIN = 32
ASCII_PRINTABLE_MAX = 126
GPS_ALTITUDE_REF_BELOW_SEA_LEVEL = 1

try:
    from PIL import Image  # noqa: F401

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import piexif
    from piexif import TAGS  # noqa: F401

    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False

try:
    from iptcinfo3 import IPTCInfo

    IPTCINFO_AVAILABLE = True
except ImportError:
    IPTCINFO_AVAILABLE = False


class MetadataError(Exception):
    """Base exception for metadata operations."""

    pass


class ExifError(MetadataError):
    """Exception raised for EXIF-related errors."""

    pass


class IPTCError(MetadataError):
    """Exception raised for IPTC-related errors."""

    pass


@dataclass
class ExifData:
    """Structured EXIF data container."""

    # Camera information
    make: str | None = None
    model: str | None = None
    lens_make: str | None = None
    lens_model: str | None = None

    # Shooting parameters
    exposure_time: str | None = None
    f_number: str | None = None
    aperture: float | None = None  # Added aperture field
    shutter_speed: float | None = None  # Added shutter_speed field
    iso_speed: int | None = None
    focal_length: str | None = None
    flash: str | None = None

    # Image dimensions and properties
    width: int | None = None
    height: int | None = None
    orientation: int | None = None
    resolution_unit: int | None = None
    x_resolution: float | None = None
    y_resolution: float | None = None

    # Date and time
    datetime_original: str | None = None
    datetime_digitized: str | None = None
    datetime: str | None = None

    # Color information
    color_space: str | None = None
    white_balance: str | None = None

    # GPS data
    gps_latitude: float | None = None
    gps_longitude: float | None = None
    gps_altitude: float | None = None
    gps_timestamp: str | None = None

    # Creator and copyright
    creator: str | None = None
    copyright: str | None = None

    # Additional metadata
    software: str | None = None
    processing_software: str | None = None
    datetime_modified: str | None = None  # Added for comprehensive tests

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class IPTCData:
    """Structured IPTC data container."""

    # Title and description
    title: str | None = None
    description: str | None = None
    caption: str | None = None  # Added caption field
    keywords: list[str] = field(default_factory=list)
    category: str | None = None
    supplemental_categories: list[str] = field(default_factory=list)

    # Creator information
    byline: str | None = None
    byline_title: str | None = None
    credit: str | None = None
    source: str | None = None

    # Copyright and usage
    copyright_notice: str | None = None
    usage_terms: str | None = None

    # Location information
    location: str | None = None
    city: str | None = None
    state: str | None = None
    country: str | None = None
    country_code: str | None = None

    # Date information
    date_created: str | None = None
    digital_creation_date: str | None = None

    # Editorial information
    headline: str | None = None
    instructions: str | None = None
    transmission_reference: str | None = None
    urgency: int | None = None

    # Raw IPTC data for compatibility
    raw_iptc: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result = {}
        for k, v in self.__dict__.items():
            if v is not None and v != []:
                result[k] = v
        return result


class MetadataReader:
    """Read metadata from images."""

    def __init__(self) -> None:
        """Initialize metadata reader."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL (Pillow) is required for metadata operations")
        self.exif_reader = ExifReader() if PIEXIF_AVAILABLE else None
        self.iptc_reader = IPTCReader() if IPTCINFO_AVAILABLE else None

    def read(self, image_path: Path) -> tuple[ExifData, IPTCData]:
        """Read all metadata from image."""
        exif_data = ExifData()
        iptc_data = IPTCData()

        if self.exif_reader:
            with contextlib.suppress(ExifError):
                exif_data = self.exif_reader.read(image_path)

        if self.iptc_reader:
            with contextlib.suppress(IPTCError):
                iptc_data = self.iptc_reader.read(image_path)

        return exif_data, iptc_data

    def read_exif(self, image_path: Path) -> ExifData:
        """Read EXIF metadata from image."""
        if not self.exif_reader:
            raise ExifError("piexif library is required for EXIF operations")
        return self.exif_reader.read(image_path)

    def read_iptc(self, image_path: Path) -> IPTCData:
        """Read IPTC metadata from image."""
        if not self.iptc_reader:
            raise IPTCError("iptcinfo3 library is required for IPTC operations")
        return self.iptc_reader.read(image_path)

    def _dms_to_decimal(self, dms_tuple: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]) -> float:
        """Convert DMS (degrees, minutes, seconds) to decimal degrees."""
        if not self.exif_reader:
            raise ExifError("EXIF reader not available")
        # Use the conversion method from ExifReader
        return self.exif_reader._convert_gps_coordinate(list(dms_tuple), "N")

    # Private method delegations for test compatibility
    def _get_string_value(self, ifd: dict[int, Any], tags: dict[int, dict[str, str]], tag_name: str) -> str | None:
        """Get string value from EXIF IFD."""
        if not self.exif_reader:
            return None

        # Find the tag ID for the given name
        tag_id = next((k for k, v in tags.items() if v.get("name") == tag_name), None)
        if tag_id is None or tag_id not in ifd:
            return None

        value = ifd[tag_id]
        if value is None:
            return None

        if isinstance(value, bytes):
            try:
                decoded = value.decode("utf-8", errors="ignore").rstrip("\x00")
                # Check if the decoded value has valid characters
                if decoded and not all(ord(c) < ASCII_PRINTABLE_MIN or ord(c) > ASCII_PRINTABLE_MAX for c in decoded):
                    return decoded
            except (UnicodeDecodeError, AttributeError):
                pass
            return None

        return str(value)

    def _get_int_value(self, ifd: dict[int, Any], tags: dict[int, dict[str, str]], tag_name: str) -> int | None:
        """Get integer value from EXIF IFD."""
        if not self.exif_reader:
            return None
        tag_id = next((k for k, v in tags.items() if v.get("name") == tag_name), None)
        if tag_id is None or tag_id not in ifd:
            return None
        value = ifd[tag_id]
        if isinstance(value, int):
            return value
        return None

    def _get_rational_value(self, ifd: dict[int, Any], tags: dict[int, dict[str, str]], tag_name: str) -> str | None:
        """Get rational value from EXIF IFD."""
        if not self.exif_reader:
            return None
        tag_id = next((k for k, v in tags.items() if v.get("name") == tag_name), None)
        if tag_id is None or tag_id not in ifd:
            return None
        value = ifd[tag_id]
        if isinstance(value, tuple) and len(value) == RATIONAL_TUPLE_LENGTH:
            return self.exif_reader._rational_to_string((value[0], value[1]))
        return None

    def _get_datetime_value(self, ifd: dict[int, Any], tags: dict[int, dict[str, str]], tag_name: str) -> str | None:
        """Get datetime value from EXIF IFD."""
        if not self.exif_reader:
            return None
        tag_id = next((k for k, v in tags.items() if v.get("name") == tag_name), None)
        if tag_id is None or tag_id not in ifd:
            return None
        value = ifd[tag_id]
        if isinstance(value, str):
            return value
        elif isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore").rstrip("\x00")
        return None

    def _get_shutter_speed(self, ifd: dict[int, Any], tags: dict[int, dict[str, str]]) -> str | None:  # noqa: ARG002
        """Get shutter speed from EXIF IFD."""
        if not self.exif_reader:
            return None
        # Look for ExposureTime tag (33434 in decimal)
        exposure_time = ifd.get(33434)  # ExposureTime
        if exposure_time and isinstance(exposure_time, tuple) and len(exposure_time) == RATIONAL_TUPLE_LENGTH:
            return self.exif_reader._rational_to_string((exposure_time[0], exposure_time[1]))
        return None

    def _get_color_space(self, ifd: dict[int, Any], tags: dict[int, dict[str, str]]) -> str | None:  # noqa: ARG002
        """Get color space from EXIF IFD."""
        if not self.exif_reader:
            return None
        # Look for ColorSpace tag (40961 in decimal)
        color_space = ifd.get(40961)  # ColorSpace
        if color_space is not None:
            return self.exif_reader._parse_color_space(color_space)
        return None

    def _parse_gps_coordinates(self, gps_dict: dict[int, Any]) -> tuple[float | None, float | None]:
        """Parse GPS coordinates from GPS IFD."""
        if not self.exif_reader:
            return None, None

        lat_ref = gps_dict.get(1)  # GPSLatitudeRef
        lat_coord = gps_dict.get(2)  # GPSLatitude
        lng_ref = gps_dict.get(3)  # GPSLongitudeRef
        lng_coord = gps_dict.get(4)  # GPSLongitude

        latitude = None
        longitude = None

        if lat_coord and lat_ref:
            latitude = self.exif_reader._convert_gps_coordinate(lat_coord, lat_ref)
        if lng_coord and lng_ref:
            longitude = self.exif_reader._convert_gps_coordinate(lng_coord, lng_ref)

        return latitude, longitude

    def _get_gps_altitude(self, gps_dict: dict[int, Any]) -> float | None:
        """Get GPS altitude from GPS IFD."""
        if not self.exif_reader:
            return None

        alt_ref = gps_dict.get(5)  # GPSAltitudeRef
        alt_coord = gps_dict.get(6)  # GPSAltitude

        if alt_coord and isinstance(alt_coord, tuple) and len(alt_coord) == RATIONAL_TUPLE_LENGTH:
            altitude = float(alt_coord[0]) / float(alt_coord[1])
            # If AltitudeRef is 1, altitude is below sea level (negative)
            if alt_ref == GPS_ALTITUDE_REF_BELOW_SEA_LEVEL:
                altitude = -altitude
            return altitude
        return None

    def _get_gps_timestamp(self, gps_dict: dict[int, Any]) -> str | None:
        """Get GPS timestamp from GPS IFD."""
        if not self.exif_reader:
            return None

        time_stamp = gps_dict.get(7)  # GPSTimeStamp
        date_stamp = gps_dict.get(29)  # GPSDateStamp

        if time_stamp and isinstance(time_stamp, list) and len(time_stamp) == TIME_STAMP_PARTS:
            # Convert rational numbers to time components
            hours = int(time_stamp[0][0] / time_stamp[0][1]) if time_stamp[0][1] != 0 else 0
            minutes = int(time_stamp[1][0] / time_stamp[1][1]) if time_stamp[1][1] != 0 else 0
            seconds = int(time_stamp[2][0] / time_stamp[2][1]) if time_stamp[2][1] != 0 else 0

            time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            if date_stamp:
                date_str = (
                    date_stamp.decode("utf-8", errors="ignore") if isinstance(date_stamp, bytes) else str(date_stamp)
                )
                return f"{date_str} {time_str}"
            return time_str
        return None

    def _parse_exif_dict(self, exif_dict: dict[str, Any]) -> ExifData:
        """Parse EXIF dictionary - delegate to ExifReader."""
        if not self.exif_reader:
            return ExifData()
        return self.exif_reader._parse_exif_dict(exif_dict)

    def _parse_iptc_info(self, iptc_info: Any) -> IPTCData:
        """Parse IPTC info - delegate to IPTCReader."""
        if not self.iptc_reader:
            return IPTCData()
        return self.iptc_reader._parse_iptc_info(iptc_info)


class ExifReader:
    """Read EXIF data from images."""

    def read(self, image_path: Path) -> ExifData:
        """Read EXIF data from image file."""
        if not PIEXIF_AVAILABLE:
            raise ExifError("piexif library not available")

        try:
            exif_dict = piexif.load(str(image_path))
        except Exception as e:
            raise ExifError(f"Failed to read EXIF data: {e}") from e

        return self._parse_exif_dict(exif_dict)

    def _safe_decode(self, value: Any) -> str:
        """Safely decode a value that might be bytes or string."""
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        elif isinstance(value, str):
            return value
        else:
            return str(value) if value is not None else ""

    def _parse_exif_dict(self, exif_dict: dict[str, Any]) -> ExifData:
        """Parse piexif dictionary into ExifData."""
        exif_data = ExifData()

        # Parse 0th IFD (main image data)
        if "0th" in exif_dict:
            ifd = exif_dict["0th"]
            exif_data.make = self._safe_decode(ifd.get(271, b"")).strip()  # Make
            exif_data.model = self._safe_decode(ifd.get(272, b"")).strip()  # Model
            exif_data.software = self._safe_decode(ifd.get(305, b"")).strip()  # Software
            exif_data.datetime = self._safe_decode(ifd.get(306, b"")).strip()  # DateTime
            exif_data.orientation = ifd.get(274)  # Orientation
            exif_data.x_resolution = self._rational_to_float(ifd.get(282))  # XResolution
            exif_data.y_resolution = self._rational_to_float(ifd.get(283))  # YResolution
            exif_data.resolution_unit = ifd.get(296)  # ResolutionUnit

        # Parse Exif IFD (camera settings)
        if "Exif" in exif_dict:
            ifd = exif_dict["Exif"]
            exif_data.datetime_original = self._safe_decode(ifd.get(36867, b"")).strip()  # DateTimeOriginal
            exif_data.datetime_digitized = self._safe_decode(ifd.get(36868, b"")).strip()  # DateTimeDigitized

            # Exposure settings
            exposure_time = ifd.get(33434)  # ExposureTime
            if exposure_time:
                exif_data.exposure_time = self._rational_to_string(exposure_time)

            f_number = ifd.get(33437)  # FNumber
            if f_number:
                exif_data.f_number = f"f/{self._rational_to_float(f_number)}"
                exif_data.aperture = self._rational_to_float(f_number)  # Set aperture field

            exif_data.iso_speed = ifd.get(34855)  # ISOSpeedRatings

            focal_length = ifd.get(37386)  # FocalLength
            if focal_length:
                focal_length_value = self._rational_to_float(focal_length)
                exif_data.focal_length = str(focal_length_value) if focal_length_value is not None else None

            exif_data.flash = self._parse_flash(ifd.get(37385))  # Flash
            exif_data.white_balance = self._parse_white_balance(ifd.get(41987))  # WhiteBalance
            exif_data.color_space = self._parse_color_space(ifd.get(40961))  # ColorSpace

            # Image dimensions
            exif_data.width = ifd.get(40962)  # PixelXDimension
            exif_data.height = ifd.get(40963)  # PixelYDimension

            # Creator and copyright
            exif_data.processing_software = self._safe_decode(ifd.get(11, b"")).strip()  # ProcessingSoftware
            exif_data.creator = self._safe_decode(ifd.get(315, b"")).strip()  # Artist

        # Parse GPS data
        if "GPS" in exif_dict:
            self._parse_gps_data(exif_dict["GPS"], exif_data)

        return exif_data

    def _rational_to_float(self, rational: tuple[int, int] | None) -> float | None:
        """Convert rational tuple to float."""
        if not rational or len(rational) != RATIONAL_TUPLE_LENGTH:
            return None
        try:
            return rational[0] / rational[1] if rational[1] != 0 else None
        except (TypeError, ZeroDivisionError):
            return None

    def _rational_to_string(self, rational: tuple[int, int]) -> str:
        """Convert rational to string representation."""
        if len(rational) != RATIONAL_TUPLE_LENGTH:
            return "Unknown"
        if rational[1] == 1:
            return str(rational[0])
        return f"{rational[0]}/{rational[1]}"

    def _parse_flash(self, flash_value: int | None) -> str | None:
        """Parse flash value to human readable string."""
        if flash_value is None:
            return None

        flash_fired = flash_value & 1
        if flash_fired:
            return "Flash fired"
        return "Flash did not fire"

    def _parse_white_balance(self, wb_value: int | None) -> str | None:
        """Parse white balance value."""
        if wb_value is None:
            return None

        wb_map = {0: "Auto", 1: "Manual"}
        return wb_map.get(wb_value, "Unknown")

    def _parse_color_space(self, cs_value: int | None) -> str | None:
        """Parse color space value."""
        if cs_value is None:
            return None

        cs_map = {
            1: "sRGB",
            ADOBE_RGB_CODE: "Adobe RGB",
            UNCALIBRATED_CODE: "Uncalibrated",
        }
        return cs_map.get(cs_value, "Unknown")

    def _parse_gps_data(self, gps_dict: dict[int, Any], exif_data: ExifData) -> None:
        """Parse GPS data from GPS IFD."""
        # GPS Latitude
        lat = gps_dict.get(piexif.GPSIFD.GPSLatitude)
        lat_ref = gps_dict.get(piexif.GPSIFD.GPSLatitudeRef, b"").decode("utf-8", errors="ignore")
        if lat and lat_ref:
            exif_data.gps_latitude = self._convert_gps_coordinate(lat, lat_ref)

        # GPS Longitude
        lon = gps_dict.get(piexif.GPSIFD.GPSLongitude)
        lon_ref = gps_dict.get(piexif.GPSIFD.GPSLongitudeRef, b"").decode("utf-8", errors="ignore")
        if lon and lon_ref:
            exif_data.gps_longitude = self._convert_gps_coordinate(lon, lon_ref)

        # GPS Altitude
        alt = gps_dict.get(piexif.GPSIFD.GPSAltitude)
        if alt:
            exif_data.gps_altitude = self._rational_to_float(alt)

    def _convert_gps_coordinate(self, coordinate: list[tuple[int, int]], reference: str) -> float:
        """Convert GPS coordinate to decimal degrees."""
        if len(coordinate) != GPS_COORDINATE_PARTS:
            return 0.0

        degrees = self._rational_to_float(coordinate[0]) or 0
        minutes = self._rational_to_float(coordinate[1]) or 0
        seconds = self._rational_to_float(coordinate[2]) or 0

        decimal = degrees + minutes / 60 + seconds / 3600

        if reference in ["S", "W"]:
            decimal = -decimal

        return decimal


class IPTCReader:
    """Read IPTC data from images."""

    def read(self, image_path: Path) -> IPTCData:
        """Read IPTC data from image file."""
        if not IPTCINFO_AVAILABLE:
            raise IPTCError("iptcinfo3 library not available")

        try:
            iptc_info = IPTCInfo(str(image_path), force=True)
        except Exception as e:
            raise IPTCError(f"Failed to read IPTC data: {e}") from e

        return self._parse_iptc_info(iptc_info)

    def _parse_iptc_info(self, iptc_info: IPTCInfo) -> IPTCData:
        """Parse IPTCInfo object into IPTCData."""
        iptc_data = IPTCData()

        # Handle both attribute access and data dictionary access
        def get_value(attr_name: str, data_key: int | None = None) -> str | None:
            """Get value from either attribute or data dictionary."""
            # Try attribute access first (for test mocks)
            if hasattr(iptc_info, attr_name):
                value = getattr(iptc_info, attr_name)
                if value is not None:
                    return str(value)

            # Fall back to data dictionary access
            if (
                data_key is not None
                and hasattr(iptc_info, "data")
                and hasattr(iptc_info.data, "get")
                and iptc_info.data.get(data_key)
            ):
                value = iptc_info.data[data_key]
                if isinstance(value, bytes):
                    return value.decode("utf-8", errors="ignore")
                return str(value)

            return None

        # Title and description
        iptc_data.title = get_value("object_name", 5)
        iptc_data.caption = get_value("caption", 120)
        iptc_data.description = get_value("caption", 120)  # Map caption to description too

        # Keywords
        if hasattr(iptc_info, "keywords") and iptc_info.keywords:
            with contextlib.suppress(TypeError):
                iptc_data.keywords = list(iptc_info.keywords)
        elif hasattr(iptc_info, "data") and hasattr(iptc_info.data, "get") and iptc_info.data.get(25):
            keywords_data = iptc_info.data.get(25, [])
            if keywords_data and isinstance(keywords_data, list):
                for kw in keywords_data:
                    if isinstance(kw, bytes):
                        try:
                            decoded_kw = kw.decode("utf-8", errors="ignore")
                            kw = decoded_kw  # noqa: PLW2901
                        except (UnicodeDecodeError, AttributeError):
                            continue
                    if kw.strip():
                        iptc_data.keywords.append(kw.strip())

        # Creator information
        iptc_data.byline = get_value("byline", 80)
        iptc_data.byline_title = get_value("byline_title", 85)

        # Copyright
        iptc_data.copyright_notice = get_value("copyright_notice", 116)

        # Instructions
        iptc_data.instructions = get_value("special_instructions", 40)

        # Location
        iptc_data.location = get_value("location", 92)
        iptc_data.city = get_value("city", 90)
        iptc_data.state = get_value("province_state", 95)
        iptc_data.country = get_value("country_name", 101)

        # Category
        iptc_data.category = get_value("category", 15)

        # Date created
        try:
            if hasattr(iptc_info, "data") and hasattr(iptc_info.data, "get") and iptc_info.data.get(55):
                date_created = iptc_info.data[55]
                if isinstance(date_created, bytes) and len(date_created) == IPTC_DATE_LENGTH:
                    with contextlib.suppress(UnicodeDecodeError):
                        iptc_data.date_created = date_created.decode("utf-8")
        except (TypeError, AttributeError):
            # Skip date processing if mock object doesn't support it
            pass

        return iptc_data


class MetadataWriter:
    """Write metadata to images."""

    def __init__(self) -> None:
        """Initialize metadata writer."""
        if not PIL_AVAILABLE:
            raise ImportError("PIL (Pillow) is required for metadata operations")
        self.exif_writer = ExifWriter() if PIEXIF_AVAILABLE else None
        self.iptc_writer = IPTCWriter() if IPTCINFO_AVAILABLE else None

    def write(
        self,
        image_path: Path,
        exif_data: ExifData | None = None,
        iptc_data: IPTCData | None = None,
        output_path: Path | None = None,
    ) -> None:
        """Write metadata to image."""
        if output_path is None:
            output_path = image_path

        # Create backup if writing to same file
        if output_path == image_path:
            backup_path = image_path.with_suffix(f"{image_path.suffix}.backup")
            shutil.copy2(image_path, backup_path)

        try:
            # Write EXIF data
            if self.exif_writer and exif_data:
                self.exif_writer.write_exif(image_path, exif_data, output_path)

            # Write IPTC data
            if self.iptc_writer and iptc_data:
                self.iptc_writer.write_iptc(image_path, iptc_data, output_path)

        except Exception:
            # Restore backup on failure
            if output_path == image_path and backup_path.exists():
                shutil.copy2(backup_path, image_path)
            raise
        finally:
            # Clean up backup
            if output_path == image_path:
                with contextlib.suppress(FileNotFoundError):
                    backup_path.unlink()

    def write_exif(self, image_path: Path, exif_data: ExifData, output_path: Path | None = None) -> None:
        """Write EXIF metadata to image."""
        if not self.exif_writer:
            raise ExifError("piexif library is required")
        if output_path is None:
            output_path = image_path
        self.exif_writer.write_exif(image_path, exif_data, output_path)

    def write_iptc(self, image_path: Path, iptc_data: IPTCData, output_path: Path | None = None) -> None:
        """Write IPTC metadata to image."""
        if not self.iptc_writer:
            raise IPTCError("iptcinfo3 library is required")
        if output_path is None:
            output_path = image_path
        self.iptc_writer.write_iptc(image_path, iptc_data, output_path)


class ExifWriter:
    """Write EXIF data to images."""

    def write_exif(self, image_path: Path, exif_data: ExifData, output_path: Path) -> None:
        """Write EXIF data to image."""
        if not PIEXIF_AVAILABLE:
            raise ExifError("piexif library not available")

        try:
            # Load existing EXIF data or create new
            try:
                exif_dict = piexif.load(str(image_path))
            except Exception:
                exif_dict = {
                    "0th": {},
                    "Exif": {},
                    "GPS": {},
                    "1st": {},
                    "thumbnail": None,
                }

            # Update EXIF dict with new data
            self._update_exif_dict(exif_dict, exif_data)

            # Dump and save
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, str(image_path), str(output_path))

        except Exception as e:
            raise ExifError(f"Failed to write EXIF data: {e}") from e

    def _format_datetime(self, dt: Any) -> str:
        """Format datetime object or string for EXIF."""
        if hasattr(dt, "strftime"):
            # It's a datetime object
            return str(dt.strftime("%Y:%m:%d %H:%M:%S"))
        elif isinstance(dt, str):
            # It's already a string
            return dt
        else:
            # Convert to string
            return str(dt) if dt is not None else ""

    def _update_exif_dict(self, exif_dict: dict[str, Any], exif_data: ExifData) -> None:
        """Update EXIF dictionary with new data."""
        # Update 0th IFD
        if exif_data.make:
            exif_dict["0th"][piexif.ImageIFD.Make] = exif_data.make.encode("utf-8")
        if exif_data.model:
            exif_dict["0th"][piexif.ImageIFD.Model] = exif_data.model.encode("utf-8")
        if exif_data.software:
            exif_dict["0th"][piexif.ImageIFD.Software] = exif_data.software.encode("utf-8")

        # Update Exif IFD
        if exif_data.datetime_original:
            dt_str = self._format_datetime(exif_data.datetime_original)
            exif_dict["Exif"][piexif.ExifIFD.DateTimeOriginal] = dt_str.encode("utf-8")

        if exif_data.datetime_digitized:
            dt_str = self._format_datetime(exif_data.datetime_digitized)
            exif_dict["Exif"][piexif.ExifIFD.DateTimeDigitized] = dt_str.encode("utf-8")

        if exif_data.datetime_modified:
            dt_str = self._format_datetime(exif_data.datetime_modified)
            exif_dict["0th"][piexif.ImageIFD.DateTime] = dt_str.encode("utf-8")


class IPTCWriter:
    """Write IPTC data to images."""

    def write_iptc(self, image_path: Path, iptc_data: IPTCData, output_path: Path) -> None:
        """Write IPTC data to image."""
        if not IPTCINFO_AVAILABLE:
            raise IPTCError("iptcinfo3 library not available")

        try:
            # Load existing IPTC data
            iptc_info = IPTCInfo(str(image_path), force=True)

            # Update with new data
            self._update_iptc_info(iptc_info, iptc_data)

            # Save to output path
            iptc_info.save_as(str(output_path))

        except Exception as e:
            raise IPTCError(f"Failed to write IPTC data: {e}") from e

    def _update_iptc_info(self, iptc_info: IPTCInfo, iptc_data: IPTCData) -> None:
        """Update IPTCInfo object with new data."""
        if iptc_data.title:
            iptc_info.data[5] = iptc_data.title.encode("utf-8")

        if iptc_data.description:
            iptc_info.data[120] = iptc_data.description.encode("utf-8")

        if iptc_data.keywords:
            iptc_info.data[25] = [kw.encode("utf-8") for kw in iptc_data.keywords]

        if iptc_data.byline:
            iptc_info.data[80] = iptc_data.byline.encode("utf-8")

        if iptc_data.copyright_notice:
            iptc_info.data[116] = iptc_data.copyright_notice.encode("utf-8")


# Convenience functions
def read_metadata(image_path: Path) -> tuple[ExifData, IPTCData]:
    """Read metadata from image file."""
    reader = MetadataReader()
    return reader.read(image_path)


def write_metadata(
    image_path: Path,
    exif_data: ExifData | None = None,
    iptc_data: IPTCData | None = None,
    output_path: Path | None = None,
) -> None:
    """Write metadata to image file."""
    writer = MetadataWriter()
    if exif_data and writer.exif_writer:
        writer.exif_writer.write_exif(image_path, exif_data, output_path or image_path)
    if iptc_data and writer.iptc_writer:
        writer.iptc_writer.write_iptc(image_path, iptc_data, output_path or image_path)
