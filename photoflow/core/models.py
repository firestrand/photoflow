"""Core domain models for PhotoFlow."""

from __future__ import annotations

import re  # used in _validate_constraints
from dataclasses import field, replace
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field, TypeAdapter, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

# Constants ---------------------------------------------------------------------

SIZE_TUPLE_LEN: int = 2  # Expected length of ImageAsset.size (width, height)


@pydantic_dataclass(frozen=True)
class ImageAsset:
    """Immutable representation of an image with rich metadata and processing context.

    Enhanced domain model that provides comprehensive image information including
    structured metadata, processing context, and template variables for dynamic
    operations. This is the core model used throughout the PhotoFlow pipeline.
    """

    path: Path
    """Path to the image file"""

    format: str
    """Image format (JPEG, PNG, WebP, etc.)"""

    size: tuple[int, int]
    """Image dimensions (width, height) in pixels"""

    mode: str
    """Color mode (RGB, RGBA, L, etc.)"""

    # Enhanced metadata system (structured)
    exif: dict[str, Any] = field(default_factory=dict)
    """EXIF metadata from image headers"""

    iptc: dict[str, Any] = field(default_factory=dict)
    """IPTC metadata (keywords, copyright, etc.)"""

    processing_metadata: dict[str, Any] = field(default_factory=dict)
    """Processing history and PhotoFlow-specific metadata"""

    # Processing context
    dpi: tuple[int, int] = field(default=(72, 72))
    """Image resolution in dots per inch (horizontal, vertical)"""

    color_profile: str | None = field(default=None)
    """ICC color profile name or identifier"""

    original_path: Path | None = field(default=None)
    """Path to original file (for tracking processing chains)"""

    # Legacy metadata field (for backward compatibility)
    metadata: dict[str, Any] = field(default_factory=dict)
    """Legacy metadata field (deprecated, use exif/iptc instead)"""

    data: bytes | None = field(default=None, repr=False)
    """Raw image data (optional, for in-memory processing)"""

    # Validation -------------------------------------------------------------------
    def __post_init__(self) -> None:
        """Validate invariants after standard dataclass __init__."""
        # Validate *size*
        if len(self.size) != SIZE_TUPLE_LEN:
            raise ValueError("size must be a 2-tuple (width, height)")

        width, height = self.size
        if not isinstance(width, int) or not isinstance(height, int):
            raise ValueError("size values must be integers")
        if width <= 0 or height <= 0:
            raise ValueError("size dimensions must be positive")

        # Basic sanity for format/mode
        if not self.format:
            raise ValueError("format cannot be empty")
        if not self.mode:
            raise ValueError("mode cannot be empty")

    # ----------------------------------------------------------------------------
    # Convenience helpers

    def with_updates(self, **kwargs: Any) -> ImageAsset:
        """Create a new ImageAsset with updated fields.

        Args:
            **kwargs: Fields to update

        Returns:
            New ImageAsset instance with updated fields
        """
        # Handle nested metadata updates
        if "metadata" in kwargs:
            new_metadata = self.metadata.copy()
            new_metadata.update(kwargs["metadata"])
            kwargs["metadata"] = new_metadata

        # Create new instance with dataclass replace
        return replace(self, **kwargs)

    @property
    def width(self) -> int:
        """Image width in pixels."""
        return self.size[0]

    @property
    def height(self) -> int:
        """Image height in pixels."""
        return self.size[1]

    @property
    def aspect_ratio(self) -> float:
        """Image aspect ratio (width/height)."""
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def file_size_mb(self) -> float:
        """File size in megabytes (if path exists)."""
        if self.path.exists():
            return self.path.stat().st_size / (1024 * 1024)
        return 0.0

    @property
    def megapixels(self) -> float:
        """Image size in megapixels."""
        return (self.width * self.height) / 1_000_000

    @property
    def template_vars(self) -> dict[str, Any]:
        """Variables available for template evaluation.

        Returns:
            Dictionary of variables that can be used in template expressions
            for file naming, parameter values, and other dynamic content.

        Example:
            Template: "<filename>_<width>x<height>"
            With template_vars: {"filename": "photo", "width": 1920, "height": 1080}
            Result: "photo_1920x1080"
        """
        # Basic image properties
        vars_dict = {
            # Dimensions
            "width": self.width,
            "height": self.height,
            "aspect_ratio": round(self.aspect_ratio, 2),
            "megapixels": round(self.megapixels, 1),
            # Format and mode
            "format": self.format,
            "format_lower": self.format.lower(),
            "mode": self.mode,
            # File path components
            "filename": self.path.stem,
            "basename": self.path.name,
            "extension": self.path.suffix.lstrip("."),
            "parent": self.path.parent.name,
            "parent_path": str(self.path.parent),
            # File properties
            "file_size_mb": round(self.file_size_mb, 2),
            "dpi_horizontal": self.dpi[0],
            "dpi_vertical": self.dpi[1],
            # Processing context
            "color_profile": self.color_profile or "",
            "original_filename": (
                self.original_path.stem if self.original_path else self.path.stem
            ),
        }

        # Add EXIF-derived variables
        if self.exif:
            vars_dict.update(
                {
                    "date_taken": self._get_exif_date(),
                    "camera_make": self.exif.get("Make", ""),
                    "camera_model": self.exif.get("Model", ""),
                    "lens_model": self.exif.get("LensModel", ""),
                    "focal_length": self._get_exif_focal_length(),
                    "aperture": self._get_exif_aperture(),
                    "iso": self.exif.get("ISOSpeedRatings", ""),
                    "exposure_time": self._get_exif_exposure_time(),
                }
            )

        # Add IPTC-derived variables
        if self.iptc:
            vars_dict.update(
                {
                    "title": self.iptc.get("ObjectName", ""),
                    "description": self.iptc.get("Caption-Abstract", ""),
                    "keywords": ", ".join(self.iptc.get("Keywords", [])),
                    "copyright": self.iptc.get("CopyrightNotice", ""),
                    "creator": self.iptc.get("By-line", ""),
                }
            )

        # Add processing metadata
        if self.processing_metadata:
            vars_dict.update(
                {
                    "processing_date": self.processing_metadata.get(
                        "processed_date", ""
                    ),
                    "pipeline_name": self.processing_metadata.get("pipeline", ""),
                    "processing_version": self.processing_metadata.get("version", ""),
                }
            )

        return vars_dict

    def _get_exif_date(self) -> str:
        """Extract formatted date from EXIF data."""
        date_fields = ["DateTime", "DateTimeOriginal", "DateTimeDigitized"]
        for date_field in date_fields:
            if self.exif.get(date_field):
                return str(self.exif[date_field])
        return ""

    def _get_exif_focal_length(self) -> str:
        """Extract formatted focal length from EXIF data."""
        focal_length = self.exif.get("FocalLength")
        if focal_length:
            try:
                # Handle fractional values
                if (
                    isinstance(focal_length, tuple)
                    and len(focal_length) == 2  # noqa: PLR2004
                ):
                    return f"{focal_length[0] / focal_length[1]:.0f}mm"
                else:
                    return f"{float(focal_length):.0f}mm"
            except (ValueError, TypeError, ZeroDivisionError):
                return str(focal_length)
        return ""

    def _get_exif_aperture(self) -> str:
        """Extract formatted aperture from EXIF data."""
        aperture = self.exif.get("FNumber") or self.exif.get("ApertureValue")
        if aperture:
            try:
                if isinstance(aperture, tuple) and len(aperture) == 2:  # noqa: PLR2004
                    f_value = aperture[0] / aperture[1]
                else:
                    f_value = float(aperture)
                return f"f/{f_value:.1f}"
            except (ValueError, TypeError, ZeroDivisionError):
                return str(aperture)
        return ""

    def _get_exif_exposure_time(self) -> str:
        """Extract formatted exposure time from EXIF data."""
        exposure = self.exif.get("ExposureTime")
        if exposure:
            try:
                if isinstance(exposure, tuple) and len(exposure) == 2:  # noqa: PLR2004
                    if exposure[0] == 1:
                        return f"1/{exposure[1]}s"
                    else:
                        return f"{exposure[0] / exposure[1]:.2f}s"
                else:
                    return f"{float(exposure):.2f}s"
            except (ValueError, TypeError, ZeroDivisionError):
                return str(exposure)
        return ""

    def with_exif_update(self, exif_data: dict[str, Any]) -> ImageAsset:
        """Create new ImageAsset with updated EXIF data.

        Args:
            exif_data: EXIF data to merge with existing data

        Returns:
            New ImageAsset with updated EXIF metadata
        """
        new_exif = self.exif.copy()
        new_exif.update(exif_data)
        return replace(self, exif=new_exif)

    def with_iptc_update(self, iptc_data: dict[str, Any]) -> ImageAsset:
        """Create new ImageAsset with updated IPTC data.

        Args:
            iptc_data: IPTC data to merge with existing data

        Returns:
            New ImageAsset with updated IPTC metadata
        """
        new_iptc = self.iptc.copy()
        new_iptc.update(iptc_data)
        return replace(self, iptc=new_iptc)

    def with_processing_update(self, processing_data: dict[str, Any]) -> ImageAsset:
        """Create new ImageAsset with updated processing metadata.

        Args:
            processing_data: Processing metadata to merge

        Returns:
            New ImageAsset with updated processing metadata
        """
        new_processing = self.processing_metadata.copy()
        new_processing.update(processing_data)
        return replace(self, processing_metadata=new_processing)


class ActionParameter(BaseModel):
    """Value object for action parameters with validation and defaults.

    Provides type-safe parameter handling with validation, default values,
    and schema information for GUI generation.
    """

    name: str = Field(..., description="Parameter name")
    value: Any = Field(..., description="Parameter value")
    param_type: str = Field(
        ..., description="Parameter type (int, float, str, bool, etc.)"
    )
    default: Any = Field(None, description="Default value")
    description: str = Field("", description="Human-readable parameter description")
    required: bool = Field(True, description="Whether parameter is required")
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="Validation constraints"
    )

    @field_validator("param_type")
    @classmethod
    def validate_param_type(cls, v: str) -> str:
        """Validate parameter type."""
        valid_types = {"int", "float", "str", "bool", "list", "dict", "path", "enum"}
        if v not in valid_types:
            raise ValueError(
                f"Invalid parameter type: {v}. Must be one of {valid_types}"
            )
        return v

    def validate_value(self) -> ActionParameter:
        """Validate the parameter value against constraints.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If value doesn't meet constraints
        """
        if self.required and self.value is None:
            raise ValueError(f"Required parameter '{self.name}' cannot be None")

        # Type validation
        if self.value is not None:
            if self.param_type == "int" and not isinstance(self.value, int):
                try:
                    self.value = int(self.value)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"Parameter '{self.name}' must be an integer"
                    ) from e

            elif self.param_type == "float" and not isinstance(
                self.value, (int, float)
            ):
                try:
                    self.value = float(self.value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Parameter '{self.name}' must be a number") from e

            elif self.param_type == "bool" and not isinstance(self.value, bool):
                if isinstance(self.value, str):
                    self.value = self.value.lower() in {"true", "yes", "1", "on"}
                else:
                    self.value = bool(self.value)

            elif self.param_type == "path":
                if isinstance(self.value, str):
                    self.value = Path(self.value)
                elif not isinstance(self.value, Path):
                    raise ValueError(f"Parameter '{self.name}' must be a path")

        # Constraint validation
        if self.constraints and self.value is not None:
            self._validate_constraints()

        return self

    def _validate_constraints(self) -> None:
        """Validate value against constraints."""
        if "min" in self.constraints and self.value < self.constraints["min"]:
            raise ValueError(
                f"Parameter '{self.name}' must be >= {self.constraints['min']}"
            )

        if "max" in self.constraints and self.value > self.constraints["max"]:
            raise ValueError(
                f"Parameter '{self.name}' must be <= {self.constraints['max']}"
            )

        if (
            "choices" in self.constraints
            and self.value not in self.constraints["choices"]
        ):
            raise ValueError(
                f"Parameter '{self.name}' must be one of {self.constraints['choices']}"
            )

        if (
            "pattern" in self.constraints
            and isinstance(self.value, str)
            and not re.match(self.constraints["pattern"], self.value)
        ):
            raise ValueError(f"Parameter '{self.name}' doesn't match required pattern")


@pydantic_dataclass(frozen=True)
class ActionSpec:
    """Specification for an action with validated parameters.

    Represents a single action in a processing pipeline with its parameters.
    Immutable to ensure pipeline consistency.
    """

    name: str
    """Action name (snake_case)"""

    parameters: dict[str, Any] = field(default_factory=dict)
    """Action parameters"""

    enabled: bool = field(default=True)
    """Whether action is enabled"""

    description: str = field(default="")
    """Optional description for documentation"""

    def __post_init__(self) -> None:
        """Validate action specification after creation."""
        if not self.name:
            raise ValueError("Action name cannot be empty")

        if not self.name.replace("_", "").isalnum():
            raise ValueError("Action name must be alphanumeric with underscores")

    def with_parameter(self, name: str, value: Any) -> ActionSpec:
        """Create new ActionSpec with additional parameter.

        Args:
            name: Parameter name
            value: Parameter value

        Returns:
            New ActionSpec with parameter added
        """
        new_params = self.parameters.copy()
        new_params[name] = value

        return replace(self, parameters=new_params)

    def without_parameter(self, name: str) -> ActionSpec:
        """Create new ActionSpec without specified parameter.

        Args:
            name: Parameter name to remove

        Returns:
            New ActionSpec with parameter removed
        """
        new_params = self.parameters.copy()
        new_params.pop(name, None)

        return replace(self, parameters=new_params)


@pydantic_dataclass(frozen=True)
class Pipeline:
    """Ordered sequence of actions to apply to images.

    Represents a complete processing pipeline that can be saved, loaded,
    and executed. Immutable to ensure reproducibility.
    """

    name: str
    """Pipeline name"""

    actions: tuple[ActionSpec, ...] = field(default_factory=tuple)
    """Ordered tuple of actions"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Pipeline metadata (version, author, description, etc.)"""

    def __post_init__(self) -> None:
        """Validate pipeline after creation."""
        if not self.name:
            raise ValueError("Pipeline name cannot be empty")

        # Ensure actions is a tuple (for immutability)
        object.__setattr__(self, "actions", tuple(self.actions))

    def with_action(self, action: ActionSpec, index: int | None = None) -> Pipeline:
        """Create new Pipeline with action added.

        Args:
            action: Action to add
            index: Position to insert (None = append)

        Returns:
            New Pipeline with action added
        """
        actions_list = list(self.actions)
        if index is None:
            actions_list.append(action)
        else:
            actions_list.insert(index, action)

        return replace(self, actions=tuple(actions_list))

    def without_action(self, index: int) -> Pipeline:
        """Create new Pipeline without action at index.

        Args:
            index: Index of action to remove

        Returns:
            New Pipeline without action
        """
        if not 0 <= index < len(self.actions):
            raise IndexError(f"Action index {index} out of range")

        actions_list = list(self.actions)
        actions_list.pop(index)

        return replace(self, actions=tuple(actions_list))

    def with_updated_action(self, index: int, action: ActionSpec) -> Pipeline:
        """Create new Pipeline with action updated at index.

        Args:
            index: Index of action to update
            action: New action specification

        Returns:
            New Pipeline with action updated
        """
        if not 0 <= index < len(self.actions):
            raise IndexError(f"Action index {index} out of range")

        actions_list = list(self.actions)
        actions_list[index] = action

        return replace(self, actions=tuple(actions_list))

    @property
    def action_count(self) -> int:
        """Number of actions in pipeline."""
        return len(self.actions)

    @property
    def enabled_actions(self) -> tuple[ActionSpec, ...]:
        """Get only enabled actions."""
        return tuple(action for action in self.actions if action.enabled)

    # Serialization is provided by Pydantic's `model_dump()` / `model_validate()`.

    # While pydantic.dataclasses currently lacks these helper methods on the
    # generated dataclass itself, we expose thin wrappers that delegate to
    # `TypeAdapter` so callers can use a familiar `.model_dump()` /
    # `.model_validate()` API consistent with `BaseModel`.

    def model_dump(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return cast("dict[str, Any]", TypeAdapter(Pipeline).dump_python(self))

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> Pipeline:
        """Validate and construct a Pipeline from raw data."""
        return TypeAdapter(cls).validate_python(data)
