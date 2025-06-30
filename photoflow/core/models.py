"""Core domain models for PhotoFlow."""

from __future__ import annotations

from dataclasses import field, replace
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, Field, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass

# Constants ---------------------------------------------------------------------

SIZE_TUPLE_LEN: int = 2  # Expected length of ImageAsset.size (width, height)


@pydantic_dataclass(frozen=True)
class ImageAsset:
    """Immutable representation of an image with its metadata and processing state.

    This is the core domain model representing an image throughout the processing
    pipeline. It contains all necessary information about the image including its
    location, format, dimensions, and metadata.
    """

    path: Path
    """Path to the image file"""

    format: str
    """Image format (JPEG, PNG, WebP, etc.)"""

    size: tuple[int, int]
    """Image dimensions (width, height) in pixels"""

    mode: str
    """Color mode (RGB, RGBA, L, etc.)"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """EXIF, IPTC, and other metadata"""

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
                    self.value = self.value.lower() in ("true", "yes", "1", "on")
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

        if "pattern" in self.constraints and isinstance(self.value, str):
            import re

            if not re.match(self.constraints["pattern"], self.value):
                raise ValueError(
                    f"Parameter '{self.name}' doesn't match required pattern"
                )


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
        if not isinstance(self.actions, tuple):
            object.__setattr__(self, "actions", tuple(self.actions))  # type: ignore[unreachable]

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
        from pydantic import TypeAdapter  # local import to avoid startup cost

        return cast(dict[str, Any], TypeAdapter(Pipeline).dump_python(self))

    @classmethod
    def model_validate(cls, data: dict[str, Any]) -> Pipeline:
        """Validate and construct a Pipeline from raw data."""
        from pydantic import TypeAdapter

        return TypeAdapter(cls).validate_python(data)
