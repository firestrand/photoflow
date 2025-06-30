"""Processing mixins for common operations.

This module provides reusable mixin classes that implement common image
processing patterns and utilities. Actions can inherit from these mixins
to gain shared functionality without code duplication.
"""

from __future__ import annotations

from typing import Any, ClassVar

from .fields import ValidationError


class GeometryMixin:
    """Mixin for actions that work with image geometry and transformations.

    Provides common utilities for calculating dimensions, handling aspect ratios,
    and validating geometric parameters.
    """

    def calculate_scaled_size(
        self,
        original_size: tuple[int, int],
        target_width: int | None = None,
        target_height: int | None = None,
        maintain_aspect: bool = True,
    ) -> tuple[int, int]:
        """Calculate new dimensions with optional aspect ratio preservation.

        Args:
            original_size: Original (width, height) tuple
            target_width: Desired width (optional)
            target_height: Desired height (optional)
            maintain_aspect: Whether to maintain aspect ratio

        Returns:
            Calculated (width, height) tuple

        Raises:
            ValueError: If no target dimensions provided

        Example:
            >>> mixin = GeometryMixin()
            >>> mixin.calculate_scaled_size((1920, 1080), target_width=960, maintain_aspect=True)
            (960, 540)
        """
        orig_w, orig_h = original_size

        if not target_width and not target_height:
            raise ValueError("Must specify at least one target dimension")

        if not maintain_aspect:
            # No aspect ratio constraints
            return (target_width or orig_w, target_height or orig_h)

        # Maintain aspect ratio
        aspect_ratio = orig_w / orig_h

        if target_width and target_height:
            # Scale to fit within both bounds
            scale_w = target_width / orig_w
            scale_h = target_height / orig_h
            scale = min(scale_w, scale_h)
            return (int(orig_w * scale), int(orig_h * scale))

        elif target_width:
            # Scale based on width
            return (target_width, int(target_width / aspect_ratio))

        elif target_height:
            # Scale based on height
            return (int(target_height * aspect_ratio), target_height)

        return original_size

    def validate_dimensions(
        self,
        width: int | None = None,
        height: int | None = None,
        min_size: int = 1,
        max_size: int = 50000,
    ) -> None:
        """Validate dimension parameters.

        Args:
            width: Width to validate
            height: Height to validate
            min_size: Minimum allowed dimension
            max_size: Maximum allowed dimension

        Raises:
            ValidationError: If dimensions are invalid
        """
        for dimension, name in [(width, "width"), (height, "height")]:
            if dimension is not None:
                if not isinstance(dimension, int):
                    raise ValidationError(
                        field_name=name,
                        expected="integer",
                        actual=f"{type(dimension).__name__}: {dimension}",
                        message=f"{name.title()} must be an integer",
                        suggestion="Provide a whole number like 800 or 1920",
                    )

                if dimension < min_size:
                    raise ValidationError(
                        field_name=name,
                        expected=f"integer >= {min_size}",
                        actual=dimension,
                        message=f"{name.title()} must be at least {min_size}",
                        suggestion=f"Use a value of {min_size} or greater",
                    )

                if dimension > max_size:
                    raise ValidationError(
                        field_name=name,
                        expected=f"integer <= {max_size}",
                        actual=dimension,
                        message=f"{name.title()} must be at most {max_size}",
                        suggestion=f"Use a value of {max_size} or smaller",
                    )

    def calculate_crop_bounds(
        self,
        image_size: tuple[int, int],
        crop_x: int,
        crop_y: int,
        crop_width: int,
        crop_height: int,
    ) -> tuple[int, int, int, int]:
        """Calculate validated crop bounds.

        Args:
            image_size: Original image (width, height)
            crop_x: X offset for crop
            crop_y: Y offset for crop
            crop_width: Width of crop area
            crop_height: Height of crop area

        Returns:
            Validated (x, y, width, height) tuple

        Raises:
            ValidationError: If crop bounds are invalid
        """
        img_w, img_h = image_size

        # Validate crop position
        if crop_x < 0:
            raise ValidationError(
                field_name="crop_x",
                expected="non-negative integer",
                actual=crop_x,
                message="Crop X position cannot be negative",
                suggestion="Use 0 or a positive value",
            )

        if crop_y < 0:
            raise ValidationError(
                field_name="crop_y",
                expected="non-negative integer",
                actual=crop_y,
                message="Crop Y position cannot be negative",
                suggestion="Use 0 or a positive value",
            )

        # Validate crop dimensions
        if crop_width <= 0:
            raise ValidationError(
                field_name="crop_width",
                expected="positive integer",
                actual=crop_width,
                message="Crop width must be positive",
                suggestion="Use a value like 100 or 500",
            )

        if crop_height <= 0:
            raise ValidationError(
                field_name="crop_height",
                expected="positive integer",
                actual=crop_height,
                message="Crop height must be positive",
                suggestion="Use a value like 100 or 500",
            )

        # Check bounds don't exceed image
        if crop_x + crop_width > img_w:
            raise ValidationError(
                field_name="crop_bounds",
                expected=f"crop area within image bounds (width: {img_w})",
                actual=f"x: {crop_x}, width: {crop_width}",
                message="Crop area extends beyond image width",
                suggestion=f"Reduce crop width or X position (max X+width: {img_w})",
            )

        if crop_y + crop_height > img_h:
            raise ValidationError(
                field_name="crop_bounds",
                expected=f"crop area within image bounds (height: {img_h})",
                actual=f"y: {crop_y}, height: {crop_height}",
                message="Crop area extends beyond image height",
                suggestion=f"Reduce crop height or Y position (max Y+height: {img_h})",
            )

        return (crop_x, crop_y, crop_width, crop_height)

    def calculate_centered_crop(
        self, image_size: tuple[int, int], target_size: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        """Calculate centered crop coordinates.

        Args:
            image_size: Original image (width, height)
            target_size: Desired crop (width, height)

        Returns:
            Centered crop bounds (x, y, width, height)

        Raises:
            ValidationError: If target size is larger than image
        """
        img_w, img_h = image_size
        target_w, target_h = target_size

        if target_w > img_w or target_h > img_h:
            raise ValidationError(
                field_name="target_size",
                expected=f"size <= image size ({img_w}x{img_h})",
                actual=f"{target_w}x{target_h}",
                message="Target crop size is larger than image",
                suggestion=f"Use dimensions no larger than {img_w}x{img_h}",
            )

        # Calculate centered position
        crop_x = (img_w - target_w) // 2
        crop_y = (img_h - target_h) // 2

        return (crop_x, crop_y, target_w, target_h)


class ColorMixin:
    """Mixin for color-related operations and validation.

    Provides utilities for working with color modes, color spaces,
    and color-related parameters.
    """

    # Common color modes and their descriptions
    COLOR_MODES: ClassVar[dict[str, str]] = {
        "RGB": "RGB Color (24-bit)",
        "RGBA": "RGB with Alpha (32-bit)",
        "L": "Grayscale (8-bit)",
        "LA": "Grayscale with Alpha (16-bit)",
        "CMYK": "CMYK (32-bit)",
        "P": "Palette (8-bit)",
        "1": "Bitmap (1-bit)",
    }

    # Modes that support transparency
    ALPHA_MODES: ClassVar[set[str]] = {"RGBA", "LA", "P"}

    # Modes suitable for web use
    WEB_MODES: ClassVar[set[str]] = {"RGB", "RGBA", "L"}

    def validate_color_mode(
        self, mode: str, supported_modes: list[str] | None = None
    ) -> str:
        """Validate that color mode is supported.

        Args:
            mode: Color mode to validate
            supported_modes: List of allowed modes (None = all known modes)

        Returns:
            Validated color mode

        Raises:
            ValidationError: If mode is not supported
        """
        if supported_modes is None:
            supported_modes = list(self.COLOR_MODES.keys())

        if mode not in supported_modes:
            raise ValidationError(
                field_name="color_mode",
                expected=f"one of {supported_modes}",
                actual=mode,
                message=f"Color mode '{mode}' not supported for this operation",
                suggestion=f"Use one of: {', '.join(supported_modes)}",
            )

        return mode

    def validate_color_value(
        self,
        color: int | float | tuple[int, ...],
        color_mode: str = "RGB",
        component_range: tuple[int, int] = (0, 255),
    ) -> tuple[int, ...] | int:
        """Validate color value for the specified mode.

        Args:
            color: Color value (int for grayscale, tuple for multi-channel)
            color_mode: Target color mode
            component_range: Valid range for color components

        Returns:
            Validated color value

        Raises:
            ValidationError: If color value is invalid
        """
        min_val, max_val = component_range

        if color_mode in {"L", "1"}:
            # Grayscale modes expect single value
            if isinstance(color, (tuple, list)):
                if len(color) == 1:
                    color = color[0]
                else:
                    raise ValidationError(
                        field_name="color",
                        expected="single color value for grayscale",
                        actual=f"tuple with {len(color)} values",
                        message="Grayscale modes require single color value",
                        suggestion="Use a single number like 128 or (128,)",
                    )

            if not isinstance(color, (int, float)):
                raise ValidationError(
                    field_name="color",
                    expected="numeric color value",
                    actual=f"{type(color).__name__}: {color}",
                    message="Color value must be numeric",
                    suggestion="Use a number between 0 and 255",
                )

            color_int = int(color)
            if not (min_val <= color_int <= max_val):
                raise ValidationError(
                    field_name="color",
                    expected=f"value between {min_val} and {max_val}",
                    actual=color_int,
                    message="Color value out of range",
                    suggestion=f"Use a value between {min_val} and {max_val}",
                )

            return color_int

        else:
            # Multi-channel modes expect tuple
            if not isinstance(color, (tuple, list)):
                raise ValidationError(
                    field_name="color",
                    expected=f"color tuple for {color_mode} mode",
                    actual=f"{type(color).__name__}: {color}",
                    message=f"{color_mode} mode requires color tuple",
                    suggestion="Use a tuple like (255, 0, 0) for red",
                )

            # Determine expected number of channels
            expected_channels = {
                "RGB": 3,
                "RGBA": 4,
                "CMYK": 4,
                "LA": 2,
            }.get(color_mode, 3)

            if len(color) != expected_channels:
                raise ValidationError(
                    field_name="color",
                    expected=f"tuple with {expected_channels} values for {color_mode}",
                    actual=f"tuple with {len(color)} values",
                    message=f"{color_mode} mode requires {expected_channels} color components",
                    suggestion=f"Use a tuple with {expected_channels} values",
                )

            # Validate each component
            validated_components = []
            for i, component in enumerate(color):
                if not isinstance(component, (int, float)):
                    raise ValidationError(
                        field_name=f"color_component_{i}",
                        expected="numeric value",
                        actual=f"{type(component).__name__}: {component}",
                        message=f"Color component {i} must be numeric",
                        suggestion="Use integers between 0 and 255",
                    )

                component_int = int(component)
                if not (min_val <= component_int <= max_val):
                    raise ValidationError(
                        field_name=f"color_component_{i}",
                        expected=f"value between {min_val} and {max_val}",
                        actual=component_int,
                        message=f"Color component {i} out of range",
                        suggestion=f"Use values between {min_val} and {max_val}",
                    )

                validated_components.append(component_int)

            return tuple(validated_components)

    def get_mode_info(self, mode: str) -> dict[str, Any]:
        """Get information about a color mode.

        Args:
            mode: Color mode

        Returns:
            Dictionary with mode information

        Example:
            {
                'description': 'RGB Color (24-bit)',
                'channels': 3,
                'has_alpha': False,
                'web_safe': True
            }
        """
        channel_counts = {
            "RGB": 3,
            "RGBA": 4,
            "L": 1,
            "LA": 2,
            "CMYK": 4,
            "P": 1,
            "1": 1,
        }

        return {
            "description": self.COLOR_MODES.get(mode, f"Unknown mode: {mode}"),
            "channels": channel_counts.get(mode, 1),
            "has_alpha": mode in self.ALPHA_MODES,
            "web_safe": mode in self.WEB_MODES,
        }

    def suggest_compatible_modes(self, current_mode: str) -> list[str]:
        """Suggest compatible color modes for conversion.

        Args:
            current_mode: Current color mode

        Returns:
            List of suggested compatible modes
        """
        if current_mode in {"RGB", "RGBA"}:
            return ["RGB", "RGBA", "L", "LA"]
        elif current_mode in {"L", "LA"}:
            return ["L", "LA", "RGB", "RGBA"]
        elif current_mode == "CMYK":
            return ["CMYK", "RGB"]
        elif current_mode == "P":
            return ["P", "RGB", "RGBA"]
        else:
            return ["RGB", "L"]


class ProcessingMixin:
    """Mixin for common processing operations and utilities.

    Provides general-purpose utilities for parameter validation,
    progress tracking, and error handling.
    """

    def validate_percentage(
        self,
        value: float | int,
        field_name: str = "percentage",
        allow_over_100: bool = False,
    ) -> float:
        """Validate percentage value.

        Args:
            value: Percentage value to validate
            field_name: Name of field for error messages
            allow_over_100: Whether to allow values over 100%

        Returns:
            Validated percentage as float

        Raises:
            ValidationError: If value is invalid
        """
        try:
            float_value = float(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(
                field_name=field_name,
                expected="numeric percentage value",
                actual=f"{type(value).__name__}: {value}",
                message="Percentage must be a number",
                suggestion="Use a number like 50 or 75.5",
            ) from e

        if float_value < 0:
            raise ValidationError(
                field_name=field_name,
                expected="non-negative percentage",
                actual=float_value,
                message="Percentage cannot be negative",
                suggestion="Use a value of 0 or greater",
            )

        if not allow_over_100 and float_value > 100:  # noqa: PLR2004
            raise ValidationError(
                field_name=field_name,
                expected="percentage <= 100",
                actual=float_value,
                message="Percentage cannot exceed 100%",
                suggestion="Use a value between 0 and 100",
            )

        return float_value

    def validate_factor(
        self,
        value: float | int,
        field_name: str = "factor",
        min_factor: float = 0.0,
        max_factor: float | None = None,
    ) -> float:
        """Validate multiplication factor.

        Args:
            value: Factor value to validate
            field_name: Name of field for error messages
            min_factor: Minimum allowed factor
            max_factor: Maximum allowed factor (None = no limit)

        Returns:
            Validated factor as float

        Raises:
            ValidationError: If value is invalid
        """
        try:
            float_value = float(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(
                field_name=field_name,
                expected="numeric factor value",
                actual=f"{type(value).__name__}: {value}",
                message="Factor must be a number",
                suggestion="Use a number like 1.0 or 2.5",
            ) from e

        if float_value < min_factor:
            raise ValidationError(
                field_name=field_name,
                expected=f"factor >= {min_factor}",
                actual=float_value,
                message=f"Factor must be at least {min_factor}",
                suggestion=f"Use a value of {min_factor} or greater",
            )

        if max_factor is not None and float_value > max_factor:
            raise ValidationError(
                field_name=field_name,
                expected=f"factor <= {max_factor}",
                actual=float_value,
                message=f"Factor must be at most {max_factor}",
                suggestion=f"Use a value of {max_factor} or smaller",
            )

        return float_value

    def clamp_value(
        self, value: int | float, min_val: int | float, max_val: int | float
    ) -> int | float:
        """Clamp value to specified range.

        Args:
            value: Value to clamp
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            Clamped value
        """
        return max(min_val, min(max_val, value))

    def normalize_value(
        self,
        value: int | float,
        input_range: tuple[int | float, int | float],
        output_range: tuple[int | float, int | float] = (0.0, 1.0),
    ) -> float:
        """Normalize value from input range to output range.

        Args:
            value: Value to normalize
            input_range: (min, max) of input range
            output_range: (min, max) of output range

        Returns:
            Normalized value
        """
        in_min, in_max = input_range
        out_min, out_max = output_range

        # Handle edge case where input range is zero
        if in_max == in_min:
            return out_min

        # Normalize to 0-1, then scale to output range
        normalized = (value - in_min) / (in_max - in_min)
        return out_min + (normalized * (out_max - out_min))
