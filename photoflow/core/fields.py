"""Form field system for PhotoFlow actions.

This module provides a field-based parameter system inspired by Django/Phatch,
replacing the generic ActionParameter with specialized field types that provide
better validation, user experience, and dynamic behavior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ValidationError(Exception):
    """Detailed validation error with context and suggestions."""

    def __init__(
        self,
        field_name: str,
        expected: str,
        actual: Any,
        message: str,
        suggestion: str = "",
    ) -> None:
        """Initialize validation error.

        Args:
            field_name: Name of the field that failed validation
            expected: Description of expected value/format
            actual: The actual value that was provided
            message: Human-readable error message
            suggestion: Optional suggestion for fixing the error
        """
        self.field_name = field_name
        self.expected = expected
        self.actual = actual
        self.message = message
        self.suggestion = suggestion

        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the complete error message."""
        msg = f"Validation failed for '{self.field_name}': {self.message}"
        msg += f"\nExpected: {self.expected}"
        msg += f"\nActual: {self.actual}"
        if self.suggestion:
            msg += f"\nSuggestion: {self.suggestion}"
        return msg


class BaseField(ABC):
    """Base class for all parameter fields.

    Provides common functionality for field validation, defaults, and metadata.
    All concrete field types inherit from this class.
    """

    def __init__(
        self,
        default: Any = None,
        label: str = "",
        help_text: str = "",
        required: bool = True,
    ) -> None:
        """Initialize base field.

        Args:
            default: Default value for the field
            label: Human-readable label for UI display
            help_text: Help text explaining the field's purpose
            required: Whether the field is required
        """
        self.default = default
        self.label = label
        self.help_text = help_text
        self.required = required

    @abstractmethod
    def validate(self, value: Any) -> Any:
        """Validate and convert the field value.

        Args:
            value: Raw value to validate

        Returns:
            Validated and converted value

        Raises:
            ValidationError: If validation fails
        """
        pass

    def get_value(self, value: Any) -> Any:
        """Get the field value, using default if value is None.

        Args:
            value: Input value (may be None)

        Returns:
            Value to use (input value or default)
        """
        if value is None:
            if self.required and self.default is None:
                raise ValidationError(
                    field_name=self.label or "field",
                    expected="non-null value",
                    actual=None,
                    message="Field is required but no value provided",
                    suggestion="Provide a value for this required field",
                )
            return self.default
        return value


class IntegerField(BaseField):
    """Integer field with range validation."""

    def __init__(
        self,
        min_value: int | None = None,
        max_value: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize integer field.

        Args:
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
            **kwargs: Base field arguments
        """
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any) -> int | None:
        """Validate and convert to integer.

        Args:
            value: Value to validate

        Returns:
            Validated integer value

        Raises:
            ValidationError: If validation fails
        """
        value = self.get_value(value)
        if value is None:
            return value

        # Convert to integer
        try:
            int_value = int(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(
                field_name=self.label or "integer_field",
                expected="integer number",
                actual=f"{type(value).__name__}: {value}",
                message="Value must be convertible to integer",
                suggestion="Try entering a whole number like 100 or 500",
            ) from e

        # Range validation
        if self.min_value is not None and int_value < self.min_value:
            raise ValidationError(
                field_name=self.label or "integer_field",
                expected=f"integer >= {self.min_value}",
                actual=int_value,
                message=f"Value must be at least {self.min_value}",
                suggestion=f"Try a value like {self.min_value} or higher",
            )

        if self.max_value is not None and int_value > self.max_value:
            raise ValidationError(
                field_name=self.label or "integer_field",
                expected=f"integer <= {self.max_value}",
                actual=int_value,
                message=f"Value must be at most {self.max_value}",
                suggestion=f"Try a value like {self.max_value} or lower",
            )

        return int_value


class FloatField(BaseField):
    """Float field with range validation."""

    def __init__(
        self,
        min_value: float | None = None,
        max_value: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize float field.

        Args:
            min_value: Minimum allowed value (inclusive)
            max_value: Maximum allowed value (inclusive)
            **kwargs: Base field arguments
        """
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, value: Any) -> float | None:
        """Validate and convert to float.

        Args:
            value: Value to validate

        Returns:
            Validated float value

        Raises:
            ValidationError: If validation fails
        """
        value = self.get_value(value)
        if value is None:
            return value

        # Convert to float
        try:
            float_value = float(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(
                field_name=self.label or "float_field",
                expected="numeric value",
                actual=f"{type(value).__name__}: {value}",
                message="Value must be convertible to number",
                suggestion="Try entering a number like 1.5 or 100.0",
            ) from e

        # Range validation
        if self.min_value is not None and float_value < self.min_value:
            raise ValidationError(
                field_name=self.label or "float_field",
                expected=f"number >= {self.min_value}",
                actual=float_value,
                message=f"Value must be at least {self.min_value}",
                suggestion=f"Try a value like {self.min_value} or higher",
            )

        if self.max_value is not None and float_value > self.max_value:
            raise ValidationError(
                field_name=self.label or "float_field",
                expected=f"number <= {self.max_value}",
                actual=float_value,
                message=f"Value must be at most {self.max_value}",
                suggestion=f"Try a value like {self.max_value} or lower",
            )

        return float_value


class BooleanField(BaseField):
    """Boolean field with flexible string conversion."""

    def validate(self, value: Any) -> bool | None:
        """Validate and convert to boolean.

        Args:
            value: Value to validate

        Returns:
            Validated boolean value

        Raises:
            ValidationError: If validation fails
        """
        value = self.get_value(value)
        if value is None:
            return value

        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            lower_value = value.lower()
            if lower_value in {"true", "yes", "1", "on", "enabled"}:
                return True
            elif lower_value in {"false", "no", "0", "off", "disabled"}:
                return False
            else:
                raise ValidationError(
                    field_name=self.label or "boolean_field",
                    expected="boolean value (true/false)",
                    actual=value,
                    message="String value not recognized as boolean",
                    suggestion="Use 'true', 'false', 'yes', 'no', '1', or '0'",
                )

        # Try to convert other types
        try:
            return bool(value)
        except (ValueError, TypeError) as e:
            raise ValidationError(
                field_name=self.label or "boolean_field",
                expected="boolean value",
                actual=f"{type(value).__name__}: {value}",
                message="Value cannot be converted to boolean",
                suggestion="Use true, false, 1, or 0",
            ) from e


class StringField(BaseField):
    """String field with length validation."""

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize string field.

        Args:
            min_length: Minimum string length
            max_length: Maximum string length
            **kwargs: Base field arguments
        """
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length

    def validate(self, value: Any) -> str | None:
        """Validate and convert to string.

        Args:
            value: Value to validate

        Returns:
            Validated string value

        Raises:
            ValidationError: If validation fails
        """
        value = self.get_value(value)
        if value is None:
            return value

        # Convert to string
        str_value = str(value)

        # Length validation
        if self.min_length is not None and len(str_value) < self.min_length:
            raise ValidationError(
                field_name=self.label or "string_field",
                expected=f"string with at least {self.min_length} characters",
                actual=f"'{str_value}' ({len(str_value)} chars)",
                message=f"String must be at least {self.min_length} characters long",
                suggestion=f"Add more characters to reach {self.min_length} minimum",
            )

        if self.max_length is not None and len(str_value) > self.max_length:
            raise ValidationError(
                field_name=self.label or "string_field",
                expected=f"string with at most {self.max_length} characters",
                actual=f"'{str_value}' ({len(str_value)} chars)",
                message=f"String must be at most {self.max_length} characters long",
                suggestion=f"Shorten to {self.max_length} characters or fewer",
            )

        return str_value


class ChoiceField(BaseField):
    """Choice field with predefined options."""

    def __init__(self, choices: list[tuple[str, str]], **kwargs: Any) -> None:
        """Initialize choice field.

        Args:
            choices: List of (value, label) tuples
            **kwargs: Base field arguments
        """
        super().__init__(**kwargs)
        self.choices = choices
        self._valid_values = {choice[0] for choice in choices}

    def validate(self, value: Any) -> str | None:
        """Validate choice value.

        Args:
            value: Value to validate

        Returns:
            Validated choice value

        Raises:
            ValidationError: If validation fails
        """
        value = self.get_value(value)
        if value is None:
            return value

        str_value = str(value)

        if str_value not in self._valid_values:
            valid_options = [f"'{choice[0]}' ({choice[1]})" for choice in self.choices]
            raise ValidationError(
                field_name=self.label or "choice_field",
                expected=f"one of: {', '.join(sorted(self._valid_values))}",
                actual=str_value,
                message="Value is not a valid choice",
                suggestion=f"Choose from: {', '.join(valid_options)}",
            )

        return str_value

    def get_choice_label(self, value: str) -> str:
        """Get the display label for a choice value.

        Args:
            value: Choice value

        Returns:
            Display label for the value
        """
        for choice_value, choice_label in self.choices:
            if choice_value == value:
                return choice_label
        return value


class ImageModeField(ChoiceField):
    """Specialized choice field for image color modes."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize image mode field with predefined choices."""
        choices = [
            ("RGB", "RGB Color (24-bit)"),
            ("RGBA", "RGB with Alpha (32-bit)"),
            ("L", "Grayscale (8-bit)"),
            ("LA", "Grayscale with Alpha (16-bit)"),
            ("CMYK", "CMYK (32-bit)"),
            ("P", "Palette (8-bit)"),
        ]
        super().__init__(choices=choices, **kwargs)


class ImageFormatField(ChoiceField):
    """Specialized choice field for image formats."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize image format field with common formats."""
        choices = [
            ("JPEG", "JPEG (lossy compression)"),
            ("PNG", "PNG (lossless, supports transparency)"),
            ("WebP", "WebP (modern format, good compression)"),
            ("TIFF", "TIFF (uncompressed, professional)"),
            ("BMP", "BMP (uncompressed bitmap)"),
            ("GIF", "GIF (supports animation)"),
        ]
        super().__init__(choices=choices, **kwargs)
