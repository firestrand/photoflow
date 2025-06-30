"""Tests for the field system."""

import pytest

from photoflow.core.fields import (
    BaseField,
    BooleanField,
    ChoiceField,
    FloatField,
    ImageFormatField,
    ImageModeField,
    IntegerField,
    StringField,
    ValidationError,
)


class ConcreteField(BaseField):
    """Concrete implementation of BaseField for testing."""

    def validate(self, value):  # type: ignore[no-untyped-def]
        """Simple validation that just returns the value."""
        return self.get_value(value)


class TestValidationError:
    """Test ValidationError formatting."""

    def test_basic_error(self) -> None:
        """Test basic error message formatting."""
        error = ValidationError(
            field_name="width",
            expected="integer",
            actual="hello",
            message="Invalid value",
        )

        expected_msg = (
            "Validation failed for 'width': Invalid value\n"
            "Expected: integer\n"
            "Actual: hello"
        )
        assert str(error) == expected_msg

    def test_error_with_suggestion(self) -> None:
        """Test error message with suggestion."""
        error = ValidationError(
            field_name="width",
            expected="integer",
            actual="hello",
            message="Invalid value",
            suggestion="Try entering a number like 800",
        )

        expected_msg = (
            "Validation failed for 'width': Invalid value\n"
            "Expected: integer\n"
            "Actual: hello\n"
            "Suggestion: Try entering a number like 800"
        )
        assert str(error) == expected_msg


class TestBaseField:
    """Test BaseField functionality."""

    def test_initialization(self) -> None:
        """Test field initialization with defaults."""
        field = ConcreteField()
        assert field.default is None
        assert field.label == ""
        assert field.help_text == ""
        assert field.required is True

    def test_initialization_with_params(self) -> None:
        """Test field initialization with custom parameters."""
        field = ConcreteField(
            default=100,
            label="Width",
            help_text="Image width in pixels",
            required=False,
        )
        assert field.default == 100
        assert field.label == "Width"
        assert field.help_text == "Image width in pixels"
        assert field.required is False

    def test_get_value_with_value(self) -> None:
        """Test get_value returns provided value."""
        field = ConcreteField(default=100)
        assert field.get_value(200) == 200

    def test_get_value_with_none_optional(self) -> None:
        """Test get_value returns default for optional field."""
        field = ConcreteField(default=100, required=False)
        assert field.get_value(None) == 100

    def test_get_value_with_none_required(self) -> None:
        """Test get_value raises error for required field with no default."""
        field = ConcreteField(required=True)
        with pytest.raises(ValidationError) as exc_info:
            field.get_value(None)

        error = exc_info.value
        assert "Field is required" in error.message
        assert error.expected == "non-null value"
        assert error.actual is None


class TestIntegerField:
    """Test IntegerField validation."""

    def test_valid_integer(self) -> None:
        """Test validation with valid integer."""
        field = IntegerField()
        assert field.validate(42) == 42
        assert field.validate("100") == 100
        assert field.validate("0") == 0

    def test_invalid_value(self) -> None:
        """Test validation with invalid value."""
        field = IntegerField(label="Width")
        with pytest.raises(ValidationError) as exc_info:
            field.validate("hello")

        error = exc_info.value
        assert error.field_name == "Width"
        assert "convertible to integer" in error.message
        assert "whole number" in error.suggestion

    def test_min_value_validation(self) -> None:
        """Test minimum value validation."""
        field = IntegerField(min_value=10, label="Size")
        assert field.validate(15) == 15

        with pytest.raises(ValidationError) as exc_info:
            field.validate(5)

        error = exc_info.value
        assert error.field_name == "Size"
        assert "at least 10" in error.message
        assert error.actual == 5

    def test_max_value_validation(self) -> None:
        """Test maximum value validation."""
        field = IntegerField(max_value=100, label="Percentage")
        assert field.validate(50) == 50

        with pytest.raises(ValidationError) as exc_info:
            field.validate(150)

        error = exc_info.value
        assert error.field_name == "Percentage"
        assert "at most 100" in error.message
        assert error.actual == 150

    def test_range_validation(self) -> None:
        """Test combined min/max validation."""
        field = IntegerField(min_value=0, max_value=100)
        assert field.validate(50) == 50
        assert field.validate(0) == 0
        assert field.validate(100) == 100

        with pytest.raises(ValidationError):
            field.validate(-1)

        with pytest.raises(ValidationError):
            field.validate(101)


class TestFloatField:
    """Test FloatField validation."""

    def test_valid_float(self) -> None:
        """Test validation with valid float."""
        field = FloatField()
        assert field.validate(3.14) == 3.14
        assert field.validate("2.5") == 2.5
        assert field.validate(42) == 42.0

    def test_invalid_value(self) -> None:
        """Test validation with invalid value."""
        field = FloatField(label="Ratio")
        with pytest.raises(ValidationError) as exc_info:
            field.validate("hello")

        error = exc_info.value
        assert error.field_name == "Ratio"
        assert "convertible to number" in error.message

    def test_range_validation(self) -> None:
        """Test range validation for floats."""
        field = FloatField(min_value=0.0, max_value=1.0)
        assert field.validate(0.5) == 0.5

        with pytest.raises(ValidationError):
            field.validate(-0.1)

        with pytest.raises(ValidationError):
            field.validate(1.1)


class TestBooleanField:
    """Test BooleanField validation."""

    def test_valid_boolean(self) -> None:
        """Test validation with valid boolean values."""
        field = BooleanField()
        assert field.validate(True) is True
        assert field.validate(False) is False

    def test_string_conversion(self) -> None:
        """Test string to boolean conversion."""
        field = BooleanField()

        # True values
        for value in ["true", "True", "TRUE", "yes", "1", "on", "enabled"]:
            assert field.validate(value) is True

        # False values
        for value in ["false", "False", "FALSE", "no", "0", "off", "disabled"]:
            assert field.validate(value) is False

    def test_invalid_string(self) -> None:
        """Test validation with invalid string."""
        field = BooleanField(label="Enabled")
        with pytest.raises(ValidationError) as exc_info:
            field.validate("maybe")

        error = exc_info.value
        assert error.field_name == "Enabled"
        assert "not recognized as boolean" in error.message

    def test_other_type_conversion(self) -> None:
        """Test conversion of other types to boolean."""
        field = BooleanField()
        assert field.validate(1) is True
        assert field.validate(0) is False
        assert field.validate([1, 2, 3]) is True
        assert field.validate([]) is False


class TestStringField:
    """Test StringField validation."""

    def test_valid_string(self) -> None:
        """Test validation with valid string."""
        field = StringField()
        assert field.validate("hello") == "hello"
        assert field.validate(123) == "123"

    def test_length_validation(self) -> None:
        """Test string length validation."""
        field = StringField(min_length=5, max_length=10, label="Name")

        assert field.validate("hello") == "hello"
        assert field.validate("world!") == "world!"

        with pytest.raises(ValidationError) as exc_info:
            field.validate("hi")
        assert "at least 5 characters" in exc_info.value.message

        with pytest.raises(ValidationError) as exc_info:
            field.validate("this is too long")
        assert "at most 10 characters" in exc_info.value.message


class TestChoiceField:
    """Test ChoiceField validation."""

    def test_valid_choice(self) -> None:
        """Test validation with valid choice."""
        choices = [("small", "Small"), ("medium", "Medium"), ("large", "Large")]
        field = ChoiceField(choices=choices)

        assert field.validate("small") == "small"
        assert field.validate("medium") == "medium"
        assert field.validate("large") == "large"

    def test_invalid_choice(self) -> None:
        """Test validation with invalid choice."""
        choices = [("small", "Small"), ("medium", "Medium")]
        field = ChoiceField(choices=choices, label="Size")

        with pytest.raises(ValidationError) as exc_info:
            field.validate("huge")

        error = exc_info.value
        assert error.field_name == "Size"
        assert "not a valid choice" in error.message
        assert "small" in error.suggestion
        assert "medium" in error.suggestion

    def test_get_choice_label(self) -> None:
        """Test getting choice labels."""
        choices = [("rgb", "RGB Color"), ("l", "Grayscale")]
        field = ChoiceField(choices=choices)

        assert field.get_choice_label("rgb") == "RGB Color"
        assert field.get_choice_label("l") == "Grayscale"
        assert field.get_choice_label("unknown") == "unknown"


class TestImageModeField:
    """Test ImageModeField predefined choices."""

    def test_predefined_choices(self) -> None:
        """Test that image mode field has correct predefined choices."""
        field = ImageModeField()

        # Test valid modes
        assert field.validate("RGB") == "RGB"
        assert field.validate("RGBA") == "RGBA"
        assert field.validate("L") == "L"
        assert field.validate("CMYK") == "CMYK"

        # Test invalid mode
        with pytest.raises(ValidationError):
            field.validate("INVALID")

    def test_choice_labels(self) -> None:
        """Test that choice labels are descriptive."""
        field = ImageModeField()
        assert "24-bit" in field.get_choice_label("RGB")
        assert "Grayscale" in field.get_choice_label("L")


class TestImageFormatField:
    """Test ImageFormatField predefined choices."""

    def test_predefined_choices(self) -> None:
        """Test that image format field has correct predefined choices."""
        field = ImageFormatField()

        # Test valid formats
        assert field.validate("JPEG") == "JPEG"
        assert field.validate("PNG") == "PNG"
        assert field.validate("WebP") == "WebP"

        # Test invalid format
        with pytest.raises(ValidationError):
            field.validate("INVALID")

    def test_choice_labels(self) -> None:
        """Test that choice labels are descriptive."""
        field = ImageFormatField()
        assert "lossy compression" in field.get_choice_label("JPEG")
        assert "transparency" in field.get_choice_label("PNG")


class TestFieldDefaults:
    """Test field default value handling."""

    def test_required_field_with_default(self) -> None:
        """Test required field with default value."""
        field = IntegerField(default=100, required=True)
        assert field.validate(None) == 100

    def test_optional_field_without_default(self) -> None:
        """Test optional field without default value."""
        field = IntegerField(required=False)
        assert field.validate(None) is None

    def test_optional_field_with_default(self) -> None:
        """Test optional field with default value."""
        field = IntegerField(default=50, required=False)
        assert field.validate(None) == 50
