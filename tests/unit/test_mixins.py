"""Tests for processing mixins."""

import pytest

from photoflow.core.fields import ValidationError
from photoflow.core.mixins import ColorMixin, GeometryMixin, ProcessingMixin


class TestGeometryMixin:
    """Test GeometryMixin functionality."""

    def setup_method(self) -> None:
        """Set up test fixture."""
        self.mixin = GeometryMixin()

    def test_calculate_scaled_size_width_only(self) -> None:
        """Test scaling with width only, maintaining aspect ratio."""
        result = self.mixin.calculate_scaled_size(
            original_size=(1920, 1080),
            target_width=960,
            maintain_aspect=True,
        )
        assert result == (960, 540)

    def test_calculate_scaled_size_height_only(self) -> None:
        """Test scaling with height only, maintaining aspect ratio."""
        result = self.mixin.calculate_scaled_size(
            original_size=(1920, 1080),
            target_height=540,
            maintain_aspect=True,
        )
        assert result == (960, 540)

    def test_calculate_scaled_size_both_dimensions_fit(self) -> None:
        """Test scaling with both dimensions, maintaining aspect ratio."""
        result = self.mixin.calculate_scaled_size(
            original_size=(1920, 1080),
            target_width=800,
            target_height=600,
            maintain_aspect=True,
        )
        # Should fit within 800x600, maintaining aspect ratio
        assert result == (800, 450)

    def test_calculate_scaled_size_no_aspect_ratio(self) -> None:
        """Test scaling without maintaining aspect ratio."""
        result = self.mixin.calculate_scaled_size(
            original_size=(1920, 1080),
            target_width=800,
            target_height=600,
            maintain_aspect=False,
        )
        assert result == (800, 600)

    def test_calculate_scaled_size_no_targets(self) -> None:
        """Test error when no target dimensions provided."""
        with pytest.raises(
            ValueError, match="Must specify at least one target dimension"
        ):
            self.mixin.calculate_scaled_size((1920, 1080))

    def test_validate_dimensions_valid(self) -> None:
        """Test valid dimension validation."""
        # Should not raise any exceptions
        self.mixin.validate_dimensions(width=800, height=600)
        self.mixin.validate_dimensions(width=1920)
        self.mixin.validate_dimensions(height=1080)

    def test_validate_dimensions_invalid_type(self) -> None:
        """Test dimension validation with invalid types."""
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.validate_dimensions(width="invalid")  # type: ignore[arg-type]

        error = exc_info.value
        assert error.field_name == "width"
        assert "must be an integer" in error.message

    def test_validate_dimensions_too_small(self) -> None:
        """Test dimension validation with values too small."""
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.validate_dimensions(width=0, min_size=1)

        error = exc_info.value
        assert error.field_name == "width"
        assert "at least 1" in error.message

    def test_validate_dimensions_too_large(self) -> None:
        """Test dimension validation with values too large."""
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.validate_dimensions(width=60000, max_size=50000)

        error = exc_info.value
        assert error.field_name == "width"
        assert "at most 50000" in error.message

    def test_calculate_crop_bounds_valid(self) -> None:
        """Test valid crop bounds calculation."""
        result = self.mixin.calculate_crop_bounds(
            image_size=(1920, 1080),
            crop_x=100,
            crop_y=50,
            crop_width=800,
            crop_height=600,
        )
        assert result == (100, 50, 800, 600)

    def test_calculate_crop_bounds_negative_position(self) -> None:
        """Test crop bounds with negative position."""
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.calculate_crop_bounds(
                image_size=(1920, 1080),
                crop_x=-10,
                crop_y=0,
                crop_width=800,
                crop_height=600,
            )

        error = exc_info.value
        assert error.field_name == "crop_x"
        assert "cannot be negative" in error.message

    def test_calculate_crop_bounds_zero_size(self) -> None:
        """Test crop bounds with zero dimensions."""
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.calculate_crop_bounds(
                image_size=(1920, 1080),
                crop_x=0,
                crop_y=0,
                crop_width=0,
                crop_height=600,
            )

        error = exc_info.value
        assert error.field_name == "crop_width"
        assert "must be positive" in error.message

    def test_calculate_crop_bounds_exceeds_image(self) -> None:
        """Test crop bounds that exceed image dimensions."""
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.calculate_crop_bounds(
                image_size=(1920, 1080),
                crop_x=1000,
                crop_y=0,
                crop_width=1000,  # 1000 + 1000 > 1920
                crop_height=600,
            )

        error = exc_info.value
        assert error.field_name == "crop_bounds"
        assert "extends beyond image width" in error.message

    def test_calculate_centered_crop_valid(self) -> None:
        """Test centered crop calculation."""
        result = self.mixin.calculate_centered_crop(
            image_size=(1920, 1080),
            target_size=(800, 600),
        )
        # Should be centered: x = (1920-800)/2 = 560, y = (1080-600)/2 = 240
        assert result == (560, 240, 800, 600)

    def test_calculate_centered_crop_too_large(self) -> None:
        """Test centered crop with target larger than image."""
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.calculate_centered_crop(
                image_size=(800, 600),
                target_size=(1000, 800),
            )

        error = exc_info.value
        assert error.field_name == "target_size"
        assert "larger than image" in error.message


class TestColorMixin:
    """Test ColorMixin functionality."""

    def setup_method(self) -> None:
        """Set up test fixture."""
        self.mixin = ColorMixin()

    def test_validate_color_mode_valid(self) -> None:
        """Test valid color mode validation."""
        assert self.mixin.validate_color_mode("RGB") == "RGB"
        assert self.mixin.validate_color_mode("RGBA") == "RGBA"
        assert self.mixin.validate_color_mode("L") == "L"

    def test_validate_color_mode_with_supported_list(self) -> None:
        """Test color mode validation with specific supported modes."""
        supported = ["RGB", "RGBA"]
        assert self.mixin.validate_color_mode("RGB", supported) == "RGB"

        with pytest.raises(ValidationError) as exc_info:
            self.mixin.validate_color_mode("CMYK", supported)

        error = exc_info.value
        assert error.field_name == "color_mode"
        assert "not supported" in error.message

    def test_validate_color_value_grayscale(self) -> None:
        """Test color value validation for grayscale modes."""
        # Valid grayscale values
        assert self.mixin.validate_color_value(128, "L") == 128
        assert self.mixin.validate_color_value((128,), "L") == 128
        assert self.mixin.validate_color_value(255.0, "L") == 255

    def test_validate_color_value_grayscale_invalid(self) -> None:
        """Test invalid grayscale color values."""
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.validate_color_value((128, 64), "L")

        error = exc_info.value
        assert "single color value" in error.message

        with pytest.raises(ValidationError) as exc_info:
            self.mixin.validate_color_value("invalid", "L")  # type: ignore[arg-type]

        error = exc_info.value
        assert "must be numeric" in error.message

    def test_validate_color_value_rgb(self) -> None:
        """Test color value validation for RGB mode."""
        # Valid RGB values
        result = self.mixin.validate_color_value((255, 128, 0), "RGB")
        assert result == (255, 128, 0)

        result = self.mixin.validate_color_value([255, 128, 0], "RGB")  # type: ignore[arg-type]
        assert result == (255, 128, 0)

    def test_validate_color_value_rgb_invalid(self) -> None:
        """Test invalid RGB color values."""
        # Wrong number of components
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.validate_color_value((255, 128), "RGB")

        error = exc_info.value
        assert "3 color components" in error.message

        # Non-numeric component
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.validate_color_value((255, "invalid", 0), "RGB")  # type: ignore[arg-type]

        error = exc_info.value
        assert "must be numeric" in error.message

        # Out of range component
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.validate_color_value((300, 128, 0), "RGB")

        error = exc_info.value
        assert "out of range" in error.message

    def test_validate_color_value_rgba(self) -> None:
        """Test color value validation for RGBA mode."""
        result = self.mixin.validate_color_value((255, 128, 0, 255), "RGBA")
        assert result == (255, 128, 0, 255)

        # Wrong number of components
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.validate_color_value((255, 128, 0), "RGBA")

        error = exc_info.value
        assert "4 color components" in error.message

    def test_validate_color_value_custom_range(self) -> None:
        """Test color value validation with custom range."""
        # Test with 0-1 range
        result = self.mixin.validate_color_value(
            (1.0, 0.5, 0.0),
            "RGB",
            component_range=(0, 1),  # type: ignore[arg-type]
        )
        assert result == (1, 0, 0)  # Converted to int

        with pytest.raises(ValidationError) as exc_info:
            self.mixin.validate_color_value(
                (2, 0.5, 0.0),  # type: ignore[arg-type]
                "RGB",
                component_range=(0, 1),  # 2 is out of range
            )

        error = exc_info.value
        assert "out of range" in error.message

    def test_get_mode_info(self) -> None:
        """Test color mode information retrieval."""
        info = self.mixin.get_mode_info("RGB")
        assert info["description"] == "RGB Color (24-bit)"
        assert info["channels"] == 3
        assert info["has_alpha"] is False
        assert info["web_safe"] is True

        info = self.mixin.get_mode_info("RGBA")
        assert info["channels"] == 4
        assert info["has_alpha"] is True

        info = self.mixin.get_mode_info("L")
        assert info["channels"] == 1
        assert info["has_alpha"] is False

    def test_suggest_compatible_modes(self) -> None:
        """Test compatible mode suggestions."""
        suggestions = self.mixin.suggest_compatible_modes("RGB")
        assert "RGB" in suggestions
        assert "RGBA" in suggestions
        assert "L" in suggestions

        suggestions = self.mixin.suggest_compatible_modes("CMYK")
        assert "CMYK" in suggestions
        assert "RGB" in suggestions

        # Unknown mode should return basic suggestions
        suggestions = self.mixin.suggest_compatible_modes("UNKNOWN")
        assert "RGB" in suggestions
        assert "L" in suggestions


class TestProcessingMixin:
    """Test ProcessingMixin functionality."""

    def setup_method(self) -> None:
        """Set up test fixture."""
        self.mixin = ProcessingMixin()

    def test_validate_percentage_valid(self) -> None:
        """Test valid percentage validation."""
        assert self.mixin.validate_percentage(50) == 50.0
        assert self.mixin.validate_percentage(75.5) == 75.5
        assert self.mixin.validate_percentage(0) == 0.0
        assert self.mixin.validate_percentage(100) == 100.0

    def test_validate_percentage_over_100(self) -> None:
        """Test percentage validation with values over 100."""
        # Should fail by default
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.validate_percentage(150)

        error = exc_info.value
        assert "cannot exceed 100%" in error.message

        # Should succeed when explicitly allowed
        assert self.mixin.validate_percentage(150, allow_over_100=True) == 150.0

    def test_validate_percentage_negative(self) -> None:
        """Test percentage validation with negative values."""
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.validate_percentage(-10)

        error = exc_info.value
        assert "cannot be negative" in error.message

    def test_validate_percentage_invalid_type(self) -> None:
        """Test percentage validation with invalid types."""
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.validate_percentage("invalid")  # type: ignore[arg-type]

        error = exc_info.value
        assert "must be a number" in error.message

    def test_validate_factor_valid(self) -> None:
        """Test valid factor validation."""
        assert self.mixin.validate_factor(1.0) == 1.0
        assert self.mixin.validate_factor(2.5) == 2.5
        assert self.mixin.validate_factor(0.5) == 0.5

    def test_validate_factor_with_limits(self) -> None:
        """Test factor validation with custom limits."""
        assert self.mixin.validate_factor(1.5, min_factor=1.0, max_factor=2.0) == 1.5

        # Below minimum
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.validate_factor(0.5, min_factor=1.0)

        error = exc_info.value
        assert "at least 1.0" in error.message

        # Above maximum
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.validate_factor(3.0, max_factor=2.0)

        error = exc_info.value
        assert "at most 2.0" in error.message

    def test_validate_factor_invalid_type(self) -> None:
        """Test factor validation with invalid types."""
        with pytest.raises(ValidationError) as exc_info:
            self.mixin.validate_factor("invalid")  # type: ignore[arg-type]

        error = exc_info.value
        assert "must be a number" in error.message

    def test_clamp_value(self) -> None:
        """Test value clamping."""
        assert self.mixin.clamp_value(5, 0, 10) == 5
        assert self.mixin.clamp_value(-5, 0, 10) == 0
        assert self.mixin.clamp_value(15, 0, 10) == 10

        # Test with floats
        assert self.mixin.clamp_value(2.5, 1.0, 3.0) == 2.5
        assert self.mixin.clamp_value(0.5, 1.0, 3.0) == 1.0

    def test_normalize_value(self) -> None:
        """Test value normalization."""
        # Default normalization to 0-1 range
        result = self.mixin.normalize_value(50, (0, 100))
        assert result == 0.5

        result = self.mixin.normalize_value(75, (0, 100))
        assert result == 0.75

        # Custom output range
        result = self.mixin.normalize_value(50, (0, 100), (0, 255))
        assert result == 127.5

        # Edge case: zero input range
        result = self.mixin.normalize_value(5, (5, 5), (0, 1))
        assert result == 0.0  # Should return output minimum

    def test_normalize_value_negative_ranges(self) -> None:
        """Test normalization with negative ranges."""
        result = self.mixin.normalize_value(0, (-100, 100), (0, 1))
        assert result == 0.5

        result = self.mixin.normalize_value(-50, (-100, 100), (0, 1))
        assert result == 0.25


class TestMixinIntegration:
    """Test mixin integration scenarios."""

    def test_combined_mixins(self) -> None:
        """Test using multiple mixins together."""

        class TestAction(GeometryMixin, ColorMixin, ProcessingMixin):
            """Test action combining multiple mixins."""

            def process_image(self, original_size, color_mode, opacity):  # type: ignore[no-untyped-def]
                """Example method using all mixins."""
                # Validate inputs using mixins
                self.validate_dimensions(*original_size)
                self.validate_color_mode(color_mode, ["RGB", "RGBA"])
                opacity_percent = self.validate_percentage(opacity)

                # Calculate new size
                new_size = self.calculate_scaled_size(
                    original_size, target_width=800, maintain_aspect=True
                )

                return {
                    "original_size": original_size,
                    "new_size": new_size,
                    "color_mode": color_mode,
                    "opacity": opacity_percent / 100.0,
                }

        action = TestAction()
        result = action.process_image((1920, 1080), "RGB", 75)  # type: ignore[no-untyped-call]

        assert result["original_size"] == (1920, 1080)
        assert result["new_size"] == (800, 450)
        assert result["color_mode"] == "RGB"
        assert result["opacity"] == 0.75

    def test_mixin_error_consistency(self) -> None:
        """Test that mixins provide consistent error handling."""

        class TestAction(GeometryMixin, ColorMixin, ProcessingMixin):
            """Test action for error consistency."""

            def test_errors(self) -> list[ValidationError]:
                """Test various error conditions."""
                errors = []

                try:
                    self.validate_dimensions(width=-1)
                except ValidationError as e:
                    errors.append(e)

                try:
                    self.validate_color_mode("INVALID")
                except ValidationError as e:
                    errors.append(e)

                try:
                    self.validate_percentage(-5)
                except ValidationError as e:
                    errors.append(e)

                return errors

        action = TestAction()
        errors = action.test_errors()

        # All should be ValidationError instances
        assert len(errors) == 3
        for error in errors:
            assert isinstance(error, ValidationError)
            assert error.field_name
            assert error.message
            assert error.suggestion  # All should have suggestions
