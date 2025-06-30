"""Tests for the BaseAction system."""

from pathlib import Path
from typing import Any, ClassVar, Union
from unittest.mock import Mock

import pytest

from photoflow.core.actions import (
    ActionError,
    ActionRegistry,
    BaseAction,
    default_registry,
    get_action,
    list_actions,
    register_action,
)
from photoflow.core.fields import (
    BooleanField,
    IntegerField,
    StringField,
    ValidationError,
)
from photoflow.core.models import ImageAsset


# Action implementations for testing
class ResizeTestAction(BaseAction):
    """Test action for resizing images."""

    name = "test_resize"
    label = "Test Resize"
    description = "Resize image to specified dimensions"
    version = "1.0"
    author = "Test Author"
    tags: ClassVar[list[str]] = ["geometry", "basic"]

    def setup_fields(self) -> None:
        """Set up resize action fields."""
        self.fields = {
            "width": IntegerField(
                default=800,
                min_value=1,
                max_value=10000,
                label="Width",
                help_text="Target width in pixels",
            ),
            "height": IntegerField(
                default=600,
                min_value=1,
                max_value=10000,
                label="Height",
                help_text="Target height in pixels",
            ),
            "maintain_aspect": BooleanField(
                default=True,
                label="Maintain Aspect Ratio",
                help_text="Keep original proportions",
            ),
        }

    def get_relevant_fields(self, params: dict[str, Any]) -> list[str]:
        """Show height field only if not maintaining aspect ratio."""
        fields = ["width", "maintain_aspect"]
        if not params.get("maintain_aspect", True):
            fields.append("height")
        return fields

    def apply(self, image: ImageAsset, **params: Union[int, bool]) -> ImageAsset:
        """Apply resize operation."""
        width = params["width"]
        height = params.get("height", image.height)
        maintain_aspect = params.get("maintain_aspect", True)

        # Simple resize logic for testing
        if maintain_aspect:
            aspect_ratio = image.width / image.height
            height = int(width / aspect_ratio)

        return image.with_updates(size=(width, height))


class WatermarkTestAction(BaseAction):
    """Test action for adding watermarks."""

    name = "test_watermark"
    label = "Test Watermark"
    description = "Add text watermark to image"
    version = "2.0"
    author = "Test Author"
    tags: ClassVar[list[str]] = ["text", "overlay"]

    def setup_fields(self) -> None:
        """Set up watermark action fields."""
        self.fields = {
            "text": StringField(
                default="Watermark",
                min_length=1,
                max_length=100,
                label="Watermark Text",
                help_text="Text to overlay on image",
            ),
        }

    def apply(self, image: ImageAsset, **params: str) -> ImageAsset:
        """Apply watermark operation."""
        text = params["text"]
        # Simple watermark logic for testing
        metadata_update = {"watermark_text": text}
        return image.with_updates(metadata=metadata_update)


class IncompleteAction(BaseAction):
    """Action missing required metadata."""

    def setup_fields(self) -> None:
        """No fields."""
        self.fields = {}

    def apply(self, image: ImageAsset, **params: str) -> ImageAsset:  # noqa: ARG002
        """No-op apply."""
        return image


class TestActionError:
    """Test ActionError functionality."""

    def test_basic_error(self) -> None:
        """Test basic error creation."""
        error = ActionError(
            action_name="test_action",
            message="Something went wrong",
        )

        assert error.action_name == "test_action"
        assert error.message == "Something went wrong"
        assert error.suggestion == ""
        assert "Action 'test_action' failed" in str(error)

    def test_error_with_suggestion(self) -> None:
        """Test error with suggestion."""
        error = ActionError(
            action_name="test_action",
            message="Invalid parameter",
            suggestion="Use a positive number",
        )

        assert error.suggestion == "Use a positive number"
        assert "Suggestion: Use a positive number" in str(error)


class TestBaseAction:
    """Test BaseAction functionality."""

    def test_action_creation(self) -> None:
        """Test action creation and metadata."""
        action = ResizeTestAction()

        assert action.name == "test_resize"
        assert action.label == "Test Resize"
        assert action.description == "Resize image to specified dimensions"
        assert action.version == "1.0"
        assert action.author == "Test Author"
        assert action.tags == ["geometry", "basic"]
        assert len(action.fields) == 3

    def test_incomplete_action_fails(self) -> None:
        """Test that actions without required metadata fail."""
        with pytest.raises(ValueError, match="must define 'name'"):
            IncompleteAction()

    def test_field_setup(self) -> None:
        """Test field setup and validation."""
        action = ResizeTestAction()

        # Check fields are properly configured
        assert "width" in action.fields
        assert "height" in action.fields
        assert "maintain_aspect" in action.fields

        width_field = action.fields["width"]
        assert width_field.default == 800
        assert width_field.label == "Width"

    def test_parameter_validation_success(self) -> None:
        """Test successful parameter validation."""
        action = ResizeTestAction()
        params = {"width": 1920, "height": 1080, "maintain_aspect": False}

        validated = action.validate_params(params)

        assert validated["width"] == 1920
        assert validated["height"] == 1080
        assert validated["maintain_aspect"] is False

    def test_parameter_validation_failure(self) -> None:
        """Test parameter validation failure."""
        action = ResizeTestAction()
        params = {"width": "invalid", "maintain_aspect": True}

        with pytest.raises(ValidationError) as exc_info:
            action.validate_params(params)

        error = exc_info.value
        assert "test_resize.width" in error.field_name
        assert "Test Resize" in error.message

    def test_get_relevant_fields(self) -> None:
        """Test dynamic field relevance."""
        action = ResizeTestAction()

        # With maintain_aspect=True, height should not be relevant
        params = {"maintain_aspect": True}
        relevant = action.get_relevant_fields(params)
        assert "width" in relevant
        assert "maintain_aspect" in relevant
        assert "height" not in relevant

        # With maintain_aspect=False, height should be relevant
        params = {"maintain_aspect": False}
        relevant = action.get_relevant_fields(params)
        assert "width" in relevant
        assert "maintain_aspect" in relevant
        assert "height" in relevant

    def test_get_field_schema(self) -> None:
        """Test field schema generation."""
        action = ResizeTestAction()
        schema = action.get_field_schema()

        assert "width" in schema
        width_schema = schema["width"]
        assert width_schema["type"] == "integer"
        assert width_schema["label"] == "Width"
        assert width_schema["default"] == 800
        assert width_schema["min_value"] == 1
        assert width_schema["max_value"] == 10000
        assert width_schema["required"] is True

    def test_execute_success(self) -> None:
        """Test successful action execution."""
        action = ResizeTestAction()
        image = ImageAsset(
            path=Path("test.jpg"),
            format="JPEG",
            size=(3000, 2000),
            mode="RGB",
        )

        params = {"width": 1500, "maintain_aspect": True}
        result = action.execute(image, **params)

        # Check image was resized
        assert result.width == 1500
        assert result.height == 1000  # Aspect ratio maintained

        # Check processing metadata was updated
        assert result.processing_metadata["last_action"] == "test_resize"
        assert result.processing_metadata["last_action_version"] == "1.0"
        assert result.processing_metadata["last_action_params"]["width"] == 1500

    def test_execute_validation_error(self) -> None:
        """Test action execution with validation error."""
        action = ResizeTestAction()
        image = ImageAsset(
            path=Path("test.jpg"),
            format="JPEG",
            size=(1920, 1080),
            mode="RGB",
        )

        params = {"width": -100}  # Invalid width

        with pytest.raises(ActionError) as exc_info:
            action.execute(image, **params)

        error = exc_info.value
        assert error.action_name == "test_resize"
        assert "Parameter validation failed" in error.message

    def test_execute_processing_error(self) -> None:
        """Test action execution with processing error."""
        # Create action that will fail during apply()
        action = ResizeTestAction()
        image = ImageAsset(
            path=Path("test.jpg"),
            format="JPEG",
            size=(1920, 1080),
            mode="RGB",
        )

        # Mock apply method to raise exception
        action.apply = Mock(side_effect=RuntimeError("Processing failed"))  # type: ignore[method-assign]

        with pytest.raises(ActionError) as exc_info:
            action.execute(image, width=800)

        error = exc_info.value
        assert error.action_name == "test_resize"
        assert "Processing failed" in error.message

    def test_get_action_info(self) -> None:
        """Test action info generation."""
        action = ResizeTestAction()
        info = action.get_action_info()

        assert info["name"] == "test_resize"
        assert info["label"] == "Test Resize"
        assert info["description"] == "Resize image to specified dimensions"
        assert info["version"] == "1.0"
        assert info["author"] == "Test Author"
        assert info["tags"] == ["geometry", "basic"]
        assert "fields" in info
        assert "width" in info["fields"]

    def test_string_representations(self) -> None:
        """Test string and repr methods."""
        action = ResizeTestAction()

        assert str(action) == "Test Resize (test_resize)"
        assert "ResizeTestAction" in repr(action)
        assert "test_resize" in repr(action)
        assert "1.0" in repr(action)


class TestActionRegistry:
    """Test ActionRegistry functionality."""

    def setup_method(self) -> None:
        """Set up test fixture."""
        self.registry = ActionRegistry()

    def test_register_action(self) -> None:
        """Test action registration."""
        self.registry.register(ResizeTestAction)

        assert len(self.registry) == 1
        assert "test_resize" in self.registry
        assert self.registry.list_actions() == ["test_resize"]

    def test_register_duplicate_action(self) -> None:
        """Test registering duplicate action."""
        self.registry.register(ResizeTestAction)

        # Registering same class again should be fine
        self.registry.register(ResizeTestAction)
        assert len(self.registry) == 1

        # But registering different class with same name should fail
        class ConflictingAction(BaseAction):
            name = "test_resize"
            label = "Conflicting"

            def setup_fields(self) -> None:
                self.fields = {}

            def apply(self, image: ImageAsset, **params: str) -> ImageAsset:
                return image

        with pytest.raises(ValueError, match="already registered"):
            self.registry.register(ConflictingAction)

    def test_register_action_without_name(self) -> None:
        """Test registering action without name."""

        class NoNameAction(BaseAction):
            label = "No Name"

            def setup_fields(self) -> None:
                self.fields = {}

            def apply(self, image: ImageAsset, **params: str) -> ImageAsset:
                return image

        with pytest.raises(ValueError, match="must define 'name'"):
            self.registry.register(NoNameAction)

    def test_get_action(self) -> None:
        """Test getting action instance."""
        self.registry.register(ResizeTestAction)

        action = self.registry.get_action("test_resize")
        assert isinstance(action, ResizeTestAction)
        assert action.name == "test_resize"

    def test_get_nonexistent_action(self) -> None:
        """Test getting non-existent action."""
        with pytest.raises(KeyError, match="Action 'nonexistent' not found"):
            self.registry.get_action("nonexistent")

    def test_unregister_action(self) -> None:
        """Test action unregistration."""
        self.registry.register(ResizeTestAction)
        assert len(self.registry) == 1

        self.registry.unregister("test_resize")
        assert len(self.registry) == 0
        assert "test_resize" not in self.registry

    def test_get_actions_by_tag(self) -> None:
        """Test getting actions by tag."""
        self.registry.register(ResizeTestAction)
        self.registry.register(WatermarkTestAction)

        geometry_actions = self.registry.get_actions_by_tag("geometry")
        assert geometry_actions == ["test_resize"]

        text_actions = self.registry.get_actions_by_tag("text")
        assert text_actions == ["test_watermark"]

        nonexistent_actions = self.registry.get_actions_by_tag("nonexistent")
        assert nonexistent_actions == []

    def test_get_action_info(self) -> None:
        """Test getting action information."""
        self.registry.register(ResizeTestAction)

        info = self.registry.get_action_info("test_resize")
        assert info["name"] == "test_resize"
        assert info["label"] == "Test Resize"

    def test_clear_registry(self) -> None:
        """Test clearing registry."""
        self.registry.register(ResizeTestAction)
        self.registry.register(WatermarkTestAction)
        assert len(self.registry) == 2

        self.registry.clear()
        assert len(self.registry) == 0
        assert self.registry.list_actions() == []


class TestGlobalFunctions:
    """Test global registry functions."""

    def setup_method(self) -> None:
        """Clear default registry before each test."""
        default_registry.clear()

    def teardown_method(self) -> None:
        """Clear default registry after each test."""
        default_registry.clear()

    def test_register_action_global(self) -> None:
        """Test global register_action function."""
        register_action(ResizeTestAction)

        assert len(default_registry) == 1
        assert "test_resize" in default_registry

    def test_get_action_global(self) -> None:
        """Test global get_action function."""
        register_action(ResizeTestAction)

        action = get_action("test_resize")
        assert isinstance(action, ResizeTestAction)

    def test_list_actions_global(self) -> None:
        """Test global list_actions function."""
        register_action(ResizeTestAction)
        register_action(WatermarkTestAction)

        actions = list_actions()
        assert set(actions) == {"test_resize", "test_watermark"}


class TestActionIntegration:
    """Test action system integration."""

    def test_complete_workflow(self) -> None:
        """Test complete action workflow."""
        # Create and register action
        registry = ActionRegistry()
        registry.register(ResizeTestAction)

        # Create test image
        image = ImageAsset(
            path=Path("test.jpg"),
            format="JPEG",
            size=(2000, 1500),
            mode="RGB",
        )

        # Get action and execute
        action = registry.get_action("test_resize")
        result = action.execute(image, width=1000, maintain_aspect=True)

        # Verify results
        assert result.width == 1000
        assert result.height == 750  # Aspect ratio maintained
        assert result.processing_metadata["last_action"] == "test_resize"

    def test_chained_actions(self) -> None:
        """Test chaining multiple actions."""
        registry = ActionRegistry()
        registry.register(ResizeTestAction)
        registry.register(WatermarkTestAction)

        # Create test image
        image = ImageAsset(
            path=Path("test.jpg"),
            format="JPEG",
            size=(2000, 1500),
            mode="RGB",
        )

        # Apply resize action
        resize_action = registry.get_action("test_resize")
        resized_image = resize_action.execute(image, width=1000, maintain_aspect=True)

        # Apply watermark action
        watermark_action = registry.get_action("test_watermark")
        final_image = watermark_action.execute(resized_image, text="© 2025")

        # Verify both actions were applied
        assert final_image.width == 1000
        assert final_image.height == 750
        assert final_image.metadata["watermark_text"] == "© 2025"
        assert final_image.processing_metadata["last_action"] == "test_watermark"
