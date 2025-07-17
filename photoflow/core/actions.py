"""Base action system with form-field architecture.

This module provides the foundation for PhotoFlow's action system, featuring
form-based parameter definitions, dynamic field relevance, and enhanced
validation inspired by Django's form system and Phatch's action architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from .fields import BaseField, ValidationError

if TYPE_CHECKING:
    from .models import ImageAsset

# Check if metadata actions are available
try:
    # This import will succeed if the metadata actions module loads correctly
    from ..actions import metadata as _metadata_actions  # noqa: F401

    METADATA_ACTIONS_AVAILABLE = True
except ImportError:
    METADATA_ACTIONS_AVAILABLE = False


class ActionError(Exception):
    """Error in action execution."""

    def __init__(self, action_name: str, message: str, suggestion: str = "") -> None:
        """Initialize action error.

        Args:
            action_name: Name of the action that failed
            message: Error message
            suggestion: Optional suggestion for fixing the error
        """
        self.action_name = action_name
        self.message = message
        self.suggestion = suggestion
        full_message = f"Action '{action_name}' failed: {message}"
        if suggestion:
            full_message += f"\nSuggestion: {suggestion}"
        super().__init__(full_message)


class BaseAction(ABC):
    """Base class for all image processing actions.

    Provides form-field architecture with dynamic parameter validation,
    field relevance calculation, and rich metadata support. All PhotoFlow
    actions inherit from this class.
    """

    name: str = ""
    """Action identifier (snake_case)"""
    label: str = ""
    """Human-readable action name"""
    description: str = ""
    """Detailed description of what the action does"""
    version: str = "1.0"
    """Action version for compatibility tracking"""
    author: str = ""
    """Action author/creator"""
    tags: ClassVar[list[str]] = []
    """Tags for categorization and search"""

    def __init__(self) -> None:
        """Initialize action with field definitions."""
        self.fields: dict[str, BaseField] = {}
        self.setup_fields()
        if not self.name:
            raise ValueError(f"Action {self.__class__.__name__} must define 'name'")
        if not self.label:
            raise ValueError(f"Action {self.__class__.__name__} must define 'label'")

    @abstractmethod
    def setup_fields(self) -> None:
        """Define the parameter fields for this action.

        This method should populate self.fields with the parameters that
        this action accepts. Each field defines validation rules, defaults,
        and UI generation hints.

        Example:
            def setup_fields(self) -> None:
                self.fields = {
                    'width': IntegerField(
                        default=800,
                        min_value=1,
                        max_value=10000,
                        label="Width",
                        help_text="Target width in pixels"
                    ),
                    'maintain_aspect': BooleanField(
                        default=True,
                        label="Maintain Aspect Ratio"
                    )
                }
        """
        pass

    @abstractmethod
    def apply(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Apply the action to an image.

        Args:
            image: Input image to process
            **params: Action parameters (already validated)

        Returns:
            Processed image

        Raises:
            ActionError: If processing fails
        """
        pass

    def get_relevant_fields(self, params: dict[str, Any]) -> list[str]:  # noqa: ARG002
        """Return list of field names relevant for current parameters.

        This method enables dynamic UIs that show/hide parameters based on
        other parameter values. Override in subclasses to implement custom
        field relevance logic.

        Args:
            params: Current parameter values

        Returns:
            List of field names that should be visible/editable

        Example:
            If 'maintain_aspect' is True, hide the 'height' field:

            def get_relevant_fields(self, params: dict[str, Any]) -> list[str]:
                fields = ['width', 'maintain_aspect']
                if not params.get('maintain_aspect', True):
                    fields.append('height')
                return fields
        """
        return list(self.fields.keys())

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Validate parameters using field definitions.

        Args:
            params: Raw parameter values to validate

        Returns:
            Validated and converted parameter values

        Raises:
            ValidationError: If any parameter fails validation
        """
        validated_params = {}
        relevant_fields = self.get_relevant_fields(params)
        for field_name in relevant_fields:
            if field_name not in self.fields:
                continue
            field = self.fields[field_name]
            raw_value = params.get(field_name)
            try:
                validated_value = field.validate(raw_value)
                validated_params[field_name] = validated_value
            except ValidationError as e:
                raise ValidationError(
                    field_name=f"{self.name}.{field_name}",
                    expected=e.expected,
                    actual=e.actual,
                    message=f"In action '{self.label}': {e.message}",
                    suggestion=e.suggestion,
                ) from e
        return validated_params

    def get_field_schema(self) -> dict[str, dict[str, Any]]:
        """Get schema information for all fields.

        Returns:
            Dictionary mapping field names to their schema information,
            useful for UI generation and documentation.

        Example:
            {
                'width': {
                    'type': 'integer',
                    'label': 'Width',
                    'help_text': 'Target width in pixels',
                    'default': 800,
                    'min_value': 1,
                    'max_value': 10000,
                    'required': True
                }
            }
        """
        schema = {}
        for field_name, field in self.fields.items():
            field_schema = {
                "type": field.__class__.__name__.lower().replace("field", ""),
                "label": field.label or field_name.replace("_", " ").title(),
                "help_text": field.help_text,
                "default": field.default,
                "required": field.required,
            }
            if hasattr(field, "min_value"):
                field_schema["min_value"] = field.min_value
            if hasattr(field, "max_value"):
                field_schema["max_value"] = field.max_value
            if hasattr(field, "min_length"):
                field_schema["min_length"] = field.min_length
            if hasattr(field, "max_length"):
                field_schema["max_length"] = field.max_length
            if hasattr(field, "choices"):
                field_schema["choices"] = field.choices
            schema[field_name] = field_schema
        return schema

    def execute(self, image: ImageAsset, **params: Any) -> ImageAsset:
        """Execute the action with parameter validation.

        This is the main entry point for action execution. It validates
        parameters, applies the action, and handles errors gracefully.

        Args:
            image: Input image to process
            **params: Raw action parameters

        Returns:
            Processed image with updated metadata

        Raises:
            ActionError: If validation or processing fails
        """
        try:
            validated_params = self.validate_params(params)
            result_image = self.apply(image, **validated_params)
            processing_update = {
                "last_action": self.name,
                "last_action_version": self.version,
                "last_action_params": validated_params,
            }
            return result_image.with_processing_update(processing_update)
        except ValidationError as e:
            raise ActionError(
                action_name=self.name,
                message=f"Parameter validation failed: {e.message}",
                suggestion=e.suggestion,
            ) from e
        except Exception as e:
            raise ActionError(
                action_name=self.name,
                message=f"Processing failed: {e!s}",
                suggestion="Check input image and parameters",
            ) from e

    def get_action_info(self) -> dict[str, Any]:
        """Get complete action information including metadata and schema.

        Returns:
            Dictionary with action metadata, field schema, and other info
            useful for documentation and UI generation.
        """
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "tags": self.tags,
            "fields": self.get_field_schema(),
        }

    def __str__(self) -> str:
        """String representation of the action."""
        return f"{self.label} ({self.name})"

    def __repr__(self) -> str:
        """Developer representation of the action."""
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"


class ActionRegistry:
    """Registry for managing available actions.

    Provides action discovery, registration, and instantiation capabilities.
    Actions can be registered explicitly or discovered automatically.
    """

    def __init__(self) -> None:
        """Initialize empty action registry."""
        self._actions: dict[str, type[BaseAction]] = {}

    def register(self, action_class: type[BaseAction]) -> None:
        """Register an action class.

        Args:
            action_class: Action class to register

        Raises:
            ValueError: If action name conflicts or is invalid
        """
        temp_instance = action_class()
        name = temp_instance.name
        if not name:
            raise ValueError(f"Action {action_class.__name__} has no name")
        if name in self._actions:
            existing_class = self._actions[name]
            if existing_class != action_class:
                raise ValueError(f"Action name '{name}' already registered by {existing_class.__name__}")
        self._actions[name] = action_class

    def unregister(self, name: str) -> None:
        """Unregister an action by name.

        Args:
            name: Action name to unregister
        """
        self._actions.pop(name, None)

    def get_action(self, name: str) -> BaseAction:
        """Get action instance by name.

        Args:
            name: Action name

        Returns:
            Action instance

        Raises:
            KeyError: If action not found
        """
        if name not in self._actions:
            available = ", ".join(sorted(self._actions.keys()))
            raise KeyError(f"Action '{name}' not found. Available actions: {available}")
        action_class = self._actions[name]
        return action_class()

    def list_actions(self) -> list[str]:
        """List all registered action names.

        Returns:
            Sorted list of action names
        """
        return sorted(self._actions.keys())

    def get_actions_by_tag(self, tag: str) -> list[str]:
        """Get actions that have a specific tag.

        Args:
            tag: Tag to search for

        Returns:
            List of action names with the specified tag
        """
        matching_actions = []
        for name, action_class in self._actions.items():
            temp_instance = action_class()
            if tag in temp_instance.tags:
                matching_actions.append(name)
        return sorted(matching_actions)

    def get_action_info(self, name: str) -> dict[str, Any]:
        """Get complete information about an action.

        Args:
            name: Action name

        Returns:
            Action information dictionary

        Raises:
            KeyError: If action not found
        """
        action = self.get_action(name)
        return action.get_action_info()

    def clear(self) -> None:
        """Clear all registered actions."""
        self._actions.clear()

    def __len__(self) -> int:
        """Get number of registered actions."""
        return len(self._actions)

    def __contains__(self, name: str) -> bool:
        """Check if action is registered."""
        return name in self._actions


default_registry = ActionRegistry()


def _register_metadata_actions() -> None:
    """Register metadata actions with the default registry."""
    if METADATA_ACTIONS_AVAILABLE:
        # Import here to avoid circular imports
        try:
            from ..actions.metadata import (
                CopyMetadataAction,
                ExtractMetadataAction,
                RemoveMetadataAction,
                WriteMetadataAction,
            )

            default_registry.register(WriteMetadataAction)
            default_registry.register(RemoveMetadataAction)
            default_registry.register(CopyMetadataAction)
            default_registry.register(ExtractMetadataAction)
        except ImportError:
            pass


_register_metadata_actions()


def register_action(action_class: type[BaseAction]) -> None:
    """Register an action with the default registry.

    Args:
        action_class: Action class to register
    """
    default_registry.register(action_class)


def get_action(name: str) -> BaseAction:
    """Get action instance from default registry.

    Args:
        name: Action name

    Returns:
        Action instance
    """
    return default_registry.get_action(name)


def list_actions() -> list[str]:
    """List all actions in default registry.

    Returns:
        List of action names
    """
    return default_registry.list_actions()
