"""Serialization utilities for PhotoFlow domain models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .models import Pipeline


class PipelineSerializer:
    """Handles serialization and deserialization of Pipeline objects to/from JSON.

    Action lists are stored in JSON format for easy parsing and tooling support.
    Supports both .actions and .json file extensions.
    """

    @staticmethod
    def save_pipeline(pipeline: Pipeline, path: str | Path) -> None:
        """Save pipeline to JSON file.

        Args:
            pipeline: Pipeline to save
            path: Output file path (.actions or .json)

        Raises:
            OSError: If file cannot be written
            ValueError: If pipeline data is invalid
        """
        path = Path(path)

        # Ensure proper file extension
        if path.suffix not in {".actions", ".json"}:
            path = path.with_suffix(".actions")

        try:
            pipeline_data = pipeline.model_dump()

            with path.open("w", encoding="utf-8") as f:
                json.dump(
                    pipeline_data,
                    f,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True,
                )
        except (OSError, TypeError, ValueError) as e:
            raise OSError(f"Failed to save pipeline to {path}: {e}") from e

    @staticmethod
    def load_pipeline(path: str | Path) -> Pipeline:
        """Load pipeline from JSON file.

        Args:
            path: Path to pipeline file (.actions or .json)

        Returns:
            Pipeline instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Pipeline file not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as f:
                pipeline_data = json.load(f)

            return Pipeline.model_validate(pipeline_data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid pipeline file format: {path}: {e}") from e

    @staticmethod
    def validate_pipeline_file(path: str | Path) -> bool:
        """Validate that a file contains a valid pipeline.

        Args:
            path: Path to pipeline file

        Returns:
            True if file contains valid pipeline, False otherwise
        """
        try:
            PipelineSerializer.load_pipeline(path)
            return True
        except (FileNotFoundError, ValueError):
            return False

    @staticmethod
    def pipeline_to_json(pipeline: Pipeline) -> str:
        """Convert pipeline to JSON string.

        Args:
            pipeline: Pipeline to serialize

        Returns:
            JSON string representation
        """
        pipeline_data = pipeline.model_dump()
        return json.dumps(
            pipeline_data,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )

    @staticmethod
    def pipeline_from_json(json_str: str) -> Pipeline:
        """Create pipeline from JSON string.

        Args:
            json_str: JSON string with pipeline data

        Returns:
            Pipeline instance

        Raises:
            ValueError: If JSON is invalid or doesn't contain valid pipeline
        """
        try:
            pipeline_data = json.loads(json_str)
            return Pipeline.model_validate(pipeline_data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid pipeline JSON: {e}") from e


class SettingsSerializer:
    """Handles serialization and deserialization of settings to/from YAML.

    Settings are stored in YAML format for human readability and comments support.
    """

    @staticmethod
    def save_settings(settings: dict[str, Any], path: str | Path) -> None:
        """Save settings to YAML file.

        Args:
            settings: Settings dictionary to save
            path: Output file path (.yaml or .yml)

        Raises:
            OSError: If file cannot be written
        """
        path = Path(path)

        # Ensure proper file extension
        if path.suffix not in {".yaml", ".yml"}:
            path = path.with_suffix(".yaml")

        try:
            with path.open("w", encoding="utf-8") as f:
                yaml.dump(
                    settings,
                    f,
                    default_flow_style=False,
                    sort_keys=True,
                    allow_unicode=True,
                    indent=2,
                )
        except (OSError, yaml.YAMLError) as e:
            raise OSError(f"Failed to save settings to {path}: {e}") from e

    @staticmethod
    def load_settings(path: str | Path) -> dict[str, Any]:
        """Load settings from YAML file.

        Args:
            path: Path to settings file (.yaml or .yml)

        Returns:
            Settings dictionary

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Settings file not found: {path}")

        try:
            with path.open("r", encoding="utf-8") as f:
                settings = yaml.safe_load(f)

            # Ensure we return a dict even if file is empty
            return settings if isinstance(settings, dict) else {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML file format: {path}: {e}") from e

    @staticmethod
    def validate_settings_file(path: str | Path) -> bool:
        """Validate that a file contains valid YAML settings.

        Args:
            path: Path to settings file

        Returns:
            True if file contains valid YAML, False otherwise
        """
        try:
            SettingsSerializer.load_settings(path)
            return True
        except (FileNotFoundError, ValueError):
            return False

    @staticmethod
    def settings_to_yaml(settings: dict[str, Any]) -> str:
        """Convert settings to YAML string.

        Args:
            settings: Settings dictionary to serialize

        Returns:
            YAML string representation
        """
        return yaml.dump(
            settings,
            default_flow_style=False,
            sort_keys=True,
            allow_unicode=True,
            indent=2,
        )

    @staticmethod
    def settings_from_yaml(yaml_str: str) -> dict[str, Any]:
        """Create settings from YAML string.

        Args:
            yaml_str: YAML string with settings data

        Returns:
            Settings dictionary

        Raises:
            ValueError: If YAML is invalid
        """
        try:
            settings = yaml.safe_load(yaml_str)
            return settings if isinstance(settings, dict) else {}
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}") from e


# Convenience functions for common operations
def save_pipeline(pipeline: Pipeline, path: str | Path) -> None:
    """Save pipeline to JSON file."""
    PipelineSerializer.save_pipeline(pipeline, path)


def load_pipeline(path: str | Path) -> Pipeline:
    """Load pipeline from JSON file."""
    return PipelineSerializer.load_pipeline(path)


def save_settings(settings: dict[str, Any], path: str | Path) -> None:
    """Save settings to YAML file."""
    SettingsSerializer.save_settings(settings, path)


def load_settings(path: str | Path) -> dict[str, Any]:
    """Load settings from YAML file."""
    return SettingsSerializer.load_settings(path)
