"""Test serialization utilities."""

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from photoflow.core.models import ActionSpec, Pipeline
from photoflow.core.serialization import (
    PipelineSerializer,
    SettingsSerializer,
    load_pipeline,
    load_settings,
    save_pipeline,
    save_settings,
)


class TestPipelineSerializer:
    """Test Pipeline JSON serialization."""

    def test_save_and_load_pipeline(self) -> None:
        """Test saving and loading pipeline to/from file."""
        original = Pipeline(
            name="test_pipeline",
            actions=(
                ActionSpec(name="resize", parameters={"width": 800}),
                ActionSpec(name="save", parameters={"format": "JPEG"}),
            ),
            metadata={"version": "1.0"},
        )

        with tempfile.NamedTemporaryFile(suffix=".actions", delete=False) as f:
            file_path = Path(f.name)

        try:
            # Save pipeline
            PipelineSerializer.save_pipeline(original, file_path)
            assert file_path.exists()

            # Load pipeline
            loaded = PipelineSerializer.load_pipeline(file_path)
            assert loaded.name == original.name
            assert len(loaded.actions) == len(original.actions)
            assert loaded.actions[0].name == original.actions[0].name
            assert loaded.metadata == original.metadata

            # Verify file content is valid JSON
            with file_path.open("r") as f:
                data = json.load(f)
            assert data["name"] == "test_pipeline"

        finally:
            file_path.unlink(missing_ok=True)

    def test_file_extension_handling(self) -> None:
        """Test automatic file extension handling."""
        pipeline = Pipeline(name="test", actions=())

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Should add .actions extension
            path_without_ext = temp_path / "pipeline"
            PipelineSerializer.save_pipeline(pipeline, path_without_ext)
            assert (temp_path / "pipeline.actions").exists()

            # Should preserve .json extension
            path_with_json = temp_path / "pipeline.json"
            PipelineSerializer.save_pipeline(pipeline, path_with_json)
            assert path_with_json.exists()

            # Should preserve .actions extension
            path_with_actions = temp_path / "pipeline.actions"
            PipelineSerializer.save_pipeline(pipeline, path_with_actions)
            assert path_with_actions.exists()

    def test_load_nonexistent_file(self) -> None:
        """Test loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PipelineSerializer.load_pipeline("/nonexistent/pipeline.actions")

    def test_load_invalid_json(self) -> None:
        """Test loading invalid JSON raises ValueError."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".actions", delete=False
        ) as f:
            f.write("invalid json content")
            f.flush()
            file_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid pipeline file format"):
                PipelineSerializer.load_pipeline(file_path)
        finally:
            file_path.unlink(missing_ok=True)

    def test_validate_pipeline_file(self) -> None:
        """Test pipeline file validation."""
        # Valid pipeline
        pipeline = Pipeline(name="test", actions=())
        with tempfile.NamedTemporaryFile(suffix=".actions", delete=False) as f:
            file_path = Path(f.name)

        try:
            PipelineSerializer.save_pipeline(pipeline, file_path)
            assert PipelineSerializer.validate_pipeline_file(file_path) is True
        finally:
            file_path.unlink(missing_ok=True)

        # Non-existent file
        assert PipelineSerializer.validate_pipeline_file("/nonexistent/file") is False

        # Invalid file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".actions", delete=False
        ) as f:
            f.write("invalid content")
            f.flush()
            file_path = Path(f.name)

        try:
            assert PipelineSerializer.validate_pipeline_file(file_path) is False
        finally:
            file_path.unlink(missing_ok=True)

    def test_json_string_serialization(self) -> None:
        """Test pipeline to/from JSON string."""
        original = Pipeline(
            name="string_test",
            actions=(ActionSpec(name="resize", parameters={"width": 1920}),),
            metadata={"test": True},
        )

        # Convert to JSON string
        json_str = PipelineSerializer.pipeline_to_json(original)
        assert isinstance(json_str, str)
        assert "string_test" in json_str

        # Convert back from JSON string
        restored = PipelineSerializer.pipeline_from_json(json_str)
        assert restored.name == original.name
        assert len(restored.actions) == len(original.actions)
        assert restored.metadata == original.metadata

    def test_json_string_invalid(self) -> None:
        """Test invalid JSON string handling."""
        with pytest.raises(ValueError, match="Invalid pipeline JSON"):
            PipelineSerializer.pipeline_from_json("invalid json")

        with pytest.raises(ValueError, match="Invalid pipeline JSON"):
            PipelineSerializer.pipeline_from_json('{"invalid": "pipeline"}')

    def test_complex_pipeline_serialization(self) -> None:
        """Test serialization of complex pipeline with various data types."""
        complex_pipeline = Pipeline(
            name="complex_test",
            actions=(
                ActionSpec(
                    name="resize",
                    parameters={
                        "width": 1920,
                        "height": 1080,
                        "maintain_aspect": True,
                        "filters": ["lanczos", "bicubic"],
                        "metadata": {"quality": "high"},
                    },
                    description="Complex resize operation",
                    enabled=False,
                ),
                ActionSpec(
                    name="watermark",
                    parameters={
                        "text": "© 2025 Test Company",
                        "position": {"x": 10, "y": 10},
                        "opacity": 0.75,
                    },
                ),
            ),
            metadata={
                "version": "2.0",
                "created": "2025-01-01",
                "tags": ["test", "complex"],
                "settings": {"parallel": True, "threads": 4},
            },
        )

        # Round trip through JSON string
        json_str = PipelineSerializer.pipeline_to_json(complex_pipeline)
        restored = PipelineSerializer.pipeline_from_json(json_str)

        assert restored == complex_pipeline


class TestSettingsSerializer:
    """Test Settings YAML serialization."""

    def test_save_and_load_settings(self) -> None:
        """Test saving and loading settings to/from file."""
        original_settings = {
            "max_workers": 8,
            "gpu_acceleration": True,
            "theme": "dark",
            "plugin_dirs": ["/path1", "/path2"],
            "nested": {"setting": "value", "number": 42},
        }

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            file_path = Path(f.name)

        try:
            # Save settings
            SettingsSerializer.save_settings(original_settings, file_path)
            assert file_path.exists()

            # Load settings
            loaded = SettingsSerializer.load_settings(file_path)
            assert loaded == original_settings

            # Verify file content is valid YAML
            with file_path.open("r") as f:
                data = yaml.safe_load(f)
            assert data == original_settings

        finally:
            file_path.unlink(missing_ok=True)

    def test_file_extension_handling(self) -> None:
        """Test automatic file extension handling."""
        settings = {"test": "value"}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Should add .yaml extension
            path_without_ext = temp_path / "settings"
            SettingsSerializer.save_settings(settings, path_without_ext)
            assert (temp_path / "settings.yaml").exists()

            # Should preserve .yml extension
            path_with_yml = temp_path / "settings.yml"
            SettingsSerializer.save_settings(settings, path_with_yml)
            assert path_with_yml.exists()

            # Should preserve .yaml extension
            path_with_yaml = temp_path / "settings.yaml"
            SettingsSerializer.save_settings(settings, path_with_yaml)
            assert path_with_yaml.exists()

    def test_load_nonexistent_file(self) -> None:
        """Test loading non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            SettingsSerializer.load_settings("/nonexistent/settings.yaml")

    def test_load_invalid_yaml(self) -> None:
        """Test loading invalid YAML raises ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            f.flush()
            file_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid YAML file format"):
                SettingsSerializer.load_settings(file_path)
        finally:
            file_path.unlink(missing_ok=True)

    def test_load_empty_file(self) -> None:
        """Test loading empty YAML file returns empty dict."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("")
            f.flush()
            file_path = Path(f.name)

        try:
            settings = SettingsSerializer.load_settings(file_path)
            assert settings == {}
        finally:
            file_path.unlink(missing_ok=True)

    def test_validate_settings_file(self) -> None:
        """Test settings file validation."""
        # Valid settings
        settings = {"test": "value"}
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            file_path = Path(f.name)

        try:
            SettingsSerializer.save_settings(settings, file_path)
            assert SettingsSerializer.validate_settings_file(file_path) is True
        finally:
            file_path.unlink(missing_ok=True)

        # Non-existent file
        assert SettingsSerializer.validate_settings_file("/nonexistent/file") is False

        # Invalid file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: [")
            f.flush()
            file_path = Path(f.name)

        try:
            assert SettingsSerializer.validate_settings_file(file_path) is False
        finally:
            file_path.unlink(missing_ok=True)

    def test_yaml_string_serialization(self) -> None:
        """Test settings to/from YAML string."""
        original_settings = {
            "max_workers": 4,
            "theme": "light",
            "features": {"ai": True, "gpu": False},
        }

        # Convert to YAML string
        yaml_str = SettingsSerializer.settings_to_yaml(original_settings)
        assert isinstance(yaml_str, str)
        assert "max_workers: 4" in yaml_str

        # Convert back from YAML string
        restored = SettingsSerializer.settings_from_yaml(yaml_str)
        assert restored == original_settings

    def test_yaml_string_invalid(self) -> None:
        """Test invalid YAML string handling."""
        with pytest.raises(ValueError, match="Invalid YAML"):
            SettingsSerializer.settings_from_yaml("invalid: yaml: [")

    def test_yaml_string_empty(self) -> None:
        """Test empty YAML string returns empty dict."""
        result = SettingsSerializer.settings_from_yaml("")
        assert result == {}

        result = SettingsSerializer.settings_from_yaml("null")
        assert result == {}


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_convenience_functions(self) -> None:
        """Test that convenience functions work correctly."""
        # Test pipeline functions
        pipeline = Pipeline(name="test", actions=())

        with tempfile.NamedTemporaryFile(suffix=".actions", delete=False) as f:
            file_path = Path(f.name)

        try:
            save_pipeline(pipeline, file_path)
            loaded_pipeline = load_pipeline(file_path)
            assert loaded_pipeline.name == pipeline.name
        finally:
            file_path.unlink(missing_ok=True)

        # Test settings functions
        settings = {"test": "value"}

        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            file_path = Path(f.name)

        try:
            save_settings(settings, file_path)
            loaded_settings = load_settings(file_path)
            assert loaded_settings == settings
        finally:
            file_path.unlink(missing_ok=True)
