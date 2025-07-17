"""Test settings management."""

import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from photoflow.core.settings import (
    FileSettingsLoader,
    PhotoFlowSettings,
    SettingsContainer,
    get_settings,
    override_settings_loader,
    reload_settings,
)


class TestPhotoFlowSettings:
    """Test settings model validation."""

    def test_default_settings(self) -> None:
        """Test default settings are valid."""
        settings = PhotoFlowSettings()
        assert settings.max_workers == 4
        assert settings.log_level == "INFO"
        assert settings.default_quality == 95
        assert settings.theme == "light"

    def test_validation_constraints(self) -> None:
        """Test settings validation constraints."""
        with pytest.raises(ValueError):
            PhotoFlowSettings(max_workers=0)
        with pytest.raises(ValueError):
            PhotoFlowSettings(log_level="INVALID")
        with pytest.raises(ValueError):
            PhotoFlowSettings(default_quality=101)

    def test_env_prefix(self) -> None:
        """Test environment variable prefix handling via loader."""
        loader = FileSettingsLoader(config_paths=[])
        with patch.dict(os.environ, {"PHOTOFLOW_MAX_WORKERS": "8"}):
            raw_settings = loader.load()
            settings = PhotoFlowSettings(**raw_settings)
            assert settings.max_workers == 8


class TestFileSettingsLoader:
    """Test file-based settings loader."""

    def test_load_from_yaml_file(self) -> None:
        """Test loading settings from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"max_workers": 8, "log_level": "DEBUG"}, f)
            f.flush()
            loader = FileSettingsLoader(config_paths=[Path(f.name)])
            settings = loader.load()
            assert settings["max_workers"] == 8
            assert settings["log_level"] == "DEBUG"
        Path(f.name).unlink()

    def test_load_with_missing_file(self) -> None:
        """Test loading when config file doesn't exist."""
        loader = FileSettingsLoader(config_paths=[Path("/nonexistent/config.yaml")])
        settings = loader.load()
        assert isinstance(settings, dict)

    def test_env_variable_override(self) -> None:
        """Test environment variables override file settings."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"max_workers": 4}, f)
            f.flush()
            with patch.dict(os.environ, {"PHOTOFLOW_MAX_WORKERS": "8"}):
                loader = FileSettingsLoader(config_paths=[Path(f.name)])
                settings = loader.load()
                assert settings["max_workers"] == 8
        Path(f.name).unlink()

    def test_env_value_conversion(self) -> None:
        """Test environment variable type conversion."""
        with patch.dict(
            os.environ,
            {
                "PHOTOFLOW_ENABLE_PARALLEL": "true",
                "PHOTOFLOW_MAX_WORKERS": "16",
                "PHOTOFLOW_MEMORY_LIMIT_MB": "2048",
                "PHOTOFLOW_PLUGIN_DIRS": "/path1,/path2",
            },
        ):
            loader = FileSettingsLoader()
            settings = loader.load()
            assert settings["enable_parallel"] is True
            assert settings["max_workers"] == 16
            assert settings["memory_limit_mb"] == 2048
            assert settings["plugin_dirs"] == ["/path1", "/path2"]

    def test_get_specific_setting(self) -> None:
        """Test getting specific setting with default."""
        loader = FileSettingsLoader(config_paths=[])
        with patch.dict(os.environ, {"PHOTOFLOW_MAX_WORKERS": "8"}):
            assert loader.get("max_workers") == 8
            assert loader.get("nonexistent", "default") == "default"

    def test_cache_invalidation(self) -> None:
        """Test settings cache invalidation."""
        loader = FileSettingsLoader(config_paths=[])
        with patch.dict(os.environ, {"PHOTOFLOW_MAX_WORKERS": "4"}):
            settings1 = loader.load()
        with patch.dict(os.environ, {"PHOTOFLOW_MAX_WORKERS": "8"}):
            settings2 = loader.load()
            assert settings1["max_workers"] == settings2["max_workers"]
            loader.invalidate_cache()
            settings3 = loader.load()
            assert settings3["max_workers"] == 8


class TestSettingsContainer:
    """Test settings dependency injection container."""

    def test_singleton_behavior(self) -> None:
        """Test settings container provides consistent instances."""
        container = SettingsContainer()
        settings1 = container.settings
        settings2 = container.settings
        assert settings1 is settings2

    def test_loader_override(self) -> None:
        """Test overriding settings loader."""

        class MockLoader:
            def load(self) -> dict[str, Any]:
                return {"max_workers": 16}

            def get(self, key: str, default: Any = None) -> Any:
                return self.load().get(key, default)

        container = SettingsContainer()
        container.override_loader(MockLoader())
        settings = container.settings
        assert settings.max_workers == 16

    def test_reload_settings(self) -> None:
        """Test reloading settings."""
        container = SettingsContainer()
        settings1 = container.settings
        container.reload()
        settings2 = container.settings
        assert settings1 is not settings2


class TestGlobalSettingsFunctions:
    """Test global settings access functions."""

    def test_get_settings(self) -> None:
        """Test global get_settings function."""
        settings = get_settings()
        assert isinstance(settings, PhotoFlowSettings)

    def test_reload_settings(self) -> None:
        """Test global reload_settings function."""
        settings1 = get_settings()
        reload_settings()
        settings2 = get_settings()
        assert settings1 is not settings2

    def test_override_settings_loader(self) -> None:
        """Test global settings loader override."""

        class MockLoader:
            def load(self) -> dict[str, Any]:
                return {"max_workers": 24}

            def get(self, key: str, default: Any = None) -> Any:
                return self.load().get(key, default)

        override_settings_loader(MockLoader())
        settings = get_settings()
        assert settings.max_workers == 24


class TestSettingsMissingLineCoverage:
    """Test specific missing lines for coverage."""

    def test_load_file_yaml_error_lines_113_116(self) -> None:
        """Test YAML error handling in _load_file (lines 113-116)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [unclosed")
            f.flush()
            file_path = Path(f.name)
        try:
            loader = FileSettingsLoader(config_paths=[file_path])
            with patch("builtins.print") as mock_print:
                result = loader._load_file(file_path)
                assert result == {}
                mock_print.assert_called_once()
                assert "Warning: Could not load config" in mock_print.call_args[0][0]
        finally:
            file_path.unlink(missing_ok=True)

    def test_load_file_os_error_lines_113_116(self) -> None:
        """Test OS error handling in _load_file (lines 113-116)."""
        loader = FileSettingsLoader()
        with (
            patch("builtins.print") as mock_print,
            patch("pathlib.Path.open", side_effect=OSError("Permission denied")),
        ):
            result = loader._load_file(Path("/fake/path.yaml"))
            assert result == {}
            mock_print.assert_called_once()
            assert "Warning: Could not load config" in mock_print.call_args[0][0]

    def test_convert_env_value_false_line_137(self) -> None:
        """Test environment value conversion for false values (line 137)."""
        loader = FileSettingsLoader()
        false_values = ["false", "no", "0", "off", "FALSE", "NO", "OFF"]
        for value in false_values:
            result = loader._convert_env_value(value)
            assert result is False, f"Expected False for '{value}', got {result}"

    def test_convert_env_value_no_comma_line_152(self) -> None:
        """Test environment value conversion when no comma present (line 152)."""
        loader = FileSettingsLoader()
        test_values = ["single_value", "some_string", "/path/to/file"]
        for value in test_values:
            result = loader._convert_env_value(value)
            assert result == value, f"Expected '{value}', got {result}"
