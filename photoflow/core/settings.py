"""Settings management with dependency injection support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, Field


class SettingsLoader(Protocol):
    """Protocol for loading application settings."""

    def load(self) -> dict[str, Any]:
        """Load settings from configured sources."""
        ...

    def get(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value."""
        ...


class PhotoFlowSettings(BaseModel):
    """Application settings with validation."""

    # Core settings
    max_workers: int = Field(default=4, ge=1, le=32)
    chunk_size: int = Field(default=100, ge=1)
    temp_dir: Path = Field(default_factory=lambda: Path.cwd() / "temp")
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".photoflow" / "cache")
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")

    # Image processing
    default_quality: int = Field(default=95, ge=1, le=100)
    preserve_metadata: bool = Field(default=True)
    auto_orient: bool = Field(default=True)

    # Performance
    enable_parallel: bool = Field(default=True)
    gpu_acceleration: bool = Field(default=False)
    memory_limit_mb: int = Field(default=1024, ge=128)

    # GUI settings
    theme: str = Field(default="light", pattern=r"^(light|dark|auto)$")
    preview_size: int = Field(default=512, ge=128, le=2048)
    auto_preview: bool = Field(default=True)

    # Plugin settings
    plugin_dirs: list[Path] = Field(default_factory=list)
    enable_unsafe_plugins: bool = Field(default=False)


class FileSettingsLoader:
    """Load settings from YAML/JSON configuration files."""

    def __init__(
        self,
        config_paths: list[Path] | None = None,
        env_prefix: str = "PHOTOFLOW_",
    ) -> None:
        """Initialize settings loader.

        Args:
            config_paths: Paths to check for config files
            env_prefix: Environment variable prefix
        """
        self.env_prefix = env_prefix
        self.config_paths = config_paths or self._default_config_paths()
        self._cached_settings: dict[str, Any] | None = None

    def _default_config_paths(self) -> list[Path]:
        """Get default configuration file paths."""
        return [
            Path.cwd() / "photoflow.yaml",
            Path.cwd() / "photoflow.yml",
            Path.cwd() / ".photoflow.yaml",
            Path.home() / ".photoflow" / "config.yaml",
            Path.home() / ".config" / "photoflow" / "config.yaml",
        ]

    def load(self) -> dict[str, Any]:
        """Load settings from all configured sources."""
        if self._cached_settings is not None:
            return self._cached_settings

        settings: dict[str, Any] = {}

        # Load from files (in order of precedence)
        for config_path in reversed(self.config_paths):
            if config_path.exists():
                file_settings = self._load_file(config_path)
                settings.update(file_settings)

        # Override with environment variables
        env_settings = self._load_env()
        settings.update(env_settings)

        self._cached_settings = settings
        return settings

    def _load_file(self, path: Path) -> dict[str, Any]:
        """Load settings from a YAML file."""
        try:
            with path.open("r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
                return content if isinstance(content, dict) else {}
        except (yaml.YAMLError, OSError) as e:
            # Log warning in production code
            print(f"Warning: Could not load config from {path}: {e}")
            return {}

    def _load_env(self) -> dict[str, Any]:
        """Load settings from environment variables."""
        settings = {}
        prefix_len = len(self.env_prefix)

        for key, value in os.environ.items():
            if key.startswith(self.env_prefix):
                setting_key = key[prefix_len:].lower()
                # Convert string values to appropriate types
                settings[setting_key] = self._convert_env_value(value)

        return settings

    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type."""
        # Handle boolean values
        if value.lower() in {"true", "yes", "1", "on"}:
            return True
        if value.lower() in {"false", "no", "0", "off"}:
            return False

        # Handle numeric values
        if value.isdigit():
            return int(value)

        try:
            return float(value)
        except ValueError:
            pass

        # Handle list values (comma-separated)
        if "," in value:
            return [item.strip() for item in value.split(",")]

        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a specific setting value."""
        settings = self.load()
        return settings.get(key, default)

    def invalidate_cache(self) -> None:
        """Clear cached settings to force reload."""
        self._cached_settings = None


class SettingsContainer:
    """Dependency injection container for settings."""

    def __init__(self, loader: SettingsLoader | None = None) -> None:
        """Initialize container with optional custom loader."""
        self._loader = loader or FileSettingsLoader()
        self._settings: PhotoFlowSettings | None = None

    @property
    def settings(self) -> PhotoFlowSettings:
        """Get validated settings instance."""
        if self._settings is None:
            raw_settings = self._loader.load()
            self._settings = PhotoFlowSettings(**raw_settings)
        return self._settings

    def reload(self) -> None:
        """Reload settings from sources."""
        if hasattr(self._loader, "invalidate_cache"):
            self._loader.invalidate_cache()
        self._settings = None

    def override_loader(self, loader: SettingsLoader) -> None:
        """Override the settings loader (for testing)."""
        self._loader = loader
        self._settings = None


# Global settings container instance
_settings_container = SettingsContainer()


def get_settings() -> PhotoFlowSettings:
    """Get the current settings instance."""
    return _settings_container.settings


def reload_settings() -> None:
    """Reload settings from all sources."""
    _settings_container.reload()


def override_settings_loader(loader: SettingsLoader) -> None:
    """Override the global settings loader (for testing)."""
    _settings_container.override_loader(loader)
