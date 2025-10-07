"""Command-line interface for PhotoFlow.

Expose the CLI entrypoint as a callable proxy so that tooling can patch
attributes on the underlying `photoflow.cli.main` module while still
invoking it like a function.
"""

from importlib import import_module
from typing import Any

_main_module = import_module("photoflow.cli.main")


class _MainProxy:
    """Callable proxy that forwards attribute access to the CLI module."""

    def __init__(self) -> None:
        object.__setattr__(self, "_module", _main_module)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._module.main(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self._module, item)

    def __setattr__(self, item: str, value: Any) -> None:
        setattr(self._module, item, value)

    def __delattr__(self, item: str) -> None:
        delattr(self._module, item)

    def __dir__(self) -> list[str]:
        return sorted(set(dir(self._module)))


main = _MainProxy()

__all__ = ["main"]
