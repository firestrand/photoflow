"""Tests for the PhotoFlow GUI scaffolding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from photoflow.gui.app import GUIImportError, PhotoFlowGUI, launch


class DummyDPG:
    """Minimal Dear PyGui stand-in used for unit testing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self._id_counter = 0
        self.values: dict[str, Any] = {}

    # Generic helper -------------------------------------------------
    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    def _allocate_tag(self, prefix: str, kwargs: dict[str, Any]) -> str:
        tag = kwargs.get("tag")
        if not tag:
            tag = f"{prefix}_{self._id_counter}"
            self._id_counter += 1
        return tag

    # Dear PyGui API surface ----------------------------------------
    def create_context(self) -> None:
        self._record("create_context")

    def destroy_context(self) -> None:
        self._record("destroy_context")

    def create_viewport(self, **kwargs: Any) -> None:
        self._record("create_viewport", **kwargs)

    def setup_dearpygui(self) -> None:
        self._record("setup_dearpygui")

    def show_viewport(self) -> None:
        self._record("show_viewport")

    def start_dearpygui(self) -> None:
        self._record("start_dearpygui")

    def add_window(self, **kwargs: Any) -> str:
        tag = self._allocate_tag("window", kwargs)
        payload = dict(kwargs)
        payload.setdefault("tag", tag)
        self._record("add_window", **payload)
        return tag

    def add_child_window(self, **kwargs: Any) -> str:
        tag = self._allocate_tag("child", kwargs)
        payload = dict(kwargs)
        payload.setdefault("tag", tag)
        self._record("add_child_window", **payload)
        return tag

    def add_text(self, *args: Any, **kwargs: Any) -> str:
        tag = self._allocate_tag("text", kwargs)
        payload = dict(kwargs)
        payload.setdefault("tag", tag)
        text_value = args[0] if args else payload.get("default_value", "")
        self.values[tag] = text_value
        self._record("add_text", *args, **payload)
        return tag

    def add_radio_button(self, **kwargs: Any) -> str:
        tag = self._allocate_tag("radio", kwargs)
        payload = dict(kwargs)
        payload.setdefault("tag", tag)
        default_value = payload.get("default_value")
        if default_value is None and payload.get("items"):
            default_value = payload["items"][0]
        self.values[tag] = default_value
        self._record("add_radio_button", **payload)
        return tag

    def add_separator(self, **kwargs: Any) -> None:
        self._record("add_separator", **kwargs)

    def add_button(self, **kwargs: Any) -> str:
        tag = self._allocate_tag("button", kwargs)
        payload = dict(kwargs)
        payload.setdefault("tag", tag)
        self._record("add_button", **payload)
        return tag

    def add_file_dialog(self, **kwargs: Any) -> str:
        tag = self._allocate_tag("dialog", kwargs)
        payload = dict(kwargs)
        payload.setdefault("tag", tag)
        self._record("add_file_dialog", **payload)
        return tag

    def add_group(self, **kwargs: Any) -> str:
        tag = self._allocate_tag("group", kwargs)
        payload = dict(kwargs)
        payload.setdefault("tag", tag)
        self._record("add_group", **payload)
        return tag

    def add_same_line(self, **kwargs: Any) -> None:
        self._record("add_same_line", **kwargs)

    def add_file_extension(self, *args: Any, **kwargs: Any) -> None:
        self._record("add_file_extension", *args, **kwargs)

    def add_collapsing_header(self, **kwargs: Any) -> str:
        tag = self._allocate_tag("header", kwargs)
        payload = dict(kwargs)
        payload.setdefault("tag", tag)
        self._record("add_collapsing_header", **payload)
        return tag

    def add_input_int(self, **kwargs: Any) -> None:
        tag = kwargs.get("tag") or self._allocate_tag("int", kwargs)
        payload = dict(kwargs)
        payload.setdefault("tag", tag)
        self.values[tag] = payload.get("default_value", 0)
        self._record("add_input_int", **payload)
        return tag

    def add_input_float(self, **kwargs: Any) -> None:
        tag = kwargs.get("tag") or self._allocate_tag("float", kwargs)
        payload = dict(kwargs)
        payload.setdefault("tag", tag)
        self.values[tag] = payload.get("default_value", 0.0)
        self._record("add_input_float", **payload)
        return tag

    def add_checkbox(self, **kwargs: Any) -> None:
        tag = kwargs.get("tag") or self._allocate_tag("checkbox", kwargs)
        payload = dict(kwargs)
        payload.setdefault("tag", tag)
        self.values[tag] = bool(payload.get("default_value", False))
        self._record("add_checkbox", **payload)
        return tag

    def add_combo(self, **kwargs: Any) -> None:
        tag = kwargs.get("tag") or self._allocate_tag("combo", kwargs)
        payload = dict(kwargs)
        payload.setdefault("tag", tag)
        items = payload.get("items", [])
        default = payload.get("default_value", items[0] if items else "")
        self.values[tag] = default
        self._record("add_combo", **payload)
        return tag

    def add_input_text(self, **kwargs: Any) -> None:
        tag = kwargs.get("tag") or self._allocate_tag("text_input", kwargs)
        payload = dict(kwargs)
        payload.setdefault("tag", tag)
        self.values[tag] = payload.get("default_value", "")
        self._record("add_input_text", **payload)
        return tag

    def add_listbox(self, **kwargs: Any) -> str:
        tag = kwargs.get("tag") or self._allocate_tag("listbox", kwargs)
        payload = dict(kwargs)
        payload.setdefault("tag", tag)
        items = payload.get("items", [])
        self.values[tag] = items[0] if items else ""
        self._record("add_listbox", **payload)
        return tag

    def show_item(self, *args: Any, **kwargs: Any) -> None:
        self._record("show_item", *args, **kwargs)

    def set_value(self, *args: Any, **kwargs: Any) -> None:
        if args:
            value = args[1] if len(args) > 1 else kwargs.get("value")
            self.values[args[0]] = value
        self._record("set_value", *args, **kwargs)

    def configure_item(self, *args: Any, **kwargs: Any) -> None:
        self._record("configure_item", *args, **kwargs)
        if args and "items" in kwargs:
            items = kwargs["items"]
            self.values[args[0]] = items[0] if items else ""

    def get_value(self, tag: str) -> Any:
        return self.values.get(tag)


@dataclass
class DummyAction:
    """Simplified action used for rendering parameter controls."""

    name: str
    label: str
    description: str
    field_schema: dict[str, dict[str, Any]]

    def get_field_schema(self) -> dict[str, dict[str, Any]]:
        return self.field_schema


class DummyRegistry:
    """Registry interface used to feed the GUI."""

    def __init__(self, action: DummyAction) -> None:
        self.action = action

    def list_actions(self) -> list[str]:
        return [self.action.name]

    def get_action(self, _: str) -> DummyAction:
        return self.action


def test_gui_requires_dearpygui(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("photoflow.gui.app._dpg", None)
    with pytest.raises(GUIImportError):
        PhotoFlowGUI(dpg_module=None)


def test_setup_builds_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    dpg = DummyDPG()
    action = DummyAction(
        name="resize",
        label="Resize Image",
        description="Resize to specific dimensions",
        field_schema={
            "width": {"type": "integer", "default": 800, "label": "Width"},
            "height": {"type": "integer", "default": 600, "label": "Height"},
            "maintain_aspect": {"type": "boolean", "default": True, "label": "Maintain Aspect"},
            "quality": {"type": "choice", "choices": [("high", "High"), ("medium", "Medium")]},
            "notes": {"type": "string", "default": ""},
        },
    )
    gui = PhotoFlowGUI(registry=DummyRegistry(action), dpg_module=dpg)
    gui.setup()

    recorded = {name for name, _, _ in dpg.calls}
    assert "create_context" in recorded
    assert "create_viewport" in recorded
    assert "add_window" in recorded
    assert "add_child_window" in recorded
    assert "add_button" in recorded
    assert "add_file_dialog" in recorded
    assert "add_input_int" in recorded
    assert "add_checkbox" in recorded
    assert "add_combo" in recorded
    assert "show_viewport" in recorded


def test_run_starts_and_destroys_context(monkeypatch: pytest.MonkeyPatch) -> None:
    dpg = DummyDPG()
    gui = PhotoFlowGUI(registry=DummyRegistry(DummyAction("noop", "Noop", "", {})), dpg_module=dpg)
    gui.run()

    sequence = [name for name, _, _ in dpg.calls]
    assert sequence.count("start_dearpygui") == 1
    assert sequence.count("destroy_context") == 1


def test_launch_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    created = {}

    class DummyGUI(PhotoFlowGUI):
        def __init__(self) -> None:  # type: ignore[override]
            created["instance"] = True
            super().__init__(registry=DummyRegistry(DummyAction("noop", "Noop", "", {})), dpg_module=DummyDPG())

        def run(self) -> None:  # type: ignore[override]
            created["ran"] = True

    monkeypatch.setattr("photoflow.gui.app.PhotoFlowGUI", DummyGUI)  # type: ignore[arg-type]
    launch()
    assert created == {"instance": True, "ran": True}


def test_file_dialog_tracks_last_directory(tmp_path) -> None:
    dpg = DummyDPG()
    gui = PhotoFlowGUI(registry=DummyRegistry(DummyAction("noop", "Noop", "", {})), dpg_module=dpg)
    gui.setup()

    new_dir = tmp_path / "images"
    new_dir.mkdir()
    gui._last_directory = new_dir
    gui._show_file_dialog()

    assert any(
        call[0] == "configure_item"
        and call[1][0] == gui._file_dialog_tag
        and call[2]["default_path"] == str(new_dir)
        and call[2].get("directory_selector") is False
        for call in dpg.calls
    )

    gui._on_file_selected(
        "sender",
        {
            "file_path_name": str(new_dir / "example.jpg"),
            "current_path": str(new_dir),
        },
        None,
    )

    assert gui._selected_image == new_dir / "example.jpg"
    assert gui._last_directory == new_dir

    gui._on_input_mode_changed("sender", "Folder")
    gui._show_file_dialog()
    assert gui._input_mode == "folder"
    assert any(
        call[0] == "configure_item" and call[1][0] == gui._file_dialog_tag and call[2].get("directory_selector") is True
        for call in dpg.calls
    )

    gui._on_file_selected(
        "sender",
        {
            "folder_path_name": str(new_dir),
            "current_path": str(new_dir),
        },
        None,
    )
    assert gui._selected_folder == new_dir


def test_add_action_updates_pipeline(tmp_path) -> None:
    dpg = DummyDPG()
    action = DummyAction(
        name="resize",
        label="Resize Image",
        description="Resize to specific dimensions",
        field_schema={
            "width": {"type": "integer", "default": 800, "label": "Width"},
            "maintain_aspect": {"type": "boolean", "default": True, "label": "Maintain Aspect"},
            "quality": {"type": "choice", "choices": [("high", "High"), ("medium", "Medium"), ("low", "Low")]},
        },
    )
    gui = PhotoFlowGUI(registry=DummyRegistry(action), dpg_module=dpg)
    gui.setup()

    bindings = gui._action_field_bindings["resize"]
    for binding in bindings:
        if binding["name"] == "width":
            dpg.values[binding["tag"]] = 1024
        if binding["name"] == "maintain_aspect":
            dpg.values[binding["tag"]] = False
        if binding["name"] == "quality":
            dpg.values[binding["tag"]] = "Medium"

    gui._on_add_action("sender", None, "resize")

    assert len(gui._pipeline) == 1
    action_name, params = gui._pipeline[0]
    assert action_name == "resize"
    assert params["width"] == 1024
    assert params["maintain_aspect"] is False
    assert params["quality"] == "medium"
    assert any(call[0] == "configure_item" and call[1][0] == gui._pipeline_list_tag for call in dpg.calls)
    assert dpg.values[gui._flow_text_tag] == "Flow: Input → resize → Output"
    assert "resize" in dpg.values[gui._flow_details_tag]


def test_folder_pattern_updates_label(tmp_path) -> None:
    dpg = DummyDPG()
    gui = PhotoFlowGUI(registry=DummyRegistry(DummyAction("noop", "Noop", "", {})), dpg_module=dpg)
    gui.setup()

    gui._on_input_mode_changed("sender", "Folder")
    gui._selected_folder = tmp_path
    gui._on_pattern_changed("photoflow::pattern_input", "*.png")

    assert gui._selected_pattern == "*.png"
    assert dpg.values["photoflow::selected_path"].endswith("Pattern: *.png")
