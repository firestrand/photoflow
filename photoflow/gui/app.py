"""Dear PyGui application scaffolding for PhotoFlow.

This module defines the initial GUI shell used in Phase 4 of the
project roadmap.  The goal is to expose a minimal-yet-extensible
structure that can later evolve into the full visual workflow
experience without blocking headless users when Dear PyGui is not
installed on their system.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

from photoflow.core.actions import default_registry

_dpg_module: Any | None
try:  # pragma: no cover - exercised indirectly via PhotoFlowGUI
    _dpg_module = importlib.import_module("dearpygui.dearpygui")
except ModuleNotFoundError:  # pragma: no cover - handled in require_dpg
    _dpg_module = None

_dpg: Any | None = _dpg_module


class GUIImportError(RuntimeError):
    """Raised when the GUI is requested but Dear PyGui is missing."""


def require_dpg() -> Any:
    """Return the Dear PyGui module or raise an import error."""

    if _dpg is None:  # pragma: no cover - covered via exception tests
        raise GUIImportError(
            "Dear PyGui is not installed. Install it with "
            "`pip install photoflow[gui]` to run the graphical interface."
        )
    return _dpg


class PhotoFlowGUI:
    """Primary Dear PyGui application shell for PhotoFlow."""

    def __init__(
        self,
        registry: Any | None = None,
        *,
        dpg_module: Any | None = None,
        viewport_title: str = "PhotoFlow",
        viewport_width: int = 1280,
        viewport_height: int = 820,
    ) -> None:
        self._dpg: Any = dpg_module or require_dpg()
        self.registry = registry or default_registry
        self.viewport_title = viewport_title
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height

        self._main_window_tag: str | None = None
        self._sidebar_tag: str | None = None
        self._preview_panel_tag: str | None = None
        self._preview_message_tag: str | None = None
        self._layout_group_tag: str | None = None
        self._file_dialog_tag: str | None = None
        self._selected_image: Path | None = None
        self._last_directory: Path = Path.cwd()
        self._pipeline_list_tag: str | None = None
        self._pipeline: list[tuple[str, dict[str, Any]]] = []
        self._action_field_bindings: dict[str, list[dict[str, Any]]] = {}
        self._input_mode: str = "single"
        self._selected_folder: Path | None = None
        self._selected_pattern: str = "*.jpg"
        self._flow_text_tag: str | None = None
        self._flow_details_tag: str | None = None

    # ------------------------------------------------------------------
    # Public API

    def run(self) -> None:
        """Start the Dear PyGui event loop."""

        self.setup()
        try:
            self._dpg.start_dearpygui()
        finally:
            self._dpg.destroy_context()

    def setup(self) -> None:
        """Initialise Dear PyGui context and build the UI."""

        self._dpg.create_context()
        self._dpg.create_viewport(
            title=self.viewport_title,
            width=self.viewport_width,
            height=self.viewport_height,
        )
        self._build_layout()
        self._dpg.setup_dearpygui()
        self._dpg.show_viewport()

    # ------------------------------------------------------------------
    # Layout assembly

    def _build_layout(self) -> None:
        """Construct the initial window layout."""

        main_window_tag = self._dpg.add_window(
            label="PhotoFlow Workspace",
            tag="photoflow::main_window",
            width=self.viewport_width,
            height=self.viewport_height - 40,
        )

        self._main_window_tag = main_window_tag

        layout_group_tag = self._dpg.add_group(
            parent=main_window_tag,
            horizontal=True,
            tag="photoflow::layout_group",
        )

        self._layout_group_tag = layout_group_tag

        sidebar_tag = self._dpg.add_child_window(
            label="Controls",
            tag="photoflow::sidebar",
            parent=layout_group_tag,
            width=int(self.viewport_width * 0.32),
            height=self.viewport_height - 80,
        )

        self._sidebar_tag = sidebar_tag

        preview_panel_tag = self._dpg.add_child_window(
            label="Preview",
            tag="photoflow::preview",
            parent=layout_group_tag,
            width=-1,
            height=self.viewport_height - 80,
        )

        self._preview_panel_tag = preview_panel_tag

        self._build_file_browser(sidebar_tag)
        self._build_action_browser(sidebar_tag)
        self._initialise_preview_panel()

    def _build_file_browser(self, parent_tag: str) -> None:
        """Create the basic file browser controls."""

        self._dpg.add_text("Input", parent=parent_tag)
        self._dpg.add_separator(parent=parent_tag)
        self._dpg.add_radio_button(
            items=["Single Image", "Folder"],
            parent=parent_tag,
            default_value="Single Image",
            tag="photoflow::input_mode",
            callback=self._on_input_mode_changed,
        )
        self._dpg.add_button(
            label="Browse…",
            parent=parent_tag,
            callback=self._show_file_dialog,
        )
        self._dpg.add_text(
            "No input selected",
            tag="photoflow::selected_path",
            parent=parent_tag,
            wrap=250,
        )
        self._dpg.add_input_text(
            label="Pattern",
            tag="photoflow::pattern_input",
            default_value=self._selected_pattern,
            parent=parent_tag,
            callback=self._on_pattern_changed,
            enabled=False,
        )

        self._file_dialog_tag = self._dpg.add_file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_file_selected,
            tag="photoflow::file_dialog",
            default_path=str(self._last_directory),
            default_filename="",
            width=700,
            height=420,
        )
        self._dpg.add_file_extension(".jpg", parent=self._file_dialog_tag)
        self._dpg.add_file_extension(".png", parent=self._file_dialog_tag)
        self._dpg.add_file_extension(".webp", parent=self._file_dialog_tag)

    def _build_action_browser(self, parent_tag: str) -> None:
        """Render available actions and their parameter controls."""

        self._dpg.add_text("Actions", parent=parent_tag)
        self._dpg.add_separator(parent=parent_tag)

        actions = sorted(self._list_actions(), key=lambda item: item[0])
        for action_name, action_instance in actions:
            header_tag = self._dpg.add_collapsing_header(
                label=f"{action_instance.label}", parent=parent_tag, default_open=False
            )
            self._dpg.add_text(action_instance.description, parent=header_tag, wrap=250)
            self._dpg.add_separator(parent=header_tag)
            schema = action_instance.get_field_schema()
            self._action_field_bindings[action_name] = []
            for field_name, field_schema in schema.items():
                binding = self._build_field_control(header_tag, action_name, field_name, field_schema)
                self._action_field_bindings[action_name].append(binding)
            self._dpg.add_button(
                label="Add Action",
                parent=header_tag,
                callback=self._on_add_action,
                user_data=action_name,
            )

    def _initialise_preview_panel(self) -> None:
        """Prepare the preview panel with placeholder content."""

        if self._preview_panel_tag is None:
            return
        self._preview_message_tag = self._dpg.add_text(
            "Preview will appear here once an image is loaded.",
            parent=self._preview_panel_tag,
            tag="photoflow::preview_message",
            wrap=600,
        )
        self._pipeline_list_tag = self._dpg.add_listbox(
            label="Pipeline",
            parent=self._preview_panel_tag,
            items=["(pipeline empty)"],
            num_items=8,
            width=-1,
            tag="photoflow::pipeline_list",
        )
        self._dpg.add_button(
            label="Clear Pipeline",
            parent=self._preview_panel_tag,
            callback=self._clear_pipeline,
        )
        self._pipeline_status_tag = self._dpg.add_text(
            "Add an action to begin building the pipeline.",
            parent=self._preview_panel_tag,
            wrap=600,
            tag="photoflow::pipeline_status",
        )
        self._dpg.add_separator(parent=self._preview_panel_tag)
        self._flow_text_tag = self._dpg.add_text(
            "Flow: Input → Output",
            parent=self._preview_panel_tag,
            wrap=600,
            tag="photoflow::flow_text",
        )
        self._flow_details_tag = self._dpg.add_text(
            "No actions queued.",
            parent=self._preview_panel_tag,
            wrap=600,
            tag="photoflow::flow_details",
        )
        self._update_pipeline_view()

    # ------------------------------------------------------------------
    # Field widgets

    def _build_field_control(
        self,
        parent: str,
        action_name: str,
        field_name: str,
        field_schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a control for the provided field schema."""

        label = field_schema.get("label") or field_name.replace("_", " ").title()
        help_text = field_schema.get("help_text")
        default = field_schema.get("default")
        required = field_schema.get("required", True)
        field_type = field_schema.get("type", "string").lower()
        widget_tag = f"photoflow::field::{action_name}::{field_name}"
        widget_factory = self._resolve_widget_factory(field_type)
        widget_factory(parent, label, default, field_schema, widget_tag)
        binding: dict[str, Any] = {
            "name": field_name,
            "tag": widget_tag,
            "schema": field_schema,
            "type": field_type,
        }
        if field_type == "choice":
            binding["choices_map"] = {display: value for value, display in field_schema.get("choices", [])}
        if help_text:
            self._dpg.add_text(help_text, parent=parent, wrap=240)
        if required:
            self._dpg.add_separator(parent=parent)
        return binding

    def _resolve_widget_factory(self, field_type: str) -> Callable[[str, str, Any, dict[str, Any], str], None]:
        """Map a field type to a widget creation function."""

        factories: dict[str, Callable[[str, str, Any, dict[str, Any], str], None]] = {
            "integer": self._add_integer_input,
            "float": self._add_float_input,
            "boolean": self._add_checkbox,
            "choice": self._add_choice_combo,
        }
        return factories.get(field_type, self._add_text_input)

    def _add_integer_input(self, parent: str, label: str, default: Any, _: dict[str, Any], widget_tag: str) -> None:
        self._dpg.add_input_int(
            label=label,
            parent=parent,
            default_value=int(default or 0),
            tag=widget_tag,
        )

    def _add_float_input(self, parent: str, label: str, default: Any, _: dict[str, Any], widget_tag: str) -> None:
        self._dpg.add_input_float(
            label=label,
            parent=parent,
            default_value=float(default or 0.0),
            tag=widget_tag,
        )

    def _add_checkbox(self, parent: str, label: str, default: Any, _: dict[str, Any], widget_tag: str) -> None:
        self._dpg.add_checkbox(label=label, parent=parent, default_value=bool(default), tag=widget_tag)

    def _add_choice_combo(self, parent: str, label: str, default: Any, schema: dict[str, Any], widget_tag: str) -> None:
        choices = schema.get("choices") or []
        items = [display for _, display in choices]
        default_value = None
        for value, display in choices:
            if value == default:
                default_value = display
                break
        self._dpg.add_combo(
            label=label,
            items=items,
            default_value=default_value,
            parent=parent,
            tag=widget_tag,
        )

    def _add_text_input(self, parent: str, label: str, default: Any, _: dict[str, Any], widget_tag: str) -> None:
        self._dpg.add_input_text(label=label, parent=parent, default_value=str(default or ""), tag=widget_tag)

    # ------------------------------------------------------------------
    # Dear PyGui callbacks

    def _show_file_dialog(self, *_: Any) -> None:
        if self._file_dialog_tag is None:
            return
        self._dpg.configure_item(
            self._file_dialog_tag,
            default_path=str(self._last_directory),
            default_filename="",
            directory_selector=self._input_mode == "folder",
        )
        self._dpg.show_item(self._file_dialog_tag)

    def _on_file_selected(
        self, _sender: str, app_data: dict[str, Any], _user_data: Any
    ) -> None:  # pragma: no cover - GUI callback
        current_path = app_data.get("current_path")
        if current_path:
            self._last_directory = Path(current_path)
        if self._input_mode == "folder":
            selected_dir = app_data.get("folder_path_name") or current_path
            if selected_dir:
                self._selected_folder = Path(selected_dir)
                self._selected_image = None
                self._dpg.set_value(
                    "photoflow::selected_path",
                    f"Folder: {self._selected_folder}\nPattern: {self._selected_pattern}",
                )
        else:
            selection = app_data.get("file_path_name")
            if selection:
                self._selected_image = Path(selection)
                self._selected_folder = None
                self._dpg.set_value("photoflow::selected_path", str(self._selected_image))
                self._dpg.set_value(
                    "photoflow::preview_message",
                    f"Loaded: {self._selected_image.name} (preview rendering TBD)",
                )
        self._update_pipeline_view()

    def _on_add_action(self, _sender: str, _app_data: Any, action_name: str) -> None:
        bindings = self._action_field_bindings.get(action_name, [])
        if not bindings:
            return
        params: dict[str, Any] = {}
        for binding in bindings:
            tag = binding["tag"]
            value = self._dpg.get_value(tag)
            field_type = binding.get("type", "string")
            schema = binding.get("schema", {})
            if field_type == "choice":
                mapping = binding.get("choices_map", {})
                value = mapping.get(value, schema.get("default"))
            elif field_type == "integer":
                value = int(value)
            elif field_type == "float":
                value = float(value)
            elif field_type == "boolean":
                value = bool(value)
            params[binding["name"]] = value
        self._pipeline.append((action_name, params))
        self._update_pipeline_view()
        if self._preview_message_tag:
            self._dpg.set_value(
                self._preview_message_tag,
                f"Queued action '{action_name}'. Preview rendering not yet implemented.",
            )

    def _clear_pipeline(self, *_: Any) -> None:
        self._pipeline.clear()
        self._update_pipeline_view()
        if self._preview_message_tag:
            self._dpg.set_value(
                self._preview_message_tag,
                "Pipeline cleared. Preview will update once implemented.",
            )

    def _update_pipeline_view(self) -> None:
        if not self._pipeline_list_tag:
            return
        if self._pipeline:
            items = [f"{idx + 1}. {action_name}" for idx, (action_name, _) in enumerate(self._pipeline)]
        else:
            items = ["(pipeline empty)"]
        self._dpg.configure_item(self._pipeline_list_tag, items=items)
        if items:
            self._dpg.set_value(self._pipeline_list_tag, items[0])
        if self._pipeline_status_tag:
            if self._pipeline:
                self._dpg.set_value(
                    self._pipeline_status_tag,
                    f"Pipeline contains {len(self._pipeline)} action(s).",
                )
            else:
                self._dpg.set_value(
                    self._pipeline_status_tag,
                    "Pipeline cleared. Add an action to begin.",
                )
        if self._flow_text_tag:
            steps = ["Input"] + [name for name, _ in self._pipeline] + ["Output"]
            flow_line = " → ".join(steps)
            self._dpg.set_value(self._flow_text_tag, f"Flow: {flow_line}")
        if self._flow_details_tag:
            if self._pipeline:
                details: list[str] = []
                for idx, (action_name, params) in enumerate(self._pipeline, start=1):
                    if params:
                        param_str = ", ".join(f"{key}={value}" for key, value in params.items())
                        details.append(f"{idx}. {action_name} ({param_str})")
                    else:
                        details.append(f"{idx}. {action_name}")
                self._dpg.set_value(self._flow_details_tag, "\n".join(details))
            else:
                self._dpg.set_value(self._flow_details_tag, "No actions queued.")

    def _on_input_mode_changed(self, _sender: str, app_data: Any, *_: Any) -> None:
        self._input_mode = "folder" if app_data == "Folder" else "single"
        self._dpg.configure_item("photoflow::pattern_input", enabled=self._input_mode == "folder")
        if self._input_mode == "folder":
            self._selected_image = None
            self._dpg.set_value(
                "photoflow::selected_path",
                f"Folder: {self._selected_folder or 'None'}\nPattern: {self._selected_pattern}",
            )
        else:
            self._selected_folder = None
            self._dpg.set_value("photoflow::selected_path", "No input selected")

    def _on_pattern_changed(self, _sender: str, app_data: Any) -> None:
        if isinstance(app_data, str):
            self._selected_pattern = app_data or "*.jpg"
            if self._selected_folder:
                self._dpg.set_value(
                    "photoflow::selected_path",
                    f"Folder: {self._selected_folder}\nPattern: {self._selected_pattern}",
                )

    # ------------------------------------------------------------------
    # Helpers

    def _list_actions(self) -> Iterable[tuple[str, Any]]:
        """Enumerate available actions from the registry."""

        if hasattr(self.registry, "list_actions") and hasattr(self.registry, "get_action"):
            for action_name in self.registry.list_actions():
                yield action_name, self.registry.get_action(action_name)
        else:  # pragma: no cover - defensive fallback
            return


def launch() -> None:
    """Convenience helper to construct and run the GUI."""

    PhotoFlowGUI().run()


__all__ = ["GUIImportError", "PhotoFlowGUI", "launch"]
