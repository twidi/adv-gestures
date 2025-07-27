from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Union, get_args, get_origin

import typer
from pydantic import BaseModel
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.reactive import reactive
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Static,
    Tree,
)
from textual.widgets.tree import TreeNode

from ..cameras import CameraInfo
from ..config import Config
from ..models.hands import Hands
from .common import (
    DEFAULT_USER_CONFIG_PATH,
    app,
    pick_camera,
)


class ConfirmQuitDialog(ModalScreen[str]):
    """Modal dialog for confirming quit with unsaved changes."""

    CSS = """
    ConfirmQuitDialog {
        align: center middle;
    }

    #dialog {
        padding: 2;
        width: 70;
        height: auto;
        border: thick $background 80%;
        background: $surface;
    }

    #dialog-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #dialog-message {
        margin-bottom: 2;
    }

    #button-container {
        dock: bottom;
        height: auto;
        align: center middle;
        padding: 1 0;
    }

    Button {
        margin: 0 1;
        min-width: 16;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Unsaved Changes", id="dialog-title")
            yield Label(
                "You have unsaved changes. What would you like to do?",
                id="dialog-message",
            )
            with Horizontal(id="button-container"):
                yield Button("Save and quit", variant="success", id="save")
                yield Button("Quit without saving", variant="error", id="quit")
                yield Button("Cancel", variant="primary", id="cancel")

    @on(Button.Pressed, "#save")
    def handle_save(self) -> None:
        """Handle Save and quit button press."""
        self.dismiss("save")

    @on(Button.Pressed, "#quit")
    def handle_quit(self) -> None:
        """Handle Quit without saving button press."""
        self.dismiss("quit")

    @on(Button.Pressed, "#cancel")
    def handle_cancel(self) -> None:
        """Handle Cancel button press."""
        self.dismiss("cancel")

    def on_key(self, event: Key) -> None:
        """Handle key events."""
        if event.key == "escape":
            event.stop()
            self.dismiss("cancel")


class ValueInputDialog(ModalScreen[str | None]):
    """Modal dialog for entering a value directly."""

    CSS = """
    ValueInputDialog {
        align: center middle;
    }

    #dialog {
        padding: 2;
        width: 60;
        min-height: 12;
        height: auto;
        border: thick $background 80%;
        background: $surface;
    }

    #dialog-title {
        text-style: bold;
        margin-bottom: 1;
    }

    Label {
        margin-bottom: 0;
    }

    #value-input {
        margin: 1 0 2 0;
    }

    #button-container {
        dock: bottom;
        height: auto;
        align: center middle;
        padding: 1 0;
    }

    Button {
        margin: 0 1;
        min-width: 12;
    }
    """

    def __init__(self, title: str, current_value: Any, value_type: type) -> None:
        super().__init__()
        self.title: str = title
        self.current_value = current_value
        self.value_type = value_type

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label(self.title, id="dialog-title")
            yield Label(f"Current value: {self._format_value(self.current_value)}")
            yield Label(f"Type: {self._format_type_name(self.value_type)}")
            yield Input(
                value=str(self.current_value) if self.current_value is not None else "",
                placeholder="Enter new value",
                id="value-input",
            )
            with Horizontal(id="button-container"):
                yield Button("OK", variant="primary", id="ok")
                yield Button("Cancel", id="cancel")

    def on_mount(self) -> None:
        """Focus the input when dialog opens."""
        self.query_one("#value-input", Input).focus()

    @on(Button.Pressed, "#ok")
    def handle_ok(self) -> None:
        """Handle OK button press."""
        input_value = self.query_one("#value-input", Input).value
        self.dismiss(input_value)

    @on(Button.Pressed, "#cancel")
    def handle_cancel(self) -> None:
        """Handle Cancel button press."""
        self.dismiss(None)

    @on(Input.Submitted)
    def handle_input_submitted(self) -> None:
        """Handle Enter key in input."""
        input_value = self.query_one("#value-input", Input).value
        self.dismiss(input_value)

    def on_key(self, event: Key) -> None:
        """Handle key events."""
        if event.key == "escape":
            event.stop()  # Prevent the event from bubbling up
            self.dismiss(None)

    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if isinstance(value, float):
            return f"{value:.3f}"
        elif value is None:
            return "None"
        else:
            return str(value)

    def _format_type_name(self, type_hint: type) -> str:
        """Format a type for display, handling Union types."""
        origin = get_origin(type_hint)

        # Handle Union types (including Optional which is Union[X, None])
        if origin is Union:
            args = get_args(type_hint)
            # Filter out NoneType
            non_none_types = [t for t in args if t is not type(None)]
            if len(args) == 2 and type(None) in args and len(non_none_types) == 1:
                # This is Optional[T] (Union[T, None])
                return getattr(non_none_types[0], "__name__", str(non_none_types[0])) + " | None"
            else:
                # General Union
                type_names = [getattr(t, "__name__", str(t)) for t in args]
                return " | ".join(type_names)
        else:
            # Not a Union type
            return getattr(type_hint, "__name__", str(type_hint))


class CopyFromFingerDialog(ModalScreen[str | None]):
    """Modal dialog for copying values from another finger."""

    CSS = """
    CopyFromFingerDialog {
        align: center middle;
    }

    #dialog {
        padding: 2;
        width: 60;
        height: auto;
        border: thick $background 80%;
        background: $surface;
    }

    #dialog-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #dialog-message {
        margin-bottom: 2;
    }

    RadioSet {
        margin-bottom: 2;
        height: auto;
    }

    RadioButton {
        padding: 0 2;
    }

    #button-container {
        dock: bottom;
        height: auto;
        align: center middle;
        padding: 1 0;
    }

    Button {
        margin: 0 1;
        min-width: 12;
    }
    """

    def __init__(
        self,
        current_finger: str,
        available_fingers: list[str],
        copy_type: str,
        finger_values: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.current_finger = current_finger
        self.available_fingers = available_fingers
        self.copy_type = copy_type
        self.finger_values = finger_values or {}

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Copy from Finger", id="dialog-title")
            yield Label(
                f"Select a finger to copy {self.copy_type} to '{self.current_finger}':",
                id="dialog-message",
            )
            with RadioSet(id="finger-set"):
                for finger in self.available_fingers:
                    label = finger.capitalize()
                    # Add value in parentheses if available
                    if finger in self.finger_values:
                        value = self.finger_values[finger]
                        formatted_value = self._format_value(value)
                        label = f"{label} ({formatted_value})"
                    yield RadioButton(label)
            with Horizontal(id="button-container"):
                yield Button("Copy", variant="primary", id="copy")
                yield Button("Cancel", id="cancel")

    @on(Button.Pressed, "#copy")
    def handle_copy(self) -> None:
        """Handle Copy button press."""
        radio_set = self.query_one("#finger-set", RadioSet)
        pressed_idx = radio_set.pressed_index
        if pressed_idx is not None and 0 <= pressed_idx < len(self.available_fingers):
            # Get the finger name from our available_fingers list using the index
            selected_finger = self.available_fingers[pressed_idx]
            self.dismiss(selected_finger)
        else:
            self.dismiss(None)

    @on(Button.Pressed, "#cancel")
    def handle_cancel(self) -> None:
        """Handle Cancel button press."""
        self.dismiss(None)

    def on_key(self, event: Key) -> None:
        """Handle key events."""
        if event.key == "escape":
            event.stop()
            self.dismiss(None)

    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if isinstance(value, float):
            return f"{value:.3f}"
        elif value is None:
            return "None"
        else:
            return str(value)


class ConfigTree(Tree[dict[str, Any]]):
    """Tree widget for displaying config structure."""

    def __init__(self, config: Config) -> None:
        super().__init__("Configuration")
        self.config = config
        self.editor: ConfigEditor | None = None  # Will be set later
        self._build_tree()
        self._last_click_time: float = 0
        self._last_clicked_node: TreeNode[dict[str, Any]] | None = None
        self._double_click_threshold = 0.5  # 500ms

    def set_editor(self, editor: ConfigEditor) -> None:
        """Set the editor reference."""
        self.editor = editor

    def on_tree_node_selected(self, event: Tree.NodeSelected[dict[str, Any]]) -> None:
        """Handle node selection (including double-clicks)."""
        current_time = time.time()

        # Check if this is a double-click on the same node
        if (
            self._last_clicked_node == event.node
            and current_time - self._last_click_time < self._double_click_threshold
        ):
            # Double-click detected
            if self.editor and event.node and event.node.data:
                data = event.node.data
                if data.get("field_name") and not isinstance(data.get("obj"), BaseModel):
                    self.editor.current_node = event.node
                    # For boolean values, toggle instead of opening dialog
                    value = data.get("obj")
                    if isinstance(value, bool):
                        self.editor.toggle_bool_value()
                    else:
                        self.editor.open_value_input_dialog()
            # Reset to prevent triple-click from triggering
            self._last_click_time = 0
            self._last_clicked_node = None
        else:
            # Single click - record for potential double-click
            self._last_click_time = current_time
            self._last_clicked_node = event.node

    @on(Key)
    def on_key(self, event: Key) -> None:
        """Handle key events in the tree."""
        # Get the current selected node
        if self.cursor_node and self.cursor_node.data:
            data = self.cursor_node.data
            # Check if node is editable (has field_name and is not a BaseModel)
            if data.get("field_name") and not isinstance(data.get("obj"), BaseModel):
                # Handle value modification keys
                handled = False
                if event.key == "left":
                    if self.editor:
                        # Make sure editor is synced with current node
                        if self.editor.current_node != self.cursor_node:
                            self.editor.current_node = self.cursor_node
                        # Check if it's a boolean value
                        value = data.get("obj")
                        if isinstance(value, bool):
                            self.editor.toggle_bool_value()
                        else:
                            self.editor.adjust_numeric_value(-1)
                        handled = True
                elif event.key == "right":
                    if self.editor:
                        if self.editor.current_node != self.cursor_node:
                            self.editor.current_node = self.cursor_node
                        # Check if it's a boolean value
                        value = data.get("obj")
                        if isinstance(value, bool):
                            self.editor.toggle_bool_value()
                        else:
                            self.editor.adjust_numeric_value(1)
                        handled = True
                elif event.key == "space":
                    if self.editor:
                        if self.editor.current_node != self.cursor_node:
                            self.editor.current_node = self.cursor_node
                        self.editor.toggle_bool_value()
                        handled = True
                elif event.key == "ctrl+left":
                    if self.editor:
                        if self.editor.current_node != self.cursor_node:
                            self.editor.current_node = self.cursor_node
                        self.editor.adjust_numeric_value(-1, use_ctrl=True)
                        handled = True
                elif event.key == "ctrl+right":
                    if self.editor:
                        if self.editor.current_node != self.cursor_node:
                            self.editor.current_node = self.cursor_node
                        self.editor.adjust_numeric_value(1, use_ctrl=True)
                        handled = True
                elif event.key == "ctrl+shift+left":
                    if self.editor:
                        if self.editor.current_node != self.cursor_node:
                            self.editor.current_node = self.cursor_node
                        self.editor.adjust_numeric_value(-1, use_shift=True)
                        handled = True
                elif event.key == "ctrl+shift+right":
                    if self.editor:
                        if self.editor.current_node != self.cursor_node:
                            self.editor.current_node = self.cursor_node
                        self.editor.adjust_numeric_value(1, use_shift=True)
                        handled = True

                if handled:
                    event.prevent_default()

    def _build_tree(self) -> None:
        """Build the tree from config structure."""
        root = self.root
        root.data = {"path": [], "obj": self.config, "field_name": None, "parent": None}

        # Use Pydantic's model_fields to build tree
        self._add_model_fields(root, self.config, [])

    def _add_model_fields(self, parent_node: TreeNode[dict[str, Any]], obj: BaseModel, path: list[str]) -> None:
        """Add fields from a Pydantic model to the tree."""
        for field_name, field_info in obj.model_fields.items():
            field_value = getattr(obj, field_name)
            field_path = path + [field_name]

            # Create node data
            node_data = {
                "path": field_path,
                "obj": field_value,
                "field_name": field_name,
                "field_info": field_info,
                "parent": obj,
            }

            # Determine if this is a leaf node (primitive value) or a container
            if isinstance(field_value, BaseModel):
                # Container node - show just the field name
                node = parent_node.add(field_name, data=node_data)
                # Recursively add children
                self._add_model_fields(node, field_value, field_path)
            else:
                # Leaf node - show field name and value
                display_value = self._format_value(field_value)
                node = parent_node.add(f"{field_name}: {display_value}", data=node_data, allow_expand=False)

    def _format_value(self, value: Any) -> str:
        """Format a value for display in the tree."""
        if isinstance(value, float):
            return f"{value:.3f}"
        elif value is None:
            return "None"
        else:
            return str(value)

    def update_node_display(self, node: TreeNode[dict[str, Any]]) -> None:
        """Update the display of a node after its value changed."""
        if node.data and node.data.get("field_name"):
            field_name = node.data["field_name"]
            parent_obj = node.data["parent"]
            if parent_obj:
                value = getattr(parent_obj, field_name)
                display_value = self._format_value(value)
                node.label = f"{field_name}: {display_value}"
                # Update the cached value in node data
                node.data["obj"] = value


class ConfigEditor(Static):
    """Widget for editing config values."""

    current_node: reactive[TreeNode[dict[str, Any]] | None] = reactive(None)

    def __init__(self, config: Config, config_tree: ConfigTree) -> None:
        super().__init__()
        self.config = config
        self.config_tree = config_tree
        self._initial_values: dict[str, Any] = {}
        self._current_values: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Label("Select a configuration value to edit", id="editor-title")
        yield Label("", id="editor-description")
        yield Label("", id="editor-type")
        yield Label("", id="editor-default")
        yield Label("", id="editor-initial")
        yield Label("", id="editor-current")
        yield Label("", id="editor-help")

    def on_mount(self) -> None:
        """Store initial values when mounted."""
        self._store_initial_values(self.config, [])
        self._current_values = self._initial_values.copy()

    def _store_initial_values(self, obj: BaseModel, path: list[str]) -> None:
        """Recursively store initial values from config."""
        for field_name in obj.model_fields:
            field_value = getattr(obj, field_name)
            field_path = path + [field_name]
            path_str = ".".join(field_path)

            if isinstance(field_value, BaseModel):
                self._store_initial_values(field_value, field_path)
            else:
                self._initial_values[path_str] = field_value

    def watch_current_node(self, node: TreeNode[dict[str, Any]] | None) -> None:
        """Update editor when node selection changes."""
        # Check if widgets are mounted yet
        if not self.is_mounted:
            return

        if not node or not node.data:
            self.query_one("#editor-title", Label).update("Select a configuration value to edit")
            self.query_one("#editor-description", Label).update("")
            self.query_one("#editor-type", Label).update("")
            self.query_one("#editor-default", Label).update("")
            self.query_one("#editor-initial", Label).update("")
            self.query_one("#editor-current", Label).update("")
            self.query_one("#editor-help", Label).update("")
            return

        data = node.data
        if not data.get("field_name"):
            # This is the root node
            self.query_one("#editor-title", Label).update("Configuration Root")
            self.query_one("#editor-description", Label).update("Main configuration object")
            self.query_one("#editor-type", Label).update("")
            self.query_one("#editor-default", Label).update("")
            self.query_one("#editor-initial", Label).update("")
            self.query_one("#editor-current", Label).update("")
            self.query_one("#editor-help", Label).update("Navigate to child nodes to edit values")
            return

        if isinstance(data["obj"], BaseModel):
            # This is a container node, show its description
            path = data["path"]
            path_str = ".".join(path)
            field_info = data.get("field_info")

            self.query_one("#editor-title", Label).update(f"Container: {path_str}")

            # Description from field info
            description = (
                field_info.description if field_info and hasattr(field_info, "description") else "No description"
            )
            self.query_one("#editor-description", Label).update(f"Description: {description}")

            # Clear other fields for container nodes
            self.query_one("#editor-type", Label).update("Type: Configuration Section")
            self.query_one("#editor-default", Label).update("")
            self.query_one("#editor-initial", Label).update("")
            self.query_one("#editor-current", Label).update("")

            # Help text for containers
            help_text = "Navigate to child nodes to edit values"
            # Add F5 help if we're in a finger configuration
            if len(path) >= 2 and path[0] == "hands" and path[1] in ["thumb", "index", "middle", "ring", "pinky"]:
                help_text += "\nF5: copy from another finger"

            self.query_one("#editor-help", Label).update(help_text)
            return

        path = data["path"]
        field_info = data.get("field_info")
        field_name = data["field_name"]
        parent_obj = data["parent"]

        # Get the current value from the parent object, not the cached one
        value = getattr(parent_obj, field_name) if parent_obj and field_name else data["obj"]
        path_str = ".".join(path)

        self.query_one("#editor-title", Label).update(f"Editing: {path_str}")

        # Description from field info
        description = (
            field_info.description if field_info and hasattr(field_info, "description") else "No description"
        )
        self.query_one("#editor-description", Label).update(f"Description: {description}")

        # Type information from field annotation
        if field_info and hasattr(field_info, "annotation"):
            # Get the type name from the field annotation
            field_type = field_info.annotation
            origin = get_origin(field_type)

            # Handle Union types (including Optional which is Union[X, None])
            if origin is Union:
                args = get_args(field_type)
                # Filter out NoneType
                non_none_types = [t for t in args if t is not type(None)]
                if len(args) == 2 and type(None) in args and len(non_none_types) == 1:
                    # This is Optional[T] (Union[T, None])
                    type_name = getattr(non_none_types[0], "__name__", str(non_none_types[0])) + " | None"
                else:
                    # General Union
                    type_names = [getattr(t, "__name__", str(t)) for t in args]
                    type_name = " | ".join(type_names)
            else:
                # Not a Union type
                type_name = getattr(field_type, "__name__", str(field_type))
        else:
            # Fallback to actual value type
            type_name = type(value).__name__

        self.query_one("#editor-type", Label).update(f"Type: {type_name}")

        # Default value from field info
        default_value = field_info.default if field_info else None
        self.query_one("#editor-default", Label).update(f"Default: {self._format_value(default_value)}")

        # Initial value (when program started)
        initial_value = self._initial_values.get(path_str, "Unknown")
        self.query_one("#editor-initial", Label).update(f"Initial: {self._format_value(initial_value)}")

        # Current value
        self.query_one("#editor-current", Label).update(f"Current: {self._format_value(value)}")

        # Help text based on type
        if isinstance(value, bool):
            help_text = (
                "← → or Space: toggle value\nDouble-click: toggle value\nF3: reset to default\nF4: reset to initial"
            )
        elif isinstance(value, float):
            help_text = (
                "← →: adjust by 1\n"
                "Ctrl+← →: adjust by 0.1\n"
                "Ctrl+Shift+← →: adjust by 10\n"
                "F2 or double-click: enter value directly\n"
                "F3: reset to default\n"
                "F4: reset to initial"
            )
        elif isinstance(value, int):
            help_text = (
                "← →: adjust by 1\n"
                "Ctrl+Shift+← →: adjust by 10\n"
                "F2 or double-click: enter value directly\n"
                "F3: reset to default\n"
                "F4: reset to initial"
            )
        elif isinstance(value, str) or value is None:
            help_text = "F2 or double-click: enter value directly\nF3: reset to default\nF4: reset to initial"
        else:
            help_text = "This type cannot be edited"

        # Add F5 help if we're in a finger configuration
        if len(path) >= 2 and path[0] == "hands" and path[1] in ["thumb", "index", "middle", "ring", "pinky"]:
            help_text += "\nF5: copy from another finger"

        self.query_one("#editor-help", Label).update(help_text)

    def _format_value(self, value: Any) -> str:
        """Format a value for display."""
        if isinstance(value, float):
            return f"{value:.3f}"
        elif value is None:
            return "None"
        else:
            return str(value)

    def adjust_numeric_value(self, delta: float, use_shift: bool = False, use_ctrl: bool = False) -> None:
        """Adjust the current numeric value."""
        if not self.current_node or not self.current_node.data:
            return

        data = self.current_node.data
        if not data.get("field_name") or not data.get("parent"):
            return

        # Check if node is editable
        if isinstance(data.get("obj"), BaseModel):
            return

        field_name = data["field_name"]
        parent_obj = data["parent"]
        current_value = getattr(parent_obj, field_name)

        # Check bool first since bool is a subclass of int in Python
        if isinstance(current_value, bool):
            # Booleans should not be adjusted with numeric values
            return
        elif isinstance(current_value, float):
            # Adjust step size based on modifiers
            if use_ctrl:
                step = 0.1
            elif use_shift:
                step = 10.0
            else:
                step = 1.0
            new_value = current_value + (delta * step)
            # Round to avoid floating point precision issues
            new_value = round(new_value, 6)
        elif isinstance(current_value, int):
            if use_shift:
                step = 10
            else:
                step = 1
            new_value = current_value + int(delta * step)
        else:
            return

        # Set the new value
        setattr(parent_obj, field_name, new_value)

        # Track the modification
        path_str = ".".join(data["path"])
        self._current_values[path_str] = new_value

        # Update the tree display
        self.config_tree.update_node_display(self.current_node)

        # Update the editor display
        self.watch_current_node(self.current_node)

    def toggle_bool_value(self) -> None:
        """Toggle the current boolean value."""
        if not self.current_node or not self.current_node.data:
            return

        data = self.current_node.data
        if not data.get("field_name") or not data.get("parent"):
            return

        # Check if node is editable
        if isinstance(data.get("obj"), BaseModel):
            return

        field_name = data["field_name"]
        parent_obj = data["parent"]
        current_value = getattr(parent_obj, field_name)

        if isinstance(current_value, bool):
            new_value = not current_value
            setattr(parent_obj, field_name, new_value)

            # Track the modification
            path = data["path"]
            path_str = ".".join(path)
            self._current_values[path_str] = new_value

            # Update displays
            self.config_tree.update_node_display(self.current_node)
            self.watch_current_node(self.current_node)

    def open_value_input_dialog(self) -> None:
        """Open the value input dialog for the current node."""
        if not self.current_node or not self.current_node.data:
            return

        data = self.current_node.data
        if not data.get("field_name") or not data.get("parent"):
            return

        # Check if node is editable
        if isinstance(data.get("obj"), BaseModel):
            return

        field_name = data["field_name"]
        parent_obj = data["parent"]
        current_value = getattr(parent_obj, field_name)
        path = data["path"]
        path_str = ".".join(path)

        # Get the field type for the dialog
        field_type = type(current_value)
        field_info = data.get("field_info")
        if field_info and hasattr(field_info, "annotation"):
            field_type = field_info.annotation

        # Open the dialog
        def handle_dialog_result(result: str | None) -> None:
            """Handle the result from the dialog."""
            if result is not None:
                try:
                    # Convert the string input to the appropriate type
                    new_value: Any
                    # Check bool first since bool is a subclass of int in Python
                    if isinstance(current_value, bool):
                        # Accept various forms of boolean input
                        result_lower = result.lower()
                        if result_lower in ("true", "1", "yes", "on"):
                            new_value = True
                        elif result_lower in ("false", "0", "no", "off"):
                            new_value = False
                        else:
                            # Try to parse as a Python boolean
                            new_value = bool(result)
                    elif isinstance(current_value, float):
                        new_value = float(result)
                    elif isinstance(current_value, int):
                        new_value = int(result)
                    elif isinstance(current_value, str) or current_value is None:
                        new_value = result
                    else:
                        self.app.notify(
                            f"Cannot edit values of type {type(current_value).__name__}", severity="error"
                        )
                        return

                    # Set the new value
                    setattr(parent_obj, field_name, new_value)

                    # Track the modification
                    path_str = ".".join(path)
                    self._current_values[path_str] = new_value

                    # Update displays
                    if self.current_node:
                        self.config_tree.update_node_display(self.current_node)
                        self.watch_current_node(self.current_node)

                except ValueError as e:
                    self.app.notify(f"Invalid value: {e}", severity="error")

        # Push the dialog screen
        dialog = ValueInputDialog(f"Edit: {path_str}", current_value, field_type)
        self.app.push_screen(dialog, handle_dialog_result)

    def reset_to_default(self) -> None:
        """Reset the current value to its default."""
        if not self.current_node or not self.current_node.data:
            return

        data = self.current_node.data
        if not data.get("field_name") or not data.get("parent"):
            return

        # Check if node is editable
        if isinstance(data.get("obj"), BaseModel):
            return

        field_name = data["field_name"]
        parent_obj = data["parent"]
        field_info = data.get("field_info")

        if field_info and hasattr(field_info, "default"):
            default_value = field_info.default
            setattr(parent_obj, field_name, default_value)

            # Track the modification
            path = data["path"]
            path_str = ".".join(path)
            self._current_values[path_str] = default_value

            # Update displays
            self.config_tree.update_node_display(self.current_node)
            self.watch_current_node(self.current_node)
            self.app.notify(f"Reset to default: {self._format_value(default_value)}", severity="information")
        else:
            self.app.notify("No default value available", severity="warning")

    def reset_to_initial(self) -> None:
        """Reset the current value to its initial value."""
        if not self.current_node or not self.current_node.data:
            return

        data = self.current_node.data
        if not data.get("field_name") or not data.get("parent"):
            return

        # Check if node is editable
        if isinstance(data.get("obj"), BaseModel):
            return

        field_name = data["field_name"]
        parent_obj = data["parent"]
        path = data["path"]
        path_str = ".".join(path)

        initial_value = self._initial_values.get(path_str)
        if initial_value is not None:
            setattr(parent_obj, field_name, initial_value)

            # Track the modification (back to initial)
            self._current_values[path_str] = initial_value

            # Update displays
            self.config_tree.update_node_display(self.current_node)
            self.watch_current_node(self.current_node)
            self.app.notify(f"Reset to initial: {self._format_value(initial_value)}", severity="information")
        else:
            self.app.notify("No initial value recorded", severity="warning")

    def has_unsaved_changes(self) -> bool:
        """Check if there are any unsaved changes by comparing current values with initial values."""
        return self._current_values != self._initial_values

    def save_changes(self) -> None:
        """Mark all changes as saved by updating initial values."""
        self._initial_values = self._current_values.copy()

    def copy_from_finger(self) -> None:
        """Copy configuration from another finger."""
        if not self.current_node or not self.current_node.data:
            return

        data = self.current_node.data
        path = data.get("path", [])

        # Check if we're in a finger's configuration
        if len(path) < 2 or path[0] != "hands" or path[1] not in ["thumb", "index", "middle", "ring", "pinky"]:
            return

        current_finger = path[1]

        # Determine what we're copying and the relative path
        if len(path) == 2:
            # We're on the finger node itself - copy entire finger
            copy_type = "entire configuration"
            relative_path = []
        else:
            # We're on a sub-node - determine if it's a value or section
            obj = data.get("obj")
            if isinstance(obj, BaseModel):
                # It's a section
                copy_type = f"'{path[-1]}' section"
                relative_path = path[2:]
            else:
                # It's a value
                copy_type = f"'{path[-1]}' value"
                relative_path = path[2:]

        # Find available source fingers and their values
        available_fingers = []
        finger_values = {}
        hands_obj = getattr(self.config, "hands", None)
        if not hands_obj:
            return

        for finger_name in ["thumb", "index", "middle", "ring", "pinky"]:
            if finger_name == current_finger:
                continue

            # Check if this finger has the same structure
            finger_obj = getattr(hands_obj, finger_name, None)
            if not finger_obj:
                continue

            # For relative paths, check if the path exists in this finger
            if relative_path:
                try:
                    current_obj = finger_obj
                    for part in relative_path[:-1]:
                        current_obj = getattr(current_obj, part)
                    # Check if the final part exists
                    if hasattr(current_obj, relative_path[-1]):
                        available_fingers.append(finger_name)
                        # Get the value if this is a value node (not a section)
                        if not isinstance(obj, BaseModel):
                            value = getattr(current_obj, relative_path[-1])
                            finger_values[finger_name] = value
                except AttributeError:
                    continue
            else:
                # For entire finger copy, always available
                available_fingers.append(finger_name)

        if not available_fingers:
            self.app.notify("No compatible fingers found to copy from", severity="warning")
            return

        # Open the dialog
        def handle_dialog_result(source_finger: str | None) -> None:
            """Handle the result from the copy dialog."""
            if source_finger:
                self._perform_copy(source_finger, current_finger, relative_path)

        dialog = CopyFromFingerDialog(current_finger, available_fingers, copy_type, finger_values)
        self.app.push_screen(dialog, handle_dialog_result)

    def _perform_copy(self, source_finger: str, target_finger: str, relative_path: list[str]) -> None:
        """Perform the actual copy operation."""
        hands_obj = self.config.hands
        source_finger_obj = getattr(hands_obj, source_finger)
        target_finger_obj = getattr(hands_obj, target_finger)

        # Navigate to the source and target objects
        source_obj = source_finger_obj
        target_obj = target_finger_obj

        if relative_path:
            # Navigate to the specific path
            for part in relative_path[:-1]:
                source_obj = getattr(source_obj, part)
                target_obj = getattr(target_obj, part)

        if not relative_path:
            # Copy entire finger configuration
            self._copy_model_fields(source_finger_obj, target_finger_obj, ["hands", target_finger])
            self.app.notify(
                f"Copied entire configuration from {source_finger} to {target_finger}", severity="information"
            )
        else:
            # Copy specific value or section
            field_name = relative_path[-1]
            source_value = getattr(source_obj, field_name)

            if isinstance(source_value, BaseModel):
                # Copy a section
                target_section = getattr(target_obj, field_name)
                self._copy_model_fields(source_value, target_section, ["hands", target_finger] + relative_path)
                self.app.notify(
                    f"Copied '{field_name}' section from {source_finger} to {target_finger}", severity="information"
                )
            else:
                # Copy a single value
                setattr(target_obj, field_name, source_value)

                # Track the modification
                full_path = ["hands", target_finger] + relative_path
                path_str = ".".join(full_path)
                self._current_values[path_str] = source_value

                # Update the tree display
                self._update_tree_for_path(full_path)

                self.app.notify(
                    f"Copied '{field_name}' value from {source_finger} to {target_finger}", severity="information"
                )

        # Update the current node display
        self.watch_current_node(self.current_node)

    def _copy_model_fields(self, source: BaseModel, target: BaseModel, base_path: list[str]) -> None:
        """Recursively copy all fields from source to target model."""
        for field_name in source.model_fields:
            source_value = getattr(source, field_name)

            if isinstance(source_value, BaseModel):
                # Recursively copy nested models
                target_value = getattr(target, field_name)
                self._copy_model_fields(source_value, target_value, base_path + [field_name])
            else:
                # Copy the value
                setattr(target, field_name, source_value)

                # Track the modification
                path_str = ".".join(base_path + [field_name])
                self._current_values[path_str] = source_value

                # Update the tree display for this specific path
                self._update_tree_for_path(base_path + [field_name])

    def _update_tree_for_path(self, path: list[str]) -> None:
        """Update tree nodes for a specific path."""

        # Find the node in the tree that corresponds to this path
        def find_node_by_path(
            node: TreeNode[dict[str, Any]], target_path: list[str]
        ) -> TreeNode[dict[str, Any]] | None:
            if node.data and node.data.get("path") == target_path:
                return node
            for child in node.children:
                result = find_node_by_path(child, target_path)
                if result:
                    return result
            return None

        # Start from the tree root
        tree_root = self.config_tree.root
        node = find_node_by_path(tree_root, path)
        if node:
            self.config_tree.update_node_display(node)


class TweakScreen(Screen[None]):
    """Main screen for the tweak interface."""

    BINDINGS = [
        Binding("f2", "edit_value", "Edit Value"),
        Binding("f3", "reset_default", "Reset Default"),
        Binding("f4", "reset_initial", "Reset Initial"),
        Binding("f5", "copy_from_finger", "Copy from Finger"),
        Binding("f10", "save_config", "Save Config"),
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit"),
    ]

    def __init__(
        self, config: Config, config_path: Path | None = None, stop_event: threading.Event | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.config_path = config_path or DEFAULT_USER_CONFIG_PATH
        self.stop_event = stop_event

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="tree-container"):
                self.config_tree = ConfigTree(self.config)
                yield self.config_tree
            with Vertical(id="editor-container"):
                self.editor = ConfigEditor(self.config, self.config_tree)
                yield self.editor
        yield Footer()

    def on_mount(self) -> None:
        """Set up connections when screen is mounted."""
        self.config_tree.set_editor(self.editor)

        # Wait for the editor to be mounted before updating bindings
        def update_bindings() -> None:
            if self.config_tree.cursor_node:
                self._update_footer_bindings(self.config_tree.cursor_node)

        # Use call_after_refresh to ensure all widgets are mounted
        self.call_after_refresh(update_bindings)

    @on(Tree.NodeSelected)
    def on_tree_node_selected(self, event: Tree.NodeSelected[dict[str, Any]]) -> None:
        """Handle tree node selection."""
        self.editor.current_node = event.node
        # The double-click handling is done in ConfigTree.on_tree_node_selected

    @on(Tree.NodeHighlighted)
    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[dict[str, Any]]) -> None:
        """Handle tree node highlighting (cursor movement)."""
        self.editor.current_node = event.node
        self._update_footer_bindings(event.node)

    def action_decrease_value(self) -> None:
        """Decrease value with normal step."""
        self.editor.adjust_numeric_value(-1)

    def action_increase_value(self) -> None:
        """Increase value with normal step."""
        self.editor.adjust_numeric_value(1)

    def action_decrease_value_large(self) -> None:
        """Decrease value with large step."""
        self.editor.adjust_numeric_value(-1, use_shift=True)

    def action_increase_value_large(self) -> None:
        """Increase value with large step."""
        self.editor.adjust_numeric_value(1, use_shift=True)

    def action_decrease_value_small(self) -> None:
        """Decrease value with small step."""
        self.editor.adjust_numeric_value(-1, use_ctrl=True)

    def action_increase_value_small(self) -> None:
        """Increase value with small step."""
        self.editor.adjust_numeric_value(1, use_ctrl=True)

    def action_toggle_value(self) -> None:
        """Toggle boolean value."""
        self.editor.toggle_bool_value()

    def action_save_config(self) -> None:
        """Save current config to file."""
        try:
            self.config.save(self.config_path)
            self.notify(f"Config saved to {self.config_path}", severity="information")
            # Mark changes as saved
            self.editor.save_changes()
        except Exception as e:
            self.notify(f"Error saving config: {e}", severity="error")

    def action_quit(self) -> None:
        """Quit the application and signal OpenCV thread to stop."""
        # Check for unsaved changes
        if hasattr(self, "editor") and self.editor.has_unsaved_changes():
            # Show confirmation dialog
            def handle_quit_confirmation(action: str | None) -> None:
                if action == "save":
                    # Save and then quit
                    self.action_save_config()
                    if self.stop_event:
                        self.stop_event.set()
                    self.app.exit()
                elif action == "quit":
                    # Quit without saving
                    if self.stop_event:
                        self.stop_event.set()
                    self.app.exit()
                # If action is "cancel" or None, do nothing

            dialog = ConfirmQuitDialog()
            self.app.push_screen(dialog, handle_quit_confirmation)
        else:
            # No unsaved changes, quit directly
            if self.stop_event:
                self.stop_event.set()
            self.app.exit()

    def check_action(self, action: str, parameters: tuple[Any, ...]) -> bool | None:
        """Check if an action should be enabled."""
        if action in ("edit_value", "reset_default", "reset_initial"):
            # These actions are only available when editing a value
            node = self.editor.current_node if hasattr(self, "editor") else None
            if node and node.data:
                data = node.data
                # Check if node is editable (has field_name and is not a BaseModel)
                if data.get("field_name") and not isinstance(data.get("obj"), BaseModel):
                    if action == "edit_value":
                        # F2 is not active for boolean values
                        value = data.get("obj")
                        return not isinstance(value, bool)
                    return True
            return False
        elif action == "copy_from_finger":
            # This action is available when on a finger node or finger sub-node
            node = self.editor.current_node if hasattr(self, "editor") else None
            if node and node.data:
                path = node.data.get("path", [])
                # Check if we're in a finger's configuration
                if len(path) >= 2 and path[0] == "hands" and path[1] in ["thumb", "index", "middle", "ring", "pinky"]:
                    return True
            return False
        return True

    def _update_footer_bindings(self, _: TreeNode[dict[str, Any]] | None) -> None:
        """Update footer bindings based on the selected node."""
        # Force footer refresh
        self.refresh_bindings()

    def action_edit_value(self) -> None:
        """Edit current value with dialog."""
        self.editor.open_value_input_dialog()

    def action_reset_default(self) -> None:
        """Reset to default value."""
        self.editor.reset_to_default()

    def action_reset_initial(self) -> None:
        """Reset to initial value."""
        self.editor.reset_to_initial()

    def action_copy_from_finger(self) -> None:
        """Copy configuration from another finger."""
        self.editor.copy_from_finger()


class TweakApp(App[None]):
    """Textual app for tweaking configuration."""

    CSS = """
    #tree-container {
        width: 50%;
        border-right: solid $primary;
    }

    #editor-container {
        width: 50%;
        padding: 1 2;
    }

    ConfigTree {
        width: 100%;
        height: 100%;
    }

    #editor-title {
        text-style: bold;
        margin-bottom: 1;
    }

    #editor-description {
        margin-bottom: 1;
    }

    #editor-type {
        color: $text-muted;
    }

    #editor-default {
        color: $text-muted;
    }

    #editor-initial {
        color: $text-muted;
    }

    #editor-current {
        text-style: bold;
        margin-top: 1;
    }

    #editor-help {
        margin-top: 2;
        color: $text-muted;
    }
    """

    def __init__(
        self, config: Config, config_path: Path | None = None, stop_event: threading.Event | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.config_path = config_path
        self.stop_event = stop_event

    def on_mount(self) -> None:
        """Set up the app when it starts."""
        self.push_screen(TweakScreen(self.config, self.config_path, self.stop_event))
        # Check stop_event periodically
        if self.stop_event:
            self.set_interval(0.1, self._check_stop_event)

    def _check_stop_event(self) -> None:
        """Check if stop_event is set and exit if so."""
        if self.stop_event and self.stop_event.is_set():
            self.exit()


def run_opencv_thread(
    camera_info: CameraInfo,
    config: Config,
    mirror: bool,
    desired_size: int,
    stop_event: threading.Event,
    recognizer_ready: threading.Event,
) -> None:
    """Run OpenCV capture and recognition in a separate thread."""
    import os
    from typing import cast

    import cv2  # type: ignore[import-untyped]

    from ..drawing import draw_hands_marks_and_info
    from ..recognizer import Recognizer
    from .common import init_camera_capture

    # Initialize global hands instance
    hands = Hands(config=config)

    cap, window_name = init_camera_capture(camera_info, show_preview=True, desired_size=desired_size)
    if cap is None:
        recognizer_ready.set()  # Signal ready even on failure to avoid deadlock
        return

    print("Loading gesture recognizer model...")

    try:
        # Create gesture recognizer with context manager
        with Recognizer(
            os.getenv("GESTURE_RECOGNIZER_MODEL_PATH", "").strip() or "gesture_recognizer.task", mirroring=mirror
        ) as recognizer:
            print("Gesture recognizer loaded successfully")
            recognizer_ready.set()  # Signal that recognizer is ready
            print("Press 'q' or ESC to quit, or use F10 to save config")

            for frame, stream_info, _ in recognizer.handle_opencv_capture(cap, hands):
                # Check if we should stop
                if stop_event.is_set():
                    break

                frame = draw_hands_marks_and_info(hands, stream_info, frame)
                cv2.imshow(cast(str, window_name), frame)

                # Check for key press (non-blocking)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    stop_event.set()
                    break

                # Check if window was closed
                try:
                    if cv2.getWindowProperty(cast(str, window_name), cv2.WND_PROP_VISIBLE) < 1:
                        stop_event.set()
                        break
                except cv2.error:
                    # Window was closed
                    stop_event.set()
                    break
    except Exception as e:
        print(f"\nError in OpenCV thread: {e}")
        recognizer_ready.set()  # Signal ready even on error to avoid deadlock
    finally:
        cap.release()
        cv2.destroyAllWindows()


@app.command(name="tweak")
def tweak_cmd(
    camera: str | None = typer.Option(None, "--camera", "--cam", help="Camera name filter (case insensitive)"),
    mirror: bool | None = typer.Option(None, "--mirror/--no-mirror", help="Mirror the video output"),
    size: int | None = typer.Option(None, "--size", "-s", help="Maximum dimension of the camera capture"),
    config_path: Path | None = typer.Option(  # noqa: B008
        None, "--config", "-c", help=f"Path to config file. Default: {DEFAULT_USER_CONFIG_PATH}"
    ),
) -> None:
    """Tweak gesture recognition configuration in real-time.

    This command provides an interactive interface to adjust configuration values
    while seeing the immediate effect on gesture detection.
    """
    # Load configuration
    config = Config.load(config_path)

    # Use config values as defaults, but CLI options take precedence
    final_camera = camera if camera is not None else config.cli.camera
    final_mirror = mirror if mirror is not None else config.cli.mirror
    final_size = size if size is not None else config.cli.size

    # Select camera
    selected = pick_camera(final_camera)
    if not selected:
        print("\nNo camera selected.")
        return

    print(f"\nSelected: {selected}")

    # Create events for thread coordination
    stop_event = threading.Event()
    recognizer_ready = threading.Event()

    # Start OpenCV thread
    opencv_thread = threading.Thread(
        target=run_opencv_thread,
        args=(selected, config, final_mirror, final_size, stop_event, recognizer_ready),
        daemon=True,
    )
    opencv_thread.start()

    # Wait for recognizer to be ready before starting Textual
    recognizer_ready.wait()

    # Run Textual app in main thread
    tweak_app = TweakApp(config, config_path, stop_event)
    tweak_app.run()

    # Signal OpenCV thread to stop if not already
    stop_event.set()
    opencv_thread.join(timeout=2.0)
