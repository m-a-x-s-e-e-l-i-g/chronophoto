from __future__ import annotations

from pathlib import Path


class DocumentController:
    """Own portable-look identity separately from source and view widgets."""

    def __init__(self) -> None:
        self.preset_path: Path | None = None
        self.preset_display_name = "Custom settings"

    def set_preset(self, path: Path | None, name: str = "Custom settings") -> None:
        self.preset_path = path
        self.preset_display_name = name

    def visible_name(self, *, modified: bool = False) -> str:
        suffix = " · modified" if modified else ""
        return f"{self.preset_display_name}{suffix}".upper()

    def tooltip(self, *, modified: bool = False) -> str:
        if self.preset_path is None:
            return "Presets contain composition controls, complete effect stacks, and outputs"
        suffix = "\nModified since it was loaded or saved" if modified else ""
        return f"{self.preset_path}{suffix}"
