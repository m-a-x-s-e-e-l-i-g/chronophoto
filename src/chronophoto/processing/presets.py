from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chronophoto import __version__
from chronophoto.processing.compositor import ComposeSettings
from chronophoto.processing.effects import EffectKeyframe, EffectTrack
from chronophoto.processing.exports import ExportKind

PRESET_SCHEMA = "com.chronophoto.complete-preset"
PRESET_VERSION = 1
PRESET_SUFFIX = ".chronophoto-preset.json"
ALIGNMENT_MODES = {"off", "translation"}
PHOTO_ORDER_MODES = {"automatic", "capture_time", "filename", "input"}
EXPORT_KINDS: set[str] = {
    "composite",
    "combined_poses",
    "individual_poses",
    "background",
    "trail_video",
    "resolve_timeline",
}


@dataclass(slots=True, frozen=True)
class ChronophotoPreset:
    """A source-independent snapshot of every creative and output control."""

    name: str
    settings: ComposeSettings
    pose_count: int
    use_all_frames: bool
    trail_duration_ms: int
    alignment: str
    photo_order: str
    outputs: tuple[ExportKind, ...]

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name or len(name) > 160:
            raise ValueError("preset name must contain 1 to 160 characters")
        object.__setattr__(self, "name", name)
        if not isinstance(self.settings, ComposeSettings):
            raise ValueError("preset settings must be ComposeSettings")
        if not 1 <= self.settings.threshold <= 90:
            raise ValueError("preset mask threshold must be between 1 and 90")
        if not 0 <= self.settings.feather <= 20:
            raise ValueError("preset edge feather must be between 0 and 20")
        if self.settings.background not in {"median", "first", "last"}:
            raise ValueError(f"unsupported preset background: {self.settings.background}")
        if isinstance(self.pose_count, bool) or not 2 <= self.pose_count <= 1_000_000:
            raise ValueError("preset pose count must be between 2 and 1000000")
        if not isinstance(self.use_all_frames, bool):
            raise ValueError("preset all-frames setting must be true or false")
        if (
            isinstance(self.trail_duration_ms, bool)
            or not 0 <= self.trail_duration_ms <= 86_400_000
        ):
            raise ValueError("preset trail duration must be between 0 and 86400000 ms")
        if self.alignment not in ALIGNMENT_MODES:
            raise ValueError(f"unsupported preset alignment: {self.alignment}")
        if self.photo_order not in PHOTO_ORDER_MODES:
            raise ValueError(f"unsupported preset photo order: {self.photo_order}")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError("preset output selections must be unique")
        if any(output not in EXPORT_KINDS for output in self.outputs):
            raise ValueError("preset contains an unsupported output selection")
        exclusive_outputs = {"resolve_timeline", "trail_video"}.intersection(self.outputs)
        if exclusive_outputs and len(self.outputs) > 1:
            selected = next(iter(exclusive_outputs)).replace("_", " ")
            raise ValueError(f"{selected} cannot be combined with other preset outputs")
        for tracks in (
            self.settings.trail_effect_tracks,
            self.settings.background_effect_tracks,
        ):
            single_kinds = [track.kind for track in tracks if track.kind != "blend_mode"]
            if len(single_kinds) != len(set(single_kinds)):
                raise ValueError("preset effect stacks may repeat only blend modes")


def preset_to_dict(preset: ChronophotoPreset) -> dict[str, object]:
    settings = preset.settings
    return {
        "schema": PRESET_SCHEMA,
        "version": PRESET_VERSION,
        "created_with": __version__,
        "name": preset.name,
        "composition": {
            "pose_count": preset.pose_count,
            "use_all_frames": preset.use_all_frames,
            "trail_duration_ms": preset.trail_duration_ms,
            "alignment": preset.alignment,
            "photo_order": preset.photo_order,
            "mask": {
                "threshold": settings.threshold,
                "feather_px": settings.feather,
            },
            "background": settings.background,
            "overlap": settings.overlap,
            "trail_style": settings.trail_style,
            "smear_style": settings.smear_style,
        },
        "effects": {
            "trail": [_effect_to_dict(track) for track in settings.trail_effect_tracks],
            "background": [_effect_to_dict(track) for track in settings.background_effect_tracks],
        },
        "outputs": list(preset.outputs),
    }


def preset_from_dict(value: object) -> ChronophotoPreset:
    root = _mapping(value, "preset")
    schema = _string(root.get("schema"), "schema")
    if schema != PRESET_SCHEMA:
        raise ValueError("not a Chronophoto complete preset")
    version = _integer(root.get("version"), "version")
    if version != PRESET_VERSION:
        raise ValueError(
            f"unsupported preset version {version}; this app supports version {PRESET_VERSION}"
        )

    composition = _mapping(root.get("composition"), "composition")
    mask = _mapping(composition.get("mask"), "composition.mask")
    effects = _mapping(root.get("effects"), "effects")
    trail_effects = _effect_list(effects.get("trail"), "effects.trail")
    background_effects = _effect_list(effects.get("background"), "effects.background")
    settings = ComposeSettings(
        threshold=_integer(mask.get("threshold"), "composition.mask.threshold"),
        feather=_integer(mask.get("feather_px"), "composition.mask.feather_px"),
        overlap=_string(composition.get("overlap"), "composition.overlap"),
        trail_style=_string(composition.get("trail_style"), "composition.trail_style"),
        smear_style=_string(composition.get("smear_style"), "composition.smear_style"),
        background=_string(composition.get("background"), "composition.background"),
        trail_effect_tracks=trail_effects,
        background_effect_tracks=background_effects,
    )
    outputs_value = _list(root.get("outputs"), "outputs")
    outputs = tuple(
        _string(output, f"outputs[{index}]") for index, output in enumerate(outputs_value)
    )
    return ChronophotoPreset(
        name=_string(root.get("name"), "name"),
        settings=settings,
        pose_count=_integer(composition.get("pose_count"), "composition.pose_count"),
        use_all_frames=_boolean(composition.get("use_all_frames"), "composition.use_all_frames"),
        trail_duration_ms=_integer(
            composition.get("trail_duration_ms"), "composition.trail_duration_ms"
        ),
        alignment=_string(composition.get("alignment"), "composition.alignment"),
        photo_order=_string(composition.get("photo_order"), "composition.photo_order"),
        outputs=outputs,  # type: ignore[arg-type]
    )


def write_preset(path: str | Path, preset: ChronophotoPreset) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}-writing-",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(preset_to_dict(preset), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def read_preset(path: str | Path) -> ChronophotoPreset:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"preset is not valid JSON: {exc.msg}") from exc
    return preset_from_dict(value)


def _effect_to_dict(track: EffectTrack) -> dict[str, object]:
    return {
        "kind": track.kind,
        "enabled": track.enabled,
        "amount": track.amount,
        "option": track.option,
        "timing_basis": track.timing_basis,
        "keyframes": [
            {"progress": point.progress, "value": point.value} for point in track.keyframes
        ],
    }


def _effect_list(value: object, field: str) -> tuple[EffectTrack, ...]:
    items = _list(value, field)
    return tuple(_effect_from_dict(item, f"{field}[{index}]") for index, item in enumerate(items))


def _effect_from_dict(value: object, field: str) -> EffectTrack:
    item = _mapping(value, field)
    keyframe_values = _list(item.get("keyframes"), f"{field}.keyframes")
    keyframes = tuple(
        EffectKeyframe(
            _number(_mapping(point, f"{field}.keyframes[{index}]").get("progress"), "progress"),
            _number(_mapping(point, f"{field}.keyframes[{index}]").get("value"), "value"),
        )
        for index, point in enumerate(keyframe_values)
    )
    return EffectTrack(
        kind=_string(item.get("kind"), f"{field}.kind"),
        keyframes=keyframes,
        enabled=_boolean(item.get("enabled"), f"{field}.enabled"),
        amount=_number(item.get("amount"), f"{field}.amount"),
        option=_string(item.get("option"), f"{field}.option"),
        timing_basis=_string(item.get("timing_basis"), f"{field}.timing_basis"),
    )


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _list(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a list")
    return value


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be true or false")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a number")
    return float(value)
