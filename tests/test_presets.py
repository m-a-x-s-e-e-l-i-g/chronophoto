from __future__ import annotations

import json
from pathlib import Path

import pytest

from chronophoto.processing import (
    PRESET_SCHEMA,
    PRESET_VERSION,
    ChronophotoPreset,
    ComposeSettings,
    EffectKeyframe,
    EffectTrack,
    preset_from_dict,
    preset_to_dict,
    read_preset,
    write_preset,
)


def _complete_preset() -> ChronophotoPreset:
    return ChronophotoPreset(
        name="Silver rush",
        settings=ComposeSettings(
            threshold=29,
            feather=8,
            overlap="oldest",
            smear_style="photographic",
            background="last",
            trail_effect_tracks=(
                EffectTrack(
                    "opacity",
                    (
                        EffectKeyframe(0.0, 15.0),
                        EffectKeyframe(0.4, 92.0),
                        EffectKeyframe(1.0, 55.0),
                    ),
                    enabled=False,
                    timing_basis="trail",
                ),
                EffectTrack(
                    "blend_mode",
                    (EffectKeyframe(0.0, 100.0), EffectKeyframe(1.0, 100.0)),
                    option="screen",
                ),
            ),
            background_effect_tracks=(
                EffectTrack(
                    "saturation",
                    (EffectKeyframe(0.0, 31.0), EffectKeyframe(1.0, 31.0)),
                ),
            ),
        ),
        pose_count=17,
        use_all_frames=False,
        trail_duration_ms=2350,
        alignment="translation",
        photo_order="capture_time",
        outputs=("composite", "individual_poses", "background"),
    )


def test_complete_preset_round_trips_all_settings_effects_and_outputs(tmp_path: Path) -> None:
    preset = _complete_preset()
    destination = tmp_path / "Silver rush.chronophoto-preset.json"

    assert preset_from_dict(preset_to_dict(preset)) == preset
    assert write_preset(destination, preset) == destination
    assert read_preset(destination) == preset
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema"] == PRESET_SCHEMA
    assert payload["version"] == PRESET_VERSION
    assert payload["effects"]["trail"][0]["keyframes"][1] == {
        "progress": 0.4,
        "value": 92.0,
    }
    assert payload["effects"]["trail"][0]["enabled"] is False
    assert payload["effects"]["trail"][0]["timing_basis"] == "trail"
    assert not list(tmp_path.glob(".*-writing-*.tmp"))


def test_preset_rejects_wrong_schema_version_and_invalid_ui_values() -> None:
    payload = preset_to_dict(_complete_preset())
    payload["version"] = PRESET_VERSION + 1
    with pytest.raises(ValueError, match="unsupported preset version"):
        preset_from_dict(payload)

    payload = preset_to_dict(_complete_preset())
    payload["composition"]["mask"]["threshold"] = 120  # type: ignore[index]
    with pytest.raises(ValueError, match="threshold"):
        preset_from_dict(payload)

    payload = preset_to_dict(_complete_preset())
    payload["outputs"] = ["resolve_timeline", "composite"]
    with pytest.raises(ValueError, match="cannot be combined"):
        preset_from_dict(payload)

    payload = preset_to_dict(_complete_preset())
    payload["outputs"] = ["trail_video", "background"]
    with pytest.raises(ValueError, match="cannot be combined"):
        preset_from_dict(payload)


def test_read_preset_reports_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.chronophoto-preset.json"
    path.write_text("{ definitely not JSON", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        read_preset(path)
