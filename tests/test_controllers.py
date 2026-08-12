from pathlib import Path

from chronophoto.processing import ComposeSettings
from chronophoto.ui.controllers import (
    DocumentController,
    ExportRecipeController,
    PreviewController,
    SourceController,
    SourceState,
    should_expand_advanced,
)


def test_export_recipe_keeps_video_recipes_exclusive() -> None:
    controller = ExportRecipeController()

    assert controller.normalized_after_check(("composite", "trail_video"), "trail_video") == (
        "trail_video",
    )
    assert controller.normalized_after_check(("resolve_timeline", "background"), "background") == (
        "background",
    )


def test_document_preview_and_source_state_are_view_independent(tmp_path: Path) -> None:
    document = DocumentController()
    preset = tmp_path / "race.chronophoto-preset.json"
    document.set_preset(preset, "Race day")
    assert document.visible_name(modified=True) == "RACE DAY · MODIFIED"
    assert str(preset) in document.tooltip(modified=False)

    preview = PreviewController()
    assert preview.mark_dirty() == 1
    assert preview.dirty
    preview.mark_current()
    assert not preview.dirty

    source = SourceController()
    source.replace(SourceState("photos", [tmp_path / "one.jpg"]))
    assert source.paths == (tmp_path / "one.jpg",)
    assert not source.is_video


def test_inspector_expands_only_for_non_default_settings() -> None:
    assert not should_expand_advanced(
        ComposeSettings(background="median"),
        alignment="off",
        photo_order="automatic",
        source_kind="video",
    )
    assert should_expand_advanced(
        ComposeSettings(threshold=30, background="median"),
        alignment="off",
        photo_order="automatic",
        source_kind="video",
    )
