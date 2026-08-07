from __future__ import annotations

import os
import sys
from pathlib import Path
from time import monotonic, sleep

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402
from PySide6.QtCore import QMimeData, QPoint, QPointF, QSignalBlocker, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QIcon, QImage, QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QSlider  # noqa: E402

from chronophoto import __version__  # noqa: E402
from chronophoto.app import application_stylesheet, main  # noqa: E402
from chronophoto.processing import BLEND_MODES, EffectKeyframe  # noqa: E402
from chronophoto.processing.sources import MediaSequence, VideoInfo  # noqa: E402
from chronophoto.ui.effects import EffectKeyframeGraph  # noqa: E402
from chronophoto.ui.widgets import (  # noqa: E402
    PreviewCanvas,
    ScrollSafeComboBox,
    ScrollSafeSlider,
)
from chronophoto.ui.window import ChronophotoWindow, SourceState, TaskWorker  # noqa: E402
from chronophoto.updates import GITHUB_REPOSITORY_URL, UpdateResult  # noqa: E402


def test_version_cli_does_not_start_the_event_loop(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["chronophoto", "--version"])

    assert main() == 0
    assert capsys.readouterr().out.strip() == f"Chronophoto {__version__}"


def test_application_icon_asset_is_valid() -> None:
    _app = QApplication.instance() or QApplication([])
    icon_path = Path(__file__).parents[1] / "src/chronophoto/assets/chronophoto-icon.png"
    icon = QIcon(str(icon_path))

    assert icon_path.is_file()
    assert not icon.isNull()


def test_terminal_font_is_applied_to_the_stylesheet() -> None:
    stylesheet = application_stylesheet("Example Mono")

    assert 'font-family: "Example Mono"' in stylesheet
    assert 'QLabel#updateStatus[state="available"] { color: #ff6b6b' in stylesheet
    assert 'QLabel#updateStatus[state="current"] { color: #70d98b' in stylesheet
    assert "__APP_MONO_FONT__" not in stylesheet
    assert "__CHECKMARK_ICON__" not in stylesheet
    assert "check.svg" in stylesheet


def test_core_custom_controls_are_keyboard_accessible() -> None:
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(application_stylesheet("Consolas"))
    window = ChronophotoWindow()

    assert window.drop_surface.focusPolicy() != Qt.FocusPolicy.NoFocus
    assert window.range_slider.focusPolicy() != Qt.FocusPolicy.NoFocus
    assert window.drop_surface.accessibleName()
    assert window.range_slider.accessibleName()

    window.close()


def test_preview_is_automatic_without_a_manual_render_action() -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()

    assert not hasattr(window, "preview_button")
    assert "Ctrl+R" not in {action.shortcut().toString() for action in window.actions()}

    window.close()


def test_header_shows_version_github_and_update_state() -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()

    assert window.version_number.text() == f"v{__version__}"
    assert window.github_button.text() == "GITHUB"
    window._apply_update_result(UpdateResult("0.2.0", "https://example.com/release", True))
    assert window.update_status.text() == "UPDATE v0.2.0 AVAILABLE"
    assert window.update_status.property("state") == "available"
    assert window._github_target_url == "https://example.com/release"
    window._apply_update_result(UpdateResult(__version__, "https://example.com/release", False))
    assert window.update_status.text() == "UP TO DATE"
    assert window.update_status.property("state") == "current"
    assert window._github_target_url == GITHUB_REPOSITORY_URL

    window.close()


def test_workspace_still_has_room_at_minimum_window_size() -> None:
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(application_stylesheet("Consolas"))
    window = ChronophotoWindow()
    window.resize(960, 680)
    window.show()
    app.processEvents()

    assert window.preview_canvas.width() >= 400
    assert window.body_splitter.sizes()[1] >= 400

    window.close()


def test_busy_state_blocks_source_replacement_but_keeps_navigation_live() -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    window.source = SourceState("photos", [])
    window._set_busy(True)

    assert not window.video_button.isEnabled()
    assert not window.photos_button.isEnabled()
    assert window.frame_list.isEnabled()
    assert window.pose_scrubber.isEnabled()
    assert not window.cancel_button.isHidden()

    window._set_busy(False)
    window.close()


def test_pose_count_is_a_non_tracking_slider_with_all_frames_mode() -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    window.source = SourceState("video", [])
    window._set_loaded_state(True)

    assert isinstance(window.pose_count, QSlider)
    assert not window.pose_count.hasTracking()
    assert window.all_frames.isChecked()
    assert not window.pose_count.isEnabled()
    assert window.pose_control.property("allFramesActive") is True

    window._reset_controls()
    window.preview_debounce.stop()
    assert window.all_frames.isChecked()
    assert not window.pose_count.isEnabled()

    window.close()


def test_mask_controls_use_requested_defaults_and_reset_to_them() -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()

    assert window.threshold.value() == 17
    assert window.threshold_value.text() == "17"
    assert window.feather.value() == 1
    assert window.feather_value.text() == "1 px"

    window.threshold.setValue(40)
    window.feather.setValue(12)
    window._reset_controls()
    window.preview_debounce.stop()

    assert window.threshold.value() == 17
    assert window.threshold_value.text() == "17"
    assert window.feather.value() == 1
    assert window.feather_value.text() == "1 px"

    window.close()


def test_trail_style_is_disabled_and_smear_defaults_to_none() -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()

    modes = [window.trail_style.itemData(index) for index in range(window.trail_style.count())]

    assert modes == ["solid"]
    assert not window.trail_style.isEnabled()
    assert window.trail_style.toolTip() == "Work in progress"
    assert window.trail_style_control.toolTip() == "Work in progress"
    window.source = SourceState("video", [])
    window._set_loaded_state(True)
    assert not window.trail_style.isEnabled()
    assert window.smear_style.isEnabled()
    modes = [window.smear_style.itemData(index) for index in range(window.smear_style.count())]
    assert modes == ["none", "photographic", "dense_clones"]
    assert window._settings_snapshot().smear_style == "none"
    window.smear_style.setCurrentIndex(window.smear_style.findData("dense_clones"))
    assert window._settings_snapshot().smear_style == "dense_clones"
    window.close()


def test_trail_effect_timeline_adds_independent_neutral_tracks() -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()

    assert window.effect_timeline.tracks() == ()
    assert window.effect_timeline.isHidden()
    window.source = SourceState("video", [])
    window._set_loaded_state(True)
    assert window.effect_timeline.title_label.text() == "TRAIL EFFECTS"
    for kind in (
        "opacity",
        "blend_mode",
        "saturation",
        "blur",
        "jpeg_quality",
        "stippling",
        "dithering",
        "halftone",
    ):
        window.effect_timeline.add_effect(kind)

    tracks = window.effect_timeline.tracks()
    assert [track.kind for track in tracks] == [
        "opacity",
        "blend_mode",
        "saturation",
        "blur",
        "jpeg_quality",
        "stippling",
        "dithering",
        "halftone",
    ]
    assert [track.value_at(0.5) for track in tracks] == [100, 100, 100, 0, 100, 0, 0, 0]
    assert {track.timing_basis for track in tracks} == {"movement"}
    opacity_lane = window.effect_timeline._lanes[0]
    assert opacity_lane.timing_combo.accessibleName() == "Effect timing"
    assert [
        opacity_lane.timing_combo.itemData(index)
        for index in range(opacity_lane.timing_combo.count())
    ] == ["movement", "trail"]
    opacity_lane.timing_combo.setCurrentIndex(opacity_lane.timing_combo.findData("trail"))
    assert opacity_lane.track.timing_basis == "trail"
    assert window._settings_snapshot().trail_effect_tracks == window.effect_timeline.tracks()
    window.close()


def test_background_effects_use_constant_values_and_a_separate_scope() -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    panel = window.background_effect_timeline

    assert panel.title_label.text() == "BACKGROUND EFFECTS"
    assert panel.summary.text() == "NO EFFECTS · CLEAN PLATE"
    assert not panel.is_expanded
    lane = panel.add_effect("blur")

    assert panel.is_expanded
    assert not window.trail_effect_timeline.is_expanded
    assert lane.graph.isHidden()
    assert lane.preset.isHidden()
    assert lane.position_spin.isHidden()
    assert lane.timing_combo.isHidden()
    assert lane.value_spin.isVisibleTo(lane)
    lane.value_spin.setValue(64)
    lane._constant_value_committed()

    track = lane.track
    assert [point.value for point in track.keyframes] == [64, 64]
    assert window._settings_snapshot().background_effect_tracks == (track,)
    window.close()


def test_background_effects_offer_the_same_effect_types_as_trail_effects() -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    with QSignalBlocker(window.background_effect_timeline):
        for kind in (
            "opacity",
            "blend_mode",
            "saturation",
            "blur",
            "jpeg_quality",
            "stippling",
            "dithering",
            "halftone",
        ):
            window.background_effect_timeline.add_effect(kind)

    assert [track.kind for track in window.background_effect_timeline.tracks()] == [
        "opacity",
        "blend_mode",
        "saturation",
        "blur",
        "jpeg_quality",
        "stippling",
        "dithering",
        "halftone",
    ]
    window.close()


def test_export_outputs_allow_any_layer_combination() -> None:
    app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    path = Path("clip.mp4")
    window.source = SourceState("video", [path], VideoInfo(path, 3.0, 640, 480, 30.0, 90))
    window._set_loaded_state(True)

    assert window._export_selections() == ("composite",)
    assert window.export_button.text() == "Export composite"
    window.export_options_button.click()
    assert not window.export_options_panel.isHidden()

    window.export_checks["combined_poses"].setChecked(True)
    window.export_checks["individual_poses"].setChecked(True)
    window.export_checks["background"].setChecked(True)
    assert window._export_selections() == (
        "composite",
        "combined_poses",
        "individual_poses",
        "background",
    )
    assert window.export_options_button.text() == "OUTPUTS · 4 SELECTED ▾"
    assert window.export_button.text() == "Export 4 outputs"
    window.resize(960, 680)
    window.show()
    app.processEvents()
    window._update_compact_workspace()
    app.processEvents()
    assert window.preview_canvas.height() >= 180
    assert window.export_button.isVisible()
    assert all(checkbox.isVisible() for checkbox in window.export_checks.values())

    for checkbox in window.export_checks.values():
        checkbox.setChecked(False)
    assert window._export_selections() == ()
    assert not window.export_button.isEnabled()
    assert window.export_options_button.text() == "OUTPUTS · NONE ▾"
    window.close()


def test_motion_trail_controls_are_video_only_and_use_seconds() -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    path = Path("clip.mp4")
    window.source = SourceState("video", [path], VideoInfo(path, 10.0, 640, 480, 30.0, 300))
    window._set_loaded_state(True)
    window.range_slider.set_values(200, 500)

    assert not window.preview_mode_buttons["trail"].isHidden()
    assert not window.trail_duration_control.isHidden()
    assert not window.trail_video_button.isHidden()
    assert window.trail_duration.maximum() == 3_000
    window.trail_duration.setValue(1_400)
    assert window.trail_duration_value.text() == "1.4 s"

    window.source = SourceState("photos", [Path("one.png"), Path("two.png")])
    window._set_loaded_state(True)
    assert window.preview_mode_buttons["trail"].isHidden()
    assert window.trail_duration_control.isHidden()
    assert window.trail_video_button.isHidden()
    window.close()


def test_trail_preview_mode_uses_rendered_video_frames() -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    path = Path("clip.mp4")
    window.source = SourceState("video", [path], VideoInfo(path, 2.0, 40, 24, 4.0, 8))
    window._set_loaded_state(True)
    window.preview_frames = [np.zeros((24, 40, 3), dtype=np.uint8) for _ in range(3)]
    window.preview_trail_frames = [
        np.full((24, 40, 3), index * 40, dtype=np.uint8) for index in range(5)
    ]
    window.preview_trail_timestamps = [0.0, 0.25, 0.5, 0.75, 1.0]

    window._set_preview_mode("trail")
    window.pose_scrubber.setValue(3)

    assert window.pose_scrubber.maximum() == 4
    assert window.preview_canvas._status.startswith("TRAIL · 00:00.75")
    assert window.preview_canvas._image is not None
    assert window.preview_canvas._image.pixelColor(0, 0).red() == 120
    window.close()


def test_motion_video_export_uses_every_frame_and_selected_duration(
    monkeypatch, tmp_path: Path
) -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    path = Path("clip.mp4")
    window.source = SourceState("video", [path], VideoInfo(path, 2.0, 48, 32, 4.0, 8))
    window._set_loaded_state(True)
    window.all_frames.setChecked(False)
    window.pose_count.setValue(2)
    window.trail_duration.setValue(750)
    frames: list[np.ndarray] = []
    for index in range(8):
        frame = np.full((32, 48, 3), (20, 40, 80), dtype=np.uint8)
        frame[10:24, 2 + index * 5 : 10 + index * 5] = (220, 60, 30)
        frames.append(frame)
    sequence = MediaSequence(
        frames,
        [f"00:00.{index}" for index in range(8)],
        (48, 32),
        [index / 4 for index in range(8)],
    )
    pose_counts: list[int | None] = []

    def load_sequence(*args, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        pose_counts.append(args[3])
        return sequence

    monkeypatch.setattr("chronophoto.ui.window.load_video_sequence", load_sequence)
    output = tmp_path / "trail.mp4"
    monkeypatch.setattr(
        "chronophoto.ui.window.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(output), "MP4 video (*.mp4)"),
    )
    exported: dict[str, object] = {}

    def write_video(output_path, source_path, render_frames, timestamps, duration, *args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        exported.update(
            output=Path(output_path),
            source=Path(source_path),
            frame_count=len(render_frames),
            timestamps=list(timestamps),
            duration=duration,
        )
        return Path(output_path)

    monkeypatch.setattr("chronophoto.ui.window.write_motion_trail_video", write_video)

    def run_now(task, on_finished, detail):  # type: ignore[no-untyped-def]
        del detail
        on_finished(task(lambda value, message: None))

    monkeypatch.setattr(window, "_start_task", run_now)
    window.export_motion_video()

    assert pose_counts == [None]
    assert exported["output"] == output
    assert exported["source"] == path
    assert exported["frame_count"] == 8
    assert exported["duration"] == pytest.approx(0.75)
    assert window.status_text.text() == "VIDEO EXPORT COMPLETE"
    window.close()


def test_layer_package_export_runs_once_and_writes_selected_outputs(monkeypatch, tmp_path) -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    path = Path("clip.mp4")
    window.source = SourceState("video", [path], VideoInfo(path, 2.0, 48, 32, 2.0, 4))
    window._set_loaded_state(True)
    window.export_checks["composite"].setChecked(False)
    window.export_checks["combined_poses"].setChecked(True)
    window.export_checks["individual_poses"].setChecked(True)
    window.export_checks["background"].setChecked(True)

    frames: list[np.ndarray] = []
    for index in range(4):
        frame = np.full((32, 48, 3), (20, 40, 80), dtype=np.uint8)
        frame[10:24, 4 + index * 9 : 12 + index * 9] = (220, 60, 30)
        frames.append(frame)
    sequence = MediaSequence(
        frames,
        [f"00:0{index}.00" for index in range(4)],
        (48, 32),
        [index / 2 for index in range(4)],
    )
    monkeypatch.setattr("chronophoto.ui.window.load_video_sequence", lambda *a, **k: sequence)
    monkeypatch.setattr(
        "chronophoto.ui.window.QFileDialog.getExistingDirectory",
        lambda *a, **k: str(tmp_path),
    )

    def run_now(task, on_finished, detail):  # type: ignore[no-untyped-def]
        del detail
        on_finished(task(lambda value, message: None))

    monkeypatch.setattr(window, "_start_task", run_now)
    window.export_composite()

    package = tmp_path / "clip-chronophoto-layers"
    assert (package / "poses.png").is_file()
    assert (package / "background.png").is_file()
    assert len(list((package / "poses").glob("pose-*.png"))) == 4
    assert not (package / "composite.png").exists()
    assert window._last_export_path == package
    assert window.status_text.text() == "EXPORT COMPLETE"

    window.export_checks["individual_poses"].setChecked(False)
    window.export_checks["background"].setChecked(False)
    single_pose_path = tmp_path / "only-poses.png"
    monkeypatch.setattr(
        "chronophoto.ui.window.QFileDialog.getSaveFileName",
        lambda *a, **k: (str(single_pose_path), "PNG image (*.png)"),
    )
    window.export_composite()
    assert single_pose_path.is_file()
    assert Image.open(single_pose_path).mode == "RGBA"
    assert window._last_export_path == single_pose_path
    window.close()


def test_effect_lane_presets_and_bypass_preserve_keyframes() -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    lane = window.effect_timeline.add_effect("opacity")
    lane.preset.setCurrentIndex(lane.preset.findData("rise_fall"))

    assert [point.value for point in lane.track.keyframes] == [0, 100, 0]
    lane.enabled_box.setChecked(False)
    assert not lane.track.enabled
    assert [point.value for point in lane.track.keyframes] == [0, 100, 0]
    assert window.effect_timeline.summary.text().startswith("0 ACTIVE / 1 TRACKS")
    lane.enabled_box.setChecked(True)
    assert lane.track.enabled
    window.close()


def test_blend_mode_lane_exposes_all_modes_and_commits_selection() -> None:
    app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    lane = window.effect_timeline.add_effect("blend_mode")
    assert lane.mode_combo is not None
    available = tuple(
        lane.mode_combo.itemData(index)
        for index in range(lane.mode_combo.count())
        if lane.mode_combo.itemData(index) is not None
    )

    assert available == BLEND_MODES
    assert lane.track.option == "multiply"
    assert lane.track.value_at(0.5) == 100.0
    lane.mode_combo.setCurrentIndex(lane.mode_combo.findData("soft_light"))
    assert lane.track.option == "soft_light"

    initial_index = lane.mode_combo.currentIndex()
    event = QWheelEvent(
        QPointF(10, 10),
        QPointF(10, 10),
        QPoint(),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    app.sendEvent(lane.mode_combo, event)
    assert lane.mode_combo.currentIndex() == initial_index
    assert not event.isAccepted()
    window.close()


def test_blend_mode_lanes_can_be_stacked_in_both_effect_scopes() -> None:
    app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    window.resize(1500, 900)
    window.show()
    app.processEvents()

    with QSignalBlocker(window.trail_effect_timeline):
        multiply = window.trail_effect_timeline.add_effect("blend_mode")
        screen = window.trail_effect_timeline.add_effect("blend_mode")
        first_blur = window.trail_effect_timeline.add_effect("blur")
        repeated_blur = window.trail_effect_timeline.add_effect("blur")
        assert screen.mode_combo is not None
        screen.mode_combo.setCurrentIndex(screen.mode_combo.findData("screen"))
    assert multiply is not screen
    assert first_blur is repeated_blur
    assert multiply.mode_combo is not None
    assert [track.option for track in window.trail_effect_timeline.tracks()[:2]] == [
        "multiply",
        "screen",
    ]
    assert multiply.name_label.text().startswith("BLEND")
    assert screen.name_label.text().startswith("BLEND")

    with QSignalBlocker(window.background_effect_timeline):
        background_multiply = window.background_effect_timeline.add_effect("blend_mode")
        background_overlay = window.background_effect_timeline.add_effect("blend_mode")
    assert background_multiply is not background_overlay
    assert [track.kind for track in window.background_effect_timeline.tracks()] == [
        "blend_mode",
        "blend_mode",
    ]
    window.close()


def test_effect_keyframe_graph_emits_live_and_committed_updates() -> None:
    _app = QApplication.instance() or QApplication([])
    graph = EffectKeyframeGraph(
        (
            EffectKeyframe(0.0, 0.0),
            EffectKeyframe(0.5, 100.0),
            EffectKeyframe(1.0, 0.0),
        )
    )
    graph.resize(600, 80)
    graph._selected = 1
    changing: list[tuple[EffectKeyframe, ...]] = []
    committed: list[tuple[EffectKeyframe, ...]] = []
    graph.keyframes_changing.connect(changing.append)
    graph.keyframes_committed.connect(committed.append)

    graph._set_selected_point(QPointF(350, 40), live=True)
    graph._set_selected_point(QPointF(360, 35), live=False)

    assert len(changing) == 1
    assert len(committed) == 1
    assert changing[0][1].progress != 0.5
    assert committed[0][1].value > changing[0][1].value
    graph.close()


def test_effect_lane_supports_exact_numeric_keyframe_editing() -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    lane = window.effect_timeline.add_effect("blur")
    lane.preset.setCurrentIndex(lane.preset.findData("rise_fall"))
    lane.graph._selected = 1
    lane.graph._emit_selection()

    lane.position_spin.setValue(65)
    lane.value_spin.setValue(72)
    lane.graph.commit_selected()

    assert lane.track.keyframes[1].progress == pytest.approx(0.65)
    assert lane.track.keyframes[1].value == 72
    assert lane.position_spin.isEnabled()
    lane.graph._selected = 0
    lane.graph._emit_selection()
    assert not lane.position_spin.isEnabled()
    window.close()


def test_effect_lanes_can_be_reordered_by_drag_position() -> None:
    app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    first = window.effect_timeline.add_effect("blur")
    second = window.effect_timeline.add_effect("halftone")
    window.effect_timeline.resize(800, 240)
    window.effect_timeline.show()
    app.processEvents()

    window.effect_timeline._drag_started(first)
    target = window.effect_timeline.lane_container.mapToGlobal(
        QPoint(20, second.geometry().bottom() + 5)
    )
    window.effect_timeline._drag_moved(first, target)
    window.effect_timeline._drag_finished(first)

    assert [track.kind for track in window.effect_timeline.tracks()] == ["halftone", "blur"]
    window.close()


def test_effect_progress_uses_source_timestamps() -> None:
    progress = ChronophotoWindow._normalized_effect_progress(
        [10.0, 10.5, 12.0, 14.0],
        4,
        10.0,
        14.0,
    )

    assert progress == pytest.approx((0.0, 0.125, 0.5, 1.0))


def test_effect_editor_has_a_usable_compact_window_state() -> None:
    app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    path = Path("clip.mp4")
    window.source = SourceState("video", [path], VideoInfo(path, 10.0, 1920, 1080, 30.0, 300))
    window._set_loaded_state(True)
    with QSignalBlocker(window.effect_timeline):
        lane = window.effect_timeline.add_effect("blend_mode")
    window.resize(960, 680)
    window.show()
    app.processEvents()
    window._update_compact_workspace()
    app.processEvents()

    assert window.workspace_header.isHidden()
    assert window.pose_navigation.isHidden()
    assert window.preview_mode_buttons["composite"].isVisible()
    assert window.preview_canvas.height() >= 180
    assert lane.graph.height() >= 50
    assert lane.more_button.isVisible()
    assert lane.reset_button.isHidden()
    assert lane.mode_combo is not None and lane.mode_combo.isVisible()
    assert lane.width() <= window.effect_timeline.scroll.viewport().width()
    assert window.export_button.isVisible()

    window.resize(1380, 880)
    app.processEvents()
    window._update_compact_workspace()
    assert not window.workspace_header.isHidden()
    assert not window.pose_navigation.isHidden()
    window.close()


def test_inspector_select_boxes_ignore_wheel_input() -> None:
    app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    select_boxes = (
        window.background_mode,
        window.overlap_mode,
        window.trail_style,
        window.smear_style,
        window.alignment_mode,
        window.photo_order_mode,
    )

    assert all(isinstance(select_box, ScrollSafeComboBox) for select_box in select_boxes)
    window.background_mode.setEnabled(True)
    initial_index = window.background_mode.currentIndex()
    event = QWheelEvent(
        QPointF(10, 10),
        QPointF(10, 10),
        QPoint(),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )

    app.sendEvent(window.background_mode, event)

    assert window.background_mode.currentIndex() == initial_index
    assert not event.isAccepted()
    window.close()


def test_sliders_ignore_wheel_input() -> None:
    app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    sliders = (
        window.pose_scrubber,
        window.pose_count,
        window.threshold,
        window.feather,
    )

    assert all(isinstance(slider, ScrollSafeSlider) for slider in sliders)
    window.threshold.setEnabled(True)
    initial_value = window.threshold.value()
    slider_event = QWheelEvent(
        QPointF(10, 10),
        QPointF(10, 10),
        QPoint(),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )

    app.sendEvent(window.threshold, slider_event)

    assert window.threshold.value() == initial_value
    assert not slider_event.isAccepted()

    initial_range = (window.range_slider.low, window.range_slider.high)
    range_event = QWheelEvent(
        QPointF(10, 10),
        QPointF(10, 10),
        QPoint(),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )

    window.range_slider.wheelEvent(range_event)

    assert (window.range_slider.low, window.range_slider.high) == initial_range
    assert not range_event.isAccepted()
    window.close()


def test_entire_window_accepts_a_footage_drop(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    window.show()
    app.processEvents()
    accepted: list[list[str]] = []
    monkeypatch.setattr(window, "_accept_paths", lambda paths: accepted.append(paths))
    mime = QMimeData()
    path = str(Path("example.mp4").resolve())
    mime.setUrls([QUrl.fromLocalFile(path)])
    enter = QDragEnterEvent(
        QPoint(400, 300),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )

    window.dragEnterEvent(enter)

    assert window.acceptDrops()
    assert enter.isAccepted()
    assert not window.drop_overlay.isHidden()

    drop = QDropEvent(
        QPointF(400, 300),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.dropEvent(drop)

    assert [[Path(value) for value in values] for values in accepted] == [[Path(path)]]
    assert window.drop_overlay.isHidden()
    window.close()


def test_all_frames_request_tracks_the_committed_range(monkeypatch) -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    path = Path("clip.mp4")
    info = VideoInfo(path, 10.0, 1920, 1080, 30.0, 300)
    window.source = SourceState("video", [path], info)
    window._set_loaded_state(True)
    window.all_frames.setChecked(True)
    window.preview_debounce.stop()
    window.range_slider.set_values(100, 900)
    request = window._render_request(window.PREVIEW_MAX_DIMENSION)
    rendered: list[bool] = []
    monkeypatch.setattr(window, "render_preview", lambda: rendered.append(True))

    window._range_committed(100, 900)

    assert request.pose_count is None
    assert request.start == pytest.approx(1.0)
    assert request.end == pytest.approx(9.0)
    assert rendered == [True]
    window.close()


def test_range_render_reuses_the_file_level_video_cache(monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    path = Path("cached-clip.mp4")
    info = VideoInfo(path, 1.0, 96, 64, 10.0, 10)
    window.source = SourceState("video", [path], info)
    window._set_loaded_state(True)
    window.all_frames.setChecked(True)
    window.preview_debounce.stop()
    window.range_slider.set_values(200, 600)
    window.preview_debounce.stop()
    frames = []
    for index in range(10):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[20:42, 8 + index * 7 : 18 + index * 7, 0] = 220
        frames.append(frame)
    cached = MediaSequence(
        frames,
        [f"00:00.{index}" for index in range(10)],
        (96, 64),
        [index / 10 for index in range(10)],
    )
    request = window._render_request(window.PREVIEW_MAX_DIMENSION)
    window._video_preview_cache_key = request.video_cache_key
    window._video_preview_cache_sequence = cached
    decoder_calls: list[bool] = []

    def unexpected_decode(*args, **kwargs):  # type: ignore[no-untyped-def]
        decoder_calls.append(True)
        raise AssertionError("Range edit decoded the video again")

    monkeypatch.setattr("chronophoto.ui.window.load_video_sequence", unexpected_decode)
    window.render_preview()
    deadline = monotonic() + 5
    while window._thread is not None and monotonic() < deadline:
        app.processEvents()
        sleep(0.01)

    assert window._thread is None
    assert decoder_calls == []
    assert len(window.preview_frames) == 5
    assert "frames and masks" in window.status_detail.text()
    assert window._video_analysis_cache is not None

    analysis_builds: list[bool] = []

    def unexpected_analysis(*args, **kwargs):  # type: ignore[no-untyped-def]
        analysis_builds.append(True)
        raise AssertionError("Range edit analyzed the poses again")

    monkeypatch.setattr("chronophoto.ui.window.build_compose_cache", unexpected_analysis)
    window.range_slider.set_values(300, 700)
    window.preview_debounce.stop()
    window.render_preview()
    deadline = monotonic() + 5
    while window._thread is not None and monotonic() < deadline:
        app.processEvents()
        sleep(0.01)

    assert window._thread is None
    assert analysis_builds == []
    assert "cached frames and masks" in window.status_detail.text()

    with QSignalBlocker(window.effect_timeline):
        lane = window.effect_timeline.add_effect("opacity")
        lane.preset.setCurrentIndex(lane.preset.findData("rise_fall"))
    window.render_preview()
    deadline = monotonic() + 5
    while window._thread is not None and monotonic() < deadline:
        app.processEvents()
        sleep(0.01)

    assert window._thread is None
    assert decoder_calls == []
    assert analysis_builds == []
    assert "cached frames and masks" in window.status_detail.text()

    with QSignalBlocker(window.background_effect_timeline):
        background_lane = window.background_effect_timeline.add_effect("saturation")
        background_lane.value_spin.setValue(0)
        background_lane._constant_value_committed()
    window.render_preview()
    deadline = monotonic() + 5
    while window._thread is not None and monotonic() < deadline:
        app.processEvents()
        sleep(0.01)

    assert window._thread is None
    assert decoder_calls == []
    assert analysis_builds == []
    assert "cached frames and masks" in window.status_detail.text()
    window.close()


def test_range_handles_preview_the_nearest_cached_source_frame() -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    path = Path("cached-clip.mp4")
    window.source = SourceState("video", [path], VideoInfo(path, 1.0, 4, 3, 4.0, 4))
    frames = [np.full((3, 4, 3), index * 50, dtype=np.uint8) for index in range(4)]
    window._video_preview_cache_sequence = MediaSequence(
        frames,
        ["0", "1", "2", "3"],
        (4, 3),
        [0.0, 0.25, 0.5, 0.75],
    )

    window.range_slider.resize(1000, 78)
    window.range_slider._active = "low"
    window.range_slider._move_active(window.range_slider._x_for_value(480))

    assert window.preview_canvas._status == "IN · 00:00.50"
    assert window.preview_canvas._image is not None
    assert window.preview_canvas._image.pixelColor(0, 0).red() == 100

    window.range_slider._active = "high"
    window.range_slider._move_active(window.range_slider._x_for_value(760))

    assert window.preview_canvas._status == "OUT · 00:00.75"
    assert window.preview_canvas._image.pixelColor(0, 0).red() == 150
    window.close()


def test_preview_canvas_zoom_is_pointer_centered_and_resets() -> None:
    _app = QApplication.instance() or QApplication([])
    canvas = PreviewCanvas()
    canvas.resize(800, 600)
    canvas.set_image(QImage(1600, 900, QImage.Format.Format_RGB888))
    anchor = QPointF(650, 300)

    initial_rect = canvas._image_rect()
    initial_image_x = (anchor.x() - initial_rect.left()) / initial_rect.width()
    canvas.zoom_by(2.0, anchor)
    zoomed_rect = canvas._image_rect()
    zoomed_image_x = (anchor.x() - zoomed_rect.left()) / zoomed_rect.width()

    assert canvas.zoom_factor == 2.0
    assert zoomed_image_x == pytest.approx(initial_image_x)

    canvas.zoom_by(100.0, anchor)
    assert canvas.zoom_factor == 12.0
    canvas.reset_view()
    assert canvas.zoom_factor == 1.0
    canvas.close()


def test_touchpad_wheel_zoom_persists_across_equal_sized_frames() -> None:
    app = QApplication.instance() or QApplication([])
    canvas = PreviewCanvas()
    canvas.resize(800, 600)
    canvas.set_image(QImage(1280, 720, QImage.Format.Format_RGB888))
    event = QWheelEvent(
        QPointF(500, 300),
        QPointF(500, 300),
        QPoint(0, 40),
        QPoint(),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )

    app.sendEvent(canvas, event)
    zoomed = canvas.zoom_factor
    canvas.set_image(QImage(1280, 720, QImage.Format.Format_RGB888))

    assert event.isAccepted()
    assert zoomed > 1.0
    assert canvas.zoom_factor == zoomed

    canvas.set_image(QImage(800, 600, QImage.Format.Format_RGB888))
    assert canvas.zoom_factor == 1.0
    canvas.close()


def test_selecting_a_frame_updates_navigation_without_scheduling_a_render() -> None:
    _app = QApplication.instance() or QApplication([])
    window = ChronophotoWindow()
    paths = [Path("first.png"), Path("second.png")]
    window.source = SourceState("photos", paths)
    window.preview_frames = [
        np.zeros((10, 10, 3), dtype=np.uint8),
        np.ones((10, 10, 3), dtype=np.uint8),
    ]
    window.preview_labels = [path.name for path in paths]
    window.pose_scrubber.setRange(0, 1)
    window._populate_photo_frames(paths, "input")
    window.preview_debounce.stop()

    window.frame_list.setCurrentRow(1)

    assert window.pose_scrubber.value() == 1
    assert not window.preview_debounce.isActive()
    window.close()


def test_worker_can_cancel_before_expensive_work() -> None:
    worker = TaskWorker(lambda progress: progress(50, "Halfway"))
    cancelled: list[bool] = []
    worker.cancelled.connect(lambda: cancelled.append(True))

    worker.request_cancel()
    worker.run()

    assert cancelled == [True]


def test_export_filter_adds_the_matching_extension() -> None:
    base = Path("motion-study")

    assert ChronophotoWindow._export_path_with_filter(base, "PNG image (*.png)").suffix == ".png"
    assert ChronophotoWindow._export_path_with_filter(base, "TIFF image (*.tif)").suffix == ".tif"
    assert ChronophotoWindow._export_path_with_filter(base, "JPEG image (*.jpg)").suffix == ".jpg"
    assert ChronophotoWindow._export_path_with_filter(base, "MP4 video (*.mp4)").suffix == ".mp4"
