"""Capture deterministic screenshots for the project README."""

from __future__ import annotations

import os
from pathlib import Path
from time import monotonic, sleep

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtCore import QPointF
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication
from render_smoke_fixture import build_frames

from chronophoto.app import _load_terminal_font, application_stylesheet
from chronophoto.processing.sources import MediaSequence, VideoInfo
from chronophoto.ui.window import ChronophotoWindow, SourceState

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "images"


def _wait_for_preview(app: QApplication, window: ChronophotoWindow) -> None:
    deadline = monotonic() + 30
    while window._thread is not None and monotonic() < deadline:
        app.processEvents()
        sleep(0.01)
    if window._thread is not None or window.preview_result is None:
        raise RuntimeError("README preview did not finish")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Chronophoto")
    font_family = _load_terminal_font()
    app.setFont(QFont(font_family, 10))
    app.setStyleSheet(application_stylesheet(font_family))
    app.setWindowIcon(QIcon(str(PROJECT_ROOT / "src/chronophoto/assets/chronophoto-icon.png")))

    frames = build_frames()
    timestamps = [index * 0.35 for index in range(len(frames))]
    labels = [f"00:{seconds:05.2f}" for seconds in timestamps]
    source_path = PROJECT_ROOT / "sample" / "jump-demo.mp4"
    sequence = MediaSequence(
        frames,
        labels,
        (frames[0].shape[1], frames[0].shape[0]),
        timestamps,
        list(range(len(frames))),
    )

    window = ChronophotoWindow(check_updates=False)
    window.resize(1500, 930)
    window.source = SourceState(
        "video",
        [source_path],
        VideoInfo(source_path, 2.8, 1280, 720, 3.21, len(frames)),
    )
    window.source_name.setText("jump-demo.mp4")
    window.source_meta.setText("VIDEO · 1280 × 720\n00:02.80 · 3.21 fps · 9 frames")
    window.preview_heading.setText("Jump study")
    window.result_meta.setText("1280 × 720 source")
    window.range_slider.set_values(0, 1000)
    window._update_pose_range(preferred=len(frames))
    window._sync_all_frames_state()
    window._set_loaded_state(True)
    request = window._render_request(window.PREVIEW_MAX_DIMENSION)
    window._video_preview_cache_key = request.video_cache_key
    window._video_preview_cache_sequence = sequence

    window.show()
    app.processEvents()
    window.render_preview()
    _wait_for_preview(app, window)

    window._set_preview_mode("composite")
    app.processEvents()
    window.grab().save(str(OUTPUT_DIR / "chronophoto-workspace.png"), "PNG")
    Image.fromarray(window.preview_result).save(OUTPUT_DIR / "chronophoto-result.png")

    window.pose_scrubber.setValue(len(frames) // 2)
    window._set_preview_mode("mask")
    window.advanced_toggle.setChecked(True)
    window.preview_canvas.zoom_by(1.45, QPointF(window.preview_canvas.rect().center()))
    app.processEvents()
    window.grab().save(str(OUTPUT_DIR / "chronophoto-mask-inspection.png"), "PNG")

    window.close()
    print(OUTPUT_DIR.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
