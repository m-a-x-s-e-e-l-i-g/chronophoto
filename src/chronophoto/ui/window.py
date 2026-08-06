from __future__ import annotations

import json
from bisect import bisect_left
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import numpy as np
from PIL import Image
from PySide6.QtCore import (
    QObject,
    QSettings,
    QSignalBlocker,
    Qt,
    QThread,
    QTimer,
    QUrl,
    Signal,
    Slot,
)
from PySide6.QtGui import QAction, QCloseEvent, QDesktopServices, QKeySequence
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from chronophoto import __version__
from chronophoto.processing import (
    ComposeCache,
    ComposeSettings,
    MediaSequence,
    align_sequence,
    build_compose_cache,
    compose_sequence,
    load_image_sequence,
    load_video_sequence,
    order_image_paths,
    select_video_sequence,
)
from chronophoto.processing.sources import VideoInfo, probe_video
from chronophoto.ui.widgets import (
    DropSurface,
    PreviewCanvas,
    RangeSlider,
    ScrollSafeComboBox,
    ScrollSafeSlider,
    classify_paths,
    image_to_qimage,
    mask_overlay_to_qimage,
)
from chronophoto.updates import (
    GITHUB_REPOSITORY_URL,
    LATEST_RELEASE_API_URL,
    UpdateResult,
    evaluate_release,
)


class TaskCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class SourceState:
    kind: str
    paths: list[Path]
    video_info: VideoInfo | None = None


@dataclass(slots=True, frozen=True)
class RenderRequest:
    kind: str
    paths: tuple[Path, ...]
    start: float
    end: float
    pose_count: int | None
    settings: ComposeSettings
    alignment: str
    max_dimension: int | None
    cache_key: tuple[object, ...]
    video_selection_key: tuple[object, ...] | None
    video_cache_key: tuple[object, ...] | None
    video_duration: float
    enabled_video_indices: tuple[int, ...] | None


@dataclass(slots=True)
class VideoAnalysisCache:
    key: tuple[object, ...]
    frames: list[np.ndarray]
    compose_cache: ComposeCache


class TaskWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    progress = Signal(int, str)

    def __init__(self, task: Callable[[Callable[[int, str], None]], object]) -> None:
        super().__init__()
        self.task = task
        self._cancel = Event()

    def request_cancel(self) -> None:
        self._cancel.set()

    @Slot()
    def run(self) -> None:
        def report(value: int, message: str) -> None:
            if self._cancel.is_set():
                raise TaskCancelled
            self.progress.emit(value, message)

        try:
            result = self.task(report)
            if self._cancel.is_set():
                raise TaskCancelled
        except TaskCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:  # noqa: BLE001 - translated into a recoverable UI state
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


class ChronophotoWindow(QMainWindow):
    PREVIEW_MAX_DIMENSION = 960

    def __init__(self, *, check_updates: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("Chronophoto — Motion, held still")
        self.resize(1380, 880)
        self.setMinimumSize(960, 680)
        self.setAcceptDrops(True)

        self.source: SourceState | None = None
        self.preview_result: np.ndarray | None = None
        self.preview_frames: list[np.ndarray] = []
        self.preview_masks: list[np.ndarray] = []
        self.preview_labels: list[str] = []
        self._preview_cache_key: tuple[object, ...] | None = None
        self._preview_cache_sequence: MediaSequence | None = None
        self._video_preview_cache_key: tuple[object, ...] | None = None
        self._video_preview_cache_sequence: MediaSequence | None = None
        self._video_analysis_cache: VideoAnalysisCache | None = None
        self._video_frame_selection_key: tuple[object, ...] | None = None
        self._timeline_thumbnails: list[np.ndarray] = []
        self._thread: QThread | None = None
        self._worker: TaskWorker | None = None
        self._pending_preview = False
        self._preview_dirty = False
        self._updating_frames = False
        self._loading_source = False
        self._close_when_done = False
        self._last_export_path: Path | None = None
        self._github_target_url = GITHUB_REPOSITORY_URL
        self._update_manager: QNetworkAccessManager | None = None
        self.settings_store = QSettings("Chronophoto", "Chronophoto")

        self.preview_debounce = QTimer(self)
        self.preview_debounce.setSingleShot(True)
        self.preview_debounce.setInterval(450)
        self.preview_debounce.timeout.connect(self.render_preview)
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(180)
        self.playback_timer.timeout.connect(self._advance_pose)

        self._build_ui()
        self._build_shortcuts()
        self._set_loaded_state(False)
        if check_updates:
            QTimer.singleShot(250, self._check_for_updates)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        root_layout.addWidget(self._build_header())

        self.body_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.body_splitter.setObjectName("bodySplitter")
        self.body_splitter.setChildrenCollapsible(False)
        self.body_splitter.addWidget(self._build_source_panel())
        self.body_splitter.addWidget(self._build_workspace())
        self.body_splitter.addWidget(self._build_inspector())
        self.body_splitter.setStretchFactor(0, 0)
        self.body_splitter.setStretchFactor(1, 1)
        self.body_splitter.setStretchFactor(2, 0)
        self.body_splitter.setSizes([232, 850, 286])
        root_layout.addWidget(self.body_splitter, 1)
        root_layout.addWidget(self._build_status_bar())
        self.setCentralWidget(root)
        self.drop_overlay = QFrame(root)
        self.drop_overlay.setObjectName("windowDropOverlay")
        self.drop_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        drop_layout = QVBoxLayout(self.drop_overlay)
        drop_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_overlay_title = QLabel("DROP FOOTAGE ANYWHERE")
        self.drop_overlay_title.setObjectName("windowDropTitle")
        drop_hint = QLabel("ONE VIDEO OR AN ORDERED PHOTO STACK")
        drop_hint.setObjectName("windowDropHint")
        drop_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(self.drop_overlay_title)
        drop_layout.addWidget(drop_hint)
        self.drop_overlay.hide()

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setObjectName("appHeader")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 11, 20, 11)
        layout.setSpacing(12)
        wordmark = QLabel("CHRONOPHOTO")
        wordmark.setObjectName("wordmark")
        strapline = QLabel("MOTION, HELD STILL")
        strapline.setObjectName("strapline")
        local = QLabel("LOCAL")
        local.setObjectName("privacyBadge")
        local.setToolTip("Video and photos stay on this computer")
        self.version_number = QLabel(f"v{__version__}")
        self.version_number.setObjectName("versionNumber")
        self.version_number.setAccessibleName(f"Chronophoto version {__version__}")
        self.update_status = QLabel("CHECKING")
        self.update_status.setObjectName("updateStatus")
        self.update_status.setProperty("state", "checking")
        self.update_status.setAccessibleName("Update status")
        self.github_button = QPushButton("GITHUB")
        self.github_button.setObjectName("headerButton")
        self.github_button.setAccessibleName("Open Chronophoto on GitHub")
        self.github_button.clicked.connect(self._open_github)
        layout.addWidget(wordmark)
        layout.addWidget(strapline)
        layout.addStretch()
        layout.addWidget(local)
        layout.addWidget(self.version_number)
        layout.addWidget(self.update_status)
        layout.addWidget(self.github_button)
        return header

    @Slot()
    def _open_github(self) -> None:
        QDesktopServices.openUrl(QUrl(self._github_target_url))

    @Slot()
    def _check_for_updates(self) -> None:
        self._set_update_status("CHECKING", "checking", "Checking the latest GitHub release")
        if self._update_manager is None:
            self._update_manager = QNetworkAccessManager(self)
            self._update_manager.finished.connect(self._update_check_finished)
        request = QNetworkRequest(QUrl(LATEST_RELEASE_API_URL))
        request.setRawHeader(b"Accept", b"application/vnd.github+json")
        request.setRawHeader(b"User-Agent", b"Chronophoto-update-check")
        request.setRawHeader(b"X-GitHub-Api-Version", b"2022-11-28")
        request.setTransferTimeout(5000)
        self._update_manager.get(request)

    @Slot(QNetworkReply)
    def _update_check_finished(self, reply: QNetworkReply) -> None:
        status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        try:
            if status == 404:
                self._github_target_url = GITHUB_REPOSITORY_URL
                self._set_update_status(
                    "NO RELEASE YET",
                    "unavailable",
                    "GitHub has no published Chronophoto release yet",
                )
                return
            if reply.error() != QNetworkReply.NetworkError.NoError or status != 200:
                self._set_update_status(
                    "CHECK FAILED",
                    "unavailable",
                    "Could not check GitHub. The application remains fully usable offline.",
                )
                return
            payload = json.loads(bytes(reply.readAll()).decode("utf-8"))
            result = evaluate_release(
                __version__,
                str(payload["tag_name"]),
                str(payload["html_url"]),
            )
            self._apply_update_result(result)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            self._set_update_status(
                "CHECK FAILED",
                "unavailable",
                "GitHub returned release information the application could not read",
            )
        finally:
            reply.deleteLater()

    def _apply_update_result(self, result: UpdateResult) -> None:
        if result.update_available:
            self._github_target_url = result.release_url
            self._set_update_status(
                f"UPDATE v{result.latest_version} AVAILABLE",
                "available",
                f"GitHub has Chronophoto v{result.latest_version}. Open the release to update.",
            )
            self.github_button.setToolTip(f"Open the v{result.latest_version} GitHub release")
        else:
            self._github_target_url = GITHUB_REPOSITORY_URL
            self._set_update_status(
                "UP TO DATE",
                "current",
                f"Chronophoto v{__version__} is the latest published release",
            )
            self.github_button.setToolTip("Open the Chronophoto GitHub repository")

    def _set_update_status(self, text: str, state: str, tooltip: str) -> None:
        self.update_status.setText(text)
        self.update_status.setProperty("state", state)
        self.update_status.setToolTip(tooltip)
        self.update_status.updateGeometry()
        self.update_status.style().unpolish(self.update_status)
        self.update_status.style().polish(self.update_status)

    def _build_source_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("sourcePanel")
        panel.setMinimumWidth(196)
        panel.setMaximumWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(14)

        self.source_intro = QWidget()
        intro_layout = QVBoxLayout(self.source_intro)
        intro_layout.setContentsMargins(0, 0, 0, 0)
        intro_layout.setSpacing(10)
        eyebrow = QLabel("01 / SOURCE")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("Bring in the\naction.")
        title.setObjectName("panelTitle")
        subtitle = QLabel("One video clip, or an ordered photo stack.")
        subtitle.setObjectName("bodyMuted")
        subtitle.setWordWrap(True)
        self.drop_surface = DropSurface()
        self.drop_surface.activated.connect(self._choose_source)
        self.drop_surface.paths_dropped.connect(self._accept_paths)
        self.drop_surface.drag_active.connect(self._set_drop_overlay_active)
        intro_layout.addWidget(eyebrow)
        intro_layout.addWidget(title)
        intro_layout.addWidget(subtitle)
        intro_layout.addWidget(self.drop_surface)
        layout.addWidget(self.source_intro)

        source_buttons = QHBoxLayout()
        source_buttons.setSpacing(8)
        self.video_button = QPushButton("Video")
        self.video_button.setObjectName("quietButton")
        self.video_button.clicked.connect(self._open_video)
        self.photos_button = QPushButton("Photo stack")
        self.photos_button.setObjectName("quietButton")
        self.photos_button.clicked.connect(self._open_photos)
        source_buttons.addWidget(self.video_button)
        source_buttons.addWidget(self.photos_button)
        layout.addLayout(source_buttons)

        self.source_summary = QFrame()
        self.source_summary.setObjectName("sourceSummary")
        summary_layout = QVBoxLayout(self.source_summary)
        summary_layout.setContentsMargins(12, 12, 12, 12)
        summary_layout.setSpacing(3)
        self.source_name = QLabel("No source selected")
        self.source_name.setObjectName("sourceName")
        self.source_name.setWordWrap(True)
        self.source_meta = QLabel("—")
        self.source_meta.setObjectName("bodyMuted")
        self.source_meta.setWordWrap(True)
        summary_layout.addWidget(self.source_name)
        summary_layout.addWidget(self.source_meta)
        layout.addWidget(self.source_summary)

        self.frames_section = QWidget()
        frames_layout = QVBoxLayout(self.frames_section)
        frames_layout.setContentsMargins(0, 0, 0, 0)
        frames_layout.setSpacing(7)
        frames_header = QHBoxLayout()
        frame_label = QLabel("FRAMES")
        frame_label.setObjectName("controlLabel")
        self.frame_count_label = QLabel("0 enabled")
        self.frame_count_label.setObjectName("valueLabel")
        frames_header.addWidget(frame_label)
        frames_header.addStretch()
        frames_header.addWidget(self.frame_count_label)
        self.frame_list = QListWidget()
        self.frame_list.setObjectName("frameList")
        self.frame_list.setAccessibleName("Included poses and photographs")
        self.frame_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.frame_list.itemChanged.connect(self._frame_state_changed)
        self.frame_list.currentRowChanged.connect(self._frame_selected)
        self.frame_list.model().rowsMoved.connect(self._frame_order_moved)
        move_row = QHBoxLayout()
        self.move_up_button = QPushButton("Move up")
        self.move_up_button.setObjectName("quietButton")
        self.move_up_button.clicked.connect(lambda: self._move_selected_frame(-1))
        self.move_down_button = QPushButton("Move down")
        self.move_down_button.setObjectName("quietButton")
        self.move_down_button.clicked.connect(lambda: self._move_selected_frame(1))
        move_row.addWidget(self.move_up_button)
        move_row.addWidget(self.move_down_button)
        frames_layout.addLayout(frames_header)
        frames_layout.addWidget(self.frame_list, 1)
        frames_layout.addLayout(move_row)
        layout.addWidget(self.frames_section, 1)

        privacy = QLabel("No upload. No cloud render.")
        privacy.setObjectName("privacyNote")
        layout.addWidget(privacy)
        return panel

    def _build_workspace(self) -> QWidget:
        workspace = QWidget()
        workspace.setObjectName("workspace")
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(12)

        top = QHBoxLayout()
        top.setSpacing(10)
        self.preview_heading = QLabel("Untitled study")
        self.preview_heading.setObjectName("workspaceTitle")
        self.result_meta = QLabel("Load footage to begin")
        self.result_meta.setObjectName("bodyMuted")
        top.addWidget(self.preview_heading)
        top.addStretch()
        top.addWidget(self.result_meta)
        layout.addLayout(top)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(0)
        self.preview_mode_group = QButtonGroup(self)
        self.preview_mode_group.setExclusive(True)
        self.preview_mode_buttons: dict[str, QToolButton] = {}
        for mode, label in (("source", "Source"), ("composite", "Composite"), ("mask", "Mask")):
            button = QToolButton()
            button.setText(label)
            button.setCheckable(True)
            button.setObjectName("modeButton")
            button.setAccessibleName(f"Show {label.lower()} preview")
            button.clicked.connect(
                lambda checked=False, selected=mode: self._set_preview_mode(selected)
            )
            self.preview_mode_group.addButton(button)
            self.preview_mode_buttons[mode] = button
            mode_row.addWidget(button)
        self.preview_mode_buttons["composite"].setChecked(True)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        self.preview_canvas = PreviewCanvas()
        self.preview_canvas.setMinimumSize(400, 260)
        layout.addWidget(self.preview_canvas, 1)

        self.pose_navigation = QWidget()
        pose_layout = QHBoxLayout(self.pose_navigation)
        pose_layout.setContentsMargins(0, 0, 0, 0)
        pose_layout.setSpacing(9)
        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("quietButton")
        self.play_button.setAccessibleName("Play sampled source frames")
        self.play_button.clicked.connect(self._toggle_playback)
        self.pose_scrubber = ScrollSafeSlider(Qt.Orientation.Horizontal)
        self.pose_scrubber.setRange(0, 0)
        self.pose_scrubber.setAccessibleName("Previewed pose")
        self.pose_scrubber.valueChanged.connect(self._refresh_preview_canvas)
        self.pose_position = QLabel("0 / 0")
        self.pose_position.setObjectName("timecode")
        pose_layout.addWidget(self.play_button)
        pose_layout.addWidget(self.pose_scrubber, 1)
        pose_layout.addWidget(self.pose_position)
        layout.addWidget(self.pose_navigation)

        self.timeline_panel = QFrame()
        self.timeline_panel.setObjectName("timelinePanel")
        timeline_layout = QVBoxLayout(self.timeline_panel)
        timeline_layout.setContentsMargins(12, 10, 12, 9)
        timeline_layout.setSpacing(4)
        time_header = QHBoxLayout()
        self.start_time = QLabel("00:00.00")
        self.start_time.setObjectName("timecode")
        range_caption = QLabel("IN / OUT · SHIFT + ARROWS ADJUST OUT")
        range_caption.setObjectName("eyebrow")
        self.end_time = QLabel("00:00.00")
        self.end_time.setObjectName("timecode")
        time_header.addWidget(self.start_time)
        time_header.addStretch()
        time_header.addWidget(range_caption)
        time_header.addStretch()
        time_header.addWidget(self.end_time)
        self.range_slider = RangeSlider()
        self.range_slider.range_changed.connect(self._range_changed)
        self.range_slider.handle_changed.connect(self._range_handle_changed)
        self.range_slider.range_committed.connect(self._range_committed)
        timeline_layout.addLayout(time_header)
        timeline_layout.addWidget(self.range_slider)
        layout.addWidget(self.timeline_panel)

        actions = QHBoxLayout()
        actions.setSpacing(9)
        self.export_button = QPushButton("Export composite")
        self.export_button.setObjectName("primaryButton")
        self.export_button.clicked.connect(self.export_composite)
        actions.addStretch()
        actions.addWidget(self.export_button)
        layout.addLayout(actions)
        return workspace

    def _build_inspector(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setObjectName("inspectorScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(248)
        scroll.setMaximumWidth(330)
        panel = QWidget()
        panel.setObjectName("inspector")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 18, 16, 18)
        layout.setSpacing(18)

        heading_row = QHBoxLayout()
        eyebrow = QLabel("02 / COMPOSE")
        eyebrow.setObjectName("eyebrow")
        self.reset_button = QPushButton("Reset")
        self.reset_button.setObjectName("quietButton")
        self.reset_button.clicked.connect(self._reset_controls)
        heading_row.addWidget(eyebrow)
        heading_row.addStretch()
        heading_row.addWidget(self.reset_button)
        title = QLabel("Shape the sequence.")
        title.setObjectName("panelTitleCompact")
        layout.addLayout(heading_row)
        layout.addWidget(title)

        self.pose_count = ScrollSafeSlider(Qt.Orientation.Horizontal)
        self.pose_count.setRange(2, 40)
        self.pose_count.setValue(10)
        self.pose_count.setSingleStep(1)
        self.pose_count.setPageStep(5)
        self.pose_count.setTracking(False)
        self.pose_count_value = QLabel("10 poses")
        self.pose_count_value.setObjectName("valueLabel")
        self.pose_count.sliderMoved.connect(self._update_pose_count_label)
        self.pose_count.valueChanged.connect(self._pose_count_changed)
        self.all_frames = QCheckBox("Use every selected frame")
        self.all_frames.setAccessibleName("Use every video frame")
        self.all_frames.setAccessibleDescription(
            "Include every decoded frame in the selected video range"
        )
        self.all_frames.setChecked(True)
        self.all_frames.toggled.connect(self._all_frames_changed)
        self.pose_control = self._pose_count_control()
        self.pose_control.setObjectName("poseControl")
        self.pose_control.setProperty("allFramesActive", True)
        layout.addWidget(self.pose_control)

        self.pose_count.setAccessibleDescription(
            "Choose any number of evenly spaced poses in the selected video range"
        )

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setObjectName("advancedToggle")
        self.advanced_toggle.setText("Advanced mask controls")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        layout.addWidget(self.advanced_toggle)

        self.advanced_controls = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_controls)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(18)
        self.threshold = ScrollSafeSlider(Qt.Orientation.Horizontal)
        self.threshold.setRange(1, 90)
        self.threshold.setValue(17)
        self.threshold_value = QLabel("17")
        self.threshold_value.setObjectName("valueLabel")
        self.threshold.valueChanged.connect(lambda value: self.threshold_value.setText(str(value)))
        self.threshold.valueChanged.connect(lambda: self._schedule_preview())
        advanced_layout.addWidget(
            self._slider_control(
                "MASK SENSITIVITY",
                "Lower values keep more movement. Inspect the Mask view.",
                self.threshold,
                self.threshold_value,
            )
        )
        self.feather = ScrollSafeSlider(Qt.Orientation.Horizontal)
        self.feather.setRange(0, 20)
        self.feather.setValue(1)
        self.feather_value = QLabel("1 px")
        self.feather_value.setObjectName("valueLabel")
        self.feather.valueChanged.connect(lambda value: self.feather_value.setText(f"{value} px"))
        self.feather.valueChanged.connect(lambda: self._schedule_preview())
        advanced_layout.addWidget(
            self._slider_control(
                "EDGE FEATHER",
                "Softens the edge of each detected pose.",
                self.feather,
                self.feather_value,
            )
        )
        self.background_mode = ScrollSafeComboBox()
        self.background_mode.addItem("Median clean plate", "median")
        self.background_mode.addItem("First frame", "first")
        self.background_mode.addItem("Last frame", "last")
        self.background_mode.currentIndexChanged.connect(lambda: self._schedule_preview())
        advanced_layout.addWidget(
            self._control("CLEAN PLATE", "The background beneath every pose.", self.background_mode)
        )
        self.overlap_mode = ScrollSafeComboBox()
        self.overlap_mode.addItem("Newest pose on top", "newest")
        self.overlap_mode.addItem("Oldest pose on top", "oldest")
        self.overlap_mode.currentIndexChanged.connect(lambda: self._schedule_preview())
        advanced_layout.addWidget(
            self._control("OVERLAP", "Which pose wins where bodies cross.", self.overlap_mode)
        )
        self.trail_style = ScrollSafeComboBox()
        self.trail_style.addItem("Solid", "solid")
        self.trail_style.setEnabled(False)
        self.trail_style.setToolTip("Work in progress")
        self.trail_style_control = self._control(
            "TRAIL STYLE",
            "Additional styles are work in progress.",
            self.trail_style,
        )
        self.trail_style_control.setToolTip("Work in progress")
        advanced_layout.addWidget(self.trail_style_control)
        self.smear_style = ScrollSafeComboBox()
        self.smear_style.addItem("None", "none")
        self.smear_style.addItem("Photographic stretch", "photographic")
        self.smear_style.addItem("Dense cloned copies", "dense_clones")
        self.smear_style.currentIndexChanged.connect(lambda: self._schedule_preview())
        advanced_layout.addWidget(
            self._control(
                "SMEAR APPEARANCE",
                "None keeps every pose distinct. Or connect them with texture or dense copies.",
                self.smear_style,
            )
        )
        self.alignment_mode = ScrollSafeComboBox()
        self.alignment_mode.addItem("Off — locked camera", "off")
        self.alignment_mode.addItem("Translation — minor movement", "translation")
        self.alignment_mode.currentIndexChanged.connect(lambda: self._schedule_preview())
        advanced_layout.addWidget(
            self._control(
                "ALIGNMENT",
                "Compensates for small camera shifts in photo stacks.",
                self.alignment_mode,
            )
        )
        self.photo_order_mode = ScrollSafeComboBox()
        self.photo_order_mode.addItem("Automatic", "automatic")
        self.photo_order_mode.addItem("Capture time (EXIF)", "capture_time")
        self.photo_order_mode.addItem("Filename", "filename")
        self.photo_order_mode.addItem("Manual", "input")
        self.photo_order_mode.currentIndexChanged.connect(self._photo_order_changed)
        self.photo_order_control = self._control(
            "PHOTO ORDER",
            "Drag frames or use Move up/down to correct the order.",
            self.photo_order_mode,
        )
        advanced_layout.addWidget(self.photo_order_control)
        self.advanced_controls.setVisible(False)
        layout.addWidget(self.advanced_controls)
        layout.addStretch()

        tip = QLabel("BEST RESULTS\nLocked camera · stable exposure · clear subject separation")
        tip.setObjectName("inspectorTip")
        tip.setWordWrap(True)
        layout.addWidget(tip)
        scroll.setWidget(panel)
        return scroll

    def _pose_count_control(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        row = QHBoxLayout()
        heading = QLabel("POSE COUNT")
        heading.setObjectName("controlLabel")
        heading.setBuddy(self.pose_count)
        row.addWidget(heading)
        row.addStretch()
        row.addWidget(self.pose_count_value)
        description = QLabel("Choose any count, or include every frame in the selected range.")
        description.setObjectName("controlHint")
        description.setWordWrap(True)
        self.pose_count.setAccessibleName("Pose count")
        layout.addLayout(row)
        layout.addWidget(description)
        layout.addWidget(self.pose_count)
        layout.addWidget(self.all_frames)
        return wrapper

    @staticmethod
    def _control(label: str, hint: str, control: QWidget) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        heading = QLabel(label)
        heading.setObjectName("controlLabel")
        heading.setBuddy(control)
        description = QLabel(hint)
        description.setObjectName("controlHint")
        description.setWordWrap(True)
        control.setAccessibleName(label.title())
        control.setAccessibleDescription(hint)
        layout.addWidget(heading)
        layout.addWidget(description)
        layout.addWidget(control)
        return wrapper

    @staticmethod
    def _slider_control(
        label: str,
        hint: str,
        slider: ScrollSafeSlider,
        value_label: QLabel,
    ) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        row = QHBoxLayout()
        heading = QLabel(label)
        heading.setObjectName("controlLabel")
        heading.setBuddy(slider)
        row.addWidget(heading)
        row.addStretch()
        row.addWidget(value_label)
        description = QLabel(hint)
        description.setObjectName("controlHint")
        description.setWordWrap(True)
        slider.setAccessibleName(label.title())
        slider.setAccessibleDescription(hint)
        layout.addLayout(row)
        layout.addWidget(description)
        layout.addWidget(slider)
        return wrapper

    def _build_status_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("statusBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(10)
        self.status_text = QLabel("READY")
        self.status_text.setObjectName("statusText")
        self.status_detail = QLabel("Choose a source to start a motion study")
        self.status_detail.setObjectName("bodyMuted")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("quietButton")
        self.cancel_button.clicked.connect(self._cancel_task)
        self.open_export_button = QPushButton("Open folder")
        self.open_export_button.setObjectName("quietButton")
        self.open_export_button.clicked.connect(self._open_export_folder)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedWidth(150)
        layout.addWidget(self.status_text)
        layout.addWidget(self.status_detail)
        layout.addStretch()
        layout.addWidget(self.cancel_button)
        layout.addWidget(self.open_export_button)
        layout.addWidget(self.progress)
        self.cancel_button.hide()
        self.open_export_button.hide()
        return bar

    def _build_shortcuts(self) -> None:
        for shortcut, handler in (
            (QKeySequence.StandardKey.Open, self._choose_source),
            (QKeySequence.StandardKey.SaveAs, self.export_composite),
            (QKeySequence("Space"), self._toggle_playback),
        ):
            action = QAction(self)
            action.setShortcut(shortcut)
            action.triggered.connect(handler)
            self.addAction(action)

    @Slot(bool)
    def _set_drop_overlay_active(self, active: bool) -> None:
        if not hasattr(self, "drop_overlay"):
            return
        active = active and self._thread is None
        if not active:
            self.drop_overlay.hide()
            return
        self.drop_overlay_title.setText(
            "DROP TO REPLACE FOOTAGE" if self.source else "DROP FOOTAGE ANYWHERE"
        )
        if self.centralWidget():
            self.drop_overlay.setGeometry(self.centralWidget().rect())
        self.drop_overlay.raise_()
        self.drop_overlay.show()

    def dragEnterEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._thread is None and event.mimeData().hasUrls():
            self._set_drop_overlay_active(True)
            event.acceptProposedAction()
            return
        event.ignore()

    def dragMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._thread is None and event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._set_drop_overlay_active(False)
        event.accept()

    def dropEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._set_drop_overlay_active(False)
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths and self._thread is None:
            self._accept_paths(paths)
            event.acceptProposedAction()
            return
        event.ignore()

    def _choose_source(self) -> None:
        self._open_video()

    def _open_video(self) -> None:
        last_dir = self.settings_store.value("last_source_dir", str(Path.home()))
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose action footage",
            str(last_dir),
            "Video files (*.mp4 *.mov *.m4v *.avi *.mkv *.webm)",
        )
        if path:
            self._accept_paths([path])

    def _open_photos(self) -> None:
        last_dir = self.settings_store.value("last_source_dir", str(Path.home()))
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose an ordered photo stack",
            str(last_dir),
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.webp)",
        )
        if paths:
            self._accept_paths(paths)

    @Slot(list)
    def _accept_paths(self, raw_paths: list[str]) -> None:
        if self._thread:
            return
        try:
            self._loading_source = True
            kind, paths = classify_paths(raw_paths)
            self._clear_preview_state()
            if kind == "video":
                info = probe_video(paths[0])
                self.source = SourceState(kind, paths, info)
                with QSignalBlocker(self.all_frames):
                    self.all_frames.setChecked(True)
                self.source_name.setText(paths[0].name)
                duration = self._format_time(info.duration)
                self.source_meta.setText(
                    f"VIDEO · {info.width} × {info.height}\n"
                    f"{duration} · {info.frame_rate:.2f} fps · {info.frame_count} frames"
                )
                self.result_meta.setText(f"{info.width} × {info.height} source")
                self._clear_frame_list()
                self.range_slider.set_values(80, 850)
                self._update_pose_range(preferred=10)
                self.alignment_mode.setCurrentIndex(self.alignment_mode.findData("off"))
            else:
                self.source = SourceState(kind, paths)
                with QSignalBlocker(self.pose_count):
                    self.pose_count.setRange(2, max(2, len(paths)))
                    self.pose_count.setValue(len(paths))
                with QSignalBlocker(self.all_frames):
                    self.all_frames.setChecked(False)
                self._update_pose_count_label(len(paths))
                self.source_name.setText(paths[0].parent.name or "Photo stack")
                self.source_meta.setText(f"PHOTO STACK · {len(paths)} photographs")
                self.result_meta.setText(f"{len(paths)} selected photographs")
                self._populate_photo_frames(paths, "automatic")
                with QSignalBlocker(self.photo_order_mode):
                    self.photo_order_mode.setCurrentIndex(
                        self.photo_order_mode.findData("automatic")
                    )
                self.alignment_mode.setCurrentIndex(self.alignment_mode.findData("translation"))
            self._sync_all_frames_state()
            self.preview_heading.setText(paths[0].stem.replace("_", " "))
            self.settings_store.setValue("last_source_dir", str(paths[0].parent))
            self._set_loaded_state(True)
            self._loading_source = False
            self.render_preview()
        except Exception as exc:  # noqa: BLE001 - file input errors need friendly feedback
            self._loading_source = False
            QMessageBox.warning(self, "Could not open source", self._friendly_error(str(exc)))

    def _clear_preview_state(self) -> None:
        self.preview_result = None
        self.preview_frames = []
        self.preview_masks = []
        self.preview_labels = []
        self._preview_cache_key = None
        self._preview_cache_sequence = None
        self._video_preview_cache_key = None
        self._video_preview_cache_sequence = None
        self._video_analysis_cache = None
        self._video_frame_selection_key = None
        self._timeline_thumbnails = []
        self._last_export_path = None
        self.open_export_button.hide()
        self.range_slider.set_thumbnails([])
        self.playback_timer.stop()

    def _clear_frame_list(self) -> None:
        self._updating_frames = True
        self.frame_list.clear()
        self._updating_frames = False
        self._update_frame_count()

    def _populate_photo_frames(self, paths: list[Path], mode: str) -> None:
        ordered = order_image_paths(paths, mode)
        self._updating_frames = True
        self.frame_list.clear()
        self.frame_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        for path in ordered:
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setFlags(
                item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsDragEnabled
            )
            item.setCheckState(Qt.CheckState.Checked)
            self.frame_list.addItem(item)
        self._updating_frames = False
        self._update_frame_count()

    def _populate_video_frames(self, labels: list[str], selection_key: tuple[object, ...]) -> None:
        self._updating_frames = True
        self.frame_list.clear()
        self.frame_list.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        for index, label in enumerate(labels):
            item = QListWidgetItem(f"Pose {index + 1} · {label}")
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.frame_list.addItem(item)
        self._updating_frames = False
        self._video_frame_selection_key = selection_key
        self._update_frame_count()

    def _ordered_photo_paths(self) -> tuple[Path, ...]:
        paths: list[Path] = []
        for row in range(self.frame_list.count()):
            item = self.frame_list.item(row)
            if item.checkState() == Qt.CheckState.Checked:
                paths.append(Path(str(item.data(Qt.ItemDataRole.UserRole))))
        return tuple(paths)

    def _enabled_video_indices(
        self, pose_count: int | None, selection_key: tuple[object, ...]
    ) -> tuple[int, ...] | None:
        if not self.source or self.source.kind != "video" or self.frame_list.count() == 0:
            return None
        if self._video_frame_selection_key != selection_key:
            return None
        if pose_count is not None and self.frame_list.count() != pose_count:
            return None
        if pose_count is None and (
            self._preview_cache_sequence is None
            or self.frame_list.count() != len(self._preview_cache_sequence.frames)
        ):
            return None
        return tuple(
            int(self.frame_list.item(row).data(Qt.ItemDataRole.UserRole))
            for row in range(self.frame_list.count())
            if self.frame_list.item(row).checkState() == Qt.CheckState.Checked
        )

    def _frame_state_changed(self) -> None:
        if self._updating_frames:
            return
        self._update_frame_count()
        self._schedule_preview()

    def _frame_selected(self, row: int) -> None:
        if row < 0 or not self.preview_frames or not self.source:
            return
        item = self.frame_list.item(row)
        if self.source.kind == "video":
            source_index = int(item.data(Qt.ItemDataRole.UserRole))
            enabled = [
                int(self.frame_list.item(index).data(Qt.ItemDataRole.UserRole))
                for index in range(self.frame_list.count())
                if self.frame_list.item(index).checkState() == Qt.CheckState.Checked
            ]
            if source_index not in enabled:
                return
            preview_index = enabled.index(source_index)
        else:
            label = Path(str(item.data(Qt.ItemDataRole.UserRole))).name
            if label not in self.preview_labels:
                return
            preview_index = self.preview_labels.index(label)
        if preview_index < len(self.preview_frames):
            self.pose_scrubber.setValue(preview_index)

    def _frame_order_moved(self) -> None:
        if self._updating_frames or not self.source or self.source.kind != "photos":
            return
        with QSignalBlocker(self.photo_order_mode):
            self.photo_order_mode.setCurrentIndex(self.photo_order_mode.findData("input"))
        self._preview_cache_key = None
        self._schedule_preview()

    def _move_selected_frame(self, direction: int) -> None:
        if not self.source or self.source.kind != "photos":
            return
        row = self.frame_list.currentRow()
        target = row + direction
        if row < 0 or target < 0 or target >= self.frame_list.count():
            return
        self._updating_frames = True
        item = self.frame_list.takeItem(row)
        self.frame_list.insertItem(target, item)
        self.frame_list.setCurrentRow(target)
        self._updating_frames = False
        self._frame_order_moved()

    def _update_frame_count(self) -> None:
        enabled = sum(
            self.frame_list.item(row).checkState() == Qt.CheckState.Checked
            for row in range(self.frame_list.count())
        )
        self.frame_count_label.setText(f"{enabled} enabled")
        if self.source and self.source.kind == "photos":
            with QSignalBlocker(self.pose_count):
                self.pose_count.setValue(enabled)
            self._update_pose_count_label(enabled)

    def _estimated_selected_frame_count(self) -> int:
        if not self.source or not self.source.video_info:
            return 2
        start, end = self._video_range()
        info = self.source.video_info
        ratio = max(0.0, min(1.0, (end - start) / info.duration))
        return max(2, round(info.frame_count * ratio))

    def _update_pose_range(self, preferred: int | None = None) -> None:
        if not self.source or self.source.kind != "video":
            return
        available = self._estimated_selected_frame_count()
        value = self.pose_count.value() if preferred is None else preferred
        with QSignalBlocker(self.pose_count):
            self.pose_count.setRange(2, available)
            self.pose_count.setValue(max(2, min(value, available)))
        self._update_pose_count_label(self.pose_count.value())

    @Slot(int)
    def _update_pose_count_label(self, value: int) -> None:
        if self.source and self.source.kind == "video" and self.all_frames.isChecked():
            self.pose_count_value.setText(f"All · ~{self._estimated_selected_frame_count()}")
        elif self.source and self.source.kind == "photos":
            self.pose_count_value.setText(f"{value} photos")
        else:
            self.pose_count_value.setText(f"{value} poses")

    @Slot(int)
    def _pose_count_changed(self, value: int) -> None:
        self._update_pose_count_label(value)
        self._schedule_preview()

    @Slot(bool)
    def _all_frames_changed(self, checked: bool) -> None:
        self._sync_all_frames_state()
        self._schedule_preview()

    def _sync_all_frames_state(self) -> None:
        checked = self.all_frames.isChecked()
        self.pose_count.setEnabled(
            not checked and bool(self.source and self.source.kind == "video")
        )
        self.pose_control.setProperty("allFramesActive", checked)
        self.pose_control.style().unpolish(self.pose_control)
        self.pose_control.style().polish(self.pose_control)
        self._update_pose_count_label(self.pose_count.value())

    def _photo_order_changed(self) -> None:
        if self._loading_source or not self.source or self.source.kind != "photos":
            return
        mode = str(self.photo_order_mode.currentData())
        if mode != "input":
            self._populate_photo_frames(self.source.paths, mode)
        self._preview_cache_key = None
        self._schedule_preview()

    def _settings_snapshot(self) -> ComposeSettings:
        return ComposeSettings(
            threshold=self.threshold.value(),
            feather=self.feather.value(),
            overlap=str(self.overlap_mode.currentData()),
            trail_style=str(self.trail_style.currentData()),
            smear_style=str(self.smear_style.currentData()),
            background=str(self.background_mode.currentData()),
        )

    def _video_range(self) -> tuple[float, float]:
        assert self.source and self.source.video_info
        duration = self.source.video_info.duration
        return duration * self.range_slider.low / 1000, duration * self.range_slider.high / 1000

    def _render_request(self, max_dimension: int | None) -> RenderRequest:
        if not self.source:
            raise ValueError("Choose a video or photo stack first")
        if self.source.kind == "video":
            start, end = self._video_range()
            paths = tuple(self.source.paths)
            assert self.source.video_info is not None
            pose_count = None if self.all_frames.isChecked() else self.pose_count.value()
            video_selection_key = (
                str(paths[0]),
                round(start, 4),
                round(end, 4),
                pose_count,
            )
            enabled_indices = self._enabled_video_indices(pose_count, video_selection_key)
            if enabled_indices is not None and len(enabled_indices) < 2:
                raise ValueError("Enable at least two video poses")
            try:
                source_stat = paths[0].stat()
                file_signature: tuple[object, ...] = (
                    source_stat.st_size,
                    source_stat.st_mtime_ns,
                )
            except OSError:
                file_signature = (None, None)
            video_cache_key = (
                str(paths[0].resolve()),
                *file_signature,
                max_dimension,
            )
            video_duration = self.source.video_info.duration
        else:
            start, end = 0.0, 0.0
            paths = self._ordered_photo_paths()
            pose_count = len(paths)
            video_selection_key = None
            enabled_indices = None
            video_cache_key = None
            video_duration = 0.0
            if len(paths) < 2:
                raise ValueError("Enable at least two photographs")
        cache_key = (
            self.source.kind,
            tuple(str(path) for path in paths),
            round(start, 4),
            round(end, 4),
            pose_count,
            max_dimension,
        )
        return RenderRequest(
            kind=self.source.kind,
            paths=paths,
            start=start,
            end=end,
            pose_count=pose_count,
            settings=self._settings_snapshot(),
            alignment=str(self.alignment_mode.currentData()),
            max_dimension=max_dimension,
            cache_key=cache_key,
            video_selection_key=video_selection_key,
            video_cache_key=video_cache_key,
            video_duration=video_duration,
            enabled_video_indices=enabled_indices,
        )

    @Slot()
    def render_preview(self) -> None:
        if not self.source:
            return
        if self._thread:
            self._pending_preview = True
            return
        self.preview_debounce.stop()
        try:
            request = self._render_request(self.PREVIEW_MAX_DIMENSION)
        except ValueError as exc:
            self.status_text.setText("CHECK FRAMES")
            self.status_detail.setText(str(exc))
            return
        cached_photo_sequence = (
            self._preview_cache_sequence
            if request.kind == "photos" and self._preview_cache_key == request.cache_key
            else None
        )
        cached_video_sequence = (
            self._video_preview_cache_sequence
            if request.kind == "video" and self._video_preview_cache_key == request.video_cache_key
            else None
        )
        analysis_key = (
            request.video_cache_key,
            request.alignment,
            request.settings.threshold,
            request.settings.feather,
            request.settings.background,
            request.settings.min_component_ratio,
        )
        cached_analysis = (
            self._video_analysis_cache
            if request.kind == "video"
            and self._video_analysis_cache is not None
            and self._video_analysis_cache.key == analysis_key
            else None
        )
        need_thumbnails = request.kind == "video" and not self._timeline_thumbnails
        frame_list_changed = (
            request.kind == "video"
            and self._video_frame_selection_key != request.video_selection_key
        )

        def task(progress: Callable[[int, str], None]):
            video_cache = cached_video_sequence
            analysis_cache = cached_analysis
            cache_was_built = False
            analysis_was_built = False
            if request.kind == "video":
                if video_cache is None:
                    video_cache = load_video_sequence(
                        request.paths[0],
                        0.0,
                        request.video_duration,
                        None,
                        max_dimension=request.max_dimension,
                        progress=lambda value, message: progress(int(value * 0.28), message),
                    )
                    cache_was_built = True
                else:
                    progress(24, f"Using {len(video_cache.frames)} cached video frames")
                sequence = select_video_sequence(
                    video_cache,
                    request.start,
                    request.end,
                    request.pose_count,
                )
                progress(28, f"Selected {len(sequence.frames)} cached frames")
            elif cached_photo_sequence is not None:
                sequence = cached_photo_sequence
                progress(18, "Using cached source frames")
            else:
                sequence = load_image_sequence(
                    request.paths,
                    max_dimension=request.max_dimension,
                    sort_mode="input",
                    progress=lambda value, message: progress(int(value * 0.28), message),
                )
            timeline_frames: list[np.ndarray] = []
            if need_thumbnails and video_cache is not None:
                thumbnail_indices = np.linspace(
                    0,
                    len(video_cache.frames) - 1,
                    min(10, len(video_cache.frames)),
                    dtype=int,
                )
                timeline_frames = [video_cache.frames[int(index)] for index in thumbnail_indices]
            labels = sequence.labels
            if request.kind == "video":
                assert video_cache is not None
                if analysis_cache is None:
                    progress(29, "Analyzing the decoded video once")
                    aligned_cache_frames = align_sequence(
                        video_cache.frames,
                        request.alignment,
                        progress=lambda value, message: progress(29 + int(value * 0.11), message),
                    )
                    reusable_masks = build_compose_cache(
                        aligned_cache_frames,
                        request.settings,
                        progress=lambda value, message: progress(40 + int(value * 0.30), message),
                    )
                    analysis_cache = VideoAnalysisCache(
                        analysis_key,
                        aligned_cache_frames,
                        reusable_masks,
                    )
                    analysis_was_built = True
                else:
                    progress(40, f"Using {len(analysis_cache.compose_cache.masks)} cached masks")

                source_indices = sequence.source_indices
                if source_indices is None:
                    raise RuntimeError("Selected video frames have no cache indices")
                selected_positions = list(range(len(source_indices)))
                if request.enabled_video_indices is not None:
                    selected_positions = [
                        index
                        for index in request.enabled_video_indices
                        if index < len(source_indices)
                    ]
                selected_source_indices = [source_indices[index] for index in selected_positions]
                frames = [analysis_cache.frames[index] for index in selected_source_indices]
                labels = [labels[index] for index in selected_positions]
                selected_cache = analysis_cache.compose_cache.select(selected_source_indices)
                compose_start = 70 if analysis_was_built else 40
                compose_span = 30 if analysis_was_built else 60
                result, masks = compose_sequence(
                    frames,
                    request.settings,
                    progress=lambda value, message: progress(
                        compose_start + int(value * compose_span / 100), message
                    ),
                    cache=selected_cache,
                )
                aligned = frames
            else:
                frames = sequence.frames
                aligned = align_sequence(
                    frames,
                    request.alignment,
                    progress=lambda value, message: progress(28 + int(value * 0.12), message),
                )
                result, masks = compose_sequence(
                    aligned,
                    request.settings,
                    progress=lambda value, message: progress(40 + int(value * 0.60), message),
                )
            return {
                "result": result,
                "masks": masks,
                "frames": aligned,
                "labels": labels,
                "sequence": sequence,
                "cache_key": request.cache_key,
                "video_cache_sequence": video_cache,
                "video_cache_key": request.video_cache_key,
                "video_cache_was_built": cache_was_built,
                "video_analysis_cache": analysis_cache,
                "video_analysis_was_built": analysis_was_built,
                "frame_list_changed": frame_list_changed,
                "source_size": sequence.source_size,
                "timeline_frames": timeline_frames,
                "kind": request.kind,
                "video_selection_key": request.video_selection_key,
            }

        if request.kind == "video" and cached_video_sequence is not None:
            task_detail = "Recomposing cached frames"
        elif request.kind == "video":
            task_detail = "Building video frame cache"
        else:
            task_detail = "Rendering preview"
        self._start_task(task, self._preview_finished, task_detail)

    @Slot(object)
    def _preview_finished(self, payload: object) -> None:
        data = dict(payload)  # type: ignore[arg-type]
        self.preview_result = data["result"]
        self.preview_masks = data["masks"]
        self.preview_frames = data["frames"]
        self.preview_labels = data["labels"]
        self._preview_cache_sequence = data["sequence"]
        self._preview_cache_key = data["cache_key"]
        if data["video_cache_sequence"] is not None:
            self._video_preview_cache_sequence = data["video_cache_sequence"]
            self._video_preview_cache_key = data["video_cache_key"]
        if data["video_analysis_cache"] is not None:
            self._video_analysis_cache = data["video_analysis_cache"]
        if data["timeline_frames"]:
            self._timeline_thumbnails = data["timeline_frames"]
            self.range_slider.set_thumbnails(
                [image_to_qimage(frame) for frame in self._timeline_thumbnails]
            )
        if data["kind"] == "video" and data["frame_list_changed"]:
            self._populate_video_frames(
                data["sequence"].labels,
                data["video_selection_key"],
            )
        pose_total = len(self.preview_frames)
        self.pose_scrubber.setRange(0, max(0, pose_total - 1))
        self.pose_scrubber.setValue(min(self.pose_scrubber.value(), max(0, pose_total - 1)))
        source_size = data["source_size"]
        unit = "frames" if self.all_frames.isChecked() and data["kind"] == "video" else "poses"
        self.result_meta.setText(
            f"{pose_total} {unit} · {source_size[0]} × {source_size[1]} preview"
        )
        if self.all_frames.isChecked() and data["kind"] == "video":
            self.pose_count_value.setText(f"All · {len(data['sequence'].frames)}")
        self._preview_dirty = False
        self._refresh_preview_canvas()
        if data["kind"] == "video":
            cached_total = len(data["video_cache_sequence"].frames)
            if data["video_analysis_was_built"]:
                detail = f"Cached {cached_total} frames and masks · in/out reuses both"
            else:
                detail = f"Recomposed from {cached_total} cached frames and masks"
        else:
            detail = "Inspect Source or Mask, then export"
        self._finish_task("PREVIEW READY", detail)

    @Slot()
    def export_composite(self) -> None:
        if not self.source or self._thread:
            return
        self.preview_debounce.stop()
        self._pending_preview = False
        try:
            request = self._render_request(None)
        except ValueError as exc:
            self.status_text.setText("CHECK FRAMES")
            self.status_detail.setText(str(exc))
            return
        last_dir = Path(str(self.settings_store.value("last_export_dir", str(Path.home()))))
        suggested = last_dir / f"{request.paths[0].stem}-chronophoto.png"
        path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export full-resolution composite",
            str(suggested),
            "PNG image (*.png);;TIFF image (*.tif *.tiff);;JPEG image (*.jpg *.jpeg)",
        )
        if not path:
            return
        export_path = self._export_path_with_filter(Path(path), selected_filter)
        self.settings_store.setValue("last_export_dir", str(export_path.parent))

        def task(progress: Callable[[int, str], None]):
            progress(2, "Reading full-resolution source")
            if request.kind == "video":
                sequence = load_video_sequence(
                    request.paths[0],
                    request.start,
                    request.end,
                    request.pose_count,
                    progress=lambda value, message: progress(int(value * 0.30), message),
                )
            else:
                sequence = load_image_sequence(
                    request.paths,
                    sort_mode="input",
                    progress=lambda value, message: progress(int(value * 0.30), message),
                )
            frames = sequence.frames
            if request.kind == "video" and request.enabled_video_indices is not None:
                frames = [
                    frames[index] for index in request.enabled_video_indices if index < len(frames)
                ]
            aligned = align_sequence(
                frames,
                request.alignment,
                progress=lambda value, message: progress(30 + int(value * 0.12), message),
            )
            result, _ = compose_sequence(
                aligned,
                request.settings,
                progress=lambda value, message: progress(42 + int(value * 0.53), message),
                return_masks=False,
            )
            progress(97, "Writing image")
            image = Image.fromarray(result)
            suffix = export_path.suffix.casefold()
            if suffix in {".tif", ".tiff"}:
                image.save(export_path, compression="tiff_lzw")
            elif suffix in {".jpg", ".jpeg"}:
                image.save(export_path, quality=95, subsampling=0)
            else:
                image.save(export_path, compress_level=6)
            return export_path, result

        self._start_task(task, self._export_finished, "Composing full resolution")

    @Slot(object)
    def _export_finished(self, payload: object) -> None:
        path, result = payload  # type: ignore[misc]
        self.preview_result = result
        self._preview_dirty = False
        self._last_export_path = Path(path)
        self._set_preview_mode("composite")
        self.result_meta.setText(f"Full resolution · {result.shape[1]} × {result.shape[0]}")
        self.open_export_button.show()
        self._finish_task("EXPORT COMPLETE", Path(path).name)

    def _start_task(
        self,
        task: Callable[[Callable[[int, str], None]], object],
        on_finished: Callable[[object], None],
        detail: str,
    ) -> None:
        self._set_busy(True)
        self.status_text.setText("WORKING")
        self.status_detail.setText(detail)
        self.progress.setValue(1)
        thread = QThread(self)
        worker = TaskWorker(task)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._update_progress)
        worker.finished.connect(on_finished)
        worker.failed.connect(self._task_failed)
        worker.cancelled.connect(self._task_cancelled)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        worker.cancelled.connect(worker.deleteLater)
        thread.finished.connect(self._thread_stopped)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot(int, str)
    def _update_progress(self, value: int, message: str) -> None:
        self.progress.setValue(value)
        self.status_detail.setText(message)

    @Slot(str)
    def _task_failed(self, message: str) -> None:
        friendly = self._friendly_error(message)
        self._finish_task("COULD NOT RENDER", friendly)
        QMessageBox.warning(self, "Processing stopped", friendly)

    @Slot()
    def _task_cancelled(self) -> None:
        self._finish_task("CANCELLED", "The previous successful preview is still available")

    def _finish_task(self, status: str, detail: str) -> None:
        self.status_text.setText(status)
        self.status_detail.setText(detail)
        self.progress.setValue(0)

    @Slot()
    def _thread_stopped(self) -> None:
        self._thread = None
        self._worker = None
        self._set_busy(False)
        if self._close_when_done:
            self._close_when_done = False
            QTimer.singleShot(0, self.close)
            return
        if self._pending_preview:
            self._pending_preview = False
            QTimer.singleShot(0, self.render_preview)

    def _cancel_task(self) -> None:
        if self._worker:
            self.status_text.setText("CANCELLING")
            self.status_detail.setText("Stopping after the current processing step")
            self._worker.request_cancel()
            self.cancel_button.setEnabled(False)

    def _set_busy(self, busy: bool) -> None:
        loaded = self.source is not None
        for control in (
            self.video_button,
            self.photos_button,
            self.drop_surface,
            self.export_button,
            self.threshold,
            self.feather,
            self.background_mode,
            self.overlap_mode,
            self.smear_style,
            self.alignment_mode,
            self.photo_order_mode,
            self.move_up_button,
            self.move_down_button,
            self.reset_button,
        ):
            control.setEnabled(
                not busy
                and (
                    loaded or control in (self.video_button, self.photos_button, self.drop_surface)
                )
            )
        is_video = loaded and bool(self.source and self.source.kind == "video")
        self.pose_count.setEnabled(is_video and not self.all_frames.isChecked())
        self.all_frames.setEnabled(is_video)
        self.range_slider.setEnabled(is_video)
        self.frame_list.setEnabled(loaded)
        self.cancel_button.setVisible(busy)
        self.cancel_button.setEnabled(busy)

    def _set_loaded_state(self, loaded: bool) -> None:
        self.source_intro.setVisible(not loaded)
        self.source_summary.setVisible(loaded)
        self.frames_section.setVisible(loaded)
        self.timeline_panel.setVisible(loaded and bool(self.source and self.source.kind == "video"))
        self.pose_navigation.setVisible(loaded)
        self.export_button.setEnabled(loaded)
        self.reset_button.setEnabled(loaded)
        self.threshold.setEnabled(loaded)
        self.feather.setEnabled(loaded)
        self.background_mode.setEnabled(loaded)
        self.overlap_mode.setEnabled(loaded)
        self.trail_style.setEnabled(False)
        self.smear_style.setEnabled(loaded)
        self.alignment_mode.setEnabled(loaded)
        is_video = loaded and bool(self.source and self.source.kind == "video")
        self.pose_control.setVisible(is_video)
        self.pose_count.setEnabled(is_video and not self.all_frames.isChecked())
        self.all_frames.setEnabled(is_video)
        self.range_slider.setEnabled(is_video)
        is_photo = loaded and bool(self.source and self.source.kind == "photos")
        self.photo_order_control.setVisible(is_photo)
        self.move_up_button.setVisible(is_photo)
        self.move_down_button.setVisible(is_photo)

    def _schedule_preview(self) -> None:
        if self._loading_source or not self.source:
            return
        self._mark_preview_dirty("Updating after your changes")
        if self._thread:
            self._pending_preview = True
            return
        self.preview_debounce.start()

    def _mark_preview_dirty(self, detail: str) -> None:
        self._preview_dirty = True
        self.status_text.setText("PREVIEW OUT OF DATE")
        self.status_detail.setText(detail)
        self._refresh_preview_canvas()

    @Slot(int, int)
    def _range_changed(self, low: int, high: int) -> None:
        if not self.source or not self.source.video_info:
            return
        duration = self.source.video_info.duration
        self.start_time.setText(self._format_time(duration * low / 1000))
        self.end_time.setText(self._format_time(duration * high / 1000))
        self._update_pose_range()
        self._preview_dirty = True
        self.status_text.setText("PREVIEW OUT OF DATE")
        self.status_detail.setText("Release the range handle to update")
        if not self.range_slider.is_adjusting:
            self._refresh_preview_canvas()

    @Slot(str, int)
    def _range_handle_changed(self, handle: str, value: int) -> None:
        if handle not in {"low", "high"} or not self.source or not self.source.video_info:
            return
        sequence = self._video_preview_cache_sequence
        if sequence is None or not sequence.frames:
            return

        target_time = self.source.video_info.duration * value / 1000
        timestamps = sequence.timestamps
        if len(timestamps) == len(sequence.frames) and timestamps:
            insertion = bisect_left(timestamps, target_time)
            candidates = {
                max(0, insertion - 1),
                min(len(timestamps) - 1, insertion),
            }
            frame_index = min(candidates, key=lambda index: abs(timestamps[index] - target_time))
            frame_time = timestamps[frame_index]
        else:
            frame_index = round(value / 1000 * (len(sequence.frames) - 1))
            frame_time = target_time

        marker = "IN" if handle == "low" else "OUT"
        self.preview_canvas.set_image(
            image_to_qimage(sequence.frames[frame_index]),
            f"{marker} · {self._format_time(frame_time)}",
        )

    @Slot(int, int)
    def _range_committed(self, low: int, high: int) -> None:
        del low, high
        self.preview_debounce.stop()
        self.render_preview()

    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
        self.advanced_controls.setVisible(checked)

    def _reset_controls(self) -> None:
        self._loading_source = True
        with QSignalBlocker(self.all_frames):
            self.all_frames.setChecked(bool(self.source and self.source.kind == "video"))
        self._sync_all_frames_state()
        self._update_pose_range(preferred=10)
        self.threshold.setValue(17)
        self.feather.setValue(1)
        self.background_mode.setCurrentIndex(self.background_mode.findData("median"))
        self.overlap_mode.setCurrentIndex(self.overlap_mode.findData("newest"))
        self.trail_style.setCurrentIndex(self.trail_style.findData("solid"))
        self.smear_style.setCurrentIndex(self.smear_style.findData("none"))
        if self.source and self.source.kind == "video":
            self.range_slider.set_values(80, 850)
            self.alignment_mode.setCurrentIndex(self.alignment_mode.findData("off"))
        else:
            self.alignment_mode.setCurrentIndex(self.alignment_mode.findData("translation"))
        self._loading_source = False
        self._schedule_preview()

    def _set_preview_mode(self, mode: str) -> None:
        button = self.preview_mode_buttons.get(mode)
        if button:
            button.setChecked(True)
        if mode == "composite":
            self.playback_timer.stop()
            self.play_button.setText("Play")
        self._refresh_preview_canvas()

    def _current_preview_mode(self) -> str:
        for mode, button in self.preview_mode_buttons.items():
            if button.isChecked():
                return mode
        return "composite"

    def _refresh_preview_canvas(self) -> None:
        mode = self._current_preview_mode()
        pose_total = len(self.preview_frames)
        position = min(self.pose_scrubber.value(), max(0, pose_total - 1))
        self.pose_position.setText(f"{position + 1 if pose_total else 0} / {pose_total}")
        if mode == "composite" and self.preview_result is not None:
            status = "PREVIEW OUTDATED" if self._preview_dirty else "COMPOSITE"
            self.preview_canvas.set_image(image_to_qimage(self.preview_result), status)
        elif mode == "source" and pose_total:
            label = (
                self.preview_labels[position] if position < len(self.preview_labels) else "SOURCE"
            )
            self.preview_canvas.set_image(image_to_qimage(self.preview_frames[position]), label)
        elif mode == "mask" and pose_total and position < len(self.preview_masks):
            self.preview_canvas.set_image(
                mask_overlay_to_qimage(self.preview_frames[position], self.preview_masks[position]),
                f"MASK · POSE {position + 1}",
            )
        else:
            self.preview_canvas.set_image(None, "WAITING FOR FOOTAGE")

    def _toggle_playback(self) -> None:
        if not self.preview_frames:
            return
        if self._current_preview_mode() == "composite":
            self._set_preview_mode("source")
        if self.playback_timer.isActive():
            self.playback_timer.stop()
            self.play_button.setText("Play")
        else:
            self.playback_timer.start()
            self.play_button.setText("Pause")

    def _advance_pose(self) -> None:
        if not self.preview_frames:
            self.playback_timer.stop()
            return
        self.pose_scrubber.setValue((self.pose_scrubber.value() + 1) % len(self.preview_frames))

    def _open_export_folder(self) -> None:
        if self._last_export_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_export_path.parent)))

    @staticmethod
    def _friendly_error(message: str) -> str:
        if "Could not decode" in message:
            return (
                "The selected range contains fewer usable frames than requested. "
                "Widen the range or reduce the pose count."
            )
        if "dimensions" in message:
            return (
                "The photographs have different dimensions. "
                "Export them at one shared size before stacking."
            )
        if "At least two" in message:
            return "Enable at least two frames before rendering."
        return message

    @staticmethod
    def _export_path_with_filter(path: Path, selected_filter: str) -> Path:
        if path.suffix:
            return path
        if selected_filter.startswith("TIFF"):
            return path.with_suffix(".tif")
        if selected_filter.startswith("JPEG"):
            return path.with_suffix(".jpg")
        return path.with_suffix(".png")

    @staticmethod
    def _format_time(seconds: float) -> str:
        minutes, remainder = divmod(max(0.0, seconds), 60)
        return f"{int(minutes):02d}:{remainder:05.2f}"

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._thread and self._worker:
            self._close_when_done = True
            self._worker.request_cancel()
            self.status_text.setText("CANCELLING")
            self.status_detail.setText("Closing after the current processing step")
            event.ignore()
            return
        event.accept()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if hasattr(self, "drop_overlay") and self.centralWidget():
            self.drop_overlay.setGeometry(self.centralWidget().rect())
        super().resizeEvent(event)
