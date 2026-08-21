from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from chronophoto.processing import (
    TimeSurface,
    load_video_sequence,
    silhouette_edge_stretch,
    slice_video_volume,
)
from chronophoto.processing.sources import probe_video
from chronophoto.ui.widgets import PreviewCanvas, ScrollSafeComboBox, image_to_qimage
from chronophoto.ui.window import ChronophotoWindow


class EffectCard(QFrame):
    def __init__(self, number: str, title: str, copy: str, action: Callable[[], None]):
        super().__init__()
        self.setObjectName("effectCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 22, 22, 22)
        for text, name in ((number, "eyebrow"), (title, "effectCardTitle")):
            label = QLabel(text)
            label.setObjectName(name)
            layout.addWidget(label)
        description = QLabel(copy)
        description.setObjectName("bodyMuted")
        description.setWordWrap(True)
        layout.addWidget(description, 1)
        button = QPushButton("OPEN WORKSPACE")
        button.setObjectName("secondaryButton")
        button.clicked.connect(action)
        layout.addWidget(button)


class EffectLauncher(QMainWindow):
    """Application home, separating effects into purpose-built workflows."""

    def __init__(self, *, check_updates: bool = False) -> None:
        super().__init__()
        self.check_updates = check_updates
        self.workspace: QMainWindow | None = None
        self.setWindowTitle("Chronophoto — Choose an effect")
        self.resize(1120, 700)
        root = QWidget()
        root.setObjectName("launcher")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(54, 42, 54, 48)
        eyebrow = QLabel("CHRONOPHOTO / EFFECT LIBRARY")
        eyebrow.setObjectName("eyebrow")
        title = QLabel("What do you want to make?")
        title.setObjectName("launcherTitle")
        intro = QLabel(
            "Choose a workflow. Your media stays local, and Photostack keeps its complete "
            "original interface."
        )
        intro.setObjectName("bodyMuted")
        layout.addWidget(eyebrow)
        layout.addWidget(title)
        layout.addWidget(intro)
        cards = QHBoxLayout()
        cards.setSpacing(16)
        specs = (
            ("01 / PHOTOSTACK", "Motion, held still.",
             "Layer selected poses from video or a photo sequence. Mask, order, grade, "
             "animate, and export.", "photostack"),
            ("02 / EDGE STRETCH", "Pull a silhouette apart.",
             "Detect a boundary, sample its edge per scanline, then extrude the colour "
             "horizontally or vertically.", "edge"),
            ("03 / VIDEO VOLUME", "Cut through space and time.",
             "Treat a clip as X × Y × Time. Explore XT, YT, diagonal, time-plane, and "
             "animated arbitrary surfaces.", "volume"),
        )
        for number, heading, copy, kind in specs:
            cards.addWidget(EffectCard(number, heading, copy, lambda k=kind: self._open(k)))
        layout.addLayout(cards, 1)
        self.setCentralWidget(root)

    def _open(self, kind: str) -> None:
        window = (
            ChronophotoWindow(check_updates=self.check_updates)
            if kind == "photostack" else ExperimentalEffectWindow(kind)
        )
        window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        window.destroyed.connect(self._return_home)
        self.workspace = window
        window.show()
        self.hide()

    def _return_home(self) -> None:
        self.workspace = None
        self.show()


class ExperimentalEffectWindow(QMainWindow):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.kind = kind
        self.frames: np.ndarray | None = None
        self.result: np.ndarray | None = None
        self.phase = 0
        effect_name = "Silhouette Edge Stretch" if kind == "edge" else "Video Volume"
        self.setWindowTitle(f"Chronophoto — {effect_name}")
        self.resize(1220, 780)
        self._build_ui()
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._animate)

    def _build_ui(self) -> None:
        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        header = QFrame()
        header.setObjectName("appHeader")
        header_layout = QHBoxLayout(header)
        mark = QLabel("CHRONOPHOTO")
        mark.setObjectName("wordmark")
        name = QLabel("EDGE STRETCH" if self.kind == "edge" else "VIDEO VOLUME")
        name.setObjectName("strapline")
        header_layout.addWidget(mark)
        header_layout.addWidget(name)
        header_layout.addStretch()
        header_layout.addWidget(QLabel("CLOSE WORKSPACE TO RETURN HOME"))
        outer.addWidget(header)
        body = QHBoxLayout()
        body.setContentsMargins(18, 18, 18, 18)
        controls = QFrame()
        controls.setObjectName("sourcePanel")
        controls.setFixedWidth(300)
        form = QVBoxLayout(controls)
        title = QLabel("Build the stretch." if self.kind == "edge" else "Slice the volume.")
        title.setObjectName("panelTitleCompact")
        hint = QLabel(
            "The first frame becomes the clean-plate reference for silhouette detection."
            if self.kind == "edge" else "A video becomes a complete X × Y × Time volume."
        )
        hint.setObjectName("bodyMuted")
        hint.setWordWrap(True)
        load = QPushButton("LOAD VIDEO")
        load.setObjectName("primaryButton")
        load.clicked.connect(self._load_video)
        form.addWidget(title)
        form.addWidget(hint)
        form.addWidget(load)
        self.mode = ScrollSafeComboBox()
        choices = (
            (("Stretch right", "right"), ("Stretch left", "left"),
             ("Both horizontal", "both-horizontal"), ("Stretch down", "down"),
             ("Stretch up", "up"), ("Both vertical", "both-vertical"))
            if self.kind == "edge" else
            (("XY · time frame", "xy"), ("XT · horizontal cut", "xt"),
             ("YT · vertical cut", "yt"), ("Diagonal through time", "diagonal"),
             ("Arbitrary t=f(x,y)", "surface"))
        )
        for label, value in choices:
            self.mode.addItem(label, value)
        self.mode.currentIndexChanged.connect(self._render)
        form.addWidget(self._labelled("SLICE / DIRECTION", self.mode))
        self.position = self._slider(50)
        self.amount = self._slider(35 if self.kind == "edge" else 50)
        self.secondary = self._slider(50)
        self.tertiary = self._slider(0)
        labels = (
            ("POSITION", "MASK SENSITIVITY", "STRETCH LENGTH", "EDGE FADE")
            if self.kind == "edge" else
            ("POSITION", "X SLOPE", "Y SLOPE", "TIME OFFSET")
        )
        sliders = (self.position, self.amount, self.secondary, self.tertiary)
        for label, slider in zip(labels, sliders, strict=True):
            form.addWidget(self._labelled(label, slider))
        if self.kind == "volume":
            self.animate_button = QPushButton("PLAY ANIMATED SLICE")
            self.animate_button.setObjectName("secondaryButton")
            self.animate_button.clicked.connect(self._toggle_animation)
            form.addWidget(self.animate_button)
        form.addStretch()
        export = QPushButton("EXPORT FRAME")
        export.setObjectName("primaryButton")
        export.clicked.connect(self._export)
        form.addWidget(export)
        self.canvas = PreviewCanvas()
        self.canvas.set_image(None, "LOAD A VIDEO TO BEGIN")
        body.addWidget(controls)
        body.addWidget(self.canvas, 1)
        outer.addLayout(body, 1)
        self.setCentralWidget(root)

    def _slider(self, value: int) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(value)
        slider.valueChanged.connect(self._render)
        return slider

    @staticmethod
    def _labelled(text: str, widget: QWidget) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 7, 0, 2)
        label = QLabel(text)
        label.setObjectName("controlLabel")
        layout.addWidget(label)
        layout.addWidget(widget)
        return box

    def _load_video(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open video", "", "Video files (*.mp4 *.mov *.mkv *.avi)"
        )
        if not filename:
            return
        try:
            info = probe_video(filename)
            sequence = load_video_sequence(filename, 0.0, info.duration, None, max_dimension=720)
            self.frames = np.stack(sequence.frames)
            self._render()
        except Exception as exc:
            QMessageBox.critical(self, "Could not load video", str(exc))

    def _render(self) -> None:
        if self.frames is None:
            return
        if self.kind == "edge":
            index = round(self.position.value() / 100 * (len(self.frames) - 1))
            frame = self.frames[index]
            delta = np.max(
                np.abs(frame.astype(np.int16) - self.frames[0].astype(np.int16)), axis=2
            )
            mask = delta >= max(2, self.amount.value())
            self.result = silhouette_edge_stretch(
                frame, mask, direction=self.mode.currentData(),
                distance=round(self.secondary.value() / 100 * max(frame.shape[:2])),
                fade=self.tertiary.value() / 100,
            )
        else:
            surface = TimeSurface(
                base=self.tertiary.value() / 100,
                x_slope=(self.amount.value() - 50) / 25,
                y_slope=(self.secondary.value() - 50) / 25,
            )
            self.result = slice_video_volume(
                self.frames, self.mode.currentData(), position=self.position.value() / 100,
                surface=surface, phase=self.phase / 100,
            )
        self.canvas.set_image(image_to_qimage(self.result), "LIVE PREVIEW")

    def _toggle_animation(self) -> None:
        active = not self.timer.isActive()
        self.animate_button.setText("STOP ANIMATED SLICE" if active else "PLAY ANIMATED SLICE")
        self.timer.start() if active else self.timer.stop()

    def _animate(self) -> None:
        self.phase = (self.phase + 1) % 101
        self._render()

    def _export(self) -> None:
        if self.result is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export frame", "chronophoto.png", "PNG image (*.png)"
        )
        if filename:
            Image.fromarray(self.result).save(Path(filename).with_suffix(".png"))
