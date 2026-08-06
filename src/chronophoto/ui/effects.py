from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRectF, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from chronophoto.processing import (
    BLEND_MODE_LABELS,
    BLEND_MODES,
    EFFECT_KINDS,
    EFFECT_LABELS,
    EffectKeyframe,
    EffectTrack,
    effect_preset,
    neutral_effect_track,
)
from chronophoto.ui.widgets import ScrollSafeComboBox


class ScrollSafeSpinBox(QSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class EffectKeyframeGraph(QWidget):
    keyframes_changing = Signal(object)
    keyframes_committed = Signal(object)
    selection_changed = Signal(float, float)

    def __init__(
        self, keyframes: tuple[EffectKeyframe, ...], parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setObjectName("effectGraph")
        self.setMinimumHeight(60)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Linear effect keyframes")
        self.setAccessibleDescription(
            "Double-click to add a keyframe. Drag or use arrow keys to adjust it."
        )
        self.setToolTip(
            "Double-click to add · drag to shape · arrows adjust · Delete removes selected"
        )
        self._keyframes = list(keyframes)
        self._selected = 0
        self._dragging = False

    @property
    def keyframes(self) -> tuple[EffectKeyframe, ...]:
        return tuple(self._keyframes)

    @property
    def selected_index(self) -> int:
        return self._selected

    def set_keyframes(self, keyframes: tuple[EffectKeyframe, ...]) -> None:
        self._keyframes = list(keyframes)
        self._selected = min(self._selected, len(self._keyframes) - 1)
        self._emit_selection()
        self.update()

    def _plot(self) -> QRectF:
        return QRectF(12, 9, max(1, self.width() - 24), max(1, self.height() - 18))

    def _point(self, keyframe: EffectKeyframe) -> QPointF:
        plot = self._plot()
        return QPointF(
            plot.left() + keyframe.progress * plot.width(),
            plot.bottom() - keyframe.value / 100.0 * plot.height(),
        )

    def _keyframe_at(self, point: QPointF) -> int | None:
        distances = [
            abs(point.x() - self._point(keyframe).x()) + abs(point.y() - self._point(keyframe).y())
            for keyframe in self._keyframes
        ]
        if not distances:
            return None
        closest = min(range(len(distances)), key=distances.__getitem__)
        return closest if distances[closest] <= 18 else None

    def _values_for_point(self, point: QPointF) -> tuple[float, float]:
        plot = self._plot()
        progress = (point.x() - plot.left()) / max(1.0, plot.width())
        value = (plot.bottom() - point.y()) / max(1.0, plot.height()) * 100.0
        return max(0.0, min(1.0, progress)), max(0.0, min(100.0, value))

    def _set_selected_point(self, point: QPointF, *, live: bool) -> None:
        progress, value = self._values_for_point(point)
        index = self._selected
        if index == 0:
            progress = 0.0
        elif index == len(self._keyframes) - 1:
            progress = 1.0
        else:
            minimum = self._keyframes[index - 1].progress + 0.01
            maximum = self._keyframes[index + 1].progress - 0.01
            progress = max(minimum, min(maximum, progress))
        self._keyframes[index] = EffectKeyframe(progress, value)
        self._emit_selection()
        self.update()
        signal = self.keyframes_changing if live else self.keyframes_committed
        signal.emit(self.keyframes)

    def _emit_selection(self) -> None:
        if self._keyframes:
            selected = self._keyframes[self._selected]
            self.selection_changed.emit(selected.progress, selected.value)

    def set_selected_values(self, progress: float, value: float, *, live: bool) -> None:
        index = self._selected
        if index == 0:
            progress = 0.0
        elif index == len(self._keyframes) - 1:
            progress = 1.0
        else:
            progress = max(
                self._keyframes[index - 1].progress + 0.01,
                min(self._keyframes[index + 1].progress - 0.01, progress),
            )
        self._keyframes[index] = EffectKeyframe(progress, max(0.0, min(100.0, value)))
        self._emit_selection()
        self.update()
        signal = self.keyframes_changing if live else self.keyframes_committed
        signal.emit(self.keyframes)

    def commit_selected(self) -> None:
        self.keyframes_committed.emit(self.keyframes)

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        plot = self._plot()
        painter.fillRect(self.rect(), QColor("#0a0a0a"))
        painter.setPen(QPen(QColor("#292929"), 1))
        for fraction in (0.25, 0.5, 0.75):
            x = plot.left() + fraction * plot.width()
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
        painter.drawLine(
            QPointF(plot.left(), plot.center().y()), QPointF(plot.right(), plot.center().y())
        )
        painter.setPen(QPen(QColor("#d8d8d8"), 1.5))
        points = [self._point(keyframe) for keyframe in self._keyframes]
        for left, right in zip(points, points[1:], strict=False):
            painter.drawLine(left, right)
        for index, point in enumerate(points):
            size = 5.5 if index == self._selected else 4.5
            diamond = (
                QPointF(point.x(), point.y() - size),
                QPointF(point.x() + size, point.y()),
                QPointF(point.x(), point.y() + size),
                QPointF(point.x() - size, point.y()),
            )
            painter.setBrush(QColor("#f0f0f0") if index == self._selected else QColor("#9a9a9a"))
            painter.setPen(QPen(QColor("#050505"), 1))
            painter.drawPolygon(diamond)
        if self.hasFocus():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#eeeeee"), 1))
            painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        selected = self._keyframe_at(event.position())
        if selected is not None:
            self._selected = selected
            self._dragging = True
            self._emit_selection()
            self.update()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self._set_selected_point(event.position(), live=True)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._set_selected_point(event.position(), live=False)
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseDoubleClickEvent(event)
            return
        progress, value = self._values_for_point(event.position())
        if progress <= 0.01 or progress >= 0.99:
            return
        if any(abs(keyframe.progress - progress) < 0.015 for keyframe in self._keyframes):
            return
        self._keyframes.append(EffectKeyframe(progress, value))
        self._keyframes.sort(key=lambda keyframe: keyframe.progress)
        self._selected = next(
            index for index, keyframe in enumerate(self._keyframes) if keyframe.progress == progress
        )
        self._emit_selection()
        self.update()
        self.keyframes_committed.emit(self.keyframes)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key.Key_Delete and 0 < self._selected < len(self._keyframes) - 1:
            self._keyframes.pop(self._selected)
            self._selected = min(self._selected, len(self._keyframes) - 1)
            self._emit_selection()
            self.update()
            self.keyframes_committed.emit(self.keyframes)
            event.accept()
            return
        if event.key() not in {
            Qt.Key.Key_Left,
            Qt.Key.Key_Right,
            Qt.Key.Key_Up,
            Qt.Key.Key_Down,
        }:
            super().keyPressEvent(event)
            return
        step = 5.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
        current = self._keyframes[self._selected]
        progress = current.progress
        value = current.value
        if (
            event.key() in {Qt.Key.Key_Left, Qt.Key.Key_Right}
            and 0 < self._selected < len(self._keyframes) - 1
        ):
            direction = -1.0 if event.key() == Qt.Key.Key_Left else 1.0
            minimum = self._keyframes[self._selected - 1].progress + 0.01
            maximum = self._keyframes[self._selected + 1].progress - 0.01
            progress = max(minimum, min(maximum, progress + direction * step / 100.0))
        elif event.key() in {Qt.Key.Key_Up, Qt.Key.Key_Down}:
            direction = 1.0 if event.key() == Qt.Key.Key_Up else -1.0
            value = max(0.0, min(100.0, value + direction * step))
        self._keyframes[self._selected] = EffectKeyframe(progress, value)
        self._emit_selection()
        self.update()
        self.keyframes_committed.emit(self.keyframes)
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class LaneDragHandle(QLabel):
    drag_started = Signal()
    drag_moved = Signal(QPoint)
    drag_finished = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("::", parent)
        self.setObjectName("effectDragHandle")
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setAccessibleName("Drag to reorder effect")
        self._dragging = False

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self.drag_started.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self.drag_moved.emit(event.globalPosition().toPoint())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            self.drag_finished.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class EffectLane(QFrame):
    track_changing = Signal(object)
    track_committed = Signal(object)
    remove_requested = Signal(object)
    drag_started = Signal(object)
    drag_moved = Signal(object, QPoint)
    drag_finished = Signal(object)

    def __init__(
        self,
        track: EffectTrack,
        *,
        keyframed: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("effectLane")
        self._track = track
        self._keyframed = keyframed
        self._narrow = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(5)

        header = QHBoxLayout()
        header.setSpacing(5)
        self.drag_handle = LaneDragHandle()
        self.drag_handle.drag_started.connect(lambda: self.drag_started.emit(self))
        self.drag_handle.drag_moved.connect(lambda point: self.drag_moved.emit(self, point))
        self.drag_handle.drag_finished.connect(lambda: self.drag_finished.emit(self))
        self.collapse_button = QToolButton()
        self.collapse_button.setObjectName("effectIconButton")
        self.collapse_button.setArrowType(Qt.ArrowType.DownArrow)
        self.collapse_button.setToolTip("Collapse effect lane")
        self.collapse_button.clicked.connect(self._toggle_collapsed)
        self.name_label = QLabel(EFFECT_LABELS[track.kind].upper())
        self.name_label.setObjectName("effectName")
        self.enabled_box = QCheckBox("ON")
        self.enabled_box.setChecked(track.enabled)
        self.enabled_box.setToolTip("Bypass this effect without losing its settings")
        self.enabled_box.toggled.connect(self._enabled_changed)
        self.preset = ScrollSafeComboBox()
        self.preset.setObjectName("effectPreset")
        self.preset.addItem("CUSTOM", "custom")
        self.preset.addItem("100 CONSTANT", "full")
        self.preset.addItem("0 → 100 → 0", "rise_fall")
        self.preset.addItem("0 → 100", "rise")
        self.preset.addItem("100 → 0", "fall")
        self.preset.setFixedWidth(126)
        self.preset.currentIndexChanged.connect(self._preset_changed)
        self.reset_button = QPushButton("RESET")
        self.reset_button.setObjectName("effectTextButton")
        self.reset_button.clicked.connect(self._reset)
        self.remove_button = QPushButton("REMOVE")
        self.remove_button.setObjectName("effectTextButton")
        self.remove_button.clicked.connect(lambda: self.remove_requested.emit(self))
        self.more_button = QToolButton()
        self.more_button.setObjectName("effectMoreButton")
        self.more_button.setText("···")
        self.more_button.setFixedWidth(29)
        self.more_button.setToolTip("More effect actions")
        self.more_button.clicked.connect(self._show_more_menu)
        self.more_button.hide()
        header.addWidget(self.drag_handle)
        header.addWidget(self.collapse_button)
        header.addWidget(self.name_label)
        header.addWidget(self.enabled_box)
        header.addStretch()
        header.addWidget(self.preset)
        header.addWidget(self.reset_button)
        header.addWidget(self.remove_button)
        header.addWidget(self.more_button)
        layout.addLayout(header)

        self.details = QWidget()
        details_layout = QVBoxLayout(self.details)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(4)
        value_row = QHBoxLayout()
        value_row.setSpacing(7)
        self.position_label = QLabel("POSITION")
        self.position_label.setObjectName("effectMeta")
        self.position_spin = ScrollSafeSpinBox()
        self.position_spin.setObjectName("effectKeyframeSpin")
        self.position_spin.setAccessibleName("Keyframe position")
        self.position_spin.setRange(0, 100)
        self.position_spin.setSuffix("%")
        self.position_spin.valueChanged.connect(self._numeric_keyframe_changing)
        self.value_label = QLabel("VALUE")
        self.value_label.setObjectName("effectMeta")
        self.value_spin = ScrollSafeSpinBox()
        self.value_spin.setObjectName("effectKeyframeSpin")
        self.value_spin.setAccessibleName("Keyframe value")
        self.value_spin.setRange(0, 100)
        self.value_spin.setSuffix("%")
        self.value_spin.valueChanged.connect(self._numeric_keyframe_changing)
        value_row.addWidget(self.position_label)
        value_row.addWidget(self.position_spin)
        value_row.addWidget(self.value_label)
        value_row.addWidget(self.value_spin)
        self.mode_combo: ScrollSafeComboBox | None = None
        self.mode_label: QLabel | None = None
        if track.kind == "blend_mode":
            self.mode_label = QLabel("MODE")
            self.mode_label.setObjectName("effectMeta")
            self.mode_combo = ScrollSafeComboBox()
            self.mode_combo.setObjectName("effectMode")
            self.mode_combo.setAccessibleName("Blend mode")
            self.mode_combo.setFixedWidth(184)
            self.mode_combo.setMinimumContentsLength(16)
            self.mode_combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            self.mode_combo.setToolTip("Blend each masked pose with the composite underneath it")
            for index, mode in enumerate(BLEND_MODES):
                self.mode_combo.addItem(BLEND_MODE_LABELS[mode].upper(), mode)
                if index in {1, 6, 11, 18, 22}:
                    self.mode_combo.insertSeparator(self.mode_combo.count())
            self.mode_combo.setCurrentIndex(self.mode_combo.findData(track.option))
            self.mode_combo.currentIndexChanged.connect(self._mode_changed)
            value_row.addWidget(self.mode_label)
            value_row.addWidget(self.mode_combo)
        value_row.addStretch()
        self.amount_spin: ScrollSafeSpinBox | None = None
        if track.kind in {"blur", "stippling", "dithering", "halftone"}:
            amount_labels = {
                "blur": "MAX BLUR",
                "stippling": "DOT SIZE",
                "dithering": "PIXEL SIZE",
                "halftone": "CELL SIZE",
            }
            amount_label = QLabel(amount_labels[track.kind])
            amount_label.setObjectName("effectMeta")
            self.amount_spin = ScrollSafeSpinBox()
            self.amount_spin.setObjectName("effectAmount")
            self.amount_spin.setRange(1, 200 if track.kind == "blur" else 64)
            self.amount_spin.setSuffix(" px")
            self.amount_spin.setValue(round(track.amount))
            self.amount_spin.valueChanged.connect(self._amount_changed)
            value_row.addWidget(amount_label)
            value_row.addWidget(self.amount_spin)
        details_layout.addLayout(value_row)
        self.graph = EffectKeyframeGraph(track.keyframes)
        self.graph.keyframes_changing.connect(self._keyframes_changing)
        self.graph.keyframes_committed.connect(self._keyframes_committed)
        self.graph.selection_changed.connect(self._selection_changed)
        if keyframed:
            self.position_spin.editingFinished.connect(self.graph.commit_selected)
            self.value_spin.editingFinished.connect(self.graph.commit_selected)
        else:
            self.value_spin.editingFinished.connect(self._constant_value_committed)
        details_layout.addWidget(self.graph)
        layout.addWidget(self.details)
        self.graph._emit_selection()
        if all(point.value == 100.0 for point in track.keyframes):
            with QSignalBlocker(self.preset):
                self.preset.setCurrentIndex(self.preset.findData("full"))
        self.preset.setVisible(keyframed)
        self.position_label.setVisible(keyframed)
        self.position_spin.setVisible(keyframed)
        self.graph.setVisible(keyframed)
        self._sync_name_label()
        self._sync_enabled_style()

    @property
    def track(self) -> EffectTrack:
        return self._track

    def _replace_track(
        self,
        *,
        keyframes: tuple[EffectKeyframe, ...] | None = None,
        enabled: bool | None = None,
        amount: float | None = None,
        option: str | None = None,
    ) -> EffectTrack:
        self._track = EffectTrack(
            self._track.kind,
            self._track.keyframes if keyframes is None else keyframes,
            self._track.enabled if enabled is None else enabled,
            self._track.amount if amount is None else amount,
            self._track.option if option is None else option,
        )
        return self._track

    def _toggle_collapsed(self) -> None:
        visible = not self.details.isVisible()
        self.details.setVisible(visible)
        self.collapse_button.setArrowType(
            Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow
        )
        self.collapse_button.setToolTip("Collapse effect lane" if visible else "Expand effect lane")

    def _show_more_menu(self) -> None:
        menu = QMenu(self)
        reset = menu.addAction("Reset effect")
        remove = menu.addAction("Remove effect")
        selected = menu.exec(self.more_button.mapToGlobal(self.more_button.rect().bottomLeft()))
        if selected is reset:
            self._reset()
        elif selected is remove:
            self.remove_requested.emit(self)

    def _enabled_changed(self, enabled: bool) -> None:
        self._replace_track(enabled=enabled)
        self._sync_enabled_style()
        self.track_committed.emit(self._track)

    def _sync_enabled_style(self) -> None:
        self.setProperty("bypassed", not self._track.enabled)
        self.style().unpolish(self)
        self.style().polish(self)

    def _preset_changed(self) -> None:
        preset = str(self.preset.currentData())
        if preset == "custom":
            return
        self._track = effect_preset(self._track, preset)
        self.graph.set_keyframes(self._track.keyframes)
        self.track_committed.emit(self._track)

    def _reset(self) -> None:
        self._track = neutral_effect_track(self._track.kind)
        with QSignalBlocker(self.enabled_box):
            self.enabled_box.setChecked(True)
        if self.amount_spin is not None:
            with QSignalBlocker(self.amount_spin):
                self.amount_spin.setValue(round(self._track.amount))
        if self.mode_combo is not None:
            with QSignalBlocker(self.mode_combo):
                self.mode_combo.setCurrentIndex(self.mode_combo.findData(self._track.option))
        with QSignalBlocker(self.preset):
            self.preset.setCurrentIndex(0)
        self.graph.set_keyframes(self._track.keyframes)
        self._sync_name_label()
        self._sync_enabled_style()
        self.track_committed.emit(self._track)

    def _amount_changed(self, value: int) -> None:
        self._replace_track(amount=float(value))
        self.track_committed.emit(self._track)

    def _mode_changed(self) -> None:
        if self.mode_combo is None or self.mode_combo.currentData() is None:
            return
        self._replace_track(option=str(self.mode_combo.currentData()))
        self._sync_name_label()
        self.track_committed.emit(self._track)

    def _sync_name_label(self) -> None:
        if self._track.kind == "blend_mode":
            separator = "/" if self._narrow else " · "
            self.name_label.setText(
                f"BLEND{separator}{BLEND_MODE_LABELS[self._track.option].upper()}"
            )
            return
        self.name_label.setText(EFFECT_LABELS[self._track.kind].upper())

    def _keyframes_changing(self, keyframes: object) -> None:
        with QSignalBlocker(self.preset):
            self.preset.setCurrentIndex(0)
        self._replace_track(keyframes=tuple(keyframes))  # type: ignore[arg-type]
        self.track_changing.emit(self._track)

    def _keyframes_committed(self, keyframes: object) -> None:
        with QSignalBlocker(self.preset):
            self.preset.setCurrentIndex(0)
        self._replace_track(keyframes=tuple(keyframes))  # type: ignore[arg-type]
        self.track_committed.emit(self._track)

    def _selection_changed(self, progress: float, value: float) -> None:
        with QSignalBlocker(self.position_spin):
            self.position_spin.setValue(round(progress * 100))
        with QSignalBlocker(self.value_spin):
            self.value_spin.setValue(round(value))
        self.position_spin.setEnabled(0 < self.graph.selected_index < len(self.graph.keyframes) - 1)

    def _numeric_keyframe_changing(self) -> None:
        if not self._keyframed:
            value = float(self.value_spin.value())
            self._replace_track(keyframes=(EffectKeyframe(0.0, value), EffectKeyframe(1.0, value)))
            self.track_changing.emit(self._track)
            return
        self.graph.set_selected_values(
            self.position_spin.value() / 100.0,
            float(self.value_spin.value()),
            live=True,
        )

    def _constant_value_committed(self) -> None:
        if not self._keyframed:
            self.track_committed.emit(self._track)

    def set_compact(self, compact: bool) -> None:
        self.graph.setMinimumHeight(50 if compact else 60)
        narrow = compact or self.window().width() < 1100 or self.width() < 650
        self._narrow = narrow
        self._sync_name_label()
        self.reset_button.setVisible(not narrow)
        self.remove_button.setVisible(not narrow)
        self.more_button.setVisible(narrow)
        self.preset.setVisible(self._keyframed)
        self.preset.setFixedWidth(112 if narrow else 126)
        self.position_label.setVisible(self._keyframed and not narrow)
        self.position_spin.setVisible(self._keyframed)
        self.value_label.setVisible(not narrow)
        self.position_spin.setPrefix("P " if narrow and self._keyframed else "")
        self.value_spin.setPrefix("V " if narrow else "")
        self.position_spin.setFixedWidth(74 if narrow else 58)
        self.value_spin.setFixedWidth(74 if narrow else 58)
        if self.mode_label is not None:
            self.mode_label.setVisible(not narrow)
        if self.mode_combo is not None:
            self.mode_combo.setMinimumContentsLength(10 if narrow else 16)
            self.mode_combo.setFixedWidth(126 if narrow else 184)
        self.graph.setVisible(self._keyframed)
        self.layout().invalidate()
        self.updateGeometry()
        if self.parentWidget() is not None:
            self.parentWidget().updateGeometry()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.set_compact(self.window().height() < 760)
        super().resizeEvent(event)


class EffectTimelinePanel(QFrame):
    tracks_changing = Signal()
    tracks_committed = Signal()
    expanded_changed = Signal(bool)

    def __init__(
        self,
        title: str = "TRAIL EFFECTS",
        scope: str = "MASKED POSES",
        *,
        keyframed: bool = True,
        empty_text: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("effectsPanel")
        self._title = title
        self._scope = scope
        self._keyframed = keyframed
        self._expanded = True
        self._lanes: list[EffectLane] = []
        self._drag_lane: EffectLane | None = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 8)
        layout.setSpacing(6)
        header = QHBoxLayout()
        header.setSpacing(8)
        self.toggle_button = QToolButton()
        self.toggle_button.setObjectName("effectIconButton")
        self.toggle_button.setArrowType(Qt.ArrowType.DownArrow)
        self.toggle_button.clicked.connect(self._toggle_tracks)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("controlLabel")
        self.summary = QLabel(f"NO EFFECTS · {scope}")
        self.summary.setObjectName("effectMeta")
        self.add_button = QPushButton("+ ADD EFFECT")
        self.add_button.setObjectName("effectAddButton")
        self.add_button.clicked.connect(self._show_add_menu)
        header.addWidget(self.toggle_button)
        header.addWidget(self.title_label)
        header.addWidget(self.summary)
        header.addStretch()
        header.addWidget(self.add_button)
        layout.addLayout(header)

        if empty_text is None:
            empty_text = (
                "Add a keyframed effect to shape the masked poses across motion."
                if keyframed
                else "Add a constant effect to process only the clean plate."
            )
        self.empty_label = QLabel(empty_text)
        self.empty_label.setObjectName("effectEmpty")
        self.empty_label.setWordWrap(True)
        layout.addWidget(self.empty_label)
        self.scroll = QScrollArea()
        self.scroll.setObjectName("effectsScroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setMaximumHeight(222 if keyframed else 128)
        self.scroll.setMinimumHeight(0)
        self.lane_container = QWidget()
        self.lane_container.setObjectName("effectsContainer")
        self.lane_layout = QVBoxLayout(self.lane_container)
        self.lane_layout.setContentsMargins(0, 0, 0, 0)
        self.lane_layout.setSpacing(5)
        self.lane_layout.addStretch()
        self.scroll.setWidget(self.lane_container)
        self.scroll.hide()
        layout.addWidget(self.scroll)
        self._compact = False

    def tracks(self) -> tuple[EffectTrack, ...]:
        return tuple(lane.track for lane in self._lanes)

    @property
    def is_expanded(self) -> bool:
        return self._expanded

    def clear(self) -> None:
        for lane in self._lanes:
            lane.deleteLater()
        self._lanes.clear()
        self._sync_state()

    def set_compact(self, compact: bool) -> None:
        self._compact = compact
        self.scroll.setMaximumHeight(
            148 if compact and self._keyframed else 222 if self._keyframed else 96
        )
        for lane in self._lanes:
            lane.set_compact(compact)
        self.lane_layout.invalidate()
        self.lane_container.updateGeometry()
        self.setMinimumHeight(self._expanded_minimum_height())

    def add_effect(self, kind: str) -> EffectLane:
        if kind not in EFFECT_KINDS:
            raise ValueError(f"unsupported effect: {kind}")
        existing = next(
            (lane for lane in self._lanes if lane.track.kind == kind and kind != "blend_mode"),
            None,
        )
        if existing is not None:
            existing.show()
            return existing
        track = neutral_effect_track(kind)
        if kind == "blend_mode":
            track = effect_preset(track, "full")
        lane = EffectLane(track, keyframed=self._keyframed)
        lane.track_changing.connect(lambda _track: self.tracks_changing.emit())
        lane.track_committed.connect(self._lane_committed)
        lane.remove_requested.connect(self._remove_lane)
        lane.drag_started.connect(self._drag_started)
        lane.drag_moved.connect(self._drag_moved)
        lane.drag_finished.connect(self._drag_finished)
        self._lanes.append(lane)
        self.lane_layout.insertWidget(len(self._lanes) - 1, lane)
        self._sync_state()
        lane.set_compact(self._compact)
        self.set_expanded(True)
        self.tracks_committed.emit()
        return lane

    def _lane_committed(self, track: object) -> None:
        del track
        self._sync_state()
        self.tracks_committed.emit()

    def _show_add_menu(self) -> None:
        menu = QMenu(self)
        active = {lane.track.kind for lane in self._lanes}
        for kind in EFFECT_KINDS:
            label = EFFECT_LABELS[kind]
            if kind == "blend_mode" and kind in active:
                label = "Blend mode · add another"
            action = menu.addAction(label)
            action.setEnabled(kind == "blend_mode" or kind not in active)
            action.triggered.connect(lambda checked=False, selected=kind: self.add_effect(selected))
        menu.exec(self.add_button.mapToGlobal(self.add_button.rect().bottomLeft()))

    def _remove_lane(self, lane: object) -> None:
        if lane not in self._lanes:
            return
        self._lanes.remove(lane)  # type: ignore[arg-type]
        lane.deleteLater()  # type: ignore[union-attr]
        self._sync_state()
        self.tracks_committed.emit()

    def _toggle_tracks(self) -> None:
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded: bool) -> None:
        expanded = bool(expanded)
        if expanded == self._expanded:
            self._sync_state()
            return
        self._expanded = expanded
        self._sync_state()
        self.expanded_changed.emit(expanded)

    def _sync_state(self) -> None:
        has_lanes = bool(self._lanes)
        self.empty_label.setVisible(not has_lanes and self._expanded)
        self.scroll.setVisible(has_lanes and self._expanded)
        self.toggle_button.setEnabled(True)
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if self._expanded else Qt.ArrowType.RightArrow
        )
        active = sum(lane.track.enabled for lane in self._lanes)
        self.summary.setText(
            f"{active} ACTIVE / {len(self._lanes)} TRACKS · {self._scope}"
            if has_lanes
            else f"NO EFFECTS · {self._scope}"
        )
        self.setMinimumHeight(self._expanded_minimum_height())

    def _expanded_minimum_height(self) -> int:
        if not self._lanes or not self._expanded:
            return 0
        if self._keyframed:
            return 132
        if self._compact:
            return 82
        return min(128, 82 + max(0, len(self._lanes) - 1) * 38)

    def _drag_started(self, lane: object) -> None:
        self._drag_lane = lane if isinstance(lane, EffectLane) else None

    def _drag_moved(self, lane: object, global_point: QPoint) -> None:
        if lane is not self._drag_lane or lane not in self._lanes:
            return
        local_y = self.lane_container.mapFromGlobal(global_point).y()
        target = len(self._lanes) - 1
        for index, candidate in enumerate(self._lanes):
            if local_y < candidate.geometry().center().y():
                target = index
                break
        current = self._lanes.index(lane)  # type: ignore[arg-type]
        if target == current:
            return
        moved = self._lanes.pop(current)
        self._lanes.insert(target, moved)
        self.lane_layout.removeWidget(moved)
        self.lane_layout.insertWidget(target, moved)
        self.tracks_changing.emit()

    def _drag_finished(self, lane: object) -> None:
        if lane is self._drag_lane:
            self._drag_lane = None
            self.tracks_committed.emit()
