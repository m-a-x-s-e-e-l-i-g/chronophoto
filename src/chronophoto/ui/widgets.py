from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

from PySide6.QtCore import QEvent, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QKeyEvent, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class BackgroundTaskView(QFrame):
    """Compact status-bar view for independently running background work."""

    cancel_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("backgroundTaskView")
        self.setAccessibleName("Background tasks")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 8, 7)
        layout.setSpacing(4)

        self.heading = QLabel("BACKGROUND")
        self.heading.setObjectName("backgroundTaskHeading")
        layout.addWidget(self.heading)

        self._rows: dict[str, QFrame] = {}
        self._details: dict[str, QLabel] = {}
        self._progress: dict[str, QProgressBar] = {}
        self._cancel_buttons: dict[str, QPushButton] = {}
        for kind, label in (("preview", "PREVIEW"), ("export", "EXPORT")):
            row = QFrame()
            row.setObjectName("backgroundTaskRow")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            kind_label = QLabel(label)
            kind_label.setObjectName("backgroundTaskKind")
            kind_label.setFixedWidth(58)
            detail = QLabel()
            detail.setObjectName("backgroundTaskDetail")
            detail.setMinimumWidth(150)
            detail.setMaximumWidth(260)
            progress = QProgressBar()
            progress.setObjectName("backgroundTaskProgress")
            progress.setRange(0, 100)
            progress.setValue(0)
            progress.setTextVisible(False)
            progress.setFixedWidth(104)
            cancel = QPushButton("Cancel")
            cancel.setObjectName("taskCancelButton")
            cancel.setAccessibleName(f"Cancel {label.lower()}")
            cancel.clicked.connect(
                lambda checked=False, selected=kind: self.cancel_requested.emit(selected)
            )

            row_layout.addWidget(kind_label)
            row_layout.addWidget(detail, 1)
            row_layout.addWidget(progress)
            row_layout.addWidget(cancel)
            layout.addWidget(row)
            row.hide()
            self._rows[kind] = row
            self._details[kind] = detail
            self._progress[kind] = progress
            self._cancel_buttons[kind] = cancel

        self.hide()

    def start_task(self, kind: str, detail: str) -> None:
        self._details[kind].setText(detail)
        self._details[kind].setToolTip(detail)
        self._progress[kind].setValue(1)
        self._cancel_buttons[kind].setEnabled(True)
        self._rows[kind].show()
        self.show()
        self._sync_heading()

    def update_task(self, kind: str, value: int, detail: str) -> None:
        self._progress[kind].setValue(value)
        self._details[kind].setText(detail)
        self._details[kind].setToolTip(detail)

    def set_cancelling(self, kind: str) -> None:
        self._details[kind].setText("Stopping after the current step")
        self._details[kind].setToolTip("Stopping after the current processing step")
        self._cancel_buttons[kind].setEnabled(False)

    def remove_task(self, kind: str) -> None:
        self._rows[kind].hide()
        self._progress[kind].setValue(0)
        if any(not row.isHidden() for row in self._rows.values()):
            self._sync_heading()
        else:
            self.hide()

    def is_task_visible(self, kind: str) -> bool:
        return not self._rows[kind].isHidden()

    def _sync_heading(self) -> None:
        count = sum(not row.isHidden() for row in self._rows.values())
        self.heading.setText("BACKGROUND" if count == 1 else f"BACKGROUND · {count} TASKS")


class ScrollSafeComboBox(QComboBox):
    """A combo box that leaves wheel scrolling to its containing panel."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class ScrollSafeSlider(QSlider):
    """A slider that leaves wheel scrolling to its containing panel."""

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()


class DropSurface(QFrame):
    paths_dropped = Signal(list)
    activated = Signal()
    drag_active = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("dropSurface")
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Choose source footage")
        self.setAccessibleDescription("Open one video or an ordered stack of two or more photos")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 24, 20, 24)
        layout.setSpacing(5)
        title = QLabel("DROP FOOTAGE")
        title.setObjectName("dropTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint = QLabel("Video or 2+ stills")
        hint.setObjectName("dropHint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(hint)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.LeftButton:
            self.activated.emit()
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space}:
            self.activated.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drag_active.emit(True)
            self.setProperty("dragActive", True)
            self.style().unpolish(self)
            self.style().polish(self)

    def dragLeaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self.drag_active.emit(False)
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        self.drag_active.emit(False)
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        if paths:
            self.paths_dropped.emit(paths)
            event.acceptProposedAction()


class PreviewCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(520, 360)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Image preview")
        self.setAccessibleDescription(
            "Pinch or scroll to zoom, drag to pan, and double-click to fit"
        )
        self.setToolTip("Pinch or scroll to zoom · drag to pan · double-click to fit")
        self._image: QImage | None = None
        self._status = "WAITING FOR FOOTAGE"
        self._zoom = 1.0
        self._pan = QPointF()
        self._drag_origin: QPointF | None = None
        self._drag_pan_origin = QPointF()

    def set_image(self, image: QImage | None, status: str = "PREVIEW") -> None:
        previous_size = self._image.size() if self._image and not self._image.isNull() else None
        self._image = image
        self._status = status
        next_size = image.size() if image and not image.isNull() else None
        if next_size is None or previous_size != next_size:
            self.reset_view(update=False)
        self.update()

    @property
    def zoom_factor(self) -> float:
        return self._zoom

    def reset_view(self, *, update: bool = True) -> None:
        self._zoom = 1.0
        self._pan = QPointF()
        self._drag_origin = None
        self.unsetCursor()
        if update:
            self.update()

    def zoom_by(self, factor: float, anchor: QPointF | None = None) -> None:
        if not self._image or self._image.isNull() or factor <= 0:
            return
        old_zoom = self._zoom
        new_zoom = max(1.0, min(12.0, old_zoom * factor))
        if math.isclose(old_zoom, new_zoom, rel_tol=1e-5):
            return
        anchor = anchor or QPointF(self.rect().center())
        viewport_center = QPointF(self.rect().center())
        image_center = viewport_center + self._pan
        pointer_offset = anchor - image_center
        new_center = anchor - pointer_offset * (new_zoom / old_zoom)
        self._zoom = new_zoom
        self._pan = new_center - viewport_center
        self._clamp_pan()
        cursor = Qt.CursorShape.OpenHandCursor if new_zoom > 1.0 else Qt.CursorShape.ArrowCursor
        self.setCursor(cursor)
        self.update()

    def _base_image_size(self):  # type: ignore[no-untyped-def]
        if not self._image or self._image.isNull():
            return None
        return self._image.size().scaled(
            self.rect().size() - self._image_offset(), Qt.AspectRatioMode.KeepAspectRatio
        )

    def _image_rect(self) -> QRectF:
        base_size = self._base_image_size()
        if base_size is None:
            return QRectF()
        width = base_size.width() * self._zoom
        height = base_size.height() * self._zoom
        center = QPointF(self.rect().center()) + self._pan
        return QRectF(
            center.x() - width / 2,
            center.y() - height / 2,
            width,
            height,
        )

    def _clamp_pan(self) -> None:
        base_size = self._base_image_size()
        if base_size is None or self._zoom <= 1.0:
            self._pan = QPointF()
            return
        available = self.rect().size() - self._image_offset()
        width = base_size.width() * self._zoom
        height = base_size.height() * self._zoom
        max_x = max(0.0, (width - available.width()) / 2)
        max_y = max(0.0, (height - available.height()) / 2)
        self._pan = QPointF(
            max(-max_x, min(max_x, self._pan.x())),
            max(-max_y, min(max_y, self._pan.y())),
        )

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bounds = self.rect()
        painter.fillRect(bounds, QColor("#070707"))

        tile = 18
        first = QColor("#101010")
        second = QColor("#151515")
        for y in range(0, bounds.height(), tile):
            for x in range(0, bounds.width(), tile):
                painter.fillRect(x, y, tile, tile, first if (x // tile + y // tile) % 2 else second)

        if self._image and not self._image.isNull():
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawImage(self._image_rect(), self._image)
        else:
            painter.setPen(QColor("#787878"))
            painter.setFont(self.font())
            painter.drawText(
                bounds, Qt.AlignmentFlag.AlignCenter, "Your motion study will appear here"
            )

        badge = QRectF(16, 16, max(126, len(self._status) * 8 + 24), 28)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#0d0d0d"))
        painter.drawRoundedRect(badge, 4, 4)
        painter.setPen(QColor("#c8c8c8"))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, self._status)

        if self._zoom > 1.001:
            zoom_text = f"{round(self._zoom * 100)}% · DOUBLE-CLICK TO FIT"
            zoom_badge = QRectF(
                bounds.width() - max(184, len(zoom_text) * 8 + 20) - 16,
                bounds.height() - 44,
                max(184, len(zoom_text) * 8 + 20),
                28,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#0d0d0d"))
            painter.drawRoundedRect(zoom_badge, 2, 2)
            painter.setPen(QColor("#c8c8c8"))
            painter.drawText(zoom_badge, Qt.AlignmentFlag.AlignCenter, zoom_text)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.pixelDelta().y()
        if delta == 0:
            delta = event.angleDelta().y()
        if delta == 0:
            event.ignore()
            return
        if event.inverted():
            delta = -delta
        self.zoom_by(math.exp(delta / 600.0), event.position())
        event.accept()

    def event(self, event: QEvent) -> bool:
        if event.type() == QEvent.Type.NativeGesture:
            gesture_type = event.gestureType()  # type: ignore[attr-defined]
            if gesture_type == Qt.NativeGestureType.ZoomNativeGesture:
                self.zoom_by(1.0 + event.value(), event.position())  # type: ignore[attr-defined]
                event.accept()
                return True
            if gesture_type == Qt.NativeGestureType.SmartZoomNativeGesture:
                if self._zoom > 1.0:
                    self.reset_view()
                else:
                    self.zoom_by(2.0, event.position())  # type: ignore[attr-defined]
                event.accept()
                return True
            if gesture_type == Qt.NativeGestureType.PanNativeGesture and self._zoom > 1.0:
                self._pan += event.delta()  # type: ignore[attr-defined]
                self._clamp_pan()
                self.update()
                event.accept()
                return True
        return super().event(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._zoom > 1.0:
            self._drag_origin = event.position()
            self._drag_pan_origin = QPointF(self._pan)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin is not None:
            self._pan = self._drag_pan_origin + event.position() - self._drag_origin
            self._clamp_pan()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._drag_origin is not None:
            self._drag_origin = None
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._clamp_pan()
        super().resizeEvent(event)

    @staticmethod
    def _image_offset():
        from PySide6.QtCore import QSize

        return QSize(48, 48)


class RangeSlider(QWidget):
    range_changed = Signal(int, int)
    range_committed = Signal(int, int)
    handle_changed = Signal(str, int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(78)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Selected video range")
        self.setAccessibleDescription(
            "Use left and right arrows for the start; hold Shift for the end"
        )
        self.minimum = 0
        self.maximum = 1000
        self.low = 90
        self.high = 820
        self._active: str | None = None
        self._thumbnails: list[QImage] = []
        self._compact = False

    def set_compact(self, compact: bool) -> None:
        if self._compact == compact:
            return
        self._compact = compact
        self.setMinimumHeight(42 if compact else 78)
        self.updateGeometry()
        self.update()

    def set_thumbnails(self, thumbnails: Sequence[QImage]) -> None:
        self._thumbnails = [thumbnail.copy() for thumbnail in thumbnails]
        self.update()

    @property
    def is_adjusting(self) -> bool:
        return self._active is not None

    def set_values(self, low: int, high: int) -> None:
        self.low = max(self.minimum, min(low, self.maximum))
        self.high = max(self.low + 1, min(high, self.maximum))
        self.update()
        self.range_changed.emit(self.low, self.high)

    def _track(self) -> QRectF:
        return QRectF(12, self.height() - (12 if self._compact else 17), self.width() - 24, 4)

    def _x_for_value(self, value: int) -> float:
        track = self._track()
        ratio = (value - self.minimum) / (self.maximum - self.minimum)
        return track.left() + ratio * track.width()

    def _value_for_x(self, x: float) -> int:
        track = self._track()
        ratio = max(0.0, min(1.0, (x - track.left()) / track.width()))
        return round(self.minimum + ratio * (self.maximum - self.minimum))

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = self._track()

        if self._thumbnails and not self._compact:
            strip = QRectF(12, 4, self.width() - 24, 48)
            cell_width = strip.width() / len(self._thumbnails)
            painter.save()
            painter.setClipRect(strip)
            for index, thumbnail in enumerate(self._thumbnails):
                target = QRectF(strip.left() + index * cell_width, strip.top(), cell_width + 1, 48)
                source_size = thumbnail.size().scaled(
                    int(target.width()),
                    int(target.height()),
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                )
                x_crop = max(0, (thumbnail.width() - source_size.width()) // 2)
                source = QRectF(x_crop, 0, thumbnail.width() - x_crop * 2, thumbnail.height())
                painter.drawImage(target, thumbnail, source)
            low_x = self._x_for_value(self.low)
            high_x = self._x_for_value(self.high)
            painter.fillRect(
                QRectF(strip.left(), strip.top(), low_x - strip.left(), strip.height()),
                QColor(0, 0, 0, 175),
            )
            painter.fillRect(
                QRectF(high_x, strip.top(), strip.right() - high_x, strip.height()),
                QColor(0, 0, 0, 175),
            )
            painter.restore()

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#383838"))
        painter.drawRoundedRect(track, 2, 2)

        low_x = self._x_for_value(self.low)
        high_x = self._x_for_value(self.high)
        selected = QRectF(low_x, track.top(), high_x - low_x, track.height())
        painter.setBrush(QColor("#e8e8e8"))
        painter.drawRoundedRect(selected, 2, 2)

        for x in (low_x, high_x):
            painter.setBrush(QColor("#f0f0f0"))
            painter.setPen(QPen(QColor("#101010"), 2))
            painter.drawEllipse(QPointF(x, track.center().y()), 7, 7)

        if self.hasFocus():
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#eeeeee"), 1))
            painter.drawRoundedRect(QRectF(1, 1, self.width() - 2, self.height() - 2), 4, 4)

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        low_distance = abs(event.position().x() - self._x_for_value(self.low))
        high_distance = abs(event.position().x() - self._x_for_value(self.high))
        self._active = "low" if low_distance <= high_distance else "high"
        self._move_active(event.position().x())

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._active:
            self._move_active(event.position().x())

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._active:
            self.range_committed.emit(self.low, self.high)
        self._active = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() not in {Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Home, Qt.Key.Key_End}:
            super().keyPressEvent(event)
            return
        handle = "high" if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else "low"
        step = 5
        if event.key() == Qt.Key.Key_Home:
            value = self.minimum if handle == "low" else self.low + 1
        elif event.key() == Qt.Key.Key_End:
            value = self.high - 1 if handle == "low" else self.maximum
        else:
            direction = -1 if event.key() == Qt.Key.Key_Left else 1
            value = (self.low if handle == "low" else self.high) + direction * step
        self._active = handle
        if handle == "low":
            self.low = max(self.minimum, min(value, self.high - 1))
        else:
            self.high = min(self.maximum, max(value, self.low + 1))
        self._active = None
        self.update()
        self.range_changed.emit(self.low, self.high)
        self.handle_changed.emit(handle, self.low if handle == "low" else self.high)
        self.range_committed.emit(self.low, self.high)
        event.accept()

    def wheelEvent(self, event: QWheelEvent) -> None:
        event.ignore()

    def _move_active(self, x: float) -> None:
        value = self._value_for_x(x)
        if self._active == "low":
            self.low = min(value, self.high - 1)
        elif self._active == "high":
            self.high = max(value, self.low + 1)
        self.update()
        self.range_changed.emit(self.low, self.high)
        if self._active:
            value = self.low if self._active == "low" else self.high
            self.handle_changed.emit(self._active, value)


def image_to_qimage(frame) -> QImage:  # type: ignore[no-untyped-def]
    height, width, channels = frame.shape
    image = QImage(frame.data, width, height, channels * width, QImage.Format.Format_RGB888)
    return image.copy()


def mask_overlay_to_qimage(frame, mask) -> QImage:  # type: ignore[no-untyped-def]
    import numpy as np

    alpha = np.clip(mask[..., None], 0.0, 1.0)
    luminance = (
        frame[..., 0:1].astype(np.float32) * 0.2126
        + frame[..., 1:2].astype(np.float32) * 0.7152
        + frame[..., 2:3].astype(np.float32) * 0.0722
    )
    base = np.repeat(luminance, 3, axis=2) * 0.20
    highlight = np.repeat(luminance, 3, axis=2) * 0.20 + 255.0 * 0.80
    visual = base * (1.0 - alpha) + highlight * alpha
    return image_to_qimage(np.clip(visual, 0, 255).astype(np.uint8))


def classify_paths(paths: Sequence[str]) -> tuple[str, list[Path]]:
    video_suffixes = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
    image_suffixes = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
    files = [Path(path) for path in paths if Path(path).is_file()]
    videos = [path for path in files if path.suffix.casefold() in video_suffixes]
    images = [path for path in files if path.suffix.casefold() in image_suffixes]
    if len(videos) == 1 and len(files) == 1:
        return "video", videos
    if len(images) >= 2 and len(images) == len(files):
        return "photos", images
    raise ValueError("Drop one video or a set of at least two photos")
