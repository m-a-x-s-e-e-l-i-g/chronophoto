from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase, QIcon
from PySide6.QtWidgets import QApplication

from chronophoto import __version__
from chronophoto.ui.window import ChronophotoWindow

STYLESHEET = """
QWidget {
    background: #0b0b0b;
    color: #e8e8e8;
    font-family: "__APP_MONO_FONT__";
    font-size: 13px;
}
QLabel { background: transparent; }
QMainWindow { background: #0b0b0b; }
QFrame#appHeader {
    background: #0d0d0d;
    border-bottom: 1px solid #303030;
}
QLabel#wordmark {
    color: #f4f4f4;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 2px;
}
QLabel#strapline, QLabel#eyebrow, QLabel#controlLabel, QLabel#statusText {
    color: #9a9a9a;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
}
QLabel#strapline { color: #737373; }
QLabel#privacyBadge {
    color: #d8d8d8;
    background: #111111;
    border: 1px solid #474747;
    border-radius: 2px;
    padding: 5px 9px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
}
QLabel#versionNumber, QLabel#updateStatus {
    color: #858585;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
}
QLabel#versionNumber { color: #c8c8c8; }
QLabel#updateStatus[state="available"] { color: #ff6b6b; font-weight: 700; }
QLabel#updateStatus[state="current"] { color: #70d98b; font-weight: 700; }
QLabel#updateStatus[state="unavailable"] { color: #6f6f6f; }
QPushButton#headerButton {
    min-height: 27px;
    padding: 0 10px;
    color: #d8d8d8;
    background: #111111;
    border: 1px solid #474747;
    font-size: 10px;
    letter-spacing: 1px;
}
QPushButton#headerButton:hover { background: #1b1b1b; border-color: #8a8a8a; }
QFrame#sourcePanel {
    background: #0f0f0f;
    border-right: 1px solid #303030;
}
QWidget#workspace { background: #080808; }
QWidget#inspector, QScrollArea#inspectorScroll {
    background: #0f0f0f;
    border: none;
}
QScrollArea#inspectorScroll { border-left: 1px solid #303030; }
QLabel#panelTitle {
    color: #f0f0f0;
    font-size: 28px;
    font-weight: 700;
}
QLabel#panelTitleCompact {
    color: #f0f0f0;
    font-size: 18px;
    font-weight: 700;
}
QLabel#workspaceTitle {
    color: #eeeeee;
    font-size: 18px;
    font-weight: 700;
}
QLabel#bodyMuted, QLabel#controlHint {
    color: #858585;
    font-size: 12px;
}
QLabel#privacyNote {
    color: #737373;
    font-size: 11px;
}
QFrame#dropSurface {
    background: #121212;
    border: 1px dashed #4a4a4a;
    border-radius: 2px;
}
QFrame#dropSurface:hover, QFrame#dropSurface[dragActive="true"] {
    background: #1a1a1a;
    border-color: #dedede;
}
QFrame#windowDropOverlay {
    background: rgba(7, 7, 7, 235);
    border: 2px solid #e8e8e8;
}
QLabel#windowDropTitle {
    color: #f2f2f2;
    font-size: 24px;
    font-weight: 700;
    letter-spacing: 2px;
}
QLabel#windowDropHint {
    color: #929292;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1px;
}
QLabel#dropTitle {
    background: transparent;
    color: #e8e8e8;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#dropHint { background: transparent; color: #838383; font-size: 11px; }
QFrame#sourceSummary {
    background: #141414;
    border: 1px solid #333333;
    border-radius: 2px;
}
QLabel#sourceName { color: #e0e0e0; font-weight: 700; }
QFrame#timelinePanel {
    background: #0e0e0e;
    border: 1px solid #303030;
    border-radius: 2px;
}
QFrame#effectsPanel {
    background: #0c0c0c;
    border: 1px solid #303030;
    border-radius: 2px;
}
QFrame#exportOptionsPanel {
    background: #101010;
    border: 1px solid #424242;
    border-radius: 2px;
}
QFrame#exportOptionsPanel QCheckBox {
    font-size: 11px;
    min-height: 24px;
}
QScrollArea#effectsScroll, QWidget#effectsContainer {
    background: #090909;
    border: none;
}
QFrame#effectLane {
    background: #101010;
    border: 1px solid #343434;
    border-radius: 1px;
}
QFrame#effectLane[bypassed="true"] {
    background: #0b0b0b;
    border-color: #242424;
}
QFrame#effectLane[bypassed="true"] QLabel,
QFrame#effectLane[bypassed="true"] QWidget#effectGraph {
    color: #5e5e5e;
}
QLabel#effectName {
    color: #e2e2e2;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#effectMeta, QLabel#effectValue, QLabel#effectEmpty {
    color: #777777;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
}
QLabel#effectEmpty {
    padding: 4px 2px 2px 22px;
    letter-spacing: 0;
}
QLabel#effectDragHandle {
    color: #777777;
    min-width: 18px;
    font-weight: 700;
}
QLabel#effectDragHandle:hover { color: #eeeeee; }
QToolButton#effectIconButton {
    min-width: 24px;
    min-height: 24px;
    background: transparent;
    color: #a0a0a0;
    border: 1px solid transparent;
}
QToolButton#effectIconButton:hover, QToolButton#effectIconButton:focus {
    color: #eeeeee;
    border-color: #555555;
}
QToolButton#effectMoreButton {
    min-width: 29px;
    min-height: 26px;
    background: transparent;
    color: #b0b0b0;
    border: 1px solid #3b3b3b;
    font-weight: 700;
}
QToolButton#effectMoreButton:hover, QToolButton#effectMoreButton:focus {
    color: #eeeeee;
    background: #1b1b1b;
    border-color: #777777;
}
QPushButton#effectAddButton, QPushButton#effectTextButton {
    min-height: 26px;
    background: transparent;
    color: #a8a8a8;
    border: 1px solid #3b3b3b;
    padding: 0 8px;
    font-size: 10px;
    letter-spacing: 1px;
}
QPushButton#effectAddButton:hover, QPushButton#effectTextButton:hover {
    color: #eeeeee;
    background: #1b1b1b;
    border-color: #777777;
}
QComboBox#effectPreset {
    min-height: 26px;
    max-width: 126px;
    padding: 0 7px;
    font-size: 10px;
}
QComboBox#effectMode {
    min-height: 26px;
    min-width: 136px;
    max-width: 184px;
    padding: 0 7px;
    font-size: 10px;
}
QSpinBox#effectAmount {
    min-height: 25px;
    max-width: 74px;
    padding: 0 6px;
    font-size: 10px;
}
QSpinBox#effectKeyframeSpin {
    min-height: 25px;
    max-width: 58px;
    padding: 0 5px;
    font-size: 10px;
}
QMenu {
    background: #111111;
    color: #dddddd;
    border: 1px solid #555555;
    padding: 4px;
}
QMenu::item { padding: 7px 28px 7px 10px; }
QMenu::item:selected { background: #eeeeee; color: #090909; }
QMenu::item:disabled { color: #555555; }
QLabel#timecode, QLabel#valueLabel {
    color: #d0d0d0;
    font-size: 12px;
    font-weight: 600;
}
QPushButton {
    min-height: 34px;
    border-radius: 2px;
    padding: 0 14px;
    font-weight: 600;
}
QPushButton#primaryButton {
    background: #eeeeee;
    color: #090909;
    border: 1px solid #eeeeee;
    min-height: 42px;
    padding: 0 20px;
}
QPushButton#primaryButton:hover { background: #f8f8f8; border-color: #f8f8f8; }
QPushButton#secondaryButton {
    background: #151515;
    color: #dedede;
    border: 1px solid #555555;
    min-height: 42px;
    padding: 0 18px;
}
QPushButton#secondaryButton:hover, QPushButton#quietButton:hover {
    background: #202020;
    border-color: #868686;
}
QPushButton#quietButton {
    background: transparent;
    color: #ababab;
    border: 1px solid #3d3d3d;
    padding: 0 10px;
}
QPushButton#primaryButton:disabled, QPushButton#secondaryButton:disabled {
    background: #171717;
    color: #565656;
    border-color: #292929;
}
QPushButton:focus, QToolButton:focus, QListWidget:focus {
    border: 1px solid #eeeeee;
}
QComboBox, QSpinBox {
    min-height: 36px;
    background: #151515;
    color: #dddddd;
    border: 1px solid #414141;
    border-radius: 2px;
    padding: 0 10px;
}
QComboBox:focus, QSpinBox:focus { border-color: #e5e5e5; }
QComboBox:disabled, QSpinBox:disabled {
    background: #111111;
    color: #555555;
    border-color: #292929;
}
QComboBox::drop-down { border: none; width: 26px; }
QComboBox QAbstractItemView {
    background: #151515;
    selection-background-color: #eeeeee;
    selection-color: #090909;
    border: 1px solid #545454;
}
QSlider::groove:horizontal {
    height: 3px;
    background: #3b3b3b;
    border-radius: 1px;
}
QSlider::sub-page:horizontal { background: #e6e6e6; border-radius: 1px; }
QSlider::handle:horizontal {
    width: 12px;
    margin: -5px 0;
    background: #f2f2f2;
    border: 2px solid #101010;
    border-radius: 6px;
}
QSlider:focus::groove:horizontal { background: #606060; }
QSlider::groove:horizontal:disabled { background: #242424; }
QSlider::sub-page:horizontal:disabled { background: #343434; }
QSlider::handle:horizontal:disabled {
    background: #454545;
    border-color: #202020;
}
QCheckBox {
    color: #a9a9a9;
    spacing: 8px;
    padding-top: 2px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    background: #141414;
    border: 1px solid #565656;
    border-radius: 1px;
}
QCheckBox::indicator:checked {
    background: #eeeeee;
    border-color: #eeeeee;
    image: url("__CHECKMARK_ICON__");
}
QCheckBox:checked { color: #eeeeee; }
QCheckBox:disabled { color: #555555; }
QCheckBox::indicator:disabled { background: #111111; border-color: #303030; }
QCheckBox:focus { color: #eeeeee; }
QToolButton#modeButton {
    min-height: 30px;
    min-width: 76px;
    background: #0e0e0e;
    color: #858585;
    border: 1px solid #343434;
    padding: 0 11px;
}
QToolButton#modeButton:checked {
    background: #eeeeee;
    color: #090909;
    border-color: #eeeeee;
}
QToolButton#advancedToggle {
    min-height: 38px;
    background: transparent;
    color: #a5a5a5;
    border-top: 1px solid #303030;
    border-bottom: 1px solid #303030;
    text-align: left;
    padding: 0 4px;
}
QListWidget#frameList {
    background: #0b0b0b;
    color: #b4b4b4;
    border: 1px solid #303030;
    border-radius: 2px;
    outline: none;
    padding: 3px;
}
QListWidget#frameList::item {
    min-height: 28px;
    padding: 2px 4px;
    border-bottom: 1px solid #242424;
}
QListWidget#frameList::item:selected { background: #e6e6e6; color: #090909; }
QSplitter#bodySplitter::handle { background: #303030; width: 1px; }
QLabel#inspectorTip {
    color: #818181;
    background: #121212;
    border: 1px solid #303030;
    border-radius: 2px;
    padding: 12px;
    font-size: 11px;
}
QFrame#statusBar {
    background: #0d0d0d;
    border-top: 1px solid #303030;
}
QLabel#statusText { color: #e5e5e5; }
QProgressBar {
    height: 3px;
    background: #303030;
    border: none;
}
QProgressBar::chunk { background: #eeeeee; }
QMessageBox { background: #0f0f0f; }
QToolTip {
    color: #e5e5e5;
    background: #191919;
    border: 1px solid #555555;
    padding: 6px;
}
"""


def _load_terminal_font() -> str:
    candidate_groups = (
        tuple(Path("C:/Windows/Fonts").glob("consola*.ttf")),
        (
            Path("/System/Library/Fonts/SFNSMono.ttf"),
            Path("/System/Library/Fonts/Supplemental/Courier New.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf"),
            Path("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf"),
        ),
    )
    for candidates in candidate_groups:
        families: list[str] = []
        for path in candidates:
            if not path.is_file():
                continue
            font_id = QFontDatabase.addApplicationFont(str(path))
            if font_id >= 0:
                families.extend(QFontDatabase.applicationFontFamilies(font_id))
        if families:
            return families[0]
    return QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family()


def application_stylesheet(font_family: str) -> str:
    checkmark = (Path(__file__).parent / "assets" / "check.svg").as_posix()
    return STYLESHEET.replace("__APP_MONO_FONT__", font_family).replace(
        "__CHECKMARK_ICON__", checkmark
    )


def main() -> int:
    if "--version" in sys.argv:
        print(f"Chronophoto {__version__}")
        return 0
    app = QApplication(sys.argv)
    app.setApplicationName("Chronophoto")
    app.setOrganizationName("Chronophoto")
    app.setApplicationVersion(__version__)
    app.setWindowIcon(QIcon(str(Path(__file__).parent / "assets" / "chronophoto-icon.png")))
    font_family = _load_terminal_font()
    fixed_font = QFont(font_family, 10)
    app.setFont(fixed_font)
    app.setStyleSheet(application_stylesheet(font_family))
    window = ChronophotoWindow(check_updates=True)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
