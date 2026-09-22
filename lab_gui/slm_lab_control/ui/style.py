APP_QSS = r"""
* {
    font-family: 'Segoe UI', 'Inter', 'Arial';
    font-size: 10.5pt;
}
QMainWindow, QWidget {
    background: #07101d;
    color: #eaf2ff;
}
QFrame#Sidebar {
    background: #050b14;
    border-right: 1px solid #17253a;
}
QLabel#Brand {
    font-size: 18pt;
    font-weight: 800;
    letter-spacing: 1.4px;
    color: #f7fbff;
}
QLabel#Title {
    font-size: 21pt;
    font-weight: 780;
    color: #ffffff;
}
QLabel#PageTitle {
    font-size: 18pt;
    font-weight: 760;
    color: #ffffff;
}
QLabel#PanelTitle {
    font-size: 15pt;
    font-weight: 740;
    color: #ffffff;
}
QLabel#CardTitle {
    font-size: 11.5pt;
    font-weight: 700;
    color: #f6fbff;
}
QLabel#SectionTitle {
    font-size: 12pt;
    font-weight: 740;
    color: #8fc9ff;
}
QLabel#SectionSubtitle, QLabel#Subtitle, QLabel#Muted {
    color: #91a8c6;
}
QLabel#SectionSubtitle {
    font-size: 9.7pt;
}
QLabel#TinyMuted {
    color: #6f87a6;
    font-size: 9pt;
}
QLabel#LockedValue {
    color: #b7d9ff;
    font-weight: 650;
}
QLabel#ActiveTerms {
    color: #a9c4e6;
    background: #081525;
    border: 1px solid #1b3350;
    border-radius: 8px;
    padding: 8px 10px;
}
QLabel#LockedProfile {
    color: #c7ddf5;
    background: #081525;
    border: 1px solid #1d3553;
    border-radius: 10px;
    padding: 12px;
    margin: 3px 0;
}
QLabel#StatusGood {
    color: #86efac;
    background: #0b281d;
    border: 1px solid #1b5a3d;
    border-radius: 9px;
    padding: 7px 10px;
    font-weight: 700;
}
QLabel#StatusWarn {
    color: #fbd38d;
    background: #35250a;
    border: 1px solid #7c5a13;
    border-radius: 9px;
    padding: 7px 10px;
    font-weight: 700;
}
QFrame#Card, QFrame#LockedCard, QGroupBox#ComponentGroup, QGroupBox {
    background: #0b1728;
    border: 1px solid #1b304b;
    border-radius: 13px;
}
QFrame#LockedCard {
    background: #0a1422;
    border-color: #263d5d;
}
QGroupBox {
    margin-top: 12px;
    padding: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 7px;
    color: #afd7ff;
    font-weight: 700;
}
QGroupBox::indicator {
    width: 16px;
    height: 16px;
}
QPushButton {
    background: #132a49;
    border: 1px solid #285684;
    border-radius: 9px;
    padding: 8px 12px;
    color: #f5f9ff;
    font-weight: 680;
}
QPushButton:hover {
    background: #193960;
    border-color: #3f78ad;
}
QPushButton:pressed {
    background: #0c213c;
}
QPushButton#Primary {
    background: #1368c4;
    border-color: #3f9dff;
}
QPushButton#Primary:hover {
    background: #1978da;
}
QPushButton#Danger {
    background: #6e2026;
    border-color: #c44a54;
}
QPushButton#Danger:hover {
    background: #842b33;
}
QPushButton#Quiet {
    background: #0b192b;
    border-color: #233c5c;
    color: #c8d9ec;
}
QPushButton#NavButton {
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 9px;
    padding: 10px 12px;
    color: #8ea5c3;
    font-weight: 680;
}
QPushButton#NavButton:hover {
    background: #0d1a2b;
    color: #dbeaff;
}
QPushButton#NavButton:checked {
    background: #102d50;
    color: #ffffff;
    border-color: #244f7c;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QListWidget {
    background: #071322;
    border: 1px solid #223d5c;
    border-radius: 8px;
    padding: 6px 8px;
    selection-background-color: #1767b8;
    color: #eff6ff;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QListWidget:focus {
    border-color: #4a91cf;
}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 0;
    height: 0;
    border: none;
}
QComboBox::drop-down {
    border: none;
    width: 26px;
}
QCheckBox {
    spacing: 8px;
    color: #caddf1;
}
QLabel#PreviewLabel {
    background: #020711;
    border: 1px solid #203a59;
    border-radius: 10px;
    padding: 8px;
}
QScrollArea {
    border: none;
    background: transparent;
}
QScrollBar:vertical {
    background: #07101d;
    width: 11px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #244362;
    min-height: 35px;
    border-radius: 5px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QSplitter::handle {
    background: #14253a;
    width: 2px;
}
QListWidget#PresetList::item {
    padding: 9px 8px;
    border-radius: 6px;
}
QListWidget#PresetList::item:selected {
    background: #154d80;
    color: #ffffff;
}
QToolTip {
    background: #0b1728;
    color: #f5f9ff;
    border: 1px solid #33597f;
}
"""
