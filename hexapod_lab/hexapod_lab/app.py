from __future__ import annotations

import sys
from PySide6 import QtWidgets

from .main_window import MainWindow


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Hexapod + Laser Lab")
    app.setOrganizationName("Heriot-Watt AOP")
    window = MainWindow()
    # A lab control screen wants the whole desktop: on a 1280x800 panel the
    # four-column layout does not fit in a floating window.
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
