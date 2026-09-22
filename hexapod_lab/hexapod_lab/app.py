from __future__ import annotations

import sys
from PySide6 import QtWidgets

from .main_window import MainWindow


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Hexapod + Laser Lab")
    app.setOrganizationName("Heriot-Watt AOP")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
