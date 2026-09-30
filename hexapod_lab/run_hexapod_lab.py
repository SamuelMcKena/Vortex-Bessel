"""Launcher for the Hexapod + Laser Lab controller.

Unhandled exceptions are written to ``hexapod_lab_crash.log`` next to this file
as well as to stderr. PySide6 terminates the process when a Python exception
escapes a slot or a virtual such as ``paintEvent``, and the Windows launcher
runs under ``pythonw`` with no console, so without this the process simply
vanishes with nothing to diagnose.
"""

from __future__ import annotations

import datetime as _datetime
import faulthandler
import sys
import traceback
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent / "hexapod_lab_crash.log"


def _timestamp() -> str:
    return _datetime.datetime.now().isoformat(timespec="seconds")


def _append(text: str) -> None:
    try:
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
    except Exception:
        pass


def _install_crash_logging() -> None:
    # Native faults (VTK/Qt) dump a Python stack here.
    try:
        handle = LOG_PATH.open("a", encoding="utf-8")
        handle.write(f"\n=== session started {_timestamp()} ===\n")
        handle.flush()
        faulthandler.enable(file=handle, all_threads=True)
    except Exception:
        faulthandler.enable(all_threads=True)

    previous_hook = sys.excepthook

    def hook(exc_type, exc, tb) -> None:
        _append(
            f"\n--- unhandled exception {_timestamp()} ---\n"
            + "".join(traceback.format_exception(exc_type, exc, tb))
        )
        previous_hook(exc_type, exc, tb)

    sys.excepthook = hook

    try:
        import threading

        def thread_hook(args) -> None:
            _append(
                f"\n--- unhandled thread exception {_timestamp()} "
                f"in {args.thread} ---\n"
                + "".join(
                    traceback.format_exception(
                        args.exc_type, args.exc_value, args.exc_traceback
                    )
                )
            )

        threading.excepthook = thread_hook
    except Exception:
        pass

    try:
        from PySide6 import QtCore

        def qt_message(mode, context, message) -> None:
            label = str(mode).rsplit(".", 1)[-1]
            _append(f"[{_timestamp()}] Qt {label}: {message}\n")
            print(f"Qt {label}: {message}", file=sys.stderr)

        QtCore.qInstallMessageHandler(qt_message)
    except Exception:
        pass


_install_crash_logging()

from hexapod_lab.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
