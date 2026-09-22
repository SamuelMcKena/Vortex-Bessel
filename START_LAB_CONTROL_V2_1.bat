@echo off
setlocal
cd /d "%~dp0"
python SPYDER_RUN_LAB_CONTROL_V2_1.py
if errorlevel 1 (
  echo.
  echo Lab Control exited with an error. Preserve lab_gui\logs\lab_gui_crash.log.
  pause
)
