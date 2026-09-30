@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ==============================================
echo   Hexapod + Laser Lab - first-time setup
echo ==============================================
echo.
echo This installer does NOT require the Windows "py" launcher.
echo It will use an existing local .venv, python, Anaconda/Miniconda,
echo or the Windows Python launcher if available.
echo.

if exist ".venv\Scripts\python.exe" goto :have_venv

echo Creating a private Python environment...

where python.exe >nul 2>nul
if not errorlevel 1 (
  echo   Trying: python -m venv .venv
  python -m venv .venv
)

if exist ".venv\Scripts\python.exe" goto :have_venv

if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\python.exe" (
  echo   Trying active Conda Python: %CONDA_PREFIX%\python.exe
  "%CONDA_PREFIX%\python.exe" -m venv .venv
)

if exist ".venv\Scripts\python.exe" goto :have_venv

if exist "%USERPROFILE%\anaconda3\python.exe" (
  echo   Trying Anaconda Python...
  "%USERPROFILE%\anaconda3\python.exe" -m venv .venv
)

if exist ".venv\Scripts\python.exe" goto :have_venv

if exist "%USERPROFILE%\miniconda3\python.exe" (
  echo   Trying Miniconda Python...
  "%USERPROFILE%\miniconda3\python.exe" -m venv .venv
)

if exist ".venv\Scripts\python.exe" goto :have_venv

where py.exe >nul 2>nul
if not errorlevel 1 (
  echo   Trying Windows Python launcher...
  py -3.12 -m venv .venv >nul 2>nul
  if not exist ".venv\Scripts\python.exe" py -3.11 -m venv .venv >nul 2>nul
  if not exist ".venv\Scripts\python.exe" py -3 -m venv .venv >nul 2>nul
)

if not exist ".venv\Scripts\python.exe" goto :no_python

:have_venv
echo.
echo Private Python:
".venv\Scripts\python.exe" --version
if errorlevel 1 goto :failed

echo.
echo Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed

echo.
echo Installing required GUI packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo.
echo Installing exact STEP/CAD support...
".venv\Scripts\python.exe" -m pip install "cadquery>=2.5,<3"
if errorlevel 1 (
  echo.
  echo WARNING: CadQuery did not install.
  echo The GUI can still start, but exact STEP rendering may be unavailable.
  echo You can retry later with:
  echo   .venv\Scripts\python.exe -m pip install -r requirements-cad.txt
  echo.
)

echo Checking GUI imports...
".venv\Scripts\python.exe" -c "import numpy, PySide6, pyvista, pyvistaqt; print('GUI dependencies OK')"
if errorlevel 1 goto :failed

echo Running source compile check...
".venv\Scripts\python.exe" -m compileall -q hexapod_lab run_hexapod_lab.py
if errorlevel 1 goto :failed

echo.
echo ==============================================
echo   Setup complete - starting Hexapod Lab
echo ==============================================
echo.
echo This first launch stays attached to this console so any error is visible.
echo Later launches can use START_HEXAPOD_LAB.bat.
echo.
".venv\Scripts\python.exe" run_hexapod_lab.py
if errorlevel 1 goto :failed
exit /b 0

:no_python
echo.
echo ERROR: No usable Python installation could create the private environment.
echo.
echo This PC can also launch from Spyder after setup by running SPYDER_LAUNCH.py.
echo If Python works in Spyder but not here, run this in the Spyder IPython console:
echo.
echo   import sys; print(sys.executable)
echo.
echo and send me the printed path.
pause
exit /b 1

:failed
echo.
echo ==============================================
echo   Hexapod Lab setup/start failed
echo ==============================================
echo.
echo The important error should be directly above this message.
echo If a GUI window appeared and vanished, also send hexapod_lab_crash.log.
pause
exit /b 1
