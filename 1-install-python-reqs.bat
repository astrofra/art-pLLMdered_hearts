@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHON_EXE=python"
if exist "python\python.exe" set "PYTHON_EXE=%~dp0python\python.exe"
if not exist ".venv\Scripts\python.exe" (
    "%PYTHON_EXE%" -m venv "%~dp0.venv"
    if errorlevel 1 goto :failed
)
"%~dp0.venv\Scripts\python.exe" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :failed
echo Installation complete. Run 3-run-ai-play.bat.
pause
exit /b 0
:failed
echo Installation failed. Python 3.10 or newer is required.
pause
exit /b 1
