@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHON_EXE=python"
if exist "python\python.exe" set "PYTHON_EXE=%~dp0python\python.exe"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
"%PYTHON_EXE%" -c "import ollama, pexpect, pygame" >nul 2>&1
if errorlevel 1 (
    echo Python or dependencies missing. Run 1-install-python-reqs.bat first.
    pause
    exit /b 1
)
"%PYTHON_EXE%" -u "%~dp0src\faketerm.py" %*
set "RUN_EXIT=%ERRORLEVEL%"
if not "%RUN_EXIT%"=="0" pause
exit /b %RUN_EXIT%
