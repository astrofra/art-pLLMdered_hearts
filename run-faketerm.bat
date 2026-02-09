@echo off
setlocal
cd /d %~dp0

set "ROOT=%CD%"
set "PY_HOME=%ROOT%\python"

if not exist "%PY_HOME%\python.exe" (
  echo Portable Python not found at "%PY_HOME%".
  echo See documentation\portable-python-study.md for setup steps.
  pause
  exit /b 1
)

set "PYTHONHOME=%PY_HOME%"
set "PYTHONPATH=%ROOT%\src;%PY_HOME%\Lib\site-packages"
set "PATH=%ROOT%;%PY_HOME%;%PATH%"

"%PY_HOME%\python.exe" src\faketerm.py
pause
