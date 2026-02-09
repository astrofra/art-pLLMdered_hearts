@echo off
setlocal
cd /d %~dp0

set "ROOT=%CD%"
set "PY_HOME=%ROOT%\python"
set "ZIP_PATTERN=python-3.10.*-embed-amd64.zip"
set "GET_PIP=%ROOT%\get-pip.py"

if exist "%PY_HOME%\python.exe" (
  echo Portable Python already present at "%PY_HOME%".
  goto :pip_setup
)

set "PY_ZIP=%ROOT%\python-3.10.11-embed-amd64.zip"
if not exist "%PY_ZIP%" (
  set "PY_ZIP="
  set "PY_ZIP_TMP=%TEMP%\pyzip_path.txt"
  powershell -NoProfile -Command "$files = [System.IO.Directory]::GetFiles($env:ROOT, $env:ZIP_PATTERN); if ($files.Length -gt 0) { $files[0] }" > "%PY_ZIP_TMP%"
  set /p PY_ZIP=<"%PY_ZIP_TMP%"
  del "%PY_ZIP_TMP%" >nul 2>nul
)

if not defined PY_ZIP (
  echo Could not find a Python embeddable ZIP matching:
  echo   %ZIP_PATTERN%
  echo.
  echo Download the Windows embeddable ZIP for Python 3.10 (x64)
  echo and place it in this folder, then re-run this script.
  pause
  exit /b 1
)

echo Extracting "%PY_ZIP%" to "%PY_HOME%"...
powershell -NoProfile -Command "New-Item -ItemType Directory -Path \"%PY_HOME%\" -Force | Out-Null; Expand-Archive -LiteralPath \"%PY_ZIP%\" -DestinationPath \"%PY_HOME%\" -Force"

echo Updating python310._pth...
powershell -NoProfile -Command ^
  "$pth = Join-Path '%PY_HOME%' 'python310._pth';" ^
  "if (-not (Test-Path $pth)) { Write-Host 'Missing python310._pth'; exit 1 }" ^
  "$lines = Get-Content $pth;" ^
  "$lines = $lines | ForEach-Object { if ($_ -match '^\s*#?\s*import site') { 'import site' } else { $_ } };" ^
  "if (-not ($lines -contains 'Lib\\site-packages')) { $lines += 'Lib\\site-packages' };" ^
  "Set-Content -Path $pth -Value $lines -Encoding ASCII"

:pip_setup
if not exist "%GET_PIP%" (
  echo get-pip.py not found at "%GET_PIP%".
  echo Download it from https://bootstrap.pypa.io/get-pip.py
  echo and place it in this folder, then re-run this script.
  pause
  exit /b 1
)

echo Bootstrapping pip...
"%PY_HOME%\python.exe" "%GET_PIP%"

echo Installing dependencies into portable site-packages...
"%PY_HOME%\python.exe" -m pip install -r requirements.txt --target "%PY_HOME%\Lib\site-packages"

echo Done. You can now run "run-faketerm.bat".
pause
