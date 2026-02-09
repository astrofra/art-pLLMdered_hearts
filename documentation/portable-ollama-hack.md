# Portable Ollama (Hack) - Windows

## Purpose
Provide a pragmatic, unofficial way to run Ollama from a folder without a full installer footprint, and keep model files in that same folder. This is a hack, not a supported mode.

## Source Context
The request for a Windows portable mode (including a `portable.txt` marker and storing data inside the portable folder) is tracked as a feature request in Ollama issue #6782. This document describes a workaround that partially matches that goal.

## Facts We Rely On (Official Docs)
- Windows default models/config live under `%HOMEPATH%\.ollama`, and the model location can be changed with `OLLAMA_MODELS`.
- The Windows installer can change its install location with `OllamaSetup.exe /DIR="d:\some\location"`.
- Windows docs list locations for logs, binaries, and default models/config.
- A standalone Windows CLI zip (`ollama-windows-amd64.zip`) exists, with an optional ROCm zip for AMD GPUs.

## What "Portable" Means Here
- Run `ollama` from a local folder without a system-wide install.
- Keep model files in a local folder by setting `OLLAMA_MODELS`.
- Accept that some data (logs, possibly config) may still be written to AppData or the user profile, because there is no documented portable mode.

## Approach A (Preferred): Standalone CLI Zip
This is the cleanest "portable-ish" path because it avoids the installer.

### Layout
```
portable-ollama/
  ollama/                # extracted ollama-windows-amd64.zip
  ollama-data/
    models/              # model storage via OLLAMA_MODELS
  run-ollama.bat
```

### Steps
1. Download and extract the standalone CLI zip into `portable-ollama/ollama`.
2. If you use an AMD GPU, also extract the ROCm zip into the same `ollama/` folder.
3. Create `portable-ollama/ollama-data/models`.
4. Create a launcher:
   ```bat
   @echo off
   setlocal
   cd /d %~dp0
   set "OLLAMA_MODELS=%CD%\ollama-data\models"
   set "PATH=%CD%\ollama;%PATH%"
   ollama serve
   ```
5. Keep this window running while `src/faketerm.py` or other clients use the Ollama API.

### Updating
Windows docs advise removing old directories when upgrading. For the CLI zip, replace the `ollama/` folder with the new version.

## Approach B: Installer, But Redirected
If you must use the installer, you can still keep binaries together and move models:

1. Run the installer with a custom location:
   `OllamaSetup.exe /DIR="X:\portable-ollama\ollama"`.
2. Set `OLLAMA_MODELS` to a path inside `portable-ollama\ollama-data\models`.

This still writes logs and other app data under `%LOCALAPPDATA%` and uses `%HOMEPATH%\.ollama` by default for models/config unless overridden.

## Optional `portable.txt`
Issue #6782 suggests a `portable.txt` marker. There is no documented behavior tied to this file, so treat it as a human-visible marker only.

## Verification Checklist
1. Run `ollama serve` via `run-ollama.bat`.
2. Run `ollama list` or `ollama pull <model>`.
3. Confirm model files appear under `portable-ollama\ollama-data\models` (not under the default user profile path).

## Limitations
- Logs and update artifacts are still stored under `%LOCALAPPDATA%\Ollama`.
- Default models/config live under `%HOMEPATH%\.ollama` unless overridden with `OLLAMA_MODELS`.
- This is a hack; future Ollama changes may invalidate it.

## Sources
- https://github.com/ollama/ollama/issues/6782
- https://docs.ollama.com/windows
