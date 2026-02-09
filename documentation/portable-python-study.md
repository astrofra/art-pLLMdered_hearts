# Porting Study: Run `src/faketerm.py` with Portable Python (No Install)

## Purpose
Evaluate whether `src/faketerm.py` can run as an uncompiled, source-open program on Windows 10/11 without installing Python, using the official **portable/embeddable** Python distribution. Users will still install Ollama.

## Feasibility Summary
**Yes, feasible.** The official Python **embeddable (portable) ZIP** can run the project directly with local `site-packages`. The approach is low-risk, keeps the code unchanged, and avoids a full C port. The main work is packaging and a small launcher script.

## Why This Works
`faketerm.py` is pure Python plus a few pip dependencies:
- `ollama` (HTTP client for Ollama)
- `pexpect` (used via `pexpect.popen_spawn.PopenSpawn`)
- `pygame` (SDL-based renderer and audio)

All of these have Windows wheels and can be installed into a local `site-packages` directory inside a portable Python bundle.

## Proposed Packaging Layout
```
pLLMdered_hearts/
  python/                 # portable Python 3.10.x (embeddable zip)
    python.exe
    python310.dll
    python310._pth
    Lib/
      site-packages/
  src/
  assets/
  roms/
  bin/
  llm_out/
  run-faketerm.bat
```

## Build Steps (Done Once on a Dev Machine)
### Scripted (Recommended)
1. Download the **Python 3.10 embeddable ZIP** (Windows x64) and place it in the repo root.
2. Download `get-pip.py` and place it in the repo root.
3. Run:
   ```
   0-setup-portable-python.bat
   ```
4. Launch:
   ```
   run-faketerm.bat
   ```

### Manual (If You Prefer)
1. **Download Python 3.10 embeddable ZIP** (Windows x64) from python.org.
2. **Extract to** `python/` under the repo root.
3. **Edit** `python/python310._pth` to enable local packages:
   - Add `Lib\site-packages`
   - Uncomment or add `import site`
4. **Bootstrap pip** in the portable folder:
   - Download `get-pip.py`
   - Run:
     ```
     python\python.exe get-pip.py
     ```
5. **Install dependencies into the portable tree**:
   ```
   python\python.exe -m pip install -r requirements.txt --target python\Lib\site-packages
   ```
6. **Add a launcher** `run-faketerm.bat` (example below).

## Example Launcher (`run-faketerm.bat`)
```
@echo off
setlocal
cd /d %~dp0
set "PYTHONHOME=%CD%\python"
set "PYTHONPATH=%CD%\python\Lib\site-packages"
set "PATH=%CD%;%PATH%"
python\python.exe src\faketerm.py
```

Notes:
- `PATH` includes the repo root so `frotz.exe` is found.
- Running from the repo root keeps all relative paths working.

## End-User Requirements (No Python Install)
Required:
- Windows 10/11
- This repo folder with the bundled portable Python
- Ollama installed and running locally
- `frotz.exe` and `roms/PLUNDERE.z3` present (already in repo)

Not required:
- System Python
- Admin rights

## Risks / Caveats
1. **Embeddable Python does not include pip by default**  
   Must bootstrap pip once as described.

2. **`pygame` native dependencies**  
   The wheel bundles SDL, but some systems may still need the Microsoft Visual C++ runtime. If the app crashes at import, installing the VC++ 2015-2022 Redistributable fixes it.

3. **`pexpect` on Windows**  
   `faketerm.py` uses `PopenSpawn`, which is compatible on Windows. If future code switches to `pexpect.spawn` it may break on Windows.

4. **Path assumptions**  
   The script launches `frotz` via `frotz -p roms/PLUNDERE.z3`. The launcher ensures `frotz.exe` is on the PATH by adding the repo root.

5. **Ollama dependency**  
   This is still an external install and service. The portable Python does not change this requirement.

## Validation Checklist
1. Double-click `run-faketerm.bat` on a clean Windows 10/11 machine.
2. Confirm the C64 renderer opens and the walkthrough starts.
3. Confirm Ollama responses appear and videos are queued in `llm_out/`.
4. Confirm the Godot viewer can still be launched from `bin/itw-viewer.exe`.

## Recommendation
Use **portable Python 3.10 embeddable ZIP + local site-packages**. This avoids a C port, keeps source open, and meets the "no Python install" requirement with minimal operational overhead.
