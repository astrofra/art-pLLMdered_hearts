# pLLuMdered Hearts - Agent Notes

Guidelines for maintaining the installation that plays Plundered Hearts while driving a video interview viewer. Keep everything simple, Pythonic (3.10 max), and light on dependencies.

## Scope
- Core logic lives in `src/faketerm.py`; keep new modules minimal.
- External programs: `frotz` (Z-Machine interpreter), ROM `roms/PLUNDERE.z3`, and `ollama`.
- The Godot viewer in `godot-viewer/` plays interview videos and polls `llm_out/` for new requests.
- `assets/abriggs-itw-embeddings.json` holds `filename`, `embedding`, `sequence_title`, and `duration_sec`.

## Coding style (Python 3.10)
- Prefer stdlib (os, time, re, json); avoid extra libs unless unavoidable. If you must add one, document why.
- Keep functions small and single-purpose; lean on clear names over comments.
- No pattern matching or 3.11+ syntax; target 3.10 for compatibility.
- Handle text as plain ASCII/UTF-8; sanitize terminal escapes before passing to the model or logs.

## Runtime flow (`src/faketerm.py`)
- Spawns `frotz` via `pexpect`, waits for the initial prompt, and plays the walkthrough in `plundered_hearts_commands`.
- Game output is cleaned by `clean_output()` and rendered through the C64 renderer when enabled.
- Complete game outputs and executed commands accumulate until the history exceeds `COMMENTARY_MIN_WORDS` (default 150). Commentary runs only at a game prompt, after rendering the whole output.
- Each successful LLM comment is displayed immediately, including during an interview, and clears the game-history buffer. Empty responses keep the buffer. History and the pending comment survive walkthrough restarts.
- Only after the current interview's estimated duration plus `VIDEO_END_MARGIN_SEC` (default 3) has elapsed is the latest pending comment embedded with `VIDEO_EMBED_MODEL` and matched by cosine similarity against `assets/abriggs-itw-embeddings.json`.
- The chosen interview `filename` is written to a timestamped file in `llm_out/`.
- A cooldown uses `duration_sec` to avoid queuing a new video while the current one is still expected to play. Godot sends no playback feedback to Python.
- The game loop restarts after the final command and shows a restart message to keep the installation running continuously.

## Godot viewer (`godot-viewer/main.gd`)
- Polls `../llm_out` (or `LLM_OUT_OVERRIDE`) for the newest timestamp file and enqueues the video by filename.
- Interview videos do not loop; noise videos loop by restarting when finished.
- When the queue is empty, the viewer plays noise videos.
- Toggles:
  - `USE_FRENCH_SUBTITLES` expects `-fr` subtitle suffix.
  - `PREFILL_VIDEO_QUEUE` preloads the numbered videos or starts with noise only.
  - `LLM_OUT_OVERRIDE` can force an absolute watchfolder path for exported builds.

## LLM usage
- Commentary uses `LLM_MODEL` (default `gemma4:e4b`) via `ollama.chat`, with extended thinking disabled by default for Gemma 4 (`LLM_THINK=0`).
- Video matching uses embeddings via `ollama.embeddings`.
- Keep prompts short and deterministic; avoid multi-turn chains.
- Guard against missing or slow responses to avoid blocking the loop.
- `LLM_NUM_CTX` defaults to 4096 and `LLM_NUM_PREDICT` to 256. `COMMENTARY_LOG_PATH` optionally logs full histories and token metrics as JSONL.

## Subtitle and embedding tools
- `src/embed_vtt.py` builds `abriggs-itw-embeddings.json` from subtitle `.txt` files and adds `sequence_title`.
- `src/translate_subtitles.py` produces French subtitle copies (`-fr.txt`) with context-aware translation.
- `src/compute_itw_durations.py` adds `duration_sec` from subtitle timecodes.

## Testing and ops
- Run `python src/faketerm.py` with `frotz` and the ROM present.
- Ensure `ollama` is running locally and required models are pulled.
- For the viewer, confirm the exported executable can reach `llm_out/` (use `LLM_OUT_OVERRIDE` if needed).

## Writing new features
- Favor configuration flags over hard-coding (e.g., toggles and timeouts).
- Keep file paths relative and document new assets.
- Avoid aggressive output filtering that might remove story text.
