# pLLuMdered Hearts - Agent Notes

Guidelines for maintaining the installation that plays Plundered Hearts while driving a video interview viewer. Keep everything simple, Pythonic (3.10 max), and light on dependencies.

## Scope
- Core logic lives in `src/faketerm.py`; keep new modules minimal.
- External programs: `frotz` (Z-Machine interpreter), ROM `roms/PLUNDERE.z3`, and `ollama`.
- The Godot viewer in `godot-viewer/` plays interview videos and polls `llm_out/` for new requests.
- `assets/abriggs-itw-passages.json` holds 25 clip entries with `filename`, full-clip `embedding`, `sequence_title`, `duration_sec`, and contextual `passages` (132 total), each retaining text, timecodes and its own embedding. The historical `assets/abriggs-itw-embeddings.json` remains available.

## Coding style (Python 3.10)
- Prefer stdlib (os, time, re, json); avoid extra libs unless unavoidable. If you must add one, document why.
- Keep functions small and single-purpose; lean on clear names over comments.
- No pattern matching or 3.11+ syntax; target 3.10 for compatibility.
- Handle text as plain ASCII/UTF-8; sanitize terminal escapes before passing to the model or logs.

## Runtime flow (`src/faketerm.py`)
- Spawns `frotz` via `pexpect`, waits for the initial prompt, and plays the walkthrough in `plundered_hearts_commands`.
- Game output is cleaned by `clean_output()` and rendered through the C64 renderer when enabled.
- Complete game outputs and executed commands accumulate. During an interview, commentary runs when history exceeds `COMMENTARY_MIN_WORDS` (default 150). When playback is estimated idle, a fresh comment is requested at the next game prompt even below that threshold. Always render the whole game output first.
- Each successful LLM comment is displayed immediately, including during an interview, and clears the game-history buffer. Empty responses keep the buffer. Uncommented history survives walkthrough restarts.
- The prompt includes only the two most recent successfully displayed comments as anti-repetition context, separate from the game history. This bounded memory survives walkthrough restarts; empty responses never enter it.
- Commentary receives only completed game output and executed commands, never the next walkthrough command. The LLM speaks in its own first person, using a conversational expression inspired by Briggs, and explicitly connects one concrete scene detail to a documented recollection. It attributes her memories to her, never impersonates her or the heroine, and never infers a scene's production causes from fiction.
- Only a comment just displayed at the current prompt can select a video, after the current interview's estimated duration plus `VIDEO_END_MARGIN_SEC` (default 3). Comments displayed during playback are not saved for later selection. After failed generation, embedding or request writing, try a fresh comment at a later idle prompt; never launch from an old off-screen comment. Each clip receives its best individual passage cosine, without averaging. Clip recency and durations still apply to 25 videos, not 132 passages.
- `VIDEO_MATCH_MODE=clip` uses full-clip vectors; `VIDEO_EMBEDDINGS_PATH` overrides the catalogue path (relative to the project root). Legacy clip-only entries remain supported.
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
- Commentary uses `LLM_MODEL` (default `gemma4:e4b`) via `ollama.chat`, with extended thinking enabled by default for Gemma 4 (`LLM_THINK=1`). Set `LLM_THINK=0` to disable it.
- `LLM_TEMPERATURE` defaults to 0.3. `INTERVIEW_CONTEXT_PATH` defaults to `assets/abriggs-itw-facts-brief.txt`, eight compact factual entries (191 words), including the coffer bug, expectations about familiar objects and help from colleagues. Relative overrides resolve against the project root; the 750-word narrative and longer fact sheets remain available for comparison.
- Video matching uses embeddings via `ollama.embed`, bare text, `truncate=False`, context 2048, CPU by default. Catalogue and query vectors use the same input format.
- Successful video requests are logged to `llm_out/video_matches_YYYYMMDD_HHMMSS_PID.jsonl`: source comment, selected clip, score, winning passage text/timecodes, model and catalogue path. `VIDEO_MATCH_LOG_PATH` overrides it; `VIDEO_MATCH_LOG_ENABLED=0` disables it. It records a request, never confirmed playback.
- Keep prompts short and deterministic; avoid multi-turn chains.
- Guard against missing or slow responses to avoid blocking the loop.
- `LLM_NUM_CTX` defaults to 4096 and `LLM_NUM_PREDICT` to 2560, shared between hidden reasoning and the visible response.
- Commentary attempts are logged by default to one `llm_out/commentary_YYYYMMDD_HHMMSS_PID.jsonl` per process. Records include full game histories, the exact prompt, previous comments, model settings, temperature, context path/word count/SHA-256, timings and token counts; hidden reasoning text is not logged. The viewer only reads `.txt` video requests. `COMMENTARY_LOG_PATH` overrides the log file; `COMMENTARY_LOG_ENABLED=0` disables logging. `LLM_OUT_OVERRIDE` also moves the default log.
- Commentary remains two sentences, with `COMMENTARY_WORD_LIMIT` defaulting to 90 words in total.

## Subtitle and embedding tools
- `src/embed_vtt.py` builds the legacy full-clip catalogue. `--passages` builds `abriggs-itw-passages.json` from contiguous subtitle-cue groups targeting 70 words, sharing one cue and merging short tails; it reuses existing clip titles/durations, without a title-generation LLM call. `--passage-words`, `--num-gpu`, `--base-catalog`, and `--output` configure rebuilding. Preserve source transcripts; automatic cue boundaries may split a sentence.
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
