# Porting Study: `src/faketerm.py` to C + SDL2 + llama.cpp

## Purpose
Assess feasibility, options, and risks for rewriting `src/faketerm.py` in C, replacing pygame with SDL2 and Ollama with llama.cpp. This focuses on runtime behavior, feature parity, and the embedding pipeline.

## Current Python Runtime (High-Level)
1. Launches `frotz` with `pexpect` and plays a fixed walkthrough.
2. Cleans terminal output (ANSI removal, status bar handling), renders to a C64-style display, and beeps on typing.
3. Builds a short prompt, calls `ollama.chat` for 2-sentence commentary.
4. Embeds the commentary via `ollama.embeddings` and selects a matching interview video via cosine similarity.
5. Writes a timestamped filename into `llm_out/` for the Godot viewer.
6. Uses a cooldown based on `duration_sec` to avoid overlapping video playback.
7. Restarts the game loop continuously.

## Porting Possibilities
### Feasible Architecture in C
- **Process control**: Spawn `frotz.exe` and interact via pipes or a pseudo-terminal equivalent.
- **Renderer**: Re-implement the C64 grid renderer on SDL2 surfaces/textures.
- **Audio**: Use SDL2 audio or SDL_mixer for the keyclick and buzz sounds.
- **LLM**: Link llama.cpp as a library or use a local llama.cpp server HTTP API.
- **Embedding matching**: Implement cosine similarity and a JSON loader for the precomputed embedding catalog.

### SDL2 Replacement for pygame
- The current renderer uses pre-rendered 8x7 glyphs and a custom grid. SDL2 can reproduce this with:
  - An SDL texture atlas for glyphs.
  - A logical buffer of characters and per-row colors.
  - Scaling to window or fullscreen with a border.
- SDL2 provides the same event loop, timer, and audio primitives needed for typing delays and quit shortcuts.

### llama.cpp Replacement for Ollama
Two approaches:
1. **Link llama.cpp directly** (fastest runtime, no HTTP):
   - Use the llama.cpp C API for prompt inference and embeddings.
   - Requires careful threading to avoid blocking the game loop.
2. **Use llama.cpp server** (simpler integration, more moving parts):
   - HTTP calls for `/completion` and `/embeddings`.
   - Slightly higher latency, but easier to swap models without recompiling.

## Embeddings: Main Risk Area
### What exists today
- `assets/abriggs-itw-embeddings.json` was generated with Ollama using model `embeddinggemma:300m`.
- `src/faketerm.py` embeds each new comment with the *same* model and compares cosine similarity.

### Why this is risky in the C + llama.cpp port
- Embedding vectors must be produced by the *same model* and the *same tokenization* to remain comparable.
- llama.cpp does not guarantee compatibility with Ollama model identifiers; you would need a GGUF build of the same embedding model.
- If you switch models, all existing embeddings in `assets/abriggs-itw-embeddings.json` must be regenerated.

### Options to Mitigate
1. **Keep the exact embedding model**
   - Obtain a GGUF of the embedding model used by Ollama (if available).
   - Verify vector dimension and output consistency.
2. **Switch embedding model**
   - Regenerate `assets/abriggs-itw-embeddings.json` with the new model.
   - Expect different nearest-neighbor behavior (video choices will shift).
3. **Hybrid approach**
   - Use llama.cpp for chat, but call Ollama only for embeddings.
   - Keeps video matching consistent while migrating the main LLM.

### Concrete Checks to Perform
- Validate embedding vector length for both models.
- Compare cosine distributions on a sample set (same text embedded by both systems).
- Regenerate the embedding catalog if the model or dimension changes.

## Key Engineering Risks and Constraints
1. **Interactive process control on Windows**
   - `pexpect` handles non-blocking reads and TTY nuances. C will need a replacement.
   - Risk: `frotz` behavior may differ when not attached to a console.
   - Likely mitigation: use ConPTY on Windows or a dedicated read thread with `PeekNamedPipe`.

2. **Renderer feature parity**
   - The C64 renderer has several behaviors: status bar, cursor, border, scaling, and always-on-top on Windows.
   - SDL2 can do all of this, but you will need Windows-specific `SetWindowPos` for always-on-top.

3. **Audio assets and decoding**
   - pygame plays `.ogg` directly. SDL2 requires SDL_mixer or a decoder such as `stb_vorbis`.
   - Adding SDL_mixer increases dependency footprint.

4. **Blocking LLM calls**
   - Current Python loop blocks while waiting for Ollama. The C version should use a worker thread to avoid freezing the renderer or event handling.

5. **Text sanitization and encoding**
   - The Python code strips ANSI control sequences and normalizes to ASCII for rendering.
   - A C port should keep the exact cleaning logic to avoid losing game text or breaking the C64 layout.

6. **Godot viewer integration**
   - File naming and timestamp format must match the existing polling behavior in `godot-viewer/main.gd`.

## Suggested Porting Strategy
1. **Phase 1: Process + Renderer in C**
   - Port the C64 renderer to SDL2.
   - Spawn `frotz` and reproduce the terminal cleaning and typing effect.
   - Keep LLM disabled or mocked.

2. **Phase 2: LLM Chat with llama.cpp**
   - Add llama.cpp prompt inference.
   - Keep embeddings disabled or stubbed.

3. **Phase 3: Embedding Integration**
   - Decide on embedding model.
   - If model changes, regenerate the catalog and update documentation.

4. **Phase 4: Performance and Reliability**
   - Move LLM and embedding inference to worker threads.
   - Ensure restart loop, cooldown timing, and error handling match Python behavior.

## Practical Feasibility Summary
- **Rendering**: Straightforward. SDL2 maps well to the current custom C64 renderer.
- **Process control**: Feasible but tricky on Windows. Requires careful pipe or PTY handling.
- **LLM chat**: Feasible with llama.cpp; expect more explicit memory management and threading.
- **Embeddings**: The main uncertainty. Model compatibility and catalog regeneration are the key risks.

## Recommended Next Step
Before starting the port, validate the embedding model path:
1. Confirm whether a GGUF for the embedding model exists and produces embeddings in llama.cpp.
2. If not, choose a new embedding model and regenerate `assets/abriggs-itw-embeddings.json` with a consistent pipeline.

---

If you want, I can produce a concrete C module layout and a minimal SDL2 renderer skeleton next.
