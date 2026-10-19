import argparse
import hashlib
import json
import math
import os
import queue
import re
import shutil
import sys
import threading
import time
import unicodedata
import subprocess

import ollama
import pexpect
from pexpect.popen_spawn import PopenSpawn

import pygame

from c64renderer import C64Renderer
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma4:e4b")
LLM_DISPLAY_NAME = os.environ.get("LLM_DISPLAY_NAME", LLM_MODEL.split(":")[0].upper())
AI_THINKING_STATUS = f"<{LLM_DISPLAY_NAME} IS THINKING>"
AI_COMMENT_LABEL = f"{LLM_DISPLAY_NAME} COMMENTS:"
AI_VIDEO_LABEL = "NOW PLAYING:"
AI_COMMENT_FG = (255, 255, 255)
AI_COMMENT_BG = (0, 0, 0)
LLM_THINK = os.environ.get("LLM_THINK", "0" if LLM_MODEL.split(":")[0] == "gemma4" else None)
COMMENTARY_LANGUAGE = os.environ.get("COMMENTARY_LANGUAGE", "English")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "180"))
LLM_NUM_CTX = int(os.environ.get("LLM_NUM_CTX", "4096"))
LLM_NUM_PREDICT = int(os.environ.get("LLM_NUM_PREDICT", "256"))
EMBED_TIMEOUT = float(os.environ.get("EMBED_TIMEOUT", "30"))
VIDEO_END_MARGIN_SEC = float(os.environ.get("VIDEO_END_MARGIN_SEC", "3"))
COMMENTARY_MIN_WORDS = int(os.environ.get("COMMENTARY_MIN_WORDS", "500"))
COMMENTARY_LOG_PATH = os.environ.get("COMMENTARY_LOG_PATH")
GAME_TIMEOUT = 30.0
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
ENABLE_LLM = True
ENABLE_RAW_OUTPUT = False
ENABLE_C64_RENDERER = True
ENABLE_KEYCLICK_BEEP = True
ENABLE_GODOT_VIEWER = True
ENABLE_C64_FULLSCREEN = False
C64_DISPLAY_INDEX = 1  # 1-based display number (1, 2, 3); None uses the primary monitor.
C64_WINDOW_UNDECORATED = True
C64_WINDOW_SIZE = (1440, 1080)
C64_WINDOW_POSITION = (0, 0)
C64_OUTPUT_SCALE = 2
C64_FIT_TO_DISPLAY = True

C64_FONT_PATH = None  # Using built-in fallback font; no external sprite sheet required.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY_AUDIO_DIR = os.path.join(PROJECT_DIR, "assets", "audio")
GODOT_VIEWER_PATH = os.path.join(PROJECT_DIR, "bin", "itw-viewer.exe")
GAME_ROM_PATH = os.path.join(PROJECT_DIR, "roms", "PLUNDERE.z3")
RAW_OUTPUT_PATH = os.path.join(PROJECT_DIR, "assets", "game-raw-output.json")
VIDEO_EMBEDDINGS_PATH = os.path.join(PROJECT_DIR, "assets", "abriggs-itw-embeddings.json")
VIDEO_EMBED_MODEL = os.environ.get("VIDEO_EMBED_MODEL", "embeddinggemma:300m")
VIDEO_EMBED_NUM_GPU = int(os.environ.get("VIDEO_EMBED_NUM_GPU", "0"))
LLM_OUT_DIR = os.path.abspath(os.environ.get("LLM_OUT_OVERRIDE") or os.path.join(PROJECT_DIR, "llm_out"))
FAST_OUTPUT = False
child = None
renderer = None
_godot_viewer_process = None

if ENABLE_RAW_OUTPUT and ENABLE_LLM:
    raise ValueError("ENABLE_RAW_OUTPUT requires ENABLE_LLM to be False.")

def llm_response_is_valid(llm_commentary):
    if llm_commentary is None:
        return False

    return True

def extract_and_parse_json(text):
    """
    Extracts the first JSON object found inside triple backticks or code-like blocks from the input text,
    and returns it as a Python dictionary.
    """

    stop_strings = ["```json", "```", "\n"]
    for _stop in stop_strings:
        text = text.replace(_stop, '')

    # Try to find JSON inside ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        # If no fenced block, try raw { ... } block (useful for degraded format)
        match = re.search(r"(\{[\s\S]*?\})", text)

    if match:
        try:
            json_str = match.group(1)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print("JSON parsing failed:", e)
            return None
    else:
        print("No JSON block found.")
        return None

def sanitize_renderer_text(text):
    if text is None:
        return ""
    placeholder = "__RSQ__"
    text = text.replace("\u2019", "'")
    text = text.replace("\u2018", "'")
    text = text.replace("'", placeholder)
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_text.replace(placeholder, "'")

# Match ANSI escape sequences like ESC[31m or ESC[2J
ansi_escape = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')
terminal_title = re.compile(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)')
# Match cursor position commands like [24d (often without ESC, incomplete sequences)
cursor_directives = re.compile(r'\[\d{1,3}d')
# Match charset switch like ESC(A or ESC(B
charset_switch = re.compile(r'\x1b\([A-B]')
# Match title/score bar lines that contain score and moves.
status_bar_re = re.compile(r".*Score:\s*\d+.*Moves:\s*\d+", re.IGNORECASE)
LAST_STATUS_BAR = ""

# Escape sequences that usually correspond to "clear screen", "move cursor", etc.
line_breaking_escapes = re.compile(r'\x1b\[[0-9;]*([HJfABCD])')

def clean_output(text):
    global LAST_STATUS_BAR
    text = terminal_title.sub('', text)
    # Replace certain ANSI codes with newlines (those likely to imply line moves)
    def replace_with_newline(match):
        command = match.group(1)
        if command in ['H', 'f', 'A', 'B', 'C', 'D']:
            return '\n'
        return ''

    text = line_breaking_escapes.sub(replace_with_newline, text)
    # Remove charset switch sequences
    text = charset_switch.sub('', text)
    # Remove cursor directives like [24d
    text = cursor_directives.sub('', text)
    # Remove remaining ANSI escape codes
    text = ansi_escape.sub('', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Remove status/title bar and lines containing only a number (like score or parser noise)
    lines = text.strip().splitlines()
    filtered = []
    for line in lines:
        stripped = line.strip()
        if status_bar_re.search(stripped):
            LAST_STATUS_BAR = stripped
            if "Plundered Hearts" in LAST_STATUS_BAR:
                LAST_STATUS_BAR = LAST_STATUS_BAR.replace("Plundered Hearts", "PLLMDERED_HEARTS")
            continue
        if stripped.isdigit():
            continue
        filtered.append(line)
    # Remove repeated empty lines
    clean_lines = []
    for i, line in enumerate(filtered):
        if line.strip() == '' and (i == 0 or filtered[i - 1].strip() == ''):
            continue
        clean_lines.append(line)
    return '\n'.join(clean_lines).strip()

_KEY_SOUNDS = []
_BUZZ_SOUNDS = []
_MIXER_READY = False


def _ensure_key_sounds_loaded():
    """Lazy-load key click sounds from assets/audio/*.ogg."""
    global _KEY_SOUNDS, _BUZZ_SOUNDS, _MIXER_READY
    if _MIXER_READY:
        return
    if not pygame:
        return
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=256)
        if not os.path.isdir(KEY_AUDIO_DIR):
            return
        oggs = [f for f in os.listdir(KEY_AUDIO_DIR) if f.lower().endswith(".ogg")]
        for fname in sorted(oggs):
            try:
                sound = pygame.mixer.Sound(os.path.join(KEY_AUDIO_DIR, fname))
            except Exception:
                continue
            lower = fname.lower()
            if lower.startswith("key_"):
                _KEY_SOUNDS.append(sound)
            elif lower.startswith("buzz_"):
                _BUZZ_SOUNDS.append(sound)
        _MIXER_READY = True
    except Exception:
        _MIXER_READY = False


def _play_key_beep(ch=" "):
    """Play a key click sound mapped from ASCII; fallback to terminal bell."""
    if not ENABLE_KEYCLICK_BEEP:
        return
    _ensure_key_sounds_loaded()
    if _KEY_SOUNDS:
        try:
            idx = ord(ch) % len(_KEY_SOUNDS)
            _KEY_SOUNDS[idx].play()
            return
        except Exception:
            pass
    # Fallback: terminal bell
    try:
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass


def _play_buzz_beep(token=" "):
    """Play a buzz sound mapped from ASCII; fallback to terminal bell."""
    if not ENABLE_KEYCLICK_BEEP:
        return
    _ensure_key_sounds_loaded()
    if _BUZZ_SOUNDS:
        try:
            idx = ord(token[0]) % len(_BUZZ_SOUNDS) if token else 0
            _BUZZ_SOUNDS[idx].play()
            return
        except Exception:
            pass
    try:
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass


def _start_godot_viewer():
    if not ENABLE_GODOT_VIEWER:
        return None
    viewer_path = os.path.abspath(GODOT_VIEWER_PATH)
    if not os.path.isfile(viewer_path):
        print(f"Godot viewer not found at {viewer_path}")
        return None
    try:
        viewer_env = os.environ.copy()
        viewer_env["LLM_OUT_OVERRIDE"] = LLM_OUT_DIR
        return subprocess.Popen(
            [viewer_path], cwd=os.path.dirname(viewer_path), env=viewer_env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"Unable to start Godot viewer: {exc}")
        return None


def _exit_immediately():
    raise KeyboardInterrupt


def _stop_process(process):
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream:
            stream.close()


def _handle_quit_shortcut():
    if not renderer or pygame is None:
        return
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            _exit_immediately()
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            _exit_immediately()

def _status_with_ai_thinking(text):
    if not text:
        return text
    match = re.search(r"\bScore:", text, re.IGNORECASE)
    if not match:
        return f"{text} " + AI_THINKING_STATUS
    idx = match.start()
    prefix = text[:idx].rstrip()
    suffix = text[idx:]
    new_prefix = f"{prefix} " + AI_THINKING_STATUS
    if len(new_prefix) < idx:
        new_prefix = new_prefix.ljust(idx)
    elif len(new_prefix) > idx:
        new_prefix = new_prefix[:idx]
    return f"{new_prefix}{suffix}"


def type_to_renderer(
    renderer,
    text,
    base_delay=0.015,
    min_delay=0.075,
    max_delay=0.20,
    beep=True,
    word_mode=False,
    fg_color=None,
    bg_color=None,
):
    """
    Simulate typing to the renderer: emit characters one by one with a delay
    proportional to ASCII distance from the previous character.
    """
    if not renderer or text is None:
        return
    if FAST_OUTPUT:
        _handle_quit_shortcut()
        renderer.write(text, fg_color=fg_color, bg_color=bg_color)
        renderer.render_frame()
        return
    def _sleep_with_events(delay):
        if delay <= 0:
            return
        end_time = time.time() + delay
        while time.time() < end_time:
            _handle_quit_shortcut()
            remaining = end_time - time.time()
            if remaining <= 0:
                break
            time.sleep(min(0.01, max(0.0, remaining)))
    prev = " "
    chunks = re.findall(r"\n|\S+\s*|\s+", text) if word_mode else list(text)
    # Show cursor ahead of each chunk, and keep it after each chunk except the final one.
    total_chunks = len([c for c in chunks if c])
    typed = 0
    for chunk in chunks:
        if not chunk:
            continue
        _handle_quit_shortcut()
        if renderer:
            renderer.render_frame(show_cursor=True)
        renderer.write(chunk, fg_color=fg_color, bg_color=bg_color)
        typed += 1
        renderer.render_frame(show_cursor=typed < total_chunks)
        # Use the last non-newline character of the chunk to keep delays consistent.
        ch = next((c for c in reversed(chunk) if c not in "\r\n"), " ")
        if not ch.strip():
            ch = " "
        distance = abs(ord(ch) - ord(prev))
        delay = max(min_delay, min(max_delay, distance * base_delay))
        if beep and chunk not in ["\n", " ", ">"]:
            if word_mode:
                if  len(chunk) > 6:
                    _play_buzz_beep(chunk)
            else:
                _play_key_beep(ch)
        if chunk not in ["\n", ">"]:
            _sleep_with_events(delay)
        prev = ch


def _sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_raw_output(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def write_raw_output(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True)


def update_raw_output(data, cleaned_text):
    if not cleaned_text:
        return False
    cleaned_text = cleaned_text.strip()
    if not cleaned_text:
        return False
    digest = _sha_text(cleaned_text)
    if digest in data:
        return False
    data[digest] = cleaned_text
    return True


def _vector_norm(vector):
    return math.sqrt(sum(value * value for value in vector))


def _cosine_similarity(a_vec, a_norm, b_vec, b_norm):
    if a_norm <= 0.0 or b_norm <= 0.0:
        return -1.0
    if len(a_vec) != len(b_vec):
        return -1.0
    dot = 0.0
    for i in range(len(a_vec)):
        dot += a_vec[i] * b_vec[i]
    return dot / (a_norm * b_norm)


def load_video_embeddings(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            return []
    except Exception:
        return []

    entries = []
    for item in data:
        if not isinstance(item, dict):
            continue
        filename = item.get("filename")
        embedding = item.get("embedding")
        sequence_title = item.get("sequence_title")
        duration_raw = item.get("duration_sec")
        duration_sec = None
        if duration_raw is not None:
            try:
                duration_sec = float(duration_raw)
            except (TypeError, ValueError):
                duration_sec = None
        if not filename or not isinstance(embedding, list):
            continue
        vector = [float(value) for value in embedding]
        norm = _vector_norm(vector)
        if norm <= 0.0:
            continue
        entries.append(
            {
                "filename": filename,
                "sequence_title": sequence_title,
                "duration_sec": duration_sec,
                "embedding": vector,
                "norm": norm,
            }
        )
    return entries


def embed_commentary_text(text):
    text = (text or "").strip()
    if not text:
        return None, 0.0
    try:
        response = _ollama_request(
            "embeddings", EMBED_TIMEOUT, model=VIDEO_EMBED_MODEL, prompt=text,
            options={"num_gpu": VIDEO_EMBED_NUM_GPU},
        )
    except Exception as exc:
        print(f"Embedding failed: {exc}")
        return None, 0.0
    vector = response.get("embedding")
    if not isinstance(vector, list):
        return None, 0.0
    vector = [float(value) for value in vector]
    return vector, _vector_norm(vector)


def select_best_video(comment_vector, comment_norm, catalog, recent, last_video):
    if not catalog or comment_vector is None:
        return None
    scored = []
    for item in catalog:
        if len(comment_vector) != len(item["embedding"]):
            continue
        score = _cosine_similarity(comment_vector, comment_norm, item["embedding"], item["norm"])
        scored.append((score, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    for _, item in scored:
        var_filename = item["filename"]
        if var_filename == last_video:
            continue
        if var_filename in recent:
            continue
        return item

    if recent:
        recent.clear()
        for _, item in scored:
            var_filename = item["filename"]
            if var_filename == last_video:
                continue
            return item

    return scored[0][1] if scored else None


def record_video_choice(filename, catalog_size, recent):
    if not filename:
        return
    if filename not in recent:
        recent.append(filename)
    if catalog_size and len(recent) >= catalog_size:
        recent.clear()


def _get_video_duration(entry):
    duration_sec = entry.get("duration_sec")
    if duration_sec is None or not math.isfinite(duration_sec) or duration_sec <= 0.0:
        return 200.0
    return duration_sec


def maybe_emit_video_request(renderer, entry, recent, last_played, next_allowed, catalog_size):
    if not entry:
        return None, next_allowed, last_played
    now = time.monotonic()
    if now < next_allowed:
        return entry, next_allowed, last_played
    next_video = entry["filename"]
    sequence_title = entry.get("sequence_title") or next_video
    if renderer and sequence_title:
        cleaned_title = sanitize_renderer_text(sequence_title).strip()
        if cleaned_title:
            display_title = "\n> " + AI_VIDEO_LABEL + " " + cleaned_title
        else:
            display_title = "\n> " + AI_VIDEO_LABEL
        type_to_renderer(
            renderer,
            display_title,
            base_delay=1 / 60.0,
            min_delay=1 / 240.0,
            max_delay=1 / 30.0,
            beep=False,
            word_mode=True,
            fg_color=AI_COMMENT_FG,
            bg_color=AI_COMMENT_BG,
        )
        type_to_renderer(
            renderer,
            "\n",
            base_delay=1 / 60.0,
            min_delay=1 / 240.0,
            max_delay=1 / 30.0,
            beep=False,
            word_mode=True,
        )
    if not write_llm_video_request(next_video):
        return entry, next_allowed, last_played
    record_video_choice(next_video, catalog_size, recent)
    last_played = next_video
    next_allowed = time.monotonic() + _get_video_duration(entry) + VIDEO_END_MARGIN_SEC
    return None, next_allowed, last_played


def write_llm_video_request(filename):
    if not filename:
        return False
    os.makedirs(LLM_OUT_DIR, exist_ok=True)
    now = time.time()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    millis = int((now - int(now)) * 1000)
    out_name = f"{stamp}_{millis:03d}.txt"
    out_path = os.path.join(LLM_OUT_DIR, out_name)
    temp_path = out_path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            handle.write(filename.strip() + "\n")
        os.replace(temp_path, out_path)
        return True
    except Exception as exc:
        print(f"Unable to write llm_out file: {exc}")
        if os.path.isfile(temp_path):
            os.remove(temp_path)
        return False


def load_itw_redux():
    base_dir = os.path.dirname(__file__)
    src_path = os.path.join(base_dir, "..", "assets", "abriggs-itw-750-words.txt")
    with open(src_path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def build_prompt(prev_output, cmd, interview_text=None):
    prompt = """You are a quiet observer of an installation replaying Plundered Hearts,
a 1987 interactive fiction by Amy Briggs. The game follows a fixed walkthrough.
Here is an excerpt from Amy Briggs recalling her years at Infocom:
"""
    prompt += load_itw_redux() if interview_text is None else interview_text

    prompt += f"""
Read the chronological game history accumulated since your previous comment.
Choose one concrete event or change, preferably from the latest developments.
Make a modest, specific association with Briggs's testimony only when it fits.
Otherwise, stay with the game. Her testimony is context, not a checklist of themes.
Do not invent biographical facts or causes for design choices.
You are neither Amy Briggs nor a game character, and you do not choose the moves.
Answer in TWO short sentences, 60 words maximum in total, in neutral {COMMENTARY_LANGUAGE}.
Use plain text only, NO MARKDOWN, as a fleeting inner association.
"""
    prompt += "\nGAME HISTORY (oldest first):\n" + prev_output
    prompt += "\nEND OF GAME HISTORY\nNext scripted command (not yet executed): " + cmd
    return prompt



# Official Amiga solution
plundered_hearts_commands = [
    "stand up", "inventory", "examine smelling salts", "read tag", "examine banknote",
    "examine coffer", "examine door", "open door", "z", "scream", "e", "examine falcon",
    "read missive", "falcon, yes", "examine davis", "examine ring", "examine falcon", "z", "z",
    "stand up", "look around", "look through window", "open curtain", "examine cupboard",
    "examine table", "examine bed", "z", "examine brooch", "open coffer", "take invitation",
    "read invitation", "n", "down", "n", "examine gate", "n", "take bottle", "take mirror",
    "examine bottle", "read label", "s", "s", "up", "open door", "enter", "take clothes",
    "remove dress", "wear breeches", "wear shirt", "z", "out", "s", "take coffer", "z",
    "throw coffer through window", "sit on ledge", "put all in reticule", "take ladder",
    "up", "up", "up", "up", "n", "n", "n", "examine winch", "read lever", "pull lever up",
    "s", "examine barrels", "tear dress", "put rag in water", "open hatch", "down",
    "throw rag over gate", "up", "s", "examine casks", "n", "n", "enter shack", "take dagger",
    "e", "s", "s", "look in cask", "enter cask", "take pork", "put all in reticule except dagger",
    "cut rope", "examine pork", "z", "z", "z", "leave cask", "examine skiff", "w", "n", "e",
    "pull slat", "z", "z", "falcon, yes", "z", "e", "n", "open window", "w", "examine portrait",
    "examine bookshelves", "examine globe", "take hat", "s", "w", "e", "examine lucy",
    "take garter", "w", "s", "ne", "climb vine", "remove clothes", "wear ball gown",
    "take invitation", "put garter in reticule", "n", "n", "take pistols", "s", "e", "e",
    "open door", "s", "n", "w", "down", "give invitation", "s", "dance", "dance", "dance",
    "dance", "w", "examine orchestra", "e", "dance", "examine lafond", "examine ring",
    "open door", "s", "n", "n", "e", "s", "e", "n", "n", "take treatise", "take hat",
    "press sinistra on globe", "n", "down", "e", "e", "take key and horn", "w", "s",
    "open door", "e", "w", "n", "w", "s", "squeeze bottle on pork", "throw pork at crocodile",
    "z", "z", "s", "w", "unlock door", "open door", "n", "give garter to papa", "z", "s",
    "e", "n", "n", "up", "s", "s", "n", "n", "s", "s", "s", "n", "n", "s", "up", "e",
    "knock on door", "open door", "n", "drink wine", "pour wine into green goblet",
    "squeeze bottle into green goblet", "pour wine into blue goblet", "z", "lafond, no",
    "drink wine", "take spice", "throw spice at lafond", "wave mirror at window", "s",
    "w", "down", "s", "z", "cookie, yes", "n", "e", "n", "take treatise", "take hat",
    "press sinistra", "n", "down", "s", "s", "take rapier", "kill crulley", "g",
    "close trapdoor", "pick lock", "wave smelling salts", "n", "n", "z", "up", "s",
    "s", "w", "up", "e", "s", "untie rope", "climb down rope", "take horn", "s", "s", "s",
    "look at nicholas", "push nicholas", "nicholas, yes", "take pistol", "load pistol",
    "fire pistol at crulley"
]

cmd_replace_dict = {
    "E": "EAST",
    "W": "WEST",
    "N": "NORTH",
    "S": "SOUTH",
    "Z": "WAIT",
    "NE": "NORTHEAST",
    "NW": "NORTHWEST",
    "SE": "SOUTHEAST",
    "SW": "SOUTHWEST",
    "U": "UP",
    "D": "DOWN",
    "L": "LOOK"
}

def enhance_game_command(cmd):
    cmd = cmd.upper()
    if cmd in cmd_replace_dict:
        return cmd_replace_dict[cmd]
    return cmd

def _game_executable():
    configured = os.environ.get("FROTZ_PATH")
    if configured:
        executable = shutil.which(configured)
        if executable:
            return executable
        raise FileNotFoundError(f"FROTZ_PATH not found: {configured}")
    local_path = os.path.join(PROJECT_DIR, "frotz.exe" if os.name == "nt" else "frotz")
    executable = local_path if os.path.isfile(local_path) else shutil.which("frotz")
    if not executable:
        raise FileNotFoundError("Frotz not found. Put frotz.exe in the project or set FROTZ_PATH.")
    return executable


def _start_game_process():
    return PopenSpawn(
        [_game_executable(), "-p", GAME_ROM_PATH], cwd=PROJECT_DIR,
        encoding="utf-8", codec_errors="replace", timeout=GAME_TIMEOUT,
    )


def read_game_output(game):
    """Wait for a complete prompt, acknowledging introduction and pagination."""
    output = ""
    deadline = time.monotonic() + GAME_TIMEOUT
    markers = ("***MORE***", "[Press RETURN or ENTER to continue.]",
               "[Press RETURN or ENTER to begin.]")
    while time.monotonic() < deadline:
        _handle_quit_shortcut()
        try:
            chunk = game.read_nonblocking(size=4096, timeout=0.05)
        except pexpect.TIMEOUT:
            chunk = ""
        except pexpect.EOF:
            if output:
                return output
            raise RuntimeError("Frotz stopped before the next prompt.") from None
        if not chunk:
            time.sleep(0.01)
            continue
        output += chunk
        for marker in markers:
            if marker in output:
                output = output.replace(marker, "")
                game.sendline("")
        if re.search(r">[ \t]*$", output):
            return output
    raise TimeoutError("Frotz did not reach a prompt within %.0f seconds." % GAME_TIMEOUT)


def _ollama_request(method, timeout, **kwargs):
    """Keep the display responsive while a bounded local request runs."""
    result = queue.Queue()

    def request():
        try:
            client = ollama.Client(host=OLLAMA_HOST, timeout=timeout)
            result.put((getattr(client, method)(**kwargs), None))
        except Exception as exc:
            result.put((None, exc))

    threading.Thread(target=request, daemon=True).start()
    deadline = time.monotonic() + timeout
    while True:
        _handle_quit_shortcut()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Ollama {method} exceeded {timeout:g} seconds.")
        try:
            response, error = result.get(timeout=min(0.05, remaining))
        except queue.Empty:
            continue
        if error:
            raise error
        return response


def generate_commentary(prompt, metrics=None):
    start = time.monotonic()
    try:
        thinking = {}
        if LLM_THINK is not None:
            thinking["think"] = LLM_THINK.strip().lower() in ("1", "true", "yes")
        response = _ollama_request(
            "chat", LLM_TIMEOUT, model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": LLM_NUM_PREDICT, "num_ctx": LLM_NUM_CTX},
            **thinking,
        )
        if metrics is not None:
            metrics.update({
                "elapsed_sec": round(time.monotonic() - start, 3),
                "prompt_tokens": getattr(response, "prompt_eval_count", None),
                "output_tokens": getattr(response, "eval_count", None),
                "done_reason": getattr(response, "done_reason", None),
                "thinking_characters": len(getattr(response.message, "thinking", None) or ""),
            })
        return clean_output(response.message.content or "").strip()
    except Exception as exc:
        if metrics is not None:
            metrics["error"] = str(exc)
        print(f"Commentary unavailable, continuing the game: {exc}")
        return ""


def check_runtime():
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"Frotz: {_game_executable()}")
    if not os.path.isfile(GAME_ROM_PATH):
        raise FileNotFoundError(f"Game ROM missing: {GAME_ROM_PATH}")
    if ENABLE_GODOT_VIEWER:
        if not os.path.isfile(GODOT_VIEWER_PATH):
            raise FileNotFoundError(f"Godot viewer missing: {GODOT_VIEWER_PATH}")
        if not os.path.isfile(os.path.splitext(GODOT_VIEWER_PATH)[0] + ".pck"):
            raise FileNotFoundError("Godot viewer .pck missing from bin/.")
    if ENABLE_LLM:
        if COMMENTARY_MIN_WORDS < 0 or LLM_NUM_CTX < 1 or LLM_NUM_PREDICT < 1:
            raise ValueError("Commentary threshold must be nonnegative and token limits must be positive.")
        if not math.isfinite(VIDEO_END_MARGIN_SEC) or VIDEO_END_MARGIN_SEC < 0:
            raise ValueError("VIDEO_END_MARGIN_SEC must be a finite, nonnegative duration.")
        load_itw_redux()
        catalog = load_video_embeddings(VIDEO_EMBEDDINGS_PATH)
        if not catalog:
            raise RuntimeError(f"Video embeddings missing or empty: {VIDEO_EMBEDDINGS_PATH}")
        response = _ollama_request("list", 5.0)
        available = {model.model for model in response.models}
        for model in (LLM_MODEL, VIDEO_EMBED_MODEL):
            if model not in available and model + ":latest" not in available:
                raise RuntimeError(f"Ollama model missing. Run: ollama pull {model}")
        print(f"Ollama: {OLLAMA_HOST}; {LLM_MODEL}; {VIDEO_EMBED_MODEL}")
        print(f"Video catalog: {len(catalog)} clips")
        print(f"Commentary: over {COMMENTARY_MIN_WORDS} words at a game prompt; "
              f"context {LLM_NUM_CTX}, output {LLM_NUM_PREDICT}; video margin {VIDEO_END_MARGIN_SEC:g}s")
    os.makedirs(LLM_OUT_DIR, exist_ok=True)
    print(f"Video requests: {LLM_OUT_DIR}")


def create_renderer():
    if not ENABLE_C64_RENDERER:
        return None
    display_index = max(0, int(C64_DISPLAY_INDEX) - 1) if C64_DISPLAY_INDEX else None
    return C64Renderer(
        font_path=C64_FONT_PATH, fps=50, fullscreen=ENABLE_C64_FULLSCREEN,
        display_index=display_index, window_size=C64_WINDOW_SIZE,
        window_position=C64_WINDOW_POSITION, borderless=C64_WINDOW_UNDECORATED,
        output_scale=C64_OUTPUT_SCALE, fit_to_display=C64_FIT_TO_DISPLAY,
    )


def display_game_output(raw_output):
    cleaned = clean_output(raw_output)
    print(cleaned)
    if renderer:
        if LAST_STATUS_BAR:
            renderer.set_status_bar(LAST_STATUS_BAR)
        type_to_renderer(
            renderer, cleaned + "\n", base_delay=1 / 60.0,
            min_delay=1 / 240.0, max_delay=1 / 30.0, word_mode=True,
        )
    return cleaned


def comment_on_game(previous_output, command):
    prompt = build_prompt(previous_output, command)
    metrics = {"history_words": len(previous_output.split()), "history_characters": len(previous_output),
               "history_commands": len(re.findall(r"^>> ", previous_output, re.M)),
               "context_tokens": LLM_NUM_CTX, "model": LLM_MODEL}
    status_color = getattr(renderer, "status_bar_bg", None)
    status_text = LAST_STATUS_BAR
    if renderer:
        renderer.set_status_bar(_status_with_ai_thinking(status_text))
        renderer.set_status_bar_color((0, 0, 0))
        renderer.render_frame()
    try:
        commentary = generate_commentary(prompt, metrics)
    finally:
        if renderer:
            renderer.set_status_bar_color(status_color)
            renderer.set_status_bar(status_text)
            renderer.render_frame()
    print("[Commentary buffer] %s words, %s commands; prompt %s/%s tokens; %s s" % (
        metrics["history_words"], metrics["history_commands"], metrics.get("prompt_tokens", "?"),
        LLM_NUM_CTX, metrics.get("elapsed_sec", "?")))
    if COMMENTARY_LOG_PATH:
        try:
            path = os.path.abspath(COMMENTARY_LOG_PATH)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            record = dict(metrics, timestamp=time.time(), game_history=previous_output,
                          next_command=command, commentary=commentary, prompt=prompt)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            print("Unable to log commentary metrics: %s" % exc)
    if commentary:
        print(f"<AI thinks: {commentary}>\n")
        type_to_renderer(
            renderer, "\n> " + AI_COMMENT_LABEL + " " + sanitize_renderer_text(commentary) + "\n",
            base_delay=1 / 60.0, min_delay=1 / 240.0, max_delay=1 / 30.0,
            beep=False, word_mode=True, fg_color=AI_COMMENT_FG, bg_color=AI_COMMENT_BG,
        )
    return commentary


def commentary_is_due(history_words):
    return history_words > COMMENTARY_MIN_WORDS


def play(steps=None, once=False):
    global child, LAST_STATUS_BAR
    catalog = load_video_embeddings(VIDEO_EMBEDDINGS_PATH) if ENABLE_LLM else []
    raw_outputs = load_raw_output(RAW_OUTPUT_PATH) if ENABLE_RAW_OUTPUT else {}
    recent = []
    last_video = None
    pending_video = None
    pending_commentary = None
    next_allowed = 0.0
    commands_sent = 0
    commentary_history = []
    while True:
        child = _start_game_process()
        previous_outputs = []
        LAST_STATUS_BAR = ""
        try:
            for command in plundered_hearts_commands:
                cleaned = display_game_output(read_game_output(child))
                if cleaned:
                    previous_outputs = (previous_outputs + [cleaned])[-3:]
                    if ENABLE_LLM:
                        commentary_history.append(cleaned)
                previous_output = "\n".join(previous_outputs)
                command = enhance_game_command(command)
                if ENABLE_RAW_OUTPUT and previous_output:
                    if update_raw_output(raw_outputs, previous_output + "\nYour next move will be : " + command):
                        write_raw_output(RAW_OUTPUT_PATH, raw_outputs)
                history_words = sum(len(text.split()) for text in commentary_history)
                if ENABLE_LLM and commentary_is_due(history_words):
                    commentary = comment_on_game("\n".join(commentary_history), command)
                    if commentary:
                        commentary_history.clear()
                        pending_commentary = commentary
                        pending_video = None
                if (ENABLE_LLM and pending_commentary and pending_video is None
                        and time.monotonic() >= next_allowed):
                    vector, norm = embed_commentary_text(pending_commentary)
                    pending_video = select_best_video(vector, norm, catalog, recent, last_video)
                had_pending_video = pending_video is not None
                pending_video, next_allowed, last_video = maybe_emit_video_request(
                    renderer, pending_video, recent, last_video, next_allowed, len(catalog),
                )
                if had_pending_video and pending_video is None:
                    pending_commentary = None
                print(">> " + command + "\n")
                type_to_renderer(renderer, "\n>> " + command + "\n")
                if ENABLE_LLM:
                    commentary_history.append(">> " + command)
                child.sendline(command)
                commands_sent += 1
                if steps is not None and commands_sent >= steps:
                    display_game_output(read_game_output(child))
                    return
            # Render the response to the final command before restarting.
            final_output = display_game_output(read_game_output(child))
            if ENABLE_LLM and final_output:
                commentary_history.append(final_output)
        finally:
            _stop_process(child.proc)
            child = None
        restart_message = ("Congratulations, you just finished pLLMdered_hearts.\n"
                           "The installation will now restart.")
        if once:
            return
        print(restart_message)
        type_to_renderer(renderer, "\n" + restart_message + "\n", word_mode=True)
        if ENABLE_LLM:
            commentary_history.append("[The game has restarted from the beginning.]")


def main(argv=None):
    global ENABLE_LLM, ENABLE_C64_RENDERER, ENABLE_GODOT_VIEWER, FAST_OUTPUT
    global renderer, _godot_viewer_process
    parser = argparse.ArgumentParser(description="Run the pLLMdered Hearts installation.")
    parser.add_argument("--check", action="store_true", help="Check dependencies without opening windows")
    parser.add_argument("--no-llm", action="store_true", help="Play without AI commentary")
    parser.add_argument("--no-renderer", action="store_true", help="Use console output only")
    parser.add_argument("--no-viewer", action="store_true", help="Do not start the video viewer")
    parser.add_argument("--fast", action="store_true", help="Display text without typing delays")
    parser.add_argument("--steps", type=int, help="Exit after this many commands, for diagnostics")
    parser.add_argument("--once", action="store_true", help="Exit after one complete walkthrough")
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    ENABLE_LLM = ENABLE_LLM and not args.no_llm
    ENABLE_C64_RENDERER = ENABLE_C64_RENDERER and not args.no_renderer
    ENABLE_GODOT_VIEWER = ENABLE_GODOT_VIEWER and not args.no_viewer
    FAST_OUTPUT = args.fast
    try:
        check_runtime()
        if args.check:
            print("Runtime ready.")
            return 0
        renderer = create_renderer()
        _godot_viewer_process = _start_godot_viewer()
        play(steps=args.steps, once=args.once)
        return 0
    except KeyboardInterrupt:
        print("Installation stopped.")
        return 0
    except Exception as exc:
        print(f"Unable to run the installation: {exc}", file=sys.stderr)
        return 1
    finally:
        if child is not None:
            _stop_process(child.proc)
        _stop_process(_godot_viewer_process)
        pygame.quit()


if __name__ == "__main__":
    sys.exit(main())
