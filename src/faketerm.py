import hashlib
import json
import math
import os
import re
import sys
import time
import unicodedata
import subprocess

import ollama
import pexpect

import pygame

from c64renderer import C64Renderer
from knowledge_base import plundered_hearts_wiki, plundered_hearts_fandom

print("Waiting for Ollama")
time.sleep(5)

# os.environ["OLLAMA_NO_CUDA"] = "1"

AI_THINKING_STATUS = "<MISTRAL REFLECHIT>"
AI_COMMENT_LABEL = "MISTRAL COMMENTE:"
AI_VIDEO_LABEL = "MAGNETO, JOUE LA BANDE:"
AI_COMMENT_FG = (255, 255, 255)
AI_COMMENT_BG = (0, 0, 0)
LLM_MODEL = 'ministral-3:14b' # 'ministral-3:8b' # 'qwen2.5:7b' # 'ministral-3:14b'
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
KEY_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "assets", "audio")
GODOT_VIEWER_PATH = os.path.join(os.path.dirname(__file__), "..", "bin", "itw-viewer.exe")
RAW_OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "game-raw-output.json")
VIDEO_EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "abriggs-itw-embeddings.json")
VIDEO_EMBED_MODEL = "embeddinggemma:300m"
LLM_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "llm_out")

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
ansi_escape = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
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
        return subprocess.Popen([viewer_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        print(f"Unable to start Godot viewer: {exc}")
        return None


def _exit_immediately():
    try:
        if child is not None:
            child.terminate(force=True)
    except Exception:
        pass
    try:
        pygame.quit()
    except Exception:
        pass
    sys.exit(0)


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
        response = ollama.embeddings(model=VIDEO_EMBED_MODEL, prompt=text)
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
        score = _cosine_similarity(comment_vector, comment_norm, item["embedding"], item["norm"])
        scored.append((score, item))
    scored.sort(reverse=True)

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
    if duration_sec is None:
        return 0.0
    if duration_sec <= 0.0:
        return 0.0
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
    write_llm_video_request(next_video)
    record_video_choice(next_video, catalog_size, recent)
    last_played = next_video
    next_allowed = now + _get_video_duration(entry)
    return None, next_allowed, last_played


def write_llm_video_request(filename):
    if not filename:
        return
    os.makedirs(LLM_OUT_DIR, exist_ok=True)
    now = time.time()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
    millis = int((now - int(now)) * 1000)
    out_name = f"{stamp}_{millis:03d}.txt"
    out_path = os.path.join(LLM_OUT_DIR, out_name)
    try:
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(filename.strip() + "\n")
    except Exception as exc:
        print(f"Unable to write llm_out file: {exc}")


def load_itw_redux():
    base_dir = os.path.dirname(__file__)
    src_path = os.path.join(base_dir, "..", "assets", "abriggs-itw-750-words.txt")
    with open(src_path, "r", encoding="utf-8") as handle:
        return handle.read().strip()


def build_prompt(prev_output, cmd):
    prompt = """Plundered Hearts is a 1987 interactive fiction by Amy Briggs.
Here is an excerpt from Amy Briggs recalling her years at Infocom:
"""
    prompt += load_itw_redux()

    prompt += """
While reading the following moment from the game, your response may attend to
any aspect of Amy Briggs’s testimony that feels relevant in this moment:
a memory of daily work, the group dynamics at Infocom, informal mentoring practices, 
commercial pressures, the struggle to legitimize video games as a growth engine versus 
business software (such as Cornerstone), and her personal desire to write novels, 
or technical detail, whatever seems relevant...
Answer in TWO sentences, in neutral French, plain text, NO MARKDOWN, as a fleeting inner association.
"""
    prompt += prev_output + "\nYour next move will be : " + cmd
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

# run frotz through a terminal emulator, using the ascii mode
# child = pexpect.spawn("frotz -p roms/PLUNDERE.z3", encoding='utf-8', timeout=5)
from pexpect.popen_spawn import PopenSpawn

def _start_game_process():
    return PopenSpawn("frotz -p roms/PLUNDERE.z3", encoding='utf-8', timeout=5)

child = None

renderer = None
if ENABLE_C64_RENDERER:
    try:
        display_index = None
        if C64_DISPLAY_INDEX:
            try:
                display_index = max(0, int(C64_DISPLAY_INDEX) - 1)
            except (TypeError, ValueError):
                display_index = None
        renderer = C64Renderer(
            font_path=C64_FONT_PATH,
            fps=50,
            fullscreen=ENABLE_C64_FULLSCREEN,
            display_index=display_index,
            window_size=C64_WINDOW_SIZE,
            window_position=C64_WINDOW_POSITION,
            borderless=C64_WINDOW_UNDECORATED,
            output_scale=C64_OUTPUT_SCALE,
            fit_to_display=C64_FIT_TO_DISPLAY,
        )
    except Exception as exc:
        print(f"Unable to start C64 renderer: {exc}")
        renderer = None

_godot_viewer_process = _start_godot_viewer()

raw_output_map = load_raw_output(RAW_OUTPUT_PATH) if ENABLE_RAW_OUTPUT else {}
video_embeddings = load_video_embeddings(VIDEO_EMBEDDINGS_PATH) if ENABLE_LLM else []
recent_videos = []
last_video_played = None
pending_video_entry = None
next_allowed_video_time = 0.0

restart_message = (
    "Congratulations, you just finished pLLMdered_hearts.\n"
    "The installation will now restart."
)

# Unified loop for reading, displaying, and responding.
while True:  # for step, cmd in enumerate(plundered_hearts_commands):
    child = _start_game_process()
    prev_output = ""
    prev_outputs = []
    cmd_index = 0
    prev_cmd = None
    pending_intro_ack = True
    last_cleaned = ""
    LAST_STATUS_BAR = ""

    while True:
        _handle_quit_shortcut()

        raw_output = ""
        start_time = time.time()
        timeout_seconds = 4
        while time.time() - start_time < timeout_seconds:
            try:
                chunk = child.read_nonblocking(size=1024, timeout=0.3)
                if not chunk:
                    break
                raw_output += chunk
                if "***MORE***" in raw_output or  "[Press RETURN or ENTER to continue.]" in raw_output:
                    raw_output = raw_output.replace("***MORE***", "")
                    child.sendline("")
                    continue
                if ">" in raw_output:
                    break
            except pexpect.exceptions.TIMEOUT:
                break

        if not raw_output and pending_intro_ack:
            try:
                child.expect("Press RETURN or ENTER to begin", timeout=1)
                raw_output = (child.before or "") + (child.after or "")
            except pexpect.exceptions.TIMEOUT:
                pass

        if raw_output:
            cleaned = clean_output(raw_output)
            last_cleaned = cleaned
            if renderer and LAST_STATUS_BAR:
                renderer.set_status_bar(LAST_STATUS_BAR)
            if renderer:
                if cleaned:
                    type_to_renderer(
                        renderer,
                        cleaned + "\n",
                        base_delay=1 / 60.0,
                        min_delay=1 / 240.0,
                        max_delay=1 / 30.0,
                        beep=True,
                        word_mode=True,
                    )
                else:
                    renderer.render_frame()
            print(cleaned)
            if cleaned:
                prev_outputs.append(cleaned)
                if len(prev_outputs) > 3:
                    prev_outputs = prev_outputs[-3:]
                prev_output = "\n".join(prev_outputs)

        if pending_intro_ack and ("Press RETURN or ENTER to begin" in raw_output or not raw_output):
            child.sendline("")
            pending_intro_ack = False
            continue

        # Only proceed if the game shows a prompt and we still have commands to send.
        if ">" in raw_output and cmd_index < len(plundered_hearts_commands):
            cmd = enhance_game_command(plundered_hearts_commands[cmd_index]) # Sanitize game command (remove the game's shortcuts)

            if ENABLE_RAW_OUTPUT and prev_output:
                if update_raw_output(raw_output_map, prev_output + "\nYour next move will be : " + cmd):
                    write_raw_output(RAW_OUTPUT_PATH, raw_output_map)
                prev_output = ""

            if ENABLE_LLM:
                prompt = build_prompt(prev_output, cmd)
                llm_commentary = None
                retry = 0
                status_color = None
                status_text = None
                if renderer:
                    status_color = getattr(renderer, "status_bar_bg", None)
                    status_text = LAST_STATUS_BAR
                    if status_text:
                        renderer.set_status_bar(_status_with_ai_thinking(status_text))
                    renderer.set_status_bar_color((0, 0, 0))
                    renderer.render_frame()
                while llm_commentary is None:
                    response = ollama.chat(
                        model=LLM_MODEL,
                        messages=[{
                            'role': 'user',
                            'content': prompt
                            }]
                    )
                    llm_commentary = response.message.content
                    if retry > 0:
                        print("Retry #" + str(retry))
                    retry = retry + 1
                if renderer and status_color:
                    renderer.set_status_bar_color(status_color)
                    if status_text:
                        renderer.set_status_bar(status_text)
                    renderer.render_frame()
                next_video_entry = None
                if llm_commentary and video_embeddings:
                    comment_vector, comment_norm = embed_commentary_text(llm_commentary)
                    next_video_entry = select_best_video(
                        comment_vector,
                        comment_norm,
                        video_embeddings,
                        recent_videos,
                        last_video_played,
                    )
                ai_thinking = llm_commentary + "\n"
                print("<AI thinks : '" + ai_thinking + "'>\n")
                if renderer and llm_commentary:
                    cleaned_comment = sanitize_renderer_text(llm_commentary).strip()
                    if cleaned_comment:
                        display_comment = "\n> " + AI_COMMENT_LABEL + " " + cleaned_comment
                    else:
                        display_comment = "\n> " + AI_COMMENT_LABEL
                    type_to_renderer(
                        renderer,
                        display_comment,
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
                if next_video_entry:
                    pending_video_entry = next_video_entry
                pending_video_entry, next_allowed_video_time, last_video_played = maybe_emit_video_request(
                    renderer,
                    pending_video_entry,
                    recent_videos,
                    last_video_played,
                    next_allowed_video_time,
                    len(video_embeddings),
                )
            display_cmd = ">> " + cmd.strip()
            print(display_cmd + "\n")
            if renderer:
                type_to_renderer(renderer, "\n" + display_cmd + "\n", beep=True)
            child.sendline(" " + cmd)
            prev_cmd = cmd
            cmd_index += 1

        pending_video_entry, next_allowed_video_time, last_video_played = maybe_emit_video_request(
            renderer,
            pending_video_entry,
            recent_videos,
            last_video_played,
            next_allowed_video_time,
            len(video_embeddings),
        )

        if cmd_index >= len(plundered_hearts_commands):
            break

    print(restart_message)
    if renderer:
        type_to_renderer(
            renderer,
            "\n" + restart_message + "\n",
            base_delay=1 / 60.0,
            min_delay=1 / 240.0,
            max_delay=1 / 30.0,
            beep=True,
            word_mode=True,
        )
    try:
        child.terminate(force=True)
    except Exception:
        pass
