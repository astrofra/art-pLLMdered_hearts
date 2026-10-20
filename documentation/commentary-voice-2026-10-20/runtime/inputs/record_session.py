"""Record the actual installation without changing its scheduling or model settings.

Uses only the standard library plus the installation's existing dependencies.
Arguments after -- are passed to faketerm, e.g. -- --once --no-renderer --no-viewer --fast.
"""

import argparse
import contextlib
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from urllib.request import urlopen, Request

ROOT = Path(__file__).resolve().parents[2]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def json_response(response):
    data = dict(response.model_dump()) if hasattr(response, "model_dump") else dict(response)
    message = data.get("message")
    if isinstance(message, dict):
        message = dict(message)
        thinking = message.pop("thinking", None)
        message["thinking_characters"] = len(thinking or "")
        data["message"] = message
    return data


class Recorder:
    def __init__(self, runtime, directory):
        self.runtime = runtime
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.lock = threading.Lock()
        self.originals = {}
        self.command_count = 0
        self.comment_count = 0
        self.current_comment = None
        self.pending_comment = None
        self.estimated_video_end = 0.0
        self.stop = threading.Event()
        self.handle = (self.directory / "events.jsonl").open("w", encoding="utf-8")

    def event(self, kind, **data):
        with self.lock:
            row = {"event": kind, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "elapsed_sec": round(time.monotonic() - self.started, 6),
                   "commands_sent": self.command_count, **data}
            self.handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            self.handle.flush()

    def wrap(self, name, replacement):
        self.originals[name] = getattr(self.runtime, name)
        setattr(self.runtime, name, replacement)

    def install(self):
        runtime = self.runtime
        recorder = self

        def start_game():
            game = recorder.originals["_start_game_process"]()
            recorder.event("game_start")
            original_sendline = game.sendline

            def sendline(command):
                result = original_sendline(command)
                if command.strip():
                    recorder.command_count += 1
                    recorder.event("command", command=command)
                else:
                    recorder.event("pager_ack", command=command)
                return result

            game.sendline = sendline
            return game

        def read_game(game):
            started = time.monotonic()
            result = recorder.originals["read_game_output"](game)
            recorder.event("game_output", text=runtime.clean_output(result),
                           read_sec=time.monotonic() - started)
            return result

        def display_game(raw):
            started = time.monotonic()
            result = recorder.originals["display_game_output"](raw)
            recorder.event("game_display_complete", text=result,
                           render_sec=time.monotonic() - started)
            return result

        def comment(history, recent_comments=None):
            recorder.comment_count += 1
            recorder.current_comment = recorder.comment_count
            recorder.event("comment_start", comment_id=recorder.current_comment,
                           game_history=history, history_words=len(history.split()),
                           recent_comments=list(recent_comments or ()),
                           estimated_video_remaining_sec=max(0, recorder.estimated_video_end - time.monotonic()))
            started = time.monotonic()
            result = recorder.originals["comment_on_game"](history, recent_comments=recent_comments)
            recorder.event("comment_display_complete", comment_id=recorder.current_comment,
                           commentary=result, elapsed_with_display_sec=time.monotonic() - started)
            if result:
                if recorder.pending_comment is not None:
                    recorder.event("pending_comment_replaced", previous_comment_id=recorder.pending_comment,
                                   new_comment_id=recorder.current_comment)
                recorder.pending_comment = recorder.current_comment
            recorder.current_comment = None
            return result

        def request(method, timeout, **kwargs):
            started = time.monotonic()
            recorder.event("ollama_start", method=method, timeout_sec=timeout,
                           comment_id=recorder.current_comment, request=kwargs)
            try:
                response = recorder.originals["_ollama_request"](method, timeout, **kwargs)
            except Exception as exc:
                recorder.event("ollama_error", method=method, error=str(exc),
                               elapsed_request_sec=time.monotonic() - started)
                raise
            recorder.event("ollama_response", method=method, comment_id=recorder.current_comment,
                           elapsed_request_sec=time.monotonic() - started, response=json_response(response))
            return response

        def select(vector, norm, catalog, recent, last):
            previous_recent = list(recent)
            ranked = []
            if vector is not None:
                rank_matches = getattr(runtime, "rank_video_matches", None)
                if rank_matches is not None:
                    matches = rank_matches(vector, norm, catalog)
                else:
                    matches = [(runtime._cosine_similarity(vector, norm, item["embedding"], item["norm"]), item)
                               for item in catalog if len(vector) == len(item["embedding"])]
                for score, item in matches:
                    ranked.append({"filename": item["filename"], "sequence_title": item.get("sequence_title"),
                                   "duration_sec": item.get("duration_sec"), "score": score,
                                   "matched_passage": item.get("matched_passage"),
                                   "is_recent": item["filename"] in previous_recent,
                                   "is_last": item["filename"] == last})
            ranked.sort(key=lambda item: item["score"], reverse=True)
            result = recorder.originals["select_best_video"](vector, norm, catalog, recent, last)
            recorder.event("video_ranking", source_comment_id=recorder.pending_comment,
                           vector_dimensions=len(vector) if vector is not None else 0,
                           vector_norm=norm, recent_before=previous_recent, recent_after=list(recent),
                           last_video=last, selected=result["filename"] if result else None,
                           ranking=ranked)
            return result

        def emit(renderer, entry, recent, last, next_allowed, catalog_size):
            before = time.monotonic()
            result = recorder.originals["maybe_emit_video_request"](
                renderer, entry, recent, last, next_allowed, catalog_size)
            emitted = entry is not None and result[0] is None
            if emitted:
                recorder.event("video_request", filename=entry["filename"],
                               sequence_title=entry.get("sequence_title"),
                               source_comment_id=recorder.pending_comment,
                               duration_sec=runtime._get_video_duration(entry),
                               margin_sec=runtime.VIDEO_END_MARGIN_SEC,
                               match_score=entry.get("match_score"), matched_passage=entry.get("matched_passage"),
                               estimated_next_allowed_sec=result[1] - recorder.started,
                               playback_confirmed=False)
                recorder.pending_comment = None
            else:
                recorder.event("video_gate", has_pending_selection=entry is not None,
                               pending_comment_id=recorder.pending_comment,
                               remaining_sec=max(0, next_allowed - before))
            recorder.estimated_video_end = result[1]
            return result

        for name, replacement in [("_start_game_process", start_game), ("read_game_output", read_game),
                                  ("display_game_output", display_game), ("comment_on_game", comment),
                                  ("_ollama_request", request), ("select_best_video", select),
                                  ("maybe_emit_video_request", emit)]:
            self.wrap(name, replacement)

    def sample_resources(self):
        while not self.stop.is_set():
            values = {}
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=4)
                values["gpu_csv"] = result.stdout.strip()
            except (OSError, subprocess.TimeoutExpired) as exc:
                values["gpu_error"] = str(exc)
            try:
                with urlopen(self.runtime.OLLAMA_HOST.rstrip("/") + "/api/ps", timeout=3) as response:
                    values["ollama_ps"] = json.load(response)
            except Exception as exc:
                values["ollama_ps_error"] = str(exc)
            self.event("resources", **values)
            self.stop.wait(5)

    def close(self):
        for name, original in self.originals.items():
            setattr(self.runtime, name, original)
        self.handle.close()


def snapshot(runtime, directory, runtime_args):
    names = ["LLM_MODEL", "LLM_THINK", "LLM_TEMPERATURE", "LLM_NUM_CTX", "LLM_NUM_PREDICT",
             "COMMENTARY_MIN_WORDS", "COMMENTARY_WORD_LIMIT", "COMMENTARY_LANGUAGE", "INTERVIEW_CONTEXT_PATH",
             "VIDEO_EMBED_MODEL", "VIDEO_EMBED_NUM_GPU", "VIDEO_END_MARGIN_SEC", "VIDEO_MATCH_MODE",
             "VIDEO_EMBEDDINGS_PATH", "LLM_TIMEOUT", "EMBED_TIMEOUT"]
    manifest = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "schema_version": 2, "command_counter": "Game commands only; blank pagination acknowledgements are separate.",
                "runtime_args": runtime_args, "python": sys.version, "python_executable": sys.executable,
                "settings": {name: getattr(runtime, name) for name in names},
                "scope": "Actual Frotz walkthrough and local Ollama requests. Playback is estimated, never confirmed.",
                "files": {}}
    source_dir = directory / "inputs"
    source_dir.mkdir()
    for name in ["src/faketerm.py", "src/c64renderer.py", "godot-viewer/main.gd", runtime.VIDEO_EMBEDDINGS_PATH]:
        path = ROOT / name
        data = path.read_bytes()
        manifest["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        (source_dir / path.name).write_bytes(data)
    for name in [runtime.GAME_ROM_PATH, runtime._game_executable(), runtime.INTERVIEW_CONTEXT_PATH, __file__]:
        path = Path(name)
        data = path.read_bytes()
        manifest["files"][str(path)] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    shutil.copyfile(runtime.INTERVIEW_CONTEXT_PATH, source_dir / "interview-context.txt")
    shutil.copyfile(__file__, source_dir / "record_session.py")
    for command, field in [(["git", "rev-parse", "HEAD"], "git_commit"),
                           (["git", "diff", "--", "src", "godot-viewer"], "git_runtime_diff")]:
        manifest[field] = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout
    for model in [runtime.LLM_MODEL, runtime.VIDEO_EMBED_MODEL]:
        request = Request(runtime.OLLAMA_HOST.rstrip("/") + "/api/show",
                          data=json.dumps({"model": model}).encode(), headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=30) as response:
            save_json(source_dir / (model.replace(":", "-") + ".json"), json.load(response))
    with urlopen(runtime.OLLAMA_HOST.rstrip("/") + "/api/version", timeout=5) as response:
        manifest["ollama_version"] = json.load(response)
    save_json(directory / "manifest.json", manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("runtime_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    runtime_args = args.runtime_args[1:] if args.runtime_args[:1] == ["--"] else args.runtime_args
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    import os
    os.environ["LLM_OUT_OVERRIDE"] = str(directory / "video-requests")
    os.environ["COMMENTARY_LOG_PATH"] = str(directory / "commentary.jsonl")
    os.environ["COMMENTARY_LOG_ENABLED"] = "1"
    sys.path.insert(0, str(ROOT / "src"))
    import faketerm
    snapshot(faketerm, directory, runtime_args)
    recorder = Recorder(faketerm, directory)
    recorder.install()
    sampler = threading.Thread(target=recorder.sample_resources, daemon=True)
    sampler.start()
    print("Recording to " + str(directory), flush=True)
    code = 1
    try:
        with (directory / "console.log").open("w", encoding="utf-8") as console:
            with contextlib.redirect_stdout(console), contextlib.redirect_stderr(console):
                recorder.event("session_start", runtime_args=runtime_args)
                code = faketerm.main(runtime_args)
    finally:
        recorder.stop.set()
        sampler.join(timeout=8)
        recorder.event("session_end", exit_code=code, comments=recorder.comment_count)
        recorder.close()
    print(json.dumps({"exit_code": code, "commands": recorder.command_count, "comments": recorder.comment_count,
                      "elapsed_sec": round(time.monotonic() - recorder.started, 3)}), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
