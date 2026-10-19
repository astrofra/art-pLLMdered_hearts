"""Compare commentary models on frozen scenes from the installation walkthrough."""

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time
import urllib.request

import ollama

import faketerm

DEFAULT_MODELS = ["ministral-3:14b", "qwen3.5:4b", "gemma4:e4b"]
SCENE_COMMANDS = {
    "stand up": "Confinement in the cabin",
    "read invitation": "An invitation and a social role",
    "wear breeches": "Changing clothes and identity",
    "cut rope": "Escaping the ship",
    "examine lucy": "Meeting Lucy",
    "dance": "Entering the ballroom",
    "throw pork at crocodile": "Distracting the crocodile",
    "give garter to papa": "Finding her father",
    "squeeze bottle into green goblet": "Deception over wine",
    "kill crulley": "Fighting Crulley",
    "wave smelling salts": "Reviving a companion",
    "fire pistol at crulley": "The final confrontation",
}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def collect_scenes():
    history = []
    scenes = []
    found = set()
    game = faketerm._start_game_process()
    try:
        for step, command in enumerate(faketerm.plundered_hearts_commands, 1):
            cleaned = faketerm.clean_output(faketerm.read_game_output(game))
            if cleaned:
                history = (history + [cleaned])[-3:]
            if command in SCENE_COMMANDS and command not in found:
                text = "\n".join(history)
                scenes.append({
                    "id": command.replace(" ", "_"), "label": SCENE_COMMANDS[command],
                    "step": step, "command": faketerm.enhance_game_command(command),
                    "game_text": text,
                    "prompt": faketerm.build_prompt(text, faketerm.enhance_game_command(command)),
                })
                found.add(command)
            game.sendline(faketerm.enhance_game_command(command))
        faketerm.read_game_output(game)
    finally:
        faketerm._stop_process(game.proc)
    if len(scenes) != len(SCENE_COMMANDS):
        raise RuntimeError("Missing benchmark scenes: " + str(set(SCENE_COMMANDS) - found))
    return scenes


class GpuSampler:
    def __init__(self):
        self.samples = []
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self):
        while not self.stopped.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                if result.returncode == 0:
                    self.samples.append(int(result.stdout.splitlines()[0].strip()))
            except (OSError, ValueError, subprocess.TimeoutExpired):
                return
            self.stopped.wait(0.5)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopped.set()
        self.thread.join(timeout=4)
        return max(self.samples) if self.samples else None


def match_video(client, text, catalog):
    if not text:
        return []
    response = client.embeddings(
        model=faketerm.VIDEO_EMBED_MODEL, prompt=text,
        options={"num_gpu": 0}, keep_alive="15m",
    )
    vector = response.embedding
    norm = faketerm._vector_norm(vector)
    matches = sorted(
        [(faketerm._cosine_similarity(vector, norm, entry["embedding"], entry["norm"]), entry)
         for entry in catalog if len(entry["embedding"]) == len(vector)],
        key=lambda pair: pair[0], reverse=True,
    )
    return [{"filename": entry["filename"], "title": entry["sequence_title"],
             "score": round(score, 5)} for score, entry in matches[:3]]


def run_sample(client, model, scene, repeat, options, thinking, catalog, enable_thinking=False):
    parameters = dict(options, seed=42 + repeat)
    request = {"model": model, "messages": [{"role": "user", "content": scene["prompt"]}],
               "options": parameters, "keep_alive": "15m"}
    if thinking:
        request["think"] = enable_thinking
    start = time.monotonic()
    response = client.chat(**request)
    elapsed = time.monotonic() - start
    text = faketerm.clean_output(response.message.content or "").strip()
    record = {
        "model": model, "scene_id": scene["id"], "round": repeat + 1,
        "seed": parameters["seed"], "commentary": text,
        "elapsed_sec": round(elapsed, 3), "words": len(text.split()),
        "prompt_tokens": response.prompt_eval_count, "output_tokens": response.eval_count,
        "done_reason": response.done_reason,
        "thinking_enabled": enable_thinking if thinking else None,
        "load_sec": round((response.load_duration or 0) / 1e9, 3),
        "prompt_eval_sec": round((response.prompt_eval_duration or 0) / 1e9, 6),
        "generation_sec": round((response.eval_duration or 0) / 1e9, 6),
        "server_total_sec": round((response.total_duration or 0) / 1e9, 6),
        "tokens_per_sec": round(response.eval_count * 1e9 / response.eval_duration, 2)
        if response.eval_duration and response.eval_count else None,
        "thinking_characters": len(getattr(response.message, "thinking", None) or ""),
        "technical_terms": bool(re.search(
            r"\b(memory|hardware|technical|commodore|parser|storage)\b", text, re.I)),
        "markdown": bool(re.search(r"\*|`|^#", text, re.M)),
        "loaded_models": client.ps().model_dump(mode="json"),
    }
    start = time.monotonic()
    try:
        record["video_matches"] = match_video(client, text, catalog)
    except Exception as exc:
        record["embedding_error"] = str(exc)
    record["embedding_sec"] = round(time.monotonic() - start, 3)
    return record


def summarize(data):
    summaries = []
    for model in data["models"]:
        rows = [row for row in data["results"] if row["model"] == model]
        successful = [row for row in rows if not row.get("error") and row.get("commentary")]
        summary = {"model": model, "attempts": len(rows), "responses": len(successful),
                   "errors": sum(bool(row.get("error")) for row in rows)}
        if successful:
            summary.update({
                "median_sec": round(statistics.median(row["elapsed_sec"] for row in successful), 3),
                "first_response_sec": successful[0]["elapsed_sec"],
                "median_words": statistics.median(row["words"] for row in successful),
                "within_60_words": sum(row["words"] <= 60 for row in successful),
                "truncated": sum(row["done_reason"] == "length" for row in successful),
                "markdown": sum(row["markdown"] for row in successful),
                "technical_terms": sum(row["technical_terms"] for row in successful),
                "distinct_top_videos": len({row["video_matches"][0]["filename"] for row in successful
                                            if row.get("video_matches")}),
            })
        summary.update(data["model_info"].get(model, {}))
        summaries.append(summary)
    return summaries


def write_html(directory, data):
    escape = lambda value: html.escape(str(value))
    parts = ['<!doctype html><html lang="fr"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Comparatif LLM — pLLMdered Hearts</title>',
             '<style>body{font:16px/1.55 system-ui;margin:2rem auto;padding:0 1rem;max-width:1500px;color:#222}'
             'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:.6rem;text-align:left}'
             '.answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:1rem}'
             'article{border:1px solid #ccc;border-radius:8px;padding:1rem}pre{white-space:pre-wrap}'
             'small{color:#555}.warning{color:#a33}section{margin:2rem 0}</style>',
             '<h1>Comparatif LLM — pLLMdered Hearts</h1>',
             '<p>Prompts figés issus du parcours réel : résumé de '
             + str(data.get("interview_summary", {}).get("words", 752))
             + ' mots et trois dernières sorties du jeu. '
             'Commentaires en anglais, sans historique des réponses précédentes. '
             'EmbeddingGemma 300M fonctionne sur le processeur. Aucun fichier envoyé au viewer.</p>',
             '<p>Ollama : ' + escape(data["ollama_version"]) + '. Paramètres : <code>'
             + escape(json.dumps(data["options"])) + '</code>. Raisonnement prolongé désactivé. '
             'Les autres réglages sont ceux du modèle distribué par Ollama.</p>',
             '<p>Le temps médian mesure la requête de commentaire avec modèle préchargé. '
             'Le pic VRAM couvre toute la carte et inclut le chargement ; il est échantillonné. '
             'La présence de mots techniques est un indicateur lexical, pas un jugement de pertinence.</p>',
             '<p>Ces mesures sont réalisées sans les fenêtres C64 et Godot. Les rapprochements vidéo '
             'sont calculés avant le filtre anti-répétition et ne déclenchent aucune lecture.</p>',
             '<table><tr><th>Modèle</th><th>Réponses</th><th>Médiane</th><th>≤ 60 mots</th>'
             '<th>Coupures</th><th>Termes techniques</th><th>Vidéos distinctes en tête</th><th>Pic VRAM</th></tr>']
    for summary in summarize(data):
        count = summary["responses"]
        cells = [summary["model"], str(count) + '/' + str(summary["attempts"]),
                 str(summary.get("median_sec", "—")) + ' s',
                 str(summary.get("within_60_words", 0)) + '/' + str(count),
                 summary.get("truncated", 0), summary.get("technical_terms", 0),
                 summary.get("distinct_top_videos", 0),
                 str(summary.get("peak_gpu_mib", "—")) + ' MiB']
        parts.append('<tr>' + ''.join('<td>' + escape(cell) + '</td>' for cell in cells) + '</tr>')
    parts.append('</table>')
    if data.get("review"):
        review = data["review"]
        parts.append('<section><h2>Lecture des résultats</h2>')
        for paragraph in review["findings"]:
            parts.append('<p>' + escape(paragraph) + '</p>')
        parts.append('<p><strong>' + escape(review["recommendation"]) + '</strong></p>')
        parts.append('<h3>Portée du comparatif</h3><ul>')
        for limitation in review["limitations"]:
            parts.append('<li>' + escape(limitation) + '</li>')
        parts.append('</ul></section>')
    if data.get("integration"):
        parts.append('<section><h2>Vérification dans l’installation</h2>')
        for check in data["integration"]:
            parts.append('<p>' + escape(check["model"]) + ' : ' + str(len(check["comments"]))
                         + ' commentaires, rendu C64 et viewer actifs, demande vidéo reçue, sortie '
                         + str(check["exit_code"]) + '. Pic de la carte : '
                         + str(check["peak_gpu_mib"]) + ' MiB.</p>')
        parts.append('</section>')
    for scene in data["scenes"]:
        parts.append('<section><h2>' + escape(scene["label"]) + '</h2><p>Commande : <code>'
                     + escape(scene["command"]) + '</code>, étape ' + str(scene["step"]) + '</p>')
        parts.append('<details><summary>Texte du jeu</summary><pre>' + escape(scene["game_text"]) + '</pre></details>')
        for repeat in range(data["rounds"]):
            parts.append('<h3>Passage ' + str(repeat + 1) + '</h3><div class="answers">')
            for model in data["models"]:
                row = next((row for row in data["results"] if row["model"] == model
                            and row["scene_id"] == scene["id"] and row["round"] == repeat + 1), None)
                if row is None:
                    continue
                parts.append('<article><h4>' + escape(model) + '</h4><p>'
                             + escape(row.get("commentary") or row.get("error", "Réponse vide")) + '</p>')
                parts.append('<small>' + escape(str(row.get("elapsed_sec", "—")) + ' s · '
                             + str(row.get("words", 0)) + ' mots · ' + str(row.get("done_reason", ""))) + '</small>')
                for match in row.get("video_matches", [])[:1]:
                    parts.append('<p><small>Vidéo : ' + escape(match["title"]) + ' · '
                                 + escape(match["score"]) + '</small></p>')
                parts.append('</article>')
            parts.append('</div>')
        parts.append('</section>')
    (directory / "comparison.html").write_text(''.join(parts) + '</html>\n', encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--limit", type=int, help="Limit the number of frozen scenes")
    parser.add_argument("--output-dir", default=os.path.join(faketerm.PROJECT_DIR, "documentation", "llm-comparison"))
    parser.add_argument("--scenes", type=Path, help="Reuse a saved scenes.json instead of replaying Frotz")
    parser.add_argument("--summary", type=Path, help="Replace only the interview summary in the prompts")
    args = parser.parse_args()
    if args.rounds < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--rounds and --limit must be positive")
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(args.scenes.read_text(encoding="utf-8")) if args.scenes else collect_scenes()
    if args.limit:
        scenes = scenes[:args.limit]
    interview_text = args.summary.read_text(encoding="utf-8").strip() if args.summary else faketerm.load_itw_redux()
    if args.summary:
        original_summary = faketerm.load_itw_redux()
        for scene in scenes:
            if scene["prompt"].count(original_summary) != 1:
                parser.error("Saved prompt does not contain the original summary exactly once: " + scene["id"])
            scene["prompt"] = scene["prompt"].replace(original_summary, interview_text, 1)
    save_json(directory / "scenes.json", scenes)
    client = ollama.Client(host=faketerm.OLLAMA_HOST, timeout=faketerm.LLM_TIMEOUT)
    catalog = faketerm.load_video_embeddings(faketerm.VIDEO_EMBEDDINGS_PATH)
    options = {"temperature": 0.3, "num_predict": 256, "num_ctx": 4096}
    with urllib.request.urlopen(faketerm.OLLAMA_HOST.rstrip("/") + "/api/version", timeout=5) as response:
        version = json.load(response)["version"]
    data = {"models": args.models, "rounds": args.rounds, "options": options,
            "ollama_version": version,
            "interview_summary": {
                "file": str(args.summary) if args.summary else "assets/abriggs-itw-750-words.txt",
                "words": len(interview_text.split()),
                "sha256": hashlib.sha256(interview_text.encode("utf-8")).hexdigest(),
            },
            "source_scenes": str(args.scenes) if args.scenes else None,
            "scenes": scenes, "model_info": {}, "results": [],
            "scene_sha256": hashlib.sha256(json.dumps(scenes, sort_keys=True).encode()).hexdigest()}
    client.embeddings(model=faketerm.VIDEO_EMBED_MODEL, prompt="Warm up the interview index.",
                      options={"num_gpu": 0}, keep_alive="15m")
    for model in args.models:
        sampler = GpuSampler()
        sampler.start()
        info = client.show(model)
        thinking = "thinking" in (info.capabilities or [])
        details = info.details.model_dump(mode="json") if info.details else {}
        data["model_info"][model] = {
            "details": details, "supports_thinking": thinking, "packaged_parameters": info.parameters,
        }
        start = time.monotonic()
        try:
            print("Loading " + model, flush=True)
            client.generate(model=model, prompt="", stream=False, options={"num_ctx": 4096}, keep_alive="15m")
            data["model_info"][model]["preload_sec"] = round(time.monotonic() - start, 3)
            for repeat in range(args.rounds):
                for scene in scenes:
                    try:
                        record = run_sample(client, model, scene, repeat, options, thinking, catalog)
                    except Exception as exc:
                        record = {"model": model, "scene_id": scene["id"], "round": repeat + 1,
                                  "error": str(exc)}
                    data["results"].append(record)
                    save_json(directory / "results.json", data)
                    print(model + " | round " + str(repeat + 1) + " | " + scene["id"] + " | "
                          + str(record.get("elapsed_sec", "ERROR")) + " s | "
                          + str(record.get("words", 0)) + " words | " + record.get("error", ""), flush=True)
        finally:
            data["model_info"][model]["peak_gpu_mib"] = sampler.stop()
            client.generate(model=model, keep_alive=0)
            data["summary"] = summarize(data)
            save_json(directory / "results.json", data)
            write_html(directory, data)
    print(json.dumps(data["summary"], indent=2), flush=True)
    print("Report: " + str((directory / "comparison.html").resolve()), flush=True)
    return 0 if all(not row.get("error") and row.get("commentary") for row in data["results"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
