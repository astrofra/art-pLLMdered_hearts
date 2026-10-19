"""Compare Gemma thinking with balanced order and separate warm-up measurements."""

import argparse
from collections import Counter
import datetime
import hashlib
import html
import itertools
import json
from pathlib import Path
import statistics
import sys
import time
import urllib.request

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ollama
import faketerm
import ollama_benchmark as benchmark

MODEL = "gemma4:e4b"
CONDITIONS = [
    {"id": "off256", "label": "Sans raisonnement, 256", "think": False, "num_predict": 256},
    {"id": "off1024", "label": "Sans raisonnement, 1024", "think": False, "num_predict": 1024},
    {"id": "on1024", "label": "Avec raisonnement, 1024", "think": True, "num_predict": 1024},
]


def median(rows, field):
    values = [row[field] for row in rows if row.get(field) is not None]
    return round(statistics.median(values), 3) if values else None


def summarize(data):
    summaries = []
    for condition in data["conditions"]:
        rows = [row for row in data["results"] if row["condition"] == condition["id"]]
        good = [row for row in rows if row.get("commentary") and not row.get("error")]
        videos = Counter(row["video_matches"][0]["filename"] for row in good if row.get("video_matches"))
        summaries.append(dict(condition, attempts=len(rows), responses=len(good),
            errors=sum(bool(row.get("error")) for row in rows),
            embedding_errors=sum(bool(row.get("embedding_error")) for row in rows),
            empty=sum(not row.get("commentary") and not row.get("error") for row in rows),
            truncated=sum(row.get("done_reason") == "length" for row in rows),
            median_sec=median(rows, "elapsed_sec"), median_prompt_sec=median(rows, "prompt_eval_sec"),
            median_generation_sec=median(rows, "generation_sec"),
            median_tokens_per_sec=median(rows, "tokens_per_sec"),
            median_output_tokens=median(rows, "output_tokens"),
            median_thinking_characters=median(rows, "thinking_characters"),
            median_words=median(good, "words"),
            within_60_words=sum(row["words"] <= 60 for row in good),
            technical_terms=sum(row["technical_terms"] for row in good),
            markdown=sum(row["markdown"] for row in good),
            distinct_top_videos=len(videos), top_video_counts=dict(videos.most_common())))
    return summaries


def write_report(directory, data):
    escape = lambda value: html.escape(str(value))
    parts = ['<!doctype html><html lang="fr"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Gemma E4B : comparaison du raisonnement</title>',
        '<style>body{font:16px/1.55 system-ui;margin:2rem auto;padding:0 1rem;max-width:1550px;color:#222}'
        'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:.6rem;text-align:left}'
        '.answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}'
        'article{border:1px solid #ccc;border-radius:8px;padding:1rem}pre{white-space:pre-wrap}'
        'small{color:#555}.warning{color:#a33}section{margin:2rem 0}</style>',
        '<h1>Gemma E4B : avec ou sans raisonnement</h1>',
        '<p>Mêmes douze scènes figées, résumé anglais de 752 mots, graines 42 et 43. '
        'Température 0,3, contexte 4 096 tokens, EmbeddingGemma 300M sur CPU. '
        'Aucun historique de commentaires. Aucun fichier envoyé au viewer.</p>',
        '<p>Le modèle et les deux modes sont échauffés avant les mesures. '
        'Les six permutations des trois conditions sont réparties également entre les scènes ; '
        'leur ordre est inversé au second passage. Le budget de 1 024 tokens sans raisonnement '
        'sert de témoin pour isoler l’activation du raisonnement.</p>',
        '<p>Si une réponse avec raisonnement atteint 1 024 tokens, une série complète '
        'à 2 048 tokens est ajoutée, après les trois conditions principales. '
        'Cette série complémentaire est distincte du protocole alterné.</p>',
        '<table><tr><th>Condition</th><th>Réponses</th><th>Coupures</th><th>Médiane (s)</th>'
        '<th>Lecture prompt (s)</th><th>Génération (s)</th><th>Tokens produits</th>'
        '<th>Termes techniques</th><th>Vidéos distinctes</th></tr>']
    for row in data["summary"]:
        cells = [row["label"], str(row["responses"]) + '/' + str(row["attempts"]), row["truncated"],
                 row["median_sec"], row["median_prompt_sec"], row["median_generation_sec"],
                 row["median_output_tokens"], str(row["technical_terms"]) + '/' + str(row["responses"]),
                 row["distinct_top_videos"]]
        parts.append('<tr>' + ''.join('<td>' + escape(value) + '</td>' for value in cells) + '</tr>')
    parts.append('</table><p><small>Les temps médians incluent toutes les tentatives, '
        'y compris les réponses vides. Les tokens produits comprennent le raisonnement et la réponse ; '
        'le plafond ne fixe pas la longueur effective. Le compteur lexical cherche memory, hardware, '
        'technical, commodore, parser ou storage et ne mesure pas la pertinence. '
        'Les rapprochements vidéo sont calculés avant le filtre anti-répétition.</small></p>')
    review = data.get("review")
    if review:
        parts.append('<section><h2>Lecture des résultats</h2>')
        for paragraph in review["findings"]:
            parts.append('<p>' + escape(paragraph) + '</p>')
        parts.append('<p><strong>' + escape(review["conclusion"]) + '</strong></p><ul>')
        for limitation in review.get("limitations", []):
            parts.append('<li>' + escape(limitation) + '</li>')
        parts.append('</ul></section>')
    initial = data.get("initial_timing_observation")
    if initial:
        parts.append('<section><h2>Origine du paradoxe initial</h2>'
                     '<p>Mesures ponctuelles antérieures au protocole contrôlé. Le cache réduit '
                     'la lecture du prompt d’environ 0,4 seconde ; l’écart principal vient de la '
                     'génération, dont le débit était très différent. Les journaux ne permettent '
                     'pas d’attribuer précisément ce ralentissement transitoire.</p>'
                     '<table><tr><th>Plafond</th><th>Tokens produits</th><th>Temps total (s)</th>'
                     '<th>Lecture prompt (s)</th><th>Génération (s)</th><th>Tokens/s</th></tr>')
        for row in initial["results"]:
            if row["think"]:
                cells = [row["num_predict"], row["output_tokens"], row["elapsed_sec"],
                         row["prompt_eval_sec"], row["generation_sec"], row["tokens_per_sec"]]
                parts.append('<tr>' + ''.join('<td>' + escape(value) + '</td>' for value in cells) + '</tr>')
        parts.append('</table><p><a href="initial-timing-observation.json">Données initiales</a> · '
                     '<a href="initial-server-timing.log">Extrait du journal initial</a></p></section>')
    parts.append('<section><h2>Contrôle du paradoxe 256 / 1 024</h2>'
                 '<p>Une même scène, raisonnement actif, graine 42, après échauffement. '
                 'Ordre 256–1024 puis 1024–256. Les temps de chargement sont enregistrés '
                 'séparément dans les données.</p>'
                 '<table><tr><th>Plafond</th><th>Tokens produits</th><th>Temps total (s)</th>'
                 '<th>Lecture prompt (s)</th><th>Génération (s)</th><th>Tokens/s</th><th>Réponse</th></tr>')
    for row in data["timing_probe"]:
        cells = [row["num_predict"], row.get("output_tokens"), row.get("elapsed_sec"),
                 row.get("prompt_eval_sec"), row.get("generation_sec"), row.get("tokens_per_sec"),
                 bool(row.get("commentary"))]
        parts.append('<tr>' + ''.join('<td>' + escape(value) + '</td>' for value in cells) + '</tr>')
    parts.append('</table></section><details><summary>Échauffements exclus des mesures</summary><pre>'
                 + escape(json.dumps(data["warmups"], ensure_ascii=False, indent=2)) + '</pre></details>')
    for scene in data["scenes"]:
        parts.append('<section><h2>' + escape(scene["label"]) + '</h2>'
                     '<details><summary>Texte du jeu</summary><pre>' + escape(scene["game_text"]) + '</pre></details>')
        for repeat in range(1, data["rounds"] + 1):
            parts.append('<h3>Graine ' + str(41 + repeat) + '</h3><div class="answers">')
            for condition in data["conditions"]:
                row = next((row for row in data["results"] if row["scene_id"] == scene["id"]
                            and row["round"] == repeat and row["condition"] == condition["id"]), None)
                if row is None:
                    continue
                parts.append('<article><h4>' + escape(condition["label"]) + '</h4><p>'
                             + escape(row.get("commentary") or row.get("error", "Aucun commentaire final")) + '</p>'
                             '<small>' + escape(str(row.get("elapsed_sec")) + ' s · '
                             + str(row.get("words", 0)) + ' mots · '
                             + str(row.get("output_tokens")) + ' tokens · '
                             + str(row.get("done_reason"))) + '</small>')
                for video in row.get("video_matches", [])[:1]:
                    parts.append('<p><small>Vidéo : ' + escape(video["title"]) + ' ('
                                 + str(video["score"]) + ')</small></p>')
                parts.append('</article>')
            parts.append('</div>')
        parts.append('</section>')
    (directory / "comparison.html").write_text(''.join(parts) + '</html>\n', encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=Path, default=ROOT / "documentation/llm-comparison-2026-10-19/scenes.json")
    parser.add_argument("--report-only", action="store_true", help="Regenerate the report from saved results and review.json")
    args = parser.parse_args()
    if args.report_only:
        data = json.loads((DIRECTORY / "results.json").read_text(encoding="utf-8"))
        for field, filename in [("review", "review.json"), ("initial_timing_observation", "initial-timing-observation.json")]:
            path = DIRECTORY / filename
            if path.exists():
                data[field] = json.loads(path.read_text(encoding="utf-8"))
        data["summary"] = summarize(data)
        benchmark.save_json(DIRECTORY / "results.json", data)
        write_report(DIRECTORY, data)
        print("Report: " + str(DIRECTORY / "comparison.html"))
        return 0
    scenes = json.loads(args.scenes.read_text(encoding="utf-8"))
    assert len(scenes) == 12
    for scene in scenes:
        assert scene["prompt"].count(faketerm.load_itw_redux()) == 1
    client = ollama.Client(host=faketerm.OLLAMA_HOST, timeout=faketerm.LLM_TIMEOUT)
    info = client.show(MODEL)
    assert "thinking" in info.capabilities
    with urllib.request.urlopen(faketerm.OLLAMA_HOST.rstrip("/") + "/api/version", timeout=5) as response:
        version = json.load(response)["version"]
    catalog = faketerm.load_video_embeddings(faketerm.VIDEO_EMBEDDINGS_PATH)
    data = {"started_at": datetime.datetime.now().astimezone().isoformat(), "model": MODEL,
            "ollama_version": version, "rounds": 2, "conditions": list(CONDITIONS),
            "options": {"temperature": 0.3, "num_ctx": 4096},
            "packaged_parameters": info.parameters, "details": info.details.model_dump(mode="json"),
            "scenes": scenes, "scene_sha256": hashlib.sha256(json.dumps(scenes, sort_keys=True).encode()).hexdigest(),
            "warmups": [], "timing_probe": [], "results": []}
    server_log = Path.home() / "AppData/Local/Ollama/server.log"
    log_offset = server_log.stat().st_size
    data["server_log_start_offset"] = log_offset
    benchmark.save_json(DIRECTORY / "scenes.json", scenes)

    def checkpoint():
        data["summary"] = summarize(data)
        benchmark.save_json(DIRECTORY / "results.json", data)
        write_report(DIRECTORY, data)

    def sample(scene, repeat, condition):
        options = dict(data["options"], num_predict=condition["num_predict"])
        try:
            row = benchmark.run_sample(client, MODEL, scene, repeat, options, True, catalog,
                                       enable_thinking=condition["think"])
        except Exception as exc:
            row = {"model": MODEL, "scene_id": scene["id"], "round": repeat + 1,
                   "seed": 42 + repeat, "error": str(exc)}
        row.update(condition=condition["id"], num_predict=condition["num_predict"])
        print(condition["id"] + ' | ' + scene["id"] + ' | seed ' + str(42 + repeat) + ' | '
              + str(row.get("elapsed_sec", "ERROR")) + ' s | ' + str(row.get("output_tokens"))
              + ' tokens | ' + str(row.get("words", 0)) + ' words | '
              + str(row.get("done_reason", row.get("error"))), flush=True)
        return row

    client.generate(model=MODEL, keep_alive=0)
    client.embeddings(model=faketerm.VIDEO_EMBED_MODEL, prompt="Warm up the interview index.",
                      options={"num_gpu": 0}, keep_alive="15m")
    sampler = benchmark.GpuSampler()
    sampler.start()
    try:
        start = time.monotonic()
        client.generate(model=MODEL, prompt="", options={"num_ctx": 4096}, keep_alive="15m")
        data["preload_sec"] = round(time.monotonic() - start, 3)
        probe_scene = next(scene for scene in scenes if scene["id"] == "cut_rope")
        for condition in CONDITIONS:
            data["warmups"].append(sample(probe_scene, 0, condition))
            checkpoint()
        for budget in [256, 1024, 1024, 256]:
            condition = {"id": "probe" + str(budget), "think": True, "num_predict": budget}
            data["timing_probe"].append(sample(probe_scene, 0, condition))
            checkpoint()
        orders = list(itertools.permutations(CONDITIONS))
        for repeat in range(data["rounds"]):
            for index, scene in enumerate(scenes):
                order = orders[index % len(orders)]
                if repeat % 2:
                    order = tuple(reversed(order))
                for position, condition in enumerate(order, 1):
                    row = sample(scene, repeat, condition)
                    row.update(order_position=position, sequence_index=len(data["results"]) + 1)
                    data["results"].append(row)
                    checkpoint()
        if any(row.get("done_reason") == "length" for row in data["results"] if row["condition"] == "on1024"):
            condition = {"id": "on2048", "label": "Avec raisonnement, 2048 (complément)",
                         "think": True, "num_predict": 2048}
            data["conditions"].append(condition)
            print("Adding the complete 2048-token condition after truncation at 1024.", flush=True)
            data["warmups"].append(sample(probe_scene, 0, condition))
            for repeat in range(data["rounds"]):
                for scene in scenes:
                    row = sample(scene, repeat, condition)
                    row["sequence_index"] = len(data["results"]) + 1
                    data["results"].append(row)
                    checkpoint()
    finally:
        data["peak_gpu_mib"] = sampler.stop()
        data["finished_at"] = datetime.datetime.now().astimezone().isoformat()
        client.generate(model=MODEL, keep_alive=0)
        with server_log.open("rb") as handle:
            handle.seek(log_offset)
            log_text = handle.read().decode("utf-8", errors="replace")
        lines = [line for line in log_text.splitlines() if any(fragment in line for fragment in
                 ["prompt eval time", "eval time", "total time", "draft acceptance", "cached n_tokens", "forcing full prompt", 'POST     "/api/chat"'])]
        (DIRECTORY / "server-timing.log").write_text('\n'.join(lines) + '\n', encoding="utf-8")
        data["input_context_warnings"] = [line for line in log_text.splitlines()
            if "truncating input" in line.lower() or "context shift" in line.lower() or "truncated = 1" in line.lower()]
        checkpoint()
    print(json.dumps(data["summary"], indent=2), flush=True)
    print("Report: " + str(DIRECTORY / "comparison.html"), flush=True)
    return int(any(row.get("error") or row.get("embedding_error") for row in data["results"]))


if __name__ == "__main__":
    raise SystemExit(main())
