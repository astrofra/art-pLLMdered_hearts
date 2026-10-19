"""Build a paired report from the frozen 750- and 1250-word benchmark runs."""

from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import statistics


DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
BASELINE = ROOT / "documentation" / "llm-comparison-2026-10-19"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def key(row):
    return row["model"], row["scene_id"], row["round"]


def digest(row):
    return next(item["digest"] for item in row["loaded_models"]["models"]
                if item["name"] == row["model"])


def metrics(rows):
    videos = Counter(row["video_matches"][0]["filename"] for row in rows)
    return {
        "responses": len(rows),
        "median_sec": round(statistics.median(row["elapsed_sec"] for row in rows), 3),
        "median_words": statistics.median(row["words"] for row in rows),
        "within_60_words": sum(row["words"] <= 60 for row in rows),
        "markdown": sum(row["markdown"] for row in rows),
        "technical_terms": sum(row["technical_terms"] for row in rows),
        "distinct_top_videos": len(videos),
        "top_video_counts": dict(videos.most_common()),
        "truncated_outputs": sum(row["done_reason"] == "length" for row in rows),
        "prompt_tokens_min": min(row["prompt_tokens"] for row in rows),
        "prompt_tokens_max": max(row["prompt_tokens"] for row in rows),
        "prompt_plus_output_tokens_max": max(row["prompt_tokens"] + row["output_tokens"] for row in rows),
    }


def main():
    before, after = load(BASELINE / "results.json"), load(DIRECTORY / "results.json")
    for field in ["models", "rounds", "options", "ollama_version"]:
        assert before[field] == after[field], field
    assert len(before["results"]) == len(after["results"]) == 72
    old_text = (ROOT / "assets" / "abriggs-itw-750-words.txt").read_text(encoding="utf-8").strip()
    new_text = (ROOT / "assets" / "abriggs-itw-1250-words.txt").read_text(encoding="utf-8").strip()
    old_scenes = {scene["id"]: scene for scene in before["scenes"]}
    for scene in after["scenes"]:
        original = dict(old_scenes[scene["id"]])
        assert original["prompt"].count(old_text) == 1
        original["prompt"] = original["prompt"].replace(old_text, new_text, 1)
        assert original == scene, scene["id"]
    previous = {key(row): row for row in before["results"]}
    current = {key(row): row for row in after["results"]}
    assert previous.keys() == current.keys()
    for pair_key, row in current.items():
        assert row.get("commentary") and not row.get("error") and not row.get("embedding_error")
        assert previous[pair_key]["seed"] == row["seed"]
        assert digest(previous[pair_key]) == digest(row)
    for model in after["models"]:
        for field in ["details", "supports_thinking", "packaged_parameters"]:
            assert before["model_info"][model][field] == after["model_info"][model][field]
    comparison = {
        "checks": "Only the interview summary changed; scenes, instructions, seeds, model digests, options and Ollama version match.",
        "baseline_results_sha256": hashlib.sha256((BASELINE / "results.json").read_bytes()).hexdigest(),
        "summary_words": [len(old_text.split()), len(new_text.split())],
        "models": [],
    }
    for model in after["models"]:
        old_rows = [row for row in before["results"] if row["model"] == model]
        new_rows = [row for row in after["results"] if row["model"] == model]
        comparison["models"].append({
            "model": model, "before": metrics(old_rows), "after": metrics(new_rows),
            "changed_top_video_pairs": sum(previous[key(row)]["video_matches"][0]["filename"]
                                           != row["video_matches"][0]["filename"] for row in new_rows),
            "identical_commentary_pairs": sum(previous[key(row)]["commentary"] == row["commentary"] for row in new_rows),
        })
    (DIRECTORY / "summary-comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    escape = lambda value: html.escape(str(value))
    parts = ['<!doctype html><html lang="fr"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>750 / 1250 mots : pLLMdered Hearts</title>',
             '<style>body{font:16px/1.55 system-ui;margin:2rem auto;padding:0 1rem;max-width:1300px;color:#222}'
             'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:.6rem;text-align:left}'
             '.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:1rem}'
             'article{border:1px solid #ccc;border-radius:8px;padding:1rem}pre{white-space:pre-wrap}'
             'small{color:#555}section{margin:2rem 0}@media(max-width:700px){.pair{grid-template-columns:1fr}}</style>',
             '<h1>Effet du résumé : 752 / 1 255 mots</h1>',
             '<p>Mêmes 12 scènes, 2 graines par scène et 3 modèles : 72 commentaires par série. '
             'Seul le texte du résumé change. Identité des scènes, consignes, graines, poids des modèles, '
             'paramètres et version d’Ollama vérifiée par le script.</p>',
             '<p>Température 0,3 ; contexte 4 096 tokens ; sortie maximale 256 tokens ; '
             'raisonnement prolongé désactivé pour Qwen et Gemma. '
             'EmbeddingGemma 300M sur CPU ; aucune lecture vidéo déclenchée.</p>',
             '<p><a href="../llm-comparison-2026-10-19/comparison.html">Série 752 mots</a> · '
             '<a href="comparison.html">Série 1 255 mots</a> · '
             '<a href="summary-comparison.json">Mesures comparées</a></p>',
             '<table><tr><th>Modèle</th><th>Médiane (s)</th><th>Termes techniques / 24</th>'
             '<th>Vidéos distinctes</th><th>Vidéo modifiée / 24</th><th>Prompt max. (tokens)</th></tr>']
    for model in comparison["models"]:
        cells = [model["model"]]
        for field in ["median_sec", "technical_terms", "distinct_top_videos"]:
            cells.append(str(model["before"][field]) + ' → ' + str(model["after"][field]))
        cells.append(model["changed_top_video_pairs"])
        cells.append(str(model["before"]["prompt_tokens_max"]) + ' → ' + str(model["after"]["prompt_tokens_max"]))
        parts.append('<tr>' + ''.join('<td>' + escape(cell) + '</td>' for cell in cells) + '</tr>')
    parts.append('</table><p><small>Flèches : 752 → 1 255 mots. Le compteur lexical cherche '
                 'memory, hardware, technical, commodore, parser ou storage ; il ne mesure pas '
                 'la pertinence. Diversité vidéo calculée avant le filtre anti-répétition. '
                 'Les temps incluent le traitement du prompt et la génération, avec modèle préchargé ; '
                 'les effets de cache et de charge de la machine ne sont pas isolés.</small></p>')
    review_path = DIRECTORY / "summary-review.json"
    if review_path.exists():
        review = load(review_path)
        parts.append('<section><h2>Lecture des résultats</h2>')
        for paragraph in review["findings"]:
            parts.append('<p>' + escape(paragraph) + '</p>')
        parts.append('<p><strong>' + escape(review["conclusion"]) + '</strong></p>')
        parts.append('<h3>Portée du comparatif</h3><ul>')
        for limitation in review.get("limitations", []):
            parts.append('<li>' + escape(limitation) + '</li>')
        parts.append('</ul></section>')
    for model in after["models"]:
        parts.append('<section><h2>' + escape(model) + '</h2>')
        for scene in after["scenes"]:
            parts.append('<h3>' + escape(scene["label"]) + '</h3>')
            parts.append('<details><summary>Texte du jeu</summary><pre>' + escape(scene["game_text"]) + '</pre></details>')
            for repeat in range(1, after["rounds"] + 1):
                parts.append('<h4>Graine ' + str(41 + repeat) + '</h4><div class="pair">')
                for label, rows in [('752 mots', previous), ('1 255 mots', current)]:
                    row = rows[model, scene["id"], repeat]
                    video = row["video_matches"][0]
                    parts.append('<article><strong>' + label + '</strong><p>' + escape(row["commentary"]) + '</p>'
                                 '<small>' + str(row["words"]) + ' mots · ' + str(row["elapsed_sec"]) + ' s · '
                                 + str(row["prompt_tokens"]) + ' tokens en entrée</small><p><small>Vidéo : '
                                 + escape(video["title"]) + ' (' + str(video["score"]) + ')</small></p></article>')
                parts.append('</div>')
        parts.append('</section>')
    (DIRECTORY / "summary-comparison.html").write_text(''.join(parts) + '</html>\n', encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
