"""Measure complete game histories triggered by the configured word threshold."""

import argparse
import hashlib
import html
import json
from pathlib import Path
import statistics
import sys

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ollama
import faketerm


def collect_buffers(rows):
    history = []
    samples = []
    for row in rows[:-1]:
        history.append(row["game_text"])
        words = sum(len(text.split()) for text in history)
        if faketerm.commentary_is_due(words):
            samples.append({"step": row["step"],
                            "history_words": words, "history": "\n".join(history)})
            history = []
        history.append(">> " + row["command"])
    history.append(rows[-1]["game_text"])
    return samples, sum(len(text.split()) for text in history)


def write_report(result, samples):
    escape = lambda value: html.escape(str(value))
    parts = ['<!doctype html><html lang="fr"><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Buffer de commentaire — pLLMdered Hearts</title>',
             '<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}'
             'table{border-collapse:collapse;width:100%}th,td{padding:.6rem;border:1px solid #ddd;text-align:left}'
             'pre{white-space:pre-wrap}section{margin:2rem 0}small{color:#555}</style>',
             '<h1>Buffer de commentaire : seuil de ' + escape(result['threshold_words']) + ' mots</h1>',
             '<p>Les commandes exécutées et les sorties complètes du jeu s’accumulent. '
             'Quand le buffer dépasse ' + escape(result['threshold_words']) + ' mots, le LLM commente au retour au prompt, après '
             'l’affichage complet de la sortie du jeu. Le commentaire s’affiche immédiatement, '
             'y compris pendant un extrait d’Amy.</p>',
             '<p>Python conserve le dernier commentaire. Sa vectorisation et le choix de la vidéo '
             'attendent la fin estimée de l’extrait courant, calculée depuis l’écriture de la demande '
             'avec la durée du catalogue et une marge de 3 secondes. Godot ne renvoie aucun état.</p>',
             '<p>Le nouveau prompt demande une observation concrète sur les événements du buffer '
             'et une association au témoignage seulement si elle est justifiée. La prochaine '
             'commande est exclue du prompt. Le comptage des tokens utilise un contexte de '
             + str(faketerm.LLM_NUM_CTX) + ' tokens, sans raisonnement prolongé.</p>',
             '<h2>Taille des historiques</h2><table><tr><th>Mesure</th><th>Valeur</th></tr>']
    for label, value in [
        ('Commandes du parcours capturé', result['source_commands']),
        ('Buffers reconstitués', result['samples']), ('Mots moyens', result['mean_words']),
        ('Mots médians', result['median_words']), ('Minimum / maximum', str(result['min_words']) + ' / ' + str(result['max_words'])),
        ('Tokens moyens du prompt complet', result.get('mean_prompt_tokens', 'Non mesuré')),
        ('Tokens maximum du prompt complet', result.get('max_prompt_tokens', 'Non mesuré')),
    ]:
        parts.append('<tr><td>' + escape(label) + '</td><td>' + escape(value) + '</td></tr>')
    parts.append('</table><p><small>Les historiques sont reconstitués à partir d’un parcours réel '
                 'capturé dans Frotz, en conservant les commandes et chaque sortie complète. '
                 'Les tokens incluent les consignes et le contexte documentaire courant ('
                 + str(len(faketerm.load_itw_redux().split())) + ' mots). '
                 'Le reliquat de fin de parcours est conservé au redémarrage dans l’installation. '
                 'Ces chiffres dépendent du seuil et des sorties effectivement produites par le jeu.</small></p>')
    integration_path = DIRECTORY / 'integration.json'
    if integration_path.exists():
        check = json.loads(integration_path.read_text(encoding='utf-8'))
        parts.append('<h2>Vérification avec le lanceur habituel</h2><p>Sur 50 commandes en mode '
                     'rapide, rendu C64 et viewer actifs : ' + str(check['comments']) + ' commentaires, '
                     + str(check['embedding_calls']) + ' vectorisation et '
                     + str(check['video_requests']) + ' demande vidéo. Sortie du programme : '
                     + str(check['exit_code']) + '. Tous les commentaires sont sans raisonnement.</p>'
                     '<p>Les tests automatiques couvrent le seuil, la fin complète des sorties, '
                     'l’utilisation du dernier commentaire après le délai, la conservation des '
                     'historiques au redémarrage et les nouvelles tentatives après erreur.</p>')
    results_path = DIRECTORY / 'word-triggered-results.json'
    measured = json.loads(results_path.read_text(encoding='utf-8')) if results_path.exists() else []
    by_step = {int(row['scene_id'].split('_')[1]): row for row in measured}
    parts.append('<h2>Commentaires sur les buffers complets</h2><p><small>Essais de profilage '
                 'hors lecture vidéo : ces rapprochements ne représentent pas une session de '
                 'diffusion. Aucune comparaison qualitative aveugle n’est menée ici.</small></p>')
    for sample in samples:
        row = by_step.get(sample['step'], {})
        prompt = faketerm.build_prompt(sample['history'])
        if row.get('prompt_sha256') != hashlib.sha256(prompt.encode()).hexdigest():
            row = {}
        parts.append('<section><h3>Étape ' + str(sample['step']) + '</h3><p>'
                     + escape(row.get('commentary', 'Non mesuré')) + '</p><small>'
                     + str(sample['history_words']) + ' mots dans l’historique · '
                     + str(row.get('prompt_tokens', '?')) + ' tokens dans le prompt</small>'
                     '<details><summary>Historique complet</summary><pre>'
                     + escape(sample['history']) + '</pre></details></section>')
    (DIRECTORY / 'report.html').write_text(''.join(parts) + '</html>\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure-tokens", action="store_true")
    args = parser.parse_args()
    rows = json.loads((DIRECTORY / "walkthrough.json").read_text(encoding="utf-8"))
    samples, tail = collect_buffers(rows)
    words = [sample["history_words"] for sample in samples]
    result = {"source_commands": len(rows) - 1, "samples": len(samples),
              "threshold_words": faketerm.COMMENTARY_MIN_WORDS,
              "mean_words": round(statistics.mean(words), 1), "median_words": statistics.median(words),
              "min_words": min(words), "max_words": max(words), "uncommented_tail_words": tail,
              "description": "One captured walkthrough, complete outputs and executed commands. Tail is preserved at runtime across restarts.",
              "prompt_template_sha256": hashlib.sha256(faketerm.build_prompt("").encode()).hexdigest(),
              "token_measurements": []}
    if args.measure_tokens:
        client = ollama.Client(host=faketerm.OLLAMA_HOST, timeout=faketerm.LLM_TIMEOUT)
        for sample in samples:
            prompt = faketerm.build_prompt(sample["history"])
            response = client.chat(model=faketerm.LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                                   think=False, options={"temperature": 0.3, "num_ctx": faketerm.LLM_NUM_CTX,
                                                         "num_predict": 1}, keep_alive="15m")
            measurement = {key: value for key, value in sample.items() if key != "history"}
            measurement.update(prompt_tokens=response.prompt_eval_count, output_tokens=response.eval_count,
                               context_tokens=faketerm.LLM_NUM_CTX)
            result["token_measurements"].append(measurement)
            print(json.dumps(measurement), flush=True)
    else:
        measured = DIRECTORY / "word-triggered-results.json"
        if measured.exists():
            measured = json.loads(measured.read_text(encoding="utf-8"))
            by_step = {int(row["scene_id"].split("_")[1]): row for row in measured}
            for sample in samples:
                row = by_step.get(sample["step"])
                prompt_hash = hashlib.sha256(faketerm.build_prompt(sample["history"]).encode()).hexdigest()
                if row and row.get("prompt_sha256") == prompt_hash:
                    result["token_measurements"].append({"step": sample["step"],
                        "history_words": sample["history_words"], "prompt_tokens": row["prompt_tokens"],
                        "output_tokens": row["output_tokens"], "elapsed_sec": row["elapsed_sec"]})
    if result["token_measurements"]:
        tokens = [row["prompt_tokens"] for row in result["token_measurements"]]
        result.update(mean_prompt_tokens=round(statistics.mean(tokens), 1), max_prompt_tokens=max(tokens),
                      min_prompt_tokens=min(tokens), context_tokens=faketerm.LLM_NUM_CTX)
    (DIRECTORY / "profile.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (DIRECTORY / "word-triggered-samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(result, samples)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
