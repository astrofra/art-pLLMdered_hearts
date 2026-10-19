"""Build a standalone viewer for the saved sampling experiments."""

import json
from pathlib import Path
import statistics

DIRECTORY = Path(__file__).resolve().parent


def compact(row):
    keys = ("condition", "case_id", "seed", "commentary", "elapsed_sec", "words",
            "prompt_tokens", "output_tokens", "done_reason", "error")
    return {key: row.get(key) for key in keys}


def summary(data):
    rows = data["results"]
    return {
        "condition": data["protocol"]["condition"], "responses": len(rows),
        "complete": sum(bool(row["commentary"]) and row.get("done_reason") == "stop" for row in rows),
        "median_sec": round(statistics.median(row["elapsed_sec"] for row in rows), 2) if rows else None,
        "max_prompt_tokens": max((row.get("prompt_tokens", 0) for row in rows), default=0),
        "max_total_tokens": max((row.get("prompt_tokens", 0) + row.get("output_tokens", 0)
                                 for row in rows), default=0),
    }


def main():
    screen = json.loads((DIRECTORY / "screen.json").read_text(encoding="utf-8"))
    sequences = [json.loads(path.read_text(encoding="utf-8"))
                 for path in sorted(DIRECTORY.glob("sequence_*_s42.json"))]
    decision_path = DIRECTORY / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else {}
    payload = {
        "contexts": screen["contexts"], "cases": screen["cases"],
        "screen": [compact(row) for row in screen["results"]],
        "screenSummary": screen["summary"],
        "sequences": [{"condition": data["protocol"]["condition"], "cases": data["cases"],
                       "results": [compact(row) for row in data["results"]], "summary": summary(data)}
                      for data in sequences],
        "decision": decision,
    }
    template = """<!doctype html>
<html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Températures et fiches — pLLMdered Hearts</title>
<style>
body{font:16px/1.55 system-ui;max-width:1500px;margin:2rem auto;padding:0 1rem;color:#242424;background:#fafafa}
h1,h2,h3{line-height:1.2}table{border-collapse:collapse;background:white}th,td{border:1px solid #ddd;padding:.5rem;text-align:left}
.answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem;margin:1rem 0}
article{border:1px solid #ccc;padding:1rem;background:white;border-radius:8px}article h3{margin-top:0}
select{font:inherit;padding:.4rem;margin:.3rem}pre{white-space:pre-wrap;background:#eee;padding:1rem}
small{color:#555}details{margin:.7rem 0}summary{cursor:pointer}.decision{border-left:4px solid #315d81;padding:1rem;background:#eaf2f8}
</style>
<h1>Températures et formats du témoignage</h1>
<p>Gemma E4B, raisonnement actif, contexte de 4 096 tokens et plafond de 2 560 tokens.
Le prompt et les paramètres du modèle restent identiques, sauf la température et le contexte documentaire.
Les textes ci-dessous sont les réponses réellement enregistrées, sans réécriture.</p>
<div id="decision" class="decision"></div>
<h2>Protocole</h2>
<p>Premier balayage : quatre contextes × trois températures × quatre scènes × deux graines, soit 96 réponses.
Les deux commentaires précédents sont identiques pour une scène donnée. La prose courte et les fiches détaillées
contiennent les mêmes phrases factuelles dans le même ordre ; les fiches compactes retirent des détails.</p>
<p>Validation : quatre candidats sur les 55 buffers du parcours capturé, puis une vérification supplémentaire des fiches détaillées à 0,3.
Chaque candidat reçoit ses propres deux derniers commentaires. L'ordre des quatre premières conditions tourne à chaque buffer ;
le cinquième rejeu est réalisé ensuite. Aucun fichier de demande n'est envoyé au viewer.</p>
<p>La sélection repose sur une lecture qualitative par l'assistant, sans panel humain indépendant.
Les durées et compteurs ne constituent pas des scores de qualité. Les récurrences thématiques doivent être jugées dans leur contexte.</p>
<p><a href="screen.json">Balayage complet et prompts</a> · <a href="screen-review.json">Lecture du balayage</a> ·
<a href="review-rubric.json">Critères</a> · <a href="environment.json">Environnement</a> · <a href="decision.json">Décision</a></p>
<h2>Contexte documentaire</h2><select id="context"></select><pre id="contextText"></pre>
<h2>Comparer les 96 réponses</h2>
<label>Scène <select id="scene"></select></label><label>Graine <select id="seed"><option>42</option><option>43</option></select></label>
<details><summary>Historique du jeu et mémoire identique</summary><pre id="screenHistory"></pre></details>
<div id="screenAnswers" class="answers"></div>
<h2>Validation sur le parcours</h2><div id="sequenceSummary"></div>
<label>Buffer <select id="buffer"></select></label>
<details open><summary>Historique du jeu</summary><pre id="sequenceHistory"></pre></details>
<div id="sequenceAnswers" class="answers"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);
const byId=id=>document.getElementById(id);
const names={original:'Récit actuel',prose:'Prose factuelle',cards:'Fiches détaillées',brief:'Fiches compactes'};
function addOption(select,value,label){const o=document.createElement('option');o.value=value;o.textContent=label;select.append(o)}
function answer(row){
  const article=document.createElement('article'),title=document.createElement('h3'),text=document.createElement('p'),metrics=document.createElement('small');
  title.textContent=row.condition; text.textContent=row.commentary||row.error||'Aucune réponse';
  metrics.textContent=`${row.elapsed_sec} s · ${row.words} mots · prompt ${row.prompt_tokens} tokens · génération ${row.output_tokens} tokens · ${row.done_reason}`;
  article.append(title,text,metrics);return article;
}
function updateContext(){const key=byId('context').value;byId('contextText').textContent=data.contexts[key].text}
function updateScreen(){
  const id=byId('scene').value,seed=Number(byId('seed').value),c=data.cases.find(c=>c.id===id),container=byId('screenAnswers');
  byId('screenHistory').textContent=c.game_history+'\\n\\nCOMMENTAIRES PRÉCÉDENTS\\n'+c.recent_comments.join('\\n\\n');container.replaceChildren();
  data.screen.filter(r=>r.case_id===id&&r.seed===seed).sort((a,b)=>a.condition.localeCompare(b.condition)).forEach(r=>container.append(answer(r)));
}
function updateSequence(){
  const index=Number(byId('buffer').value),container=byId('sequenceAnswers');container.replaceChildren();
  byId('sequenceHistory').textContent=data.sequences[0]?.cases[index]?.game_history||'';
  for(const sequence of data.sequences){const row=sequence.results[index];if(row)container.append(answer(row))}
}
for(const [key,context] of Object.entries(data.contexts))addOption(byId('context'),key,`${names[key]} — ${context.words} mots`);
for(const c of data.cases)addOption(byId('scene'),c.id,c.id);
if(data.sequences.length){
  data.sequences[0].cases.forEach((c,i)=>addOption(byId('buffer'),i,`${i+1} — ${c.id}`));
  const table=document.createElement('table'),head=document.createElement('tr');
  for(const t of ['Condition','Réponses complètes','Médiane (s)','Prompt maximum','Prompt + sortie maximum']){const th=document.createElement('th');th.textContent=t;head.append(th)}table.append(head);
  for(const sequence of data.sequences){const s=sequence.summary,tr=document.createElement('tr');for(const value of [s.condition,`${s.complete}/${s.responses}`,s.median_sec,s.max_prompt_tokens,s.max_total_tokens]){const td=document.createElement('td');td.textContent=value;tr.append(td)}table.append(tr)}
  byId('sequenceSummary').append(table);
}
byId('decision').textContent=data.decision.summary_fr||'Expérimentation en cours : aucun réglage retenu à ce stade.';
byId('context').onchange=updateContext;byId('scene').onchange=updateScreen;byId('seed').onchange=updateScreen;byId('buffer').onchange=updateSequence;
updateContext();updateScreen();updateSequence();
</script></html>
"""
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    (DIRECTORY / "comparison.html").write_text(template.replace("__DATA__", encoded), encoding="utf-8")
    print("comparison.html written")


if __name__ == "__main__":
    main()
