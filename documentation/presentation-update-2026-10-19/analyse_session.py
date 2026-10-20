"""Build auditable session statistics and an embedding projection for the presentation.

Plotting dependencies live only in .tools/presentation-python, outside the installation.
The optional offline embeddings do not send video requests or alter the catalogue.
"""

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / ".tools/presentation-python"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ollama
import embed_vtt


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_vectors(session):
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    catalog = json.loads((session / "inputs/abriggs-itw-embeddings.json").read_text(encoding="utf-8"))
    comments = load_rows(session / "commentary.jsonl")
    settings = manifest["settings"]
    path = HERE / "offline-embeddings.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {
        "model": settings["VIDEO_EMBED_MODEL"], "options": {"num_gpu": settings["VIDEO_EMBED_NUM_GPU"]},
        "scope": "Offline diagnostic of every displayed comment and transcript. These requests never drive playback.",
        "catalog_sha256": hashlib.sha256((session / "inputs/abriggs-itw-embeddings.json").read_bytes()).hexdigest(),
        "commentary_sha256": hashlib.sha256((session / "commentary.jsonl").read_bytes()).hexdigest(), "texts": []}
    assert data["commentary_sha256"] == hashlib.sha256((session / "commentary.jsonl").read_bytes()).hexdigest()
    client = ollama.Client(timeout=60)
    live_vectors = {}
    embedding_text = None
    for event in load_rows(session / "events.jsonl"):
        if event["event"] == "ollama_start" and event["method"] == "embeddings":
            embedding_text = event["request"]["prompt"]
        elif event["event"] == "ollama_response" and event["method"] == "embeddings":
            live_vectors[embedding_text] = event["response"]["embedding"]
    points = []
    for i, item in enumerate(catalog, 1):
        subtitle = ROOT / "godot-viewer/video" / (Path(item["filename"]).stem + ".txt")
        text = embed_vtt.extract_plain_text(str(subtitle))
        points.append({"id": "V%02d" % i, "kind": "video", "filename": item["filename"],
                       "title": item.get("sequence_title"), "text": text, "duration_sec": item.get("duration_sec"),
                       "embedding": item["embedding"]})
    for i, row in enumerate(comments, 1):
        points.append({"id": "C%02d" % i, "kind": "comment", "comment_id": i,
                       "text": row["commentary"], "game_history": row["game_history"]})
    existing = {row["id"]: row for row in data["texts"]}
    for point in points:
        digest = hashlib.sha256(point["text"].encode("utf-8")).hexdigest()
        row = existing.get(point["id"])
        if row is None:
            started = time.monotonic()
            response = client.embeddings(model=data["model"], prompt=point["text"], options=data["options"])
            row = {"id": point["id"], "text": point["text"], "text_sha256": digest,
                   "elapsed_sec": time.monotonic() - started, "response": dict(response)}
            data["texts"].append(row)
            save(path, data)
            print("Embedded " + point["id"], flush=True)
        assert row["text_sha256"] == digest
        vector = row["response"]["embedding"]
        if point["kind"] == "video":
            original = np.asarray(point["embedding"], dtype=float)
            computed = np.asarray(vector, dtype=float)
            point["catalogue_verification_cosine"] = float(np.dot(original, computed) / (np.linalg.norm(original) * np.linalg.norm(computed)))
        else:
            point["embedding"] = live_vectors.get(point["text"], vector)
            point["vector_source"] = "recorded live embedding" if point["text"] in live_vectors else "offline diagnostic"
    return points


def analyse(session, points):
    events = load_rows(session / "events.jsonl")
    comments = load_rows(session / "commentary.jsonl")
    commands = [row["command"] for row in events if row["event"] == "command" and row["command"].strip()]
    requests = [row for row in events if row["event"] == "video_request"]
    responses = [row for row in events if row["event"] == "ollama_response" and row["method"] == "chat"]
    gpu = [float(row["gpu_csv"].split(",")[2]) for row in events if row["event"] == "resources" and row.get("gpu_csv")]
    catalogue_verification = [p["catalogue_verification_cosine"] for p in points if p["kind"] == "video"]
    offline = json.loads((HERE / "offline-embeddings.json").read_text(encoding="utf-8"))
    reembedded = np.asarray([row["response"]["embedding"] for row in offline["texts"] if row["id"].startswith("V")], dtype=float)
    reembedded /= np.linalg.norm(reembedded, axis=1, keepdims=True)
    # All points are L2 normalised before fitting one shared, centred PCA.
    vectors = np.asarray([point["embedding"] for point in points], dtype=float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    mean = vectors.mean(axis=0)
    centered = vectors - mean
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    for j in range(2):
        if vt[j, np.argmax(np.abs(vt[j]))] < 0:
            coords[:, j] *= -1
            vt[j] *= -1
    explained = singular[:2] ** 2 / np.sum(singular ** 2)
    for i, point in enumerate(points):
        point["x"], point["y"] = map(float, coords[i])
        if point["kind"] == "comment":
            scores = vectors[:25] @ vectors[i]
            point["raw_ranking"] = [{"id": points[int(j)]["id"], "filename": points[int(j)]["filename"],
                                     "title": points[int(j)]["title"], "cosine": float(scores[j])}
                                    for j in np.argsort(-scores, kind="stable")]
    projection = {"method": "PCA by SVD, jointly fitted to L2-normalised vectors; deterministic component signs",
                  "dimensions": int(vectors.shape[1]), "explained_variance_ratio": explained.tolist(),
                  "retained_variance_percent": float(100 * sum(explained)),
                  "mean": mean.tolist(), "components": vt[:2].tolist(), "points": points,
                  "actual_video_requests": requests,
                  "live_rankings": [row for row in events if row["event"] == "video_ranking"],
                  "caveat": "Selection uses cosine similarity in the full embedding space, not distances on this 2D map. Offline rankings ignore recency and playback timing."}
    save(HERE / "projection.json", projection)
    stats = {"scope": "One actual, unseeded Frotz walkthrough; console and fast output; no Godot or C64 rendering.",
             "date": "2026-10-19", "game_commands": len(commands),
             "pagination_acknowledgements": sum(row["event"] == "pager_ack" or (row["event"] == "command" and not row["command"].strip()) for row in events),
             "raw_counter_caveat": "Schema 1 commands_sent includes blank pagination; game_commands here filters them out. Recorder schema 2 separates them.",
             "game_outputs": sum(row["event"] == "game_output" for row in events),
             "comments": len(comments), "successful_comments": sum(bool(row["commentary"]) for row in comments),
             "exit_code": events[-1]["exit_code"], "elapsed_sec": events[-1]["elapsed_sec"],
             "median_inference_sec": statistics.median(row["elapsed_sec"] for row in comments),
             "median_inference_excluding_first_sec": statistics.median(row["elapsed_sec"] for row in comments[1:]),
             "first_inference_sec": comments[0]["elapsed_sec"],
             "inference_range_sec": [min(row["elapsed_sec"] for row in comments), max(row["elapsed_sec"] for row in comments)],
             "history_words": {"min": min(row["history_words"] for row in comments),
                               "median": statistics.median(row["history_words"] for row in comments),
                               "max": max(row["history_words"] for row in comments)},
             "max_prompt_tokens": max(row["prompt_tokens"] for row in comments),
             "max_prompt_plus_output_tokens": max(row["prompt_tokens"] + row["output_tokens"] for row in comments),
             "output_token_definition": "Ollama eval_count includes reasoning and visible output.",
             "done_reasons": dict(Counter(row.get("done_reason") for row in comments)),
             "nonempty_error_records": sum(bool(row.get("error")) for row in comments),
             "max_comment_words": max(len(row["commentary"].split()) for row in comments),
             "sampled_gpu_peak_mib": max(gpu), "resource_sampling_interval_sec": 5,
             "video_requests": requests, "distinct_requested_videos": len(set(row["filename"] for row in requests)),
             "comments_started_during_estimated_video": sum(row["estimated_video_remaining_sec"] > 0 for row in events if row["event"] == "comment_start"),
             "pending_comments_replaced": sum(row["event"] == "pending_comment_replaced" for row in events),
             "catalogue_entries": len(catalogue_verification), "catalogue_reembedding_min_cosine": min(catalogue_verification),
             "catalogue_reembedding_self_nearest": int(np.sum(np.argmax(reembedded @ vectors[:25].T, axis=1) == np.arange(25))),
             "catalogue_reembedding_note": "Recomputed vectors are close, not bit-identical; projection and live rankings retain the original stored catalogue vectors.",
             "projection_retained_variance_percent": projection["retained_variance_percent"],
             "chat_timing_ns": [{"comment_id": row["comment_id"], **{key: row["response"].get(key) for key in
                                   ["total_duration", "load_duration", "prompt_eval_duration", "eval_duration", "prompt_eval_count", "eval_count"]}}
                                  for row in responses]}
    save(HERE / "session-summary.json", stats)
    # Verify the recorded execution against the actual walkthrough, including all pagers.
    sys.path.insert(0, str(ROOT / "src"))
    import faketerm
    assert commands == [faketerm.enhance_game_command(c) for c in faketerm.plundered_hearts_commands]
    assert stats["exit_code"] == 0 and stats["successful_comments"] == 55
    assert stats["nonempty_error_records"] == 0 and stats["max_prompt_plus_output_tokens"] < 4096
    for i, row in enumerate(comments):
        assert row["recent_comments"] == [x["commentary"] for x in comments[max(0, i-2):i]]
        assert row["done_reason"] == "stop" and row["history_words"] > 150
    assert all(np.isfinite(vectors).flat) and len({len(point["embedding"]) for point in points}) == 1
    return projection


def plot(projection):
    points = projection["points"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(12.8, 6.6), layout="constrained")
    fig.set_facecolor("#f7f7f7")
    ax.set_facecolor("#f7f7f7")
    colors = {"video": "#277da8", "comment": "#bd4966"}
    labelled_comments = {r["source_comment_id"] for r in projection["actual_video_requests"]}
    labelled_videos = {r["filename"] for r in projection["actual_video_requests"]}
    for kind, label, marker in [("video", "Interview transcripts (25)", "s"), ("comment", "Gemma comments (55)", "o")]:
        group = [p for p in points if p["kind"] == kind]
        ax.scatter([p["x"] for p in group], [p["y"] for p in group], c=colors[kind], label=label,
                   marker=marker, s=66 if kind == "video" else 34, alpha=.9, zorder=3)
        for p in group:
            if p.get("comment_id") in labelled_comments or p.get("filename") in labelled_videos:
                ax.annotate(p["id"], (p["x"], p["y"]), xytext=(6, 6), textcoords="offset points",
                            fontsize=13, color=colors[kind])
    by_comment = {p.get("comment_id"): p for p in points if p["kind"] == "comment"}
    by_file = {p.get("filename"): p for p in points if p["kind"] == "video"}
    for request in projection["actual_video_requests"]:
        a, b = by_comment[request["source_comment_id"]], by_file[request["filename"]]
        ax.annotate("", xy=(b["x"], b["y"]), xytext=(a["x"], a["y"]),
                    arrowprops={"arrowstyle": "->", "color": "#666666", "alpha": .6, "lw": 1.1}, zorder=1)
    ratio = projection["explained_variance_ratio"]
    ax.set_xlabel("Principal component 1 (%.1f%% of variance)" % (100*ratio[0]), fontsize=11)
    ax.set_ylabel("Principal component 2 (%.1f%%)" % (100*ratio[1]), fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
    ax.legend(loc="upper center", framealpha=.9, fontsize=12)
    ax.set_title("One space: interview transcripts and generated comments", loc="left", fontsize=17, pad=18)
    fig.text(.5, -.015, "PCA of L2-normalised %s-D embeddings. Arrows = actual requests; matching uses full-space cosine similarity."
             % projection["dimensions"], ha="center", fontsize=10)
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(HERE / ("embedding-map." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


def interactive(projection):
    # Keep the public viewer lightweight: no vectors, model reasoning or network libraries.
    public = {k: v for k, v in projection.items() if k not in ["points", "mean", "components"]}
    public["points"] = [{k: v for k, v in p.items() if k != "embedding"} for p in projection["points"]]
    payload = json.dumps(public, ensure_ascii=False).replace("</", "<\\/")
    template = r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>pLLMdered_hearts — an inspectable semantic map</title><style>
*{box-sizing:border-box}body{margin:0;background:#f7f7f7;color:#28282c;font:16px system-ui}header{padding:24px 32px 10px}h1{margin:0;font-size:26px}p{line-height:1.5}main{display:grid;grid-template-columns:minmax(500px,2fr) minmax(320px,1fr);gap:20px;padding:10px 32px}svg{background:white;border:1px solid #ddd;width:100%;height:auto}aside{background:white;padding:20px;border:1px solid #ddd;max-height:78vh;overflow:auto}button,select{font:inherit;padding:7px;margin:0 8px 10px 0}circle,rect.point{cursor:pointer}.muted{color:#666;font-size:13px}li{margin:8px 0}pre{white-space:pre-wrap;font:14px/1.5 system-ui}.tag{font-size:12px;pointer-events:none}table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:6px;border-bottom:1px solid #ddd}tr{cursor:pointer}a{color:#277da8}@media(max-width:900px){main{display:block;padding:12px}aside{max-height:none}header{padding:16px}}
</style><header><h1>Where a comment meets an interview</h1><p>Blue squares: 25 interview transcripts. Pink circles: 55 visible Gemma comments. Select a point to inspect its text and exact similarities.</p>
<div><select id="choice" aria-label="Choose text"></select><button id="next">Next comment</button><button id="links">Hide actual requests</button><button id="reset">Reset selection</button></div>
<p class="muted" id="method"></p></header><main><section><svg id="map" viewBox="0 0 960 660" role="img" aria-label="PCA embedding map"></svg><p class="muted">The game history, testimony cards and internal reasoning are not directly embedded. Offline rankings shown here ignore the live cooldown and recent-clip exclusions.</p></section><aside id="detail"><h2>Select a text</h2><p>The nearest point on this projection may differ from the best match in the original space.</p></aside></main>
<script id="data" type="application/json">DATA</script><script>
const D=JSON.parse(document.getElementById('data').textContent),P=D.points,N='http://www.w3.org/2000/svg',S=document.getElementById('map'),Q=document.getElementById('choice');
const ranges=['x','y'].map(k=>[Math.min(...P.map(p=>p[k])),Math.max(...P.map(p=>p[k]))]);
const X=p=>55+(p.x-ranges[0][0])/(ranges[0][1]-ranges[0][0])*850,Y=p=>600-(p.y-ranges[1][0])/(ranges[1][1]-ranges[1][0])*535;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function el(tag,attrs){let e=document.createElementNS(N,tag);for(let [k,v]of Object.entries(attrs))e.setAttribute(k,v);return e}
let selected=null,showLinks=true;
function draw(){S.replaceChildren();let defs=el('defs',{}),m=el('marker',{id:'arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'});m.appendChild(el('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#666'}));defs.append(m);S.append(defs);
 if(showLinks)for(let r of D.actual_video_requests){let a=P.find(p=>p.comment_id===r.source_comment_id),b=P.find(p=>p.filename===r.filename);S.append(el('line',{x1:X(a),y1:Y(a),x2:X(b),y2:Y(b),stroke:'#888','stroke-width':1.5,'stroke-dasharray':'5 5','marker-end':'url(#arrow)',opacity:.6}))}
 for(let p of P){let c=p.kind==='video'?'#277da8':'#bd4966',r=p.id===selected?9:6,shape=p.kind==='video'?el('rect',{x:X(p)-r,y:Y(p)-r,width:r*2,height:r*2,class:'point'}):el('circle',{cx:X(p),cy:Y(p),r});shape.setAttribute('fill',c);shape.setAttribute('stroke',p.id===selected?'#111':'white');shape.setAttribute('stroke-width',p.id===selected?3:1);shape.addEventListener('click',()=>choose(p.id));let tip=el('title',{});tip.textContent=p.id+' · '+(p.title||p.text);shape.append(tip);S.append(shape);let label=el('text',{x:X(p)+9,y:Y(p)-7,class:'tag',fill:c});label.textContent=p.id;S.append(label)} }
function choose(id){selected=id;Q.value=id;let p=P.find(p=>p.id===id),box=document.getElementById('detail');let h='<h2>'+esc(p.id)+' · '+esc(p.kind==='video'?p.title:'Gemma comment')+'</h2><pre>'+esc(p.text)+'</pre>';
if(p.kind==='comment'){let actual=D.actual_video_requests.find(r=>r.source_comment_id===p.comment_id),live=D.live_rankings.find(r=>r.source_comment_id===p.comment_id);h+='<p><strong>Live request:</strong> '+(actual?esc(actual.sequence_title):'Not used to select a clip during this run.')+'</p><h3>Original-space ranking</h3><table><tr><th>Clip</th><th>Cosine</th><th>Live filter</th></tr>'+p.raw_ranking.map(r=>{let f=live&&live.ranking.find(x=>x.filename===r.filename),status=!live?'Not evaluated':actual&&actual.filename===r.filename?'Selected':f&&f.is_last?'Previous clip':f&&f.is_recent?'Recent clip':'Eligible';return '<tr data-id="'+r.id+'"><td>'+esc(r.id+' · '+r.title)+'</td><td>'+r.cosine.toFixed(4)+'</td><td>'+status+'</td></tr>'}).join('')+'</table><details><summary>Game history given to Gemma</summary><pre>'+esc(p.game_history)+'</pre></details>'}
else h+='<p>Duration: '+p.duration_sec.toFixed(1)+' s · '+esc(p.filename)+'</p>';
box.innerHTML=h;for(let row of box.querySelectorAll('[data-id]'))row.onclick=()=>choose(row.dataset.id);draw()}
Q.innerHTML='<option value="">Choose a text…</option>'+P.map(p=>'<option value="'+p.id+'">'+esc(p.id+' · '+(p.title||p.text.slice(0,75)))+'</option>').join('');Q.onchange=()=>Q.value&&choose(Q.value);
document.getElementById('links').onclick=()=>{showLinks=!showLinks;document.getElementById('links').textContent=showLinks?'Hide actual requests':'Show actual requests';draw()};document.getElementById('reset').onclick=()=>{selected=null;Q.value='';document.getElementById('detail').innerHTML='<h2>Select a text</h2><p>The nearest point on this projection may differ from the best match in the original space.</p>';draw()};
document.getElementById('next').onclick=()=>{let p=P.find(p=>p.id===selected),n=p&&p.comment_id?p.comment_id%55+1:1;choose('C'+String(n).padStart(2,'0'))};
document.getElementById('method').textContent='Joint PCA of normalised '+D.dimensions+'-D vectors; two axes retain '+D.retained_variance_percent.toFixed(1)+'% of variance. Scores are calculated in '+D.dimensions+' dimensions, not on this map.';choose('C01');
</script></html>'''
    (HERE / "embedding-map.html").write_text(template.replace("DATA", payload), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, default=HERE / "session-headless")
    args = parser.parse_args()
    projection = analyse(args.session, collect_vectors(args.session))
    plot(projection)
    interactive(projection)
    print("Projection saved: %.2f%% retained variance" % projection["retained_variance_percent"], flush=True)


if __name__ == "__main__":
    main()
