"""Build a standalone comparison and a grouped experimental passage catalogue.

Uses numpy only for analysis, from the isolated presentation tools directory.
The installation and its default catalogue are not modified.
"""

import csv
import json
from pathlib import Path

import analyse
import experiment

HERE=Path(__file__).resolve().parent


def read(name):
    return json.loads((HERE/name).read_text(encoding="utf-8"))


def export_catalogue(cache,documents,videos):
    original=read("../../assets/abriggs-itw-embeddings.json")
    by_name={v["filename"]:v for v in original}
    clips=[]
    for video in videos:
        passages=[]
        for doc in documents:
            if doc["kind"]!="sentence70" or doc["filename"]!=video["filename"]:continue
            passages.append({"id":doc["id"],"text":doc["text"],"start":doc["start"],"end":doc["end"],
                             "embedding":cache.data["records"][doc["embeddings"]["raw"]]["embedding"]})
        clips.append({"filename":video["filename"],"sequence_title":video["title"],
                      "duration_sec":by_name[video["filename"]]["duration_sec"],"passages":passages})
    environment=read("environment.json")
    digest=next(m["digest"] for m in environment["models"] if m["name"]==experiment.EMBED_MODEL)
    experiment.save(HERE/"catalogue-sentence70.json",{
        "experimental":True,"schema":"grouped-passage-v1","model":experiment.EMBED_MODEL,"model_digest":digest,
        "embedding_format":"raw","embedding_options":cache.data["options"],"scoring":"maximum passage cosine per clip",
        "playback":"Select the whole clip; passage timecodes are evidence, not new video cuts.",
        "warning":"Not compatible with the current flat production loader. Keep clip recency and duration at clip level when integrating.",
        "clips":clips})


def main():
    granularity=read("granularity-evaluation.json")
    retrieval=read("retrieval-evaluation.json")
    screen=read("screen-evaluation.json")
    inputs=read("retrieval-inputs.json")
    documents=inputs["documents"]+read("granularity-inputs.json")["documents"]
    cache=experiment.Cache();videos,old_comments=experiment.source_data()
    export_catalogue(cache,documents,videos)
    names={v["filename"]:v["title"] for v in videos}
    queries=[]
    for q in inputs["queries"]:
        rankings={c["id"]:next(r["ranking"][:3] for r in c["records"] if r["id"]==q["id"])
                  for c in granularity["conditions"]}
        history=old_comments[q["id"]-1]["game_history"] if isinstance(q["id"],int) else None
        queries.append({"id":q["id"],"text":q["text"],"history":history,"expected":q.get("expected"),
                        "control_type":q.get("control_type"),"rankings":rankings})
    generated=[]
    passage_docs=[d for d in documents if d["kind"]=="sentence70"]
    for row in screen["results"]:
        key=row["retrieval"]["full_raw"]["embedding_key"]
        ranking=analyse.ranks(analyse.cached_vector(cache,key),passage_docs,"raw",cache)
        generated.append({k:row[k] for k in ["condition","scene","seed","history","prompt","recent_comments",
                                             "commentary","words","elapsed_sec","prompt_tokens","output_tokens","done_reason"]})
        generated[-1]["match_full"]=row["retrieval"]["full_raw"]["ranking"][:3]
        generated[-1]["match_sentence70"]=ranking[:3]
    prompt_summary=[]
    for condition in ["production","spoken","resonant","testimony","companion"]:
        summary=next(r for r in screen["summary"] if r["condition"]==condition and r["embedding"]=="raw" and r["index"]=="full")
        rows=[r for r in generated if r["condition"]==condition]
        prompt_summary.append(dict(summary,wonder_count=sum("wonder" in r["commentary"].lower() for r in rows)))
    payload={"queries":queries,"generated":generated,"names":names,
             "granularity":[{k:v for k,v in c.items() if k!="records"} for c in granularity["conditions"]],
             "retrieval":[{k:v for k,v in c.items() if k!="records"} for c in retrieval["conditions"]],
             "prompt_summary":prompt_summary,"review":read("review.json"),"clusters":read("clusters.json")}
    encoded=json.dumps(payload,ensure_ascii=False).replace("<","\\u003c")
    template=(HERE/"report-template.html").read_text(encoding="utf-8")
    (HERE/"comparison.html").write_text(template.replace("REPORT_JSON",encoded),encoding="utf-8")
    with (HERE/"matches.csv").open("w",encoding="utf-8-sig",newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow(["index","query_id","query","rank","clip","score","passage_start","passage_end","passage"])
        for q in queries:
            for kind,ranking in q["rankings"].items():
                for rank,r in enumerate(ranking,1):
                    writer.writerow([kind,q["id"],q["text"],rank,r["filename"],r["score"],r["start"],r["end"],r["excerpt"]])
    experiment.save(HERE/"generated-passage-matches.json",{"results":generated})
    print("Built comparison.html, matches.csv and experimental catalogue (25 clips / 130 passages).")


if __name__=="__main__":main()
