"""Score the saved resonance experiments without confusing proximity with relevance."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/".tools/presentation-python"))
import numpy as np
import experiment


def unit(v):
    a=np.asarray(v,dtype=float)
    return a/np.linalg.norm(a)


def cached_vector(cache,key):
    return unit(cache.data["records"][key]["embedding"])


def ranks(query_vector,documents,mode,cache):
    best={}
    for doc in documents:
        score=float(np.dot(query_vector,cached_vector(cache,doc["embeddings"][mode])))
        if doc["filename"] not in best or score>best[doc["filename"]]["score"]:
            best[doc["filename"]]={"filename":doc["filename"],"title":doc["title"],"score":score,
                                   "unit_id":doc["id"],"start":doc["start"],"end":doc["end"],"excerpt":doc["text"]}
    ranking=sorted(best.values(),key=lambda r:r["score"],reverse=True)
    # All source text remains in the input files; avoid repeating 25 full transcripts per query.
    for row in ranking[3:]:row.pop("excerpt",None)
    return ranking


def retrieval():
    data=json.loads((HERE/"retrieval-inputs.json").read_text(encoding="utf-8"));cache=experiment.Cache()
    conditions=[]
    for kind in ["full","chunk"]:
        docs=[d for d in data["documents"] if d["kind"]==kind]
        for mode in ["raw","similarity","retrieval"]:
            records=[]
            for q in data["queries"]:
                ranking=ranks(cached_vector(cache,q["embeddings"][mode]),docs,mode,cache)
                record={"id":q["id"],"text":q["text"],"ranking":ranking}
                if "expected" in q:
                    record["expected"]=q["expected"]
                    record["expected_rank"]=next(i+1 for i,r in enumerate(ranking) if r["filename"] in q["expected"])
                records.append(record)
            controls=[r for r in records if "expected_rank" in r]
            original=[r for r in records if isinstance(r["id"],int)]
            full=[d for d in data["documents"] if d["kind"]=="full"]
            vectors=np.array([cached_vector(cache,q["embeddings"][mode]) for q in data["queries"] if isinstance(q["id"],int)])
            documentary=np.array([cached_vector(cache,d["embeddings"][mode]) for d in full])
            item={"id":kind+"_"+mode,"unit_count":len(docs),"records":records,
                  "calibration_top1":sum(r["expected_rank"]==1 for r in controls),
                  "calibration_top3":sum(r["expected_rank"]<=3 for r in controls),
                  "calibration_mrr":statistics.mean(1/r["expected_rank"] for r in controls),
                  "calibration_count":len(controls),
                  "mean_top_score":statistics.mean(r["ranking"][0]["score"] for r in original),
                  "mean_cross_full_clip_cosine":float((vectors@documentary.T).mean()),
                  "top_clip_counts":dict(Counter(r["ranking"][0]["filename"] for r in original))}
            conditions.append(item)
            print({k:v for k,v in item.items() if k not in ["records","top_clip_counts"]},flush=True)
    experiment.save(HERE/"retrieval-evaluation.json",{"conditions":conditions,
        "limits":["Calibration queries are manually authored paraphrases of known clips, not independent relevance labels for game scenes.",
                  "Maximum chunk scoring provides multiple chances per clip. Absolute cosine values and increases are not directly comparable across task prefixes or candidate counts.",
                  "Cross-family geometry is task-dependent, especially with asymmetric retrieval query/document prefixes."]})


def screen():
    data=json.loads((HERE/"screen.json").read_text(encoding="utf-8"));cache=experiment.Cache()
    inputs=json.loads((HERE/"retrieval-inputs.json").read_text(encoding="utf-8"))
    cache.vector_many([experiment.format_text(r["commentary"],mode)
                       for r in data["results"] for mode in ["raw","similarity","retrieval"]])
    evaluated=[]
    for i,r in enumerate(data["results"]):
        versions={}
        for mode in ["raw","similarity","retrieval"]:
            key,vec=cache.vector(experiment.format_text(r["commentary"],mode))
            for kind in ["full","chunk"]:
                docs=[d for d in inputs["documents"] if d["kind"]==kind]
                versions[kind+"_"+mode]={"embedding_key":key,"ranking":ranks(unit(vec),docs,mode,cache)}
        evaluated.append(dict(r,retrieval=versions))
        if (i+1)%12==0:
            experiment.save(HERE/"screen-evaluation.json",{"results":evaluated})
            print("Evaluated %d/%d"%(i+1,len(data["results"])),flush=True)
    summary=[]
    for condition in sorted({r["condition"] for r in evaluated}):
        rows=[r for r in evaluated if r["condition"]==condition]
        for mode in ["raw","similarity","retrieval"]:
            vv=np.array([cached_vector(cache,r["retrieval"]["full_"+mode]["embedding_key"]) for r in rows])
            documentary=np.array([cached_vector(cache,d["embeddings"][mode]) for d in inputs["documents"] if d["kind"]=="full"])
            intra=vv@vv.T
            for kind in ["full","chunk"]:
                matches=[r["retrieval"][kind+"_"+mode]["ranking"][0] for r in rows]
                counts=Counter(r["filename"] for r in matches)
                summary.append({"condition":condition,"embedding":mode,"index":kind,"samples":len(rows),
                                "complete":sum(bool(r["commentary"]) and r["done_reason"]=="stop" for r in rows),
                                "mean_top_cosine":statistics.mean(r["score"] for r in matches),
                                "mean_full_clip_cross_cosine":float((vv@documentary.T).mean()),
                                "mean_comment_intra_cosine":float(intra[np.triu_indices(len(rows),1)].mean()),
                                "top_clip_counts":dict(counts),"distinct_top_clips":len(counts),
                                "largest_clip_share":max(counts.values())/len(rows),
                                "median_inference_sec":statistics.median(r["elapsed_sec"] for r in rows),
                                "max_total_tokens":max(r["prompt_tokens"]+r["output_tokens"] for r in rows)})
    experiment.save(HERE/"screen-evaluation.json",{"results":evaluated,"summary":summary})
    for r in summary:
        if r["embedding"]=="raw" and r["index"]=="full":print(r,flush=True)


def granularity():
    inputs=json.loads((HERE/"retrieval-inputs.json").read_text(encoding="utf-8"))
    extra=json.loads((HERE/"granularity-inputs.json").read_text(encoding="utf-8"))
    cache=experiment.Cache();conditions=[]
    documents=inputs["documents"]+extra["documents"]
    for kind in ["full","chunk","cue40","cue70","cue140","sent1","sent3","sentence70"]:
        docs=[d for d in documents if d["kind"]==kind]
        records=[]
        for q in inputs["queries"]:
            ranking=ranks(cached_vector(cache,q["embeddings"]["raw"]),docs,"raw",cache)
            record={"id":q["id"],"text":q["text"],"ranking":ranking}
            if "expected" in q:
                record["expected"]=q["expected"]
                record["expected_rank"]=next(i+1 for i,r in enumerate(ranking) if r["filename"] in q["expected"])
            if "control_type" in q:record["control_type"]=q["control_type"]
            records.append(record)
        calibration=[r for r in records if "expected_rank" in r]
        original=[r for r in records if isinstance(r["id"],int)]
        counts=Counter(r["ranking"][0]["filename"] for r in original)
        fullrank={r["id"]:r["ranking"][0]["filename"] for r in conditions[0]["records"]} if conditions else {}
        units_per_clip=dict(Counter(d["filename"] for d in docs))
        words=[len(d["text"].split()) for d in docs]
        item={"id":kind,"unit_count":len(docs),"words_min":min(words),"words_median":statistics.median(words),
              "words_max":max(words),"units_per_clip":units_per_clip,"records":records,
              "calibration_top1":sum(r["expected_rank"]==1 for r in calibration),
              "calibration_top3":sum(r["expected_rank"]<=3 for r in calibration),
              "calibration_count":len(calibration),
              "top_clip_counts":dict(counts),"distinct_top_clips":len(counts),
              "mean_top_cosine":statistics.mean(r["ranking"][0]["score"] for r in original),
              "changed_from_full":sum(r["ranking"][0]["filename"]!=fullrank.get(r["id"],r["ranking"][0]["filename"]) for r in original)}
        conditions.append(item)
        print({k:v for k,v in item.items() if k not in ["records","units_per_clip","top_clip_counts"]},flush=True)
    experiment.save(HERE/"granularity-evaluation.json",{"conditions":conditions,
        "limits":["Raw embeddings only: each clip is scored by its single best passage, never by an averaged vector.",
                  "More units give more opportunities for accidental matches. Scores are not confidence probabilities.",
                  "The 8 labelled queries test known-subject lookup; no gold clip labels exist for the 55 game comments.",
                  "Vague-reason and other controls are synthetic, not quotations attributed to Amy Briggs."]})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--stage",choices=["retrieval","screen","granularity"],required=True)
    args=p.parse_args()
    if args.stage=="retrieval":retrieval()
    elif args.stage=="screen":screen()
    else:granularity()


if __name__=="__main__":main()
