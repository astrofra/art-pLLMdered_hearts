"""Test cluster-first passage retrieval offline; no new model calls or runtime changes."""

from collections import Counter
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/".tools/projection-python"))
import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import experiment


def normalise(rows):
    values=np.asarray(rows,dtype=float)
    return values/np.linalg.norm(values,axis=1,keepdims=True)


def main():
    catalogue=json.loads((ROOT/"assets/abriggs-itw-passages.json").read_text(encoding="utf-8"))
    docs=[dict(p,filename=clip["filename"],title=clip["sequence_title"])
          for clip in catalogue for p in clip["passages"]]
    inputs=json.loads((HERE/"retrieval-inputs.json").read_text(encoding="utf-8"))
    cache=experiment.Cache()
    X=normalise([d["embedding"] for d in docs])
    queries=inputs["queries"]
    Q=normalise([cache.data["records"][q["embeddings"]["raw"]]["embedding"] for q in queries])
    scores=Q@X.T
    full_best=np.argmax(scores,axis=1)
    results=[];stability=[]
    for k in [4,6,8,10]:
        partitions=[]
        for seed in [42,43]:
            model=KMeans(n_clusters=k,n_init=20,random_state=seed)
            labels=model.fit_predict(X);partitions.append(labels)
            centres=normalise(model.cluster_centers_)
            cluster_order=np.argsort(-(Q@centres.T),axis=1)
            rows=[]
            for qi,q in enumerate(queries):
                row={"id":q["id"],"global_best_clip":docs[full_best[qi]]["filename"],
                     "global_best_passage":docs[full_best[qi]]["id"],"filters":{}}
                for count in [1,2]:
                    candidates=np.flatnonzero(np.isin(labels,cluster_order[qi,:count]))
                    winner=candidates[np.argmax(scores[qi,candidates])]
                    record={"candidate_count":len(candidates),"filename":docs[winner]["filename"],
                            "passage":docs[winner]["id"],"score":float(scores[qi,winner]),
                            "preserves_global_best_passage":int(full_best[qi]) in candidates,
                            "preserves_global_best_clip":docs[winner]["filename"]==row["global_best_clip"]}
                    if "expected" in q:record["known_subject_correct"]=docs[winner]["filename"] in q["expected"]
                    row["filters"][str(count)]=record
                rows.append(row)
            groups=[]
            for cluster in range(k):
                indices=np.flatnonzero(labels==cluster)
                representative=sorted(indices,key=lambda i:float(np.dot(X[i],centres[cluster])),reverse=True)[:3]
                groups.append({"id":cluster,"passages":len(indices),
                               "clip_counts":dict(Counter(docs[i]["filename"] for i in indices)),
                               "representatives":[{key:value for key,value in docs[i].items() if key!="embedding"} for i in representative]})
            games=[r for r in rows if isinstance(r["id"],int)]
            known=[r for r in rows if "known_subject_correct" in r["filters"]["1"]]
            summary={"clusters":k,"seed":seed,"cosine_silhouette":float(silhouette_score(X,labels,metric="cosine")),
                     "smallest_cluster":min(g["passages"] for g in groups),"largest_cluster":max(g["passages"] for g in groups)}
            for count in [1,2]:
                summary["top%d"%count]={"same_best_clip":sum(r["filters"][str(count)]["preserves_global_best_clip"] for r in games),
                    "retains_best_passage":sum(r["filters"][str(count)]["preserves_global_best_passage"] for r in games),
                    "game_queries":len(games),"known_subject_top1":sum(r["filters"][str(count)]["known_subject_correct"] for r in known),
                    "known_queries":len(known),"coffer":next(r["filters"][str(count)]["filename"] for r in games if r["id"]==2)}
            results.append({"summary":summary,"groups":groups,"queries":rows})
            print(summary,flush=True)
        stability.append({"clusters":k,"adjusted_rand_between_seeds":float(adjusted_rand_score(*partitions))})
    experiment.save(HERE/"clusters.json",{"method":"Euclidean KMeans on L2-normalised 768D embeddings; routing by cosine to normalised centroids; max individual passage within admitted clusters.",
        "sklearn_version":sklearn.__version__,"n_init":20,"passages":len(docs),"clips":len(catalogue),
        "results":results,"stability":stability,
        "limits":["No new generation or embedding calls. No cluster filter enabled in production.",
                  "Preserving the global nearest passage is a retrieval diagnostic, not an artistic relevance label.",
                  "Passages overlap; their count does not represent independent observations.",
                  "Silhouette and seed agreement measure geometric separation and partition stability, not semantic correctness.",
                  "With 132 candidates an exhaustive cosine search is inexpensive; approximate routing must justify any lost matches."],
        "reference":"https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html"})


if __name__=="__main__":main()
