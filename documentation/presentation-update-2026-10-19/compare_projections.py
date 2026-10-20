"""Compare PCA, t-SNE and UMAP on the same frozen vectors; never calls Ollama.

Analysis dependencies are isolated in .tools/projection-python. Production is unchanged.
"""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
sys.path.insert(0, str(ROOT / ".tools/projection-python"))

import numpy as np
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, trustworthiness
from umap import UMAP


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rank_without_self(distances):
    distances = distances.copy()
    np.fill_diagonal(distances, np.inf)
    return np.argsort(distances, axis=1, kind="stable")


def evaluate(x, coordinates, video, cosine):
    distances = cdist(coordinates, coordinates)
    original = rank_without_self(1-cosine)
    projected = rank_without_self(distances)
    true_cross = np.argsort(-cosine[~video][:, video], axis=1, kind="stable")
    map_cross = np.argsort(distances[~video][:, video], axis=1, kind="stable")
    return {
        "trustworthiness_k5": float(trustworthiness(x, coordinates, n_neighbors=5, metric="cosine")),
        "trustworthiness_k10": float(trustworthiness(x, coordinates, n_neighbors=10, metric="cosine")),
        "neighbour_recall_k5": float(np.mean([len(set(a[:5]) & set(b[:5]))/5 for a,b in zip(original, projected)])),
        "nearest_clip_preserved_count": int(np.sum(true_cross[:,0] == map_cross[:,0])),
        "comments_count": int((~video).sum()),
        "cross_neighbour_recall_k5": float(np.mean([len(set(a[:5]) & set(b[:5]))/5 for a,b in zip(true_cross, map_cross)])),
    }


def generate():
    source_path = HERE / "projection.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    points = source["points"]
    video = np.array([p["kind"] == "video" for p in points])
    x = np.asarray([p["embedding"] for p in points], dtype=np.float64)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    cosine = np.clip(x @ x.T, -1, 1)
    singular = np.linalg.svd(x-x.mean(axis=0), compute_uv=False)
    variance = singular**2 / np.sum(singular**2)
    pca = np.array([[p["x"],p["y"]] for p in points])
    assert np.all(np.isfinite(x)) and x.shape == (80,768)
    data = {
        "input_file": "projection.json", "input_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "normalised_vectors_sha256": hashlib.sha256(x.astype("<f8").tobytes()).hexdigest(),
        "preprocessing": "The same 80 stored 768-D vectors, L2 normalised, jointly fitted; no labels supplied to any method. No new text generation or embedding calls.",
        "packages": {name: importlib.metadata.version(name) for name in ["numpy","scipy","scikit-learn","umap-learn","numba","matplotlib"]},
        "points": [{k:v for k,v in p.items() if k != "embedding"} for p in points],
        "actual_video_requests": source["actual_video_requests"], "live_rankings": source["live_rankings"],
        "cosine_matrix": cosine.tolist(), "pca_variance_ratio": variance[:3].tolist(),
        "method_notes": {
            "PCA": "A linear view maximising retained variance. This 2D view retains 21.5%; three axes retain 25.0%. It does not preserve every global distance.",
            "t-SNE": "Emphasises local neighbourhoods. Gaps, cluster sizes and inter-cluster distances are not similarity scores.",
            "UMAP": "A neighbourhood-graph view; n_neighbors changes the local/broader-scale balance. This is a practical compromise, not guaranteed global geometry.",
        },
        "metric_definitions": {
            "neighbour_recall_k5": "Average fraction of each point's original five nearest neighbours retained in 2D, excluding self, over all 80 points.",
            "trustworthiness": "Scikit-learn trustworthiness with high-dimensional cosine distances; penalises spurious projected neighbours. Not an artistic quality measure.",
            "nearest_clip_preserved_count": "Among the 55 comments, how often the nearest VIDEO point by 2D Euclidean distance equals the top original-space cosine clip. Ignores recency and playback gates.",
            "cross_neighbour_recall_k5": "Average fraction of the original five nearest clips retained among the five nearest clips in 2D, over 55 comments.",
        },
        "references": [
            "https://scikit-learn.org/stable/modules/manifold.html#t-sne",
            "https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html",
            "https://umap-learn.readthedocs.io/en/latest/parameters.html",
            "https://umap-learn.readthedocs.io/en/latest/faq.html",
        ],
        "maps": [],
    }
    # Defaults chosen before inspecting the layouts; variants expose sensitivity rather than select the prettiest map.
    jobs = [("PCA",None,None)] + [(method, scale, seed) for method in ["t-SNE","UMAP"]
            for scale in ([10,15,30] if method == "t-SNE" else [5,15,30]) for seed in [42,43]]
    for method, scale, seed in jobs:
        started = time.monotonic()
        if method == "PCA":
            coordinates = pca
            params = {"n_components":2,"solver":"full SVD","centred":True}
            key = "pca"
        elif method == "t-SNE":
            params = {"n_components":2,"perplexity":scale,"metric":"cosine","init":"pca",
                      "learning_rate":"auto","max_iter":1500,"method":"exact","random_state":seed,"n_jobs":1}
            coordinates = TSNE(**params).fit_transform(x)
            key = "tsne-p%d-s%d" % (scale,seed)
        else:
            params = {"n_components":2,"n_neighbors":scale,"min_dist":0.1,"metric":"cosine",
                      "init":"spectral","random_state":seed,"n_jobs":1,"n_epochs":1000}
            coordinates = UMAP(**params).fit_transform(x)
            key = "umap-n%d-s%d" % (scale,seed)
        assert coordinates.shape == (80,2) and np.all(np.isfinite(coordinates))
        item = {"id":key,"method":method,"parameters":params,"coordinates":coordinates.tolist(),
                "metrics":evaluate(x, coordinates, video, cosine),"elapsed_sec":time.monotonic()-started}
        data["maps"].append(item)
        print(json.dumps({k:v for k,v in item.items() if k not in ["coordinates"]}), flush=True)
        save(HERE/"projection-comparison.json", data)
    assert len(data["maps"]) == 13
    return data


def plot(data):
    points = data["points"]
    video = np.array([p["kind"] == "video" for p in points])
    chosen = [next(m for m in data["maps"] if m["id"] == key) for key in ["pca","tsne-p15-s42","umap-n15-s42"]]
    plt.rcParams.update({"font.family":"DejaVu Sans","svg.fonttype":"none"})
    fig,axes = plt.subplots(1,3,figsize=(16,6.3))
    fig.subplots_adjust(top=.76,bottom=.17,left=.025,right=.99,wspace=.08)
    selected_comments = {data["actual_video_requests"][0]["source_comment_id"]}
    selected_videos = {data["actual_video_requests"][0]["filename"]}
    for ax,item in zip(axes,chosen):
        coords = np.asarray(item["coordinates"])
        for mask,color,marker,label in [(video,"#277da8","s","Interview transcripts"),(~video,"#bd4966","o","Gemma comments")]:
            ax.scatter(coords[mask,0],coords[mask,1],c=color,marker=marker,s=40,label=label,alpha=.9,zorder=3)
        for i,p in enumerate(points):
            if p.get("comment_id") in selected_comments or p.get("filename") in selected_videos:
                ax.annotate(p["id"],coords[i],xytext=(5,5),textcoords="offset points",fontsize=9)
        for request in data["actual_video_requests"]:
            i=next(i for i,p in enumerate(points) if p.get("comment_id")==request["source_comment_id"])
            j=next(i for i,p in enumerate(points) if p.get("filename")==request["filename"])
            ax.annotate("",xy=coords[j],xytext=coords[i],arrowprops={"arrowstyle":"->","lw":.8,"alpha":.4,"color":"#777"})
        metrics=item["metrics"]
        ax.set_title(item["method"]+"\n"+{"PCA":"Linear variance view","t-SNE":"Local neighbourhoods","UMAP":"Neighbourhood graph"}[item["method"]],fontsize=15)
        ax.set_aspect("equal",adjustable="datalim")
        ax.set_xticks([]);ax.set_yticks([])
        ax.spines[["top","right"]].set_visible(False)
        ax.set_xlabel("Original 5-neighbour overlap: %.1f%%\nNearest clip retained: %d / 55" %
                      (100*metrics["neighbour_recall_k5"],metrics["nearest_clip_preserved_count"]),fontsize=11)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="upper center",bbox_to_anchor=(.5,.91),ncol=2,frameon=False,fontsize=12)
    fig.suptitle("The same 80 vectors, three 2D layouts",fontsize=20,y=.98)
    for ext in ["png","svg","pdf"]:
        fig.savefig(HERE/("projection-comparison."+ext),dpi=200,bbox_inches="tight")
    plt.close(fig)
    # A full-space matrix complements the maps without any dimensionality reduction.
    cosine=np.array(data["cosine_matrix"])
    np.fill_diagonal(cosine,np.nan)
    fig,ax=plt.subplots(figsize=(9,8),layout="constrained")
    image=ax.imshow(cosine,vmin=-.1,vmax=.8,cmap="viridis",interpolation="nearest")
    ax.axhline(24.5,color="white",lw=1.5);ax.axvline(24.5,color="white",lw=1.5)
    ax.set_xticks([12,52],labels=["25 transcripts","55 comments"])
    ax.set_yticks([12,52],labels=["25 transcripts","55 comments"])
    ax.set_title("Actual cosine similarities in 768 dimensions",fontsize=17,pad=16)
    fig.colorbar(image,ax=ax,label="Cosine similarity (self-pairs omitted)")
    for ext in ["png","svg","pdf"]:
        fig.savefig(HERE/("cosine-matrix."+ext),dpi=200,bbox_inches="tight")
    plt.close(fig)


def interactive(data):
    payload=json.dumps(data,ensure_ascii=False).replace("</","<\\/")
    template=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Gemma semantic maps — PCA, t-SNE, UMAP</title><style>
*{box-sizing:border-box}body{margin:0;background:#f7f7f7;color:#28282c;font:16px/1.5 system-ui}header,main,footer{padding:20px 30px}h1{font-size:30px;margin:0 0 10px}h2{font-size:21px;margin:0}h3{font-size:17px;margin:12px 0}p{margin:8px 0}button,select{font:inherit;padding:6px;margin:5px 10px 5px 0}#choice{width:650px;max-width:100%}#plots{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.panel,aside,.evidence{background:white;border:1px solid #ddd;padding:16px}.panel svg{width:100%;height:auto;display:block;background:#fff}.panel .note{min-height:70px}.muted{font-size:13px;color:#666}.blue{color:#277da8}.pink{color:#bd4966}.controls{margin:12px 0}.point{cursor:pointer}.tag{font-size:13px;pointer-events:none}.metrics{font-size:13px;margin:10px 0}.metrics b{font-size:15px}#detail{margin-top:20px;display:grid;grid-template-columns:1fr 1fr;gap:30px}pre{white-space:pre-wrap;font:15px/1.5 system-ui}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:6px;border-bottom:1px solid #ddd}tr[data-id]{cursor:pointer}tr.selected{background:#f8dde6}a{color:#216b91}.evidence{margin-top:20px;display:grid;grid-template-columns:1fr 1fr;gap:25px}.evidence img{width:100%;max-width:520px}details{margin:10px 0}code{font-size:13px}#selection-note{background:#f8dde6;padding:10px}footer{font-size:13px;color:#666}.legend{margin:8px 0 14px}@media(max-width:1100px){#plots{grid-template-columns:1fr}#detail,.evidence{grid-template-columns:1fr}header,main,footer{padding:16px}.panel svg{max-height:65vh}}
</style><header><h1>One set of texts. Three different views.</h1>
<p><span class="blue">■ 25 interview transcripts</span> · <span class="pink">● 55 Gemma comments</span> · All use the same 768-dimensional embedding space.</p>
<p>Choose a text: the same point is highlighted in every map. The original-space ranking stays unchanged.</p>
<div class="controls"><select id="choice" aria-label="Select text"></select><button id="next">Next comment</button><button id="arrows">Show actual clip requests</button><label><input type="checkbox" id="labels"> All labels</label><label>View <select id="focus"><option value="all">All texts</option><option value="family">Selected text family</option></select></label></div>
<p id="selection-note"></p><p class="muted">Coordinates, orientations and scales belong to each method. These drawings do not calculate the clip selection. No label was used to fit them. Family focus only zooms the existing layout; it never refits the data.</p></header>
<main><section id="plots"></section><aside id="detail"></aside>
<section class="evidence"><div><h2>The separation is already present in 768D</h2><p>Average pairwise cosine similarity, excluding self-pairs:</p><table><tr><th>Text pairs</th><th>Mean</th></tr><tr><td>Comment / comment</td><td>0.325</td></tr><tr><td>Transcript / transcript</td><td>0.439</td></tr><tr><td>Comment / transcript</td><td>0.126</td></tr></table><p>This supports your impression of greater within-group similarity. It does not identify the cause: shared subjects, style, length and vocabulary may contribute.</p><p>A PCA in 3D would retain 25.0% of variance, versus 21.5% in 2D. A third axis would therefore add relatively little here.</p><p><strong>The current selector still chooses the closest eligible transcript even when that resemblance is weak.</strong> A cosine is not a probability or an artistic quality score.</p></div><div><img src="cosine-matrix.png" alt="Original-space cosine similarity matrix"><p class="muted">Matrix values use the full vectors; no PCA, t-SNE or UMAP. White diagonals omit self-similarity.</p></div></section>
<details><summary>How to read the diagnostic numbers</summary><p><b>5-neighbour overlap:</b> the fraction of original nearest neighbours retained by each map, averaged over all 80 points. <b>Nearest clip retained:</b> among 55 comments, how often the visually nearest video point is the highest-scoring transcript in 768D. Neither metric rates artistic relevance. Recency and playback gates are excluded from these two diagnostics.</p><p>Parameter controls expose sensitivity. Defaults were selected before viewing the layouts: t-SNE perplexity 15; UMAP 15 neighbours, min_dist 0.1; seed 42. Smaller scales emphasise smaller neighbourhoods. The 13 recorded fits cover three scales and two seeds for each non-linear method, plus PCA.</p></details></main>
<footer><p>Methods: <a href="https://scikit-learn.org/stable/modules/manifold.html#t-sne">scikit-learn manifold guide</a> · <a href="https://umap-learn.readthedocs.io/en/latest/parameters.html">UMAP parameters</a> · <a href="https://umap-learn.readthedocs.io/en/latest/faq.html">UMAP caveats</a>. Static maps, full coordinates, parameters and package versions accompany this page in the project documentation.</p></footer>
<script id="data" type="application/json">DATA</script><script>
const D=JSON.parse(document.getElementById('data').textContent),P=D.points,N='http://www.w3.org/2000/svg';let selected='C01',arrows=false;
const E=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const methods=['PCA','t-SNE','UMAP'],active={'PCA':'pca','t-SNE':'tsne-p15-s42','UMAP':'umap-n15-s42'};
const slug=s=>s==='PCA'?'pca':s==='t-SNE'?'tsne':'umap';
function node(tag,attrs){let n=document.createElementNS(N,tag);for(let[k,v]of Object.entries(attrs))n.setAttribute(k,v);return n}
for(let method of methods){let panel=document.createElement('article');panel.className='panel';panel.dataset.method=method;panel.innerHTML='<h2>'+method+'</h2><p class="note muted">'+E(D.method_notes[method])+'</p>'+(method==='PCA'?'<p class="muted">Centred PCA · same normalised vectors</p>':'<label class="muted">'+(method==='t-SNE'?'Perplexity':'Neighbours')+' <select class="scale">'+(method==='t-SNE'?[10,15,30]:[5,15,30]).map(n=>'<option '+(n===15?'selected':'')+'>'+n+'</option>').join('')+'</select></label><label class="muted">Seed <select class="seed"><option>42</option><option>43</option></select></label>')+'<svg viewBox="0 0 480 430" role="img" aria-label="'+method+' embedding map"></svg><div class="metrics"></div>';document.getElementById('plots').append(panel);if(method!=='PCA')for(let c of panel.querySelectorAll('select'))c.onchange=()=>{let scale=panel.querySelector('.scale').value,seed=panel.querySelector('.seed').value;active[method]=(method==='t-SNE'?'tsne-p':'umap-n')+scale+'-s'+seed;draw()}}
function draw(){const pick=P.findIndex(p=>p.id===selected);for(let panel of document.querySelectorAll('.panel')){let method=panel.dataset.method,map=D.maps.find(m=>m.id===active[method]),coords=map.coordinates,S=panel.querySelector('svg');S.replaceChildren();let fitted=coords.filter((c,i)=>document.getElementById('focus').value==='all'||P[i].kind===P[pick].kind),xmin=Math.min(...fitted.map(p=>p[0])),xmax=Math.max(...fitted.map(p=>p[0])),ymin=Math.min(...fitted.map(p=>p[1])),ymax=Math.max(...fitted.map(p=>p[1])),scale=Math.min(410/(xmax-xmin),350/(ymax-ymin));const X=i=>240+(coords[i][0]-(xmin+xmax)/2)*scale,Y=i=>215-(coords[i][1]-(ymin+ymax)/2)*scale;
let defs=node('defs',{}),marker=node('marker',{id:'arrow-'+slug(method),viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto'});marker.append(node('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#777'}));defs.append(marker);S.append(defs);
if(arrows)for(let r of D.actual_video_requests){let a=P.findIndex(p=>p.comment_id===r.source_comment_id),b=P.findIndex(p=>p.filename===r.filename);S.append(node('line',{class:'request',x1:X(a),y1:Y(a),x2:X(b),y2:Y(b),stroke:'#777','stroke-width':1,'stroke-dasharray':'5 4','marker-end':'url(#arrow-'+slug(method)+')',opacity:.6}))}
const raw=P[pick].raw_ranking,nearest=raw?raw[0].id:null;
const order=P.map((p,i)=>i).filter(i=>i!==pick&&P[i].id!==nearest);if(nearest)order.push(P.findIndex(p=>p.id===nearest));order.push(pick);
for(let i of order){let p=P[i],c=p.kind==='video'?'#277da8':'#bd4966',r=i===pick?8:5,s=p.kind==='video'?node('rect',{x:X(i)-r,y:Y(i)-r,width:2*r,height:2*r}):node('circle',{cx:X(i),cy:Y(i),r});s.setAttribute('class','point');s.setAttribute('data-id',p.id);s.setAttribute('fill',c);s.setAttribute('stroke',i===pick?'#111':p.id===nearest?'#b58a00':'white');s.setAttribute('stroke-width',i===pick||p.id===nearest?2.5:1);s.addEventListener('click',()=>choose(p.id));let t=node('title',{});t.textContent=p.id+' · '+(p.title||p.text);s.append(t);S.append(s);if(i===pick||p.id===nearest||document.getElementById('labels').checked){let t=node('text',{x:X(i)+9,y:Y(i)-8,class:'tag',fill:c});t.textContent=p.id;S.append(t)}}let m=map.metrics;panel.querySelector('.metrics').innerHTML='5-neighbour overlap: <b>'+(100*m.neighbour_recall_k5).toFixed(1)+'%</b><br>Nearest clip retained: <b>'+m.nearest_clip_preserved_count+' / 55</b>';}}
function choose(id){selected=id;document.getElementById('choice').value=id;let p=P.find(p=>p.id===id),detail=document.getElementById('detail');let left='<div><h2>'+E(p.id)+' · '+E(p.title||'Gemma comment')+'</h2><pre>'+E(p.text)+'</pre>';
if(p.game_history)left+='<details><summary>Game passage given to Gemma</summary><pre>'+E(p.game_history)+'</pre></details>';left+='</div>';let right='';
if(p.kind==='comment'){let actual=D.actual_video_requests.find(r=>r.source_comment_id===p.comment_id),live=D.live_rankings.find(r=>r.source_comment_id===p.comment_id);right='<div><h3>Ranking in the original 768D space</h3><p class="muted">This ranking stays identical when you change the projection.</p><table><tr><th>Clip</th><th>Cosine</th><th>Live filter</th></tr>'+p.raw_ranking.map((r,i)=>{let f=live&&live.ranking.find(v=>v.filename===r.filename),status=actual&&actual.filename===r.filename?'Selected':!live?'Not evaluated':f&&f.is_last?'Previous clip':f&&f.is_recent?'Recent clip':'Eligible';return '<tr data-id="'+r.id+'" class="'+(i===0?'selected':'')+'"><td>'+E(r.id+' · '+r.title)+'</td><td>'+r.cosine.toFixed(4)+'</td><td>'+status+'</td></tr>'}).join('')+'</table></div>';document.getElementById('selection-note').textContent=p.id+': black outline = selected comment; gold outline = its highest-scoring transcript in 768D. The maps may place another clip closer.'}
else{right='<div><p>Duration: '+p.duration_sec.toFixed(1)+' s</p><p class="muted">'+E(p.filename)+'</p></div>';document.getElementById('selection-note').textContent=p.id+': the same transcript is outlined in all three views.'}
detail.innerHTML=left+right;for(let row of detail.querySelectorAll('tr[data-id]'))row.onclick=()=>choose(row.dataset.id);draw()}
document.getElementById('choice').innerHTML=P.map(p=>'<option value="'+p.id+'">'+E(p.id+' · '+(p.title||p.text.slice(0,65)))+'</option>').join('');document.getElementById('choice').onchange=e=>choose(e.target.value);
document.getElementById('next').onclick=()=>{let p=P.find(p=>p.id===selected),n=p.comment_id?p.comment_id%55+1:1;choose('C'+String(n).padStart(2,'0'))};document.getElementById('arrows').onclick=()=>{arrows=!arrows;document.getElementById('arrows').textContent=arrows?'Hide actual clip requests':'Show actual clip requests';draw()};document.getElementById('labels').onchange=draw;document.getElementById('focus').onchange=draw;choose(selected);
</script></html>'''
    # Embed the matrix image so the comparison remains a single portable HTML file.
    import base64
    template=template.replace('src="cosine-matrix.png"','src="data:image/png;base64,'+base64.b64encode((HERE/'cosine-matrix.png').read_bytes()).decode()+'"')
    (HERE/"projection-comparison.html").write_text(template.replace("DATA",payload),encoding="utf-8")


def main():
    data=generate()
    plot(data)
    interactive(data)
    print("Saved 13 fits, static figures, cosine matrix and standalone interactive comparison.",flush=True)


if __name__ == "__main__":
    main()
