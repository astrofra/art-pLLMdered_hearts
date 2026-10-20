"""Explore testimony resonance using frozen scenes, cached embeddings and local Gemma.

This experiment does not alter production settings, its catalogue or video requests.
Dependencies: standard library and the project's existing ollama client.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import faketerm
import ollama

EMBED_MODEL = "embeddinggemma:300m"
CHAT_MODEL = "gemma4:e4b"
SOURCE = ROOT / "documentation/presentation-update-2026-10-19"
SCENES = [1,2,8,9,11,16,20,26,31,38,43,55]


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    text=json.dumps(data, indent=2, ensure_ascii=False, default=str)+"\n"
    if path.suffix==".gz":temporary.write_bytes(gzip.compress(text.encode("utf-8"),compresslevel=3,mtime=0))
    else:temporary.write_text(text,encoding="utf-8")
    temporary.replace(path)


def source_data():
    projection = json.loads((SOURCE / "projection.json").read_text(encoding="utf-8"))
    comments = [json.loads(line) for line in (SOURCE / "session-headless/commentary.jsonl").read_text(encoding="utf-8").splitlines()]
    return projection["points"][:25], comments


def prepare():
    videos, comments = source_data()
    # Verbatim source fragments are background, not text to imitate or attribute to the observer.
    fragments = [
        ("05_becoming_an_imp_0.ogv", "I had written a really off-the-wall, bizarre little game"),
        ("08_writing_a_romance_1.ogv", "I was interested in in writing romance. I thought a romance might be kind of fun and different."),
        ("09_writing_a_romance_2.ogv", "It's where does the world that you've invented break down"),
        ("13_critical_response_0.ogv", "it made a difference in their lives and they remembered it"),
        ("16_writing_a_novel_0.ogv", "So I wonder if I really do want to write the novel or if I just want to keep dreaming about writing the novel."),
        ("22_market_vision_2.ogv", "We weren't doing just games. We were doing experiences, environments, and they were more like books."),
    ]
    for filename, quote in fragments:
        text = next(v["text"] for v in videos if v["filename"] == filename)
        assert quote in text, (filename, quote)
    scenes = [{"scene": n, "history": comments[n-1]["game_history"],
               "fixed_recent_comments": comments[n-1]["recent_comments"]} for n in SCENES]
    config = {
        "source_commit": "3475243", "model": CHAT_MODEL,"embedding_model":EMBED_MODEL,
        "context_tokens":4096,"output_tokens":2560,"temperature":0.3,"think":True,"seeds":[42,43],
        "source_commentary_sha256":hashlib.sha256((SOURCE/"session-headless/commentary.jsonl").read_bytes()).hexdigest(),
        "scenes":scenes,"fragments":[{"filename":f,"verbatim":q} for f,q in fragments],
        "conditions":["production","spoken","resonant","testimony","companion"],
        "followup_note":"Companion is a post-hoc prompt revision after reading the first screening responses, run separately; not an independent blind confirmation.",
        "controls":[
            "Same frozen histories and two prior production comments for paired screening; candidates never see a next command.",
            "Each condition sees the same selected scenes and seeds. Order rotates by scene and seed.",
            "Do not equate a higher cosine, smaller centroid gap or greater clip diversity with interpretive quality.",
            "No clip title or retrieval result is supplied to commentary generation. No Amy impersonation or invented memories.",
            "No weights trained and no production configuration changed. Reasoning text counted, not saved.",
        ],
        "references":["https://huggingface.co/google/embeddinggemma-300m#prompt-instructions",
                      "https://docs.ollama.com/api/embed"],
    }
    save(HERE/"protocol.json",config)
    return config


def build_prompt(condition, history, recent, protocol):
    if condition == "production":
        return faketerm.build_prompt(history, recent_comments=recent)
    background = faketerm.load_itw_redux()
    if condition == "testimony":
        background = "\n".join('Archive excerpt (%s): "%s"' % (q["filename"],q["verbatim"]) for q in protocol["fragments"])
    role = """You are a present-day reader watching Plundered Hearts unfold.
You are not Amy Briggs and cannot remember her life or speak on her behalf.
You do not choose moves. Read only the events already shown.
DOCUMENTARY BACKGROUND (evidence about the interview, not an explanation of this scene):
""" + background + "\nEND OF BACKGROUND\n"
    if recent:
        role += "\nYOUR LAST COMMENTS, ideas already used:\n"+"\n".join(recent[-2:])+"\nDo not repeat their point or analogy. They are not evidence.\n"
    role += "\nGAME HISTORY (all events already shown):\n"+history+"\nEND OF GAME HISTORY\n"
    if condition == "companion":
        task = """Say a passing thought to someone watching beside you, not an analysis addressed to an audience.
Find what this particular moment makes you, as a reader, want, expect, doubt, enjoy or care about.
Let one concrete detail carry that feeling. In the second sentence, move gently towards what
making, reading or sharing a story asks of a person. Use the testimony as lived experience,
not a list of subjects to tick off. A small observation is better than an impressive metaphor.
The background offers pleasure in making something, the ordinary things a reader expects,
the wish to be taken seriously, trust between people, and uncertainty about what will last.
Keep the connection implicit when possible; if none is supported, stay with the scene.
You may say I about the act of reading now, but do not borrow anyone else's memories.
Contractions and a qualification are natural. Avoid a reviewer's abstract vocabulary.
STYLE EXAMPLES FROM UNRELATED IMAGINARY SCENES (not game facts; do not copy):
I like that the story pauses over an ordinary cup when everything else seems urgent.
That small bit of care is doing more to make the place feel lived in than a grand description would.
Everyone seems so sure what a proper ending ought to look like, and I'm not sure I want that one.
There is something rather freeing about taking a story seriously without having to make it respectable.
END OF STYLE EXAMPLES
Write your own point, not a paraphrase of these examples. Don't just narrate what happened.
"""
    elif condition == "spoken":
        task = """Notice one fresh, concrete detail from the latest events and deepen the same observation.
Speak conversationally, as someone finding their thought while saying it, rather than
writing an analysis. A small hesitation or qualification is welcome when it means something.
You may say I about your present reading, never about a past event or a designer's experience.
Stay inside this scene: do not append a general lesson or require a historical connection.
"""
    else:
        task = """Speak as a thoughtful reader rather than a plot summariser. Amy's testimony is candid:
pleasure in imagining worlds, uncertainty about what readers will try, taking responsibility
for omissions, being trusted or constrained, and discovering what stays with people.
Use that kind of attention to read this moment. Start from one particular gesture, object,
phrase or consequence already in the scene; find the lived tension it makes you feel.
Develop it in a second sentence: an expectation that shifts, an ordinary detail that matters,
or something a person can and cannot do. Let it sound like a modest thought spoken aloud.
A connection may be implicit. Do not name interview topics just to make a connection.
If no documentary resonance is warranted, a precise uncertainty about this scene is enough.
Do not reduce every moment to restrictions, freedom or the difference between games and books.
You may use I only for your current reading, never for invented personal or historical memories.
"""
    return role+"\n"+task+"""Do not predict events, advise the player, invent motives, or explain why Briggs designed a scene.
No claim of authorial intention or production cause can be inferred from a fictional detail.
Do not add names, hardware, company lore or biographical facts just to sound like the background.
Write TWO sentences, at most 90 words total, in plain conversational English. No Markdown.
"""


class Cache:
    def __init__(self):
        self.client = ollama.Client(host=faketerm.OLLAMA_HOST,timeout=180)
        self.path = HERE / "embedding-cache.json.gz"
        old_path=HERE/"embedding-cache.json"
        if self.path.exists():contents=gzip.decompress(self.path.read_bytes()).decode("utf-8")
        elif old_path.exists():contents=old_path.read_text(encoding="utf-8")
        else:contents=None
        self.data = json.loads(contents) if contents is not None else {
            "model":EMBED_MODEL,"options":{"num_gpu":0,"num_ctx":2048},"records":{}}

    def vector(self, text):
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row = self.data["records"].get(key)
        if row is None:
            started=time.monotonic()
            response=self.client.embed(model=EMBED_MODEL,input=text,truncate=False,
                                       options=self.data["options"],keep_alive="30m")
            row={"text":text,"elapsed_sec":time.monotonic()-started,
                 "prompt_eval_count":response.prompt_eval_count,"embedding":response.embeddings[0],
                 "total_duration":response.total_duration,"load_duration":response.load_duration}
            self.data["records"][key]=row
            save(self.path,self.data)
        return key,row["embedding"]

    def vector_many(self,texts):
        missing={hashlib.sha256(text.encode("utf-8")).hexdigest():text for text in texts
                 if hashlib.sha256(text.encode("utf-8")).hexdigest() not in self.data["records"]}
        items=list(missing.items())
        for offset in range(0,len(items),8):
            batch=items[offset:offset+8];start=time.monotonic()
            response=self.client.embed(model=EMBED_MODEL,input=[text for _,text in batch],truncate=False,
                                       options=self.data["options"],keep_alive="30m")
            for (key,text),vector in zip(batch,response.embeddings):
                self.data["records"][key]={"text":text,"embedding":vector,"elapsed_batch_sec":time.monotonic()-start,
                    "batch_prompt_eval_count":response.prompt_eval_count,"batch_size":len(batch)}
            save(self.path,self.data)
        return [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]


def format_text(text, mode, document=False):
    if mode == "raw":return text
    if mode == "similarity":return "task: sentence similarity | query: "+text
    return ("title: none | text: " if document else "task: search result | query: ")+text


def subtitle_cues(video):
    path=ROOT/"godot-viewer/video"/(Path(video["filename"]).stem+".txt")
    raw=path.read_text(encoding="utf-8")
    pattern=r"(?m)^\s*(\d+:\d+:\d+[.,]\d+)\s*,\s*(\d+:\d+:\d+[.,]\d+)\s*$"
    matches=list(re.finditer(pattern,raw))
    cues=[]
    for i,m in enumerate(matches):
        text=re.sub(r"\s+"," ",raw[m.end():matches[i+1].start() if i+1<len(matches) else len(raw)]).strip()
        if text:cues.append({"start":m.group(1),"end":m.group(2),"text":text})
    return cues


def chunks(video,threshold=100):
    # Complete subtitle cues retain start/end times; windows share at most one cue.
    cues=subtitle_cues(video)
    result=[];current=[]
    for cue in cues:
        current.append(cue)
        if sum(len(c["text"].split()) for c in current)>=threshold:
            result.append({"text":" ".join(c["text"] for c in current),"start":current[0]["start"],"end":current[-1]["end"]})
            current=current[-1:]
    if current and (not result or result[-1]["end"]!=current[-1]["end"]):
        result.append({"text":" ".join(c["text"] for c in current),"start":current[0]["start"],"end":current[-1]["end"]})
    assert result,video["filename"]
    return result


def sentence_units(video,width):
    raw=(ROOT/"godot-viewer/video"/(Path(video["filename"]).stem+".txt")).read_text(encoding="utf-8")
    pattern=r"(?m)^\s*(\d+:\d+:\d+[.,]\d+)\s*,\s*(\d+:\d+:\d+[.,]\d+)\s*$"
    times=list(re.finditer(pattern,raw));cues=[];text=""
    for i,m in enumerate(times):
        body=re.sub(r"\s+"," ",raw[m.end():times[i+1].start() if i+1<len(times) else len(raw)]).strip()
        begin=len(text);text+=body+" "
        cues.append({"begin":begin,"finish":len(text),"start":m.group(1),"end":m.group(2)})
    spans=[(m.start(),m.end()) for m in re.finditer(r".+?(?:[.!?](?=\s|$)|$)",text,flags=re.S) if m.group().strip()]
    rows=[]
    for i in range(0,len(spans),1 if width==1 else 2):
        selected=spans[i:i+width]
        begin,end=selected[0][0],selected[-1][1];body=text[begin:end].strip()
        if len(body.split()) < (12 if width==1 else 20):continue
        relevant=[c for c in cues if c["finish"]>begin and c["begin"]<end]
        rows.append({"text":body,"start":relevant[0]["start"],"end":relevant[-1]["end"]})
    if not rows:rows=[{"text":video["text"],"start":None,"end":None}]
    return rows


def sentence_windows(video,threshold=70):
    # Keep complete punctuated sentences; overlap one sentence and merge a short tail.
    # Unlike the sentence-only index, retain the short sentences too.
    text=video["text"]
    bodies=[m.group().strip() for m in re.finditer(r".+?(?:[.!?](?=\s|$)|$)",text,flags=re.S) if m.group().strip()]
    cue_rows=subtitle_cues(video)
    joined=" ".join(c["text"] for c in cue_rows)
    assert re.sub(r"\s+"," ",text).strip()==joined,video["filename"]
    spans=[];position=0
    for cue in cue_rows:
        spans.append((position,position+len(cue["text"]),cue))
        position+=len(cue["text"])+1
    sentence_spans=[];position=0
    for body in bodies:
        begin=joined.index(body,position);end=begin+len(body)
        sentence_spans.append((begin,end,body));position=end
    windows=[];current=[]
    for sentence in sentence_spans:
        current.append(sentence)
        if sum(len(s[2].split()) for s in current)>=threshold:
            windows.append(current);current=current[-1:]
    if current and (not windows or current[-1][1]!=windows[-1][-1][1]):
        if windows and sum(len(s[2].split()) for s in current)<35:
            windows[-1]+=current[1:]
        else:windows.append(current)
    rows=[]
    for window in windows:
        begin,end=window[0][0],window[-1][1]
        relevant=[c for a,b,c in spans if b>begin and a<end]
        rows.append({"text":joined[begin:end],"start":relevant[0]["start"],"end":relevant[-1]["end"]})
    return rows


def granularity_stage():
    videos,_=source_data();cache=Cache();documents=[]
    for kind in ["cue40","cue70","cue140","sent1","sent3","sentence70"]:
        pending=[]
        for video in videos:
            if kind=="sentence70":rows=sentence_windows(video)
            elif kind.startswith("sent"):rows=sentence_units(video,int(kind[-1]))
            else:rows=chunks(video,int(kind[3:]))
            for i,row in enumerate(rows):
                if kind.startswith("cue") and len(row["text"].split())<20:continue
                pending.append(dict(row,kind=kind,id=video["id"]+"_"+kind+"_"+str(i),filename=video["filename"],title=video["title"]))
        keys=cache.vector_many([row["text"] for row in pending])
        for row,key in zip(pending,keys):documents.append(dict(row,embeddings={"raw":key}))
        save(HERE/"granularity-inputs.json",{"documents":documents,
            "notes":["Per clip score is the maximum passage cosine, never a mean vector.",
                     "Sentence boundaries follow transcript punctuation and may be imperfect in spontaneous speech.",
                     "One-sentence units shorter than 12 words and cue-tail units shorter than 20 words are excluded from those indexes, not from the archival transcript.",
                     "Larger clips offer more candidates; max scores are not length-corrected probabilities."]})
        print("Prepared %s: %d passage vectors"%(kind,len(pending)),flush=True)


def embedding_stage():
    videos,comments=source_data();cache=Cache()
    document_rows=[]
    for vi,v in enumerate(videos):
        units=[{"text":v["text"],"start":None,"end":None,"kind":"full"}]+[dict(c,kind="chunk") for c in chunks(v)]
        for ui,unit in enumerate(units):
            keys={mode:cache.vector(format_text(unit["text"],mode,True))[0] for mode in ["raw","similarity","retrieval"]}
            document_rows.append(dict(unit,filename=v["filename"],title=v["title"],id="%s_%d"%(v["id"],ui),embeddings=keys))
        print("Prepared documents %d/25"%(vi+1),flush=True)
    queries=[]
    for i,row in enumerate(comments,1):
        keys={mode:cache.vector(format_text(row["commentary"],mode))[0] for mode in ["raw","similarity","retrieval"]}
        queries.append({"id":i,"text":row["commentary"],"embeddings":keys})
    calibration=[
        ("office_expectations","A familiar room makes a player expect all the ordinary objects and actions that should be there.",["09_writing_a_romance_2.ogv"]),
        ("romance_choice","Choosing to write a romance was her own idea, not an assignment to make a product for girls.",["08_writing_a_romance_1.ogv"]),
        ("deaths_and_space","Killing the player was easier to implement than designing another escape route, and used less room.",["24_writing_a_romance_5.ogv"]),
        ("later_readers","A game remembered internally as a failure turned out to have mattered to readers who sought out its writer years later.",["13_critical_response_0.ogv"]),
        ("novel_expectations","The expectation of writing a great novel became so intimidating that dreaming of it displaced actually writing it.",["16_writing_a_novel_0.ogv"]),
        ("work_as_home","The office was also a home and social circle, where working late did not mean being alone.",["04_company_culture_0.ogv"]),
        ("abstract_application","She chose an absurd fantasy for her application game to avoid having to implement all the details of a realistic world.",["05_becoming_an_imp_0.ogv"]),
        ("books_not_films","They thought of their games as literary experiences and environments, more like books than ordinary games.",["22_market_vision_2.ogv"]),
    ]
    for name,text,expected in calibration:
        keys={mode:cache.vector(format_text(text,mode))[0] for mode in ["raw","similarity","retrieval"]}
        queries.append({"id":name,"text":text,"expected":expected,"embeddings":keys})
    negative_controls=[
        ("unrelated_weather","The weather forecast says there will be light rain tomorrow, with a colder wind in the afternoon."),
        ("vague_reason","You, err, I was there for a reason, you know, and I thought that was what I was supposed to do."),
        ("generic_reader","I enjoy telling stories, and I sometimes wonder whether other people see what I see in them."),
        ("biographical_decoy","Amy Briggs wrote a romance at Infocom, worked with testers, faced computer memory limits, and later discovered that readers valued her game."),
    ]
    for name,text in negative_controls:
        keys={mode:cache.vector(format_text(text,mode))[0] for mode in ["raw","similarity","retrieval"]}
        queries.append({"id":name,"text":text,"control_type":"synthetic_not_grounded_in_any_game_scene","embeddings":keys})
    save(HERE/"retrieval-inputs.json",{"documents":document_rows,"queries":queries,
         "note":"Calibration items paraphrase specified clips; useful for implementation sanity, not a blind artistic benchmark. Chunk max-pooling creates more candidates than full-clip retrieval."})
    print("Embedding stage complete: %d documents, %d queries, %d embeddings"%(len(document_rows),len(queries),len(cache.data["records"])),flush=True)


def sample(condition,scene,seed,protocol,client,recent=None):
    recent=scene["fixed_recent_comments"] if recent is None else recent
    prompt=build_prompt(condition,scene["history"],recent,protocol)
    options={"temperature":0.3,"num_ctx":4096,"num_predict":2560,"seed":seed}
    start=time.monotonic()
    response=client.chat(model=CHAT_MODEL,messages=[{"role":"user","content":prompt}],
                         think=True,options=options,keep_alive="30m")
    text=faketerm.clean_output(response.message.content or "").strip()
    return {"condition":condition,"scene":scene["scene"],"seed":seed,"prompt":prompt,
            "history":scene["history"],"recent_comments":recent,"commentary":text,"options":options,
            "prompt_tokens":response.prompt_eval_count,"output_tokens":response.eval_count,
            "done_reason":response.done_reason,"thinking_characters":len(response.message.thinking or ""),
            "elapsed_sec":round(time.monotonic()-start,3),"words":len(text.split())}


def generation_stage(protocol,conditions):
    path=HERE/"screen.json"
    data=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"results":[],"warmups":[]}
    done={(r["condition"],r["scene"],r["seed"]) for r in data["results"]}
    client=ollama.Client(host=faketerm.OLLAMA_HOST,timeout=180)
    if not data["warmups"]:
        data["warmups"].append(sample("production",protocol["scenes"][0],0,protocol,client))
        save(path,data)
    for si,scene in enumerate(protocol["scenes"]):
        for seed in protocol["seeds"]:
            offset=(si+seed)%len(conditions)
            for condition in conditions[offset:]+conditions[:offset]:
                if (condition,scene["scene"],seed) in done:continue
                result=sample(condition,scene,seed,protocol,client)
                data["results"].append(result);save(path,data)
                print(json.dumps({k:result[k] for k in ["condition","scene","seed","elapsed_sec","done_reason","words"]}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["prepare","embeddings","granularity","screen"],required=True)
    parser.add_argument("--conditions",nargs="+",default=["production","spoken","resonant","testimony"])
    args=parser.parse_args()
    protocol=prepare()
    if args.stage=="embeddings":embedding_stage()
    elif args.stage=="granularity":granularity_stage()
    elif args.stage=="screen":generation_stage(protocol,args.conditions)


if __name__=="__main__":main()
