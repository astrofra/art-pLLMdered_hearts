"""Update a copy of the English deck while preserving all unchanged OOXML/media.

Uses only the standard library. New figures come from analyse_session.py.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import struct
from xml.etree import ElementTree as E
from zipfile import ZipFile, ZIP_DEFLATED

HERE = Path(__file__).resolve().parent
P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
NS = {"p": P, "a": A, "r": R}
for prefix, uri in NS.items():
    E.register_namespace(prefix, uri)


def tag(namespace, name):
    return "{" + namespace + "}" + name


def xml(root):
    return E.tostring(root, encoding="utf-8", xml_declaration=True)


def package_xml(root):
    # Keep OPC package roots unprefixed for LibreOffice's package reader.
    E.register_namespace("", root.tag.split("}", 1)[0][1:])
    return xml(root)


def node(parent, namespace, element_name, **attributes):
    return E.SubElement(parent, tag(namespace, element_name), {k: str(v) for k, v in attributes.items()})


def set_text(shape, lines, size=None, bullet=False, color=None):
    body = shape.find("p:txBody", NS)
    first = body.find("a:p", NS)
    ppr = first.find("a:pPr", NS) if first is not None else None
    rpr = first.find("a:r/a:rPr", NS) if first is not None else None
    for p in list(body.findall("a:p", NS)):
        body.remove(p)
    for line in lines:
        p = node(body, A, "p")
        props = deepcopy(ppr) if ppr is not None else E.Element(tag(A, "pPr"))
        if not bullet:
            for child in list(props):
                if child.tag.rsplit("}", 1)[-1].startswith("bu"):
                    props.remove(child)
            node(props, A, "buNone")
            props.set("marL", "0")
            props.set("indent", "0")
        else:
            after = props.find("a:spcAft", NS)
            if after is None:
                after = node(props, A, "spcAft")
            for child in list(after):
                after.remove(child)
            node(after, A, "spcPts", val=1400)
        p.append(props)
        run = node(p, A, "r")
        style = deepcopy(rpr) if rpr is not None else E.Element(tag(A, "rPr"))
        style.set("lang", "en-GB")
        if size:
            style.set("sz", str(round(size * 100)))
        for child in list(style):
            if child.tag in [tag(A, "highlight"), tag(A, "solidFill")]:
                style.remove(child)
        if color:
            node(node(style, A, "solidFill"), A, "srgbClr", val=color)
        run.append(style)
        node(run, A, "t").text = line
        node(p, A, "endParaRPr", lang="en-GB")


def shape_by_id(slide, sid):
    return next(s for s in slide.findall(".//p:sp", NS) if s.find("p:nvSpPr/p:cNvPr", NS).get("id") == str(sid))


def notes_text(notes, text):
    for shape in notes.findall(".//p:sp", NS):
        ph = shape.find("p:nvSpPr/p:nvPr/p:ph", NS)
        if ph is not None and ph.get("type") == "body":
            set_text(shape, text.split("\n"), 12)
            return
    raise ValueError("Missing speaker-notes placeholder")


def textbox(tree, sid, x, y, w, h, lines, size=16, fill=None, border=None, align="l", color="333333"):
    sp = node(tree, P, "sp")
    nv = node(sp, P, "nvSpPr")
    node(nv, P, "cNvPr", id=sid, name="Updated text %s" % sid)
    node(nv, P, "cNvSpPr", txBox=1)
    node(nv, P, "nvPr")
    prop = node(sp, P, "spPr")
    transform = node(prop, A, "xfrm")
    node(transform, A, "off", x=round(x*12700), y=round(y*12700))
    node(transform, A, "ext", cx=round(w*12700), cy=round(h*12700))
    node(node(prop, A, "prstGeom", prst="rect"), A, "avLst")
    if fill:
        node(node(prop, A, "solidFill"), A, "srgbClr", val=fill)
    else:
        node(prop, A, "noFill")
    ln = node(prop, A, "ln", w=9525)
    if border:
        node(node(ln, A, "solidFill"), A, "srgbClr", val=border)
    else:
        node(ln, A, "noFill")
    body = node(sp, P, "txBody")
    node(body, A, "bodyPr", wrap="square", lIns=76200, rIns=76200, tIns=63500, bIns=63500, anchor="ctr" if fill else "t")
    node(body, A, "lstStyle")
    for line in lines:
        p = node(body, A, "p")
        props = node(p, A, "pPr", algn=align)
        node(props, A, "buNone")
        run = node(p, A, "r")
        style = node(run, A, "rPr", lang="en-GB", sz=round(size*100))
        node(node(style, A, "solidFill"), A, "srgbClr", val=color)
        node(style, A, "latin", typeface="Barlow Semi Condensed")
        node(run, A, "t").text = line
    return sp


def arrow(tree, sid, x1, y1, x2, y2):
    sp = node(tree, P, "cxnSp")
    nv = node(sp, P, "nvCxnSpPr")
    node(nv, P, "cNvPr", id=sid, name="Flow %s" % sid)
    node(nv, P, "cNvCxnSpPr")
    node(nv, P, "nvPr")
    props = node(sp, P, "spPr")
    transform = node(props, A, "xfrm")
    node(transform, A, "off", x=round(x1*12700), y=round(y1*12700))
    node(transform, A, "ext", cx=round((x2-x1)*12700), cy=round((y2-y1)*12700))
    node(node(props, A, "prstGeom", prst="line"), A, "avLst")
    ln = node(props, A, "ln", w=15240)
    node(node(ln, A, "solidFill"), A, "srgbClr", val="777777")
    node(ln, A, "tailEnd", type="triangle")


def blank_content(slide):
    tree = slide.find("p:cSld/p:spTree", NS)
    for item in list(tree)[2:]:
        tree.remove(item)
    return tree


def add_title(tree, text):
    textbox(tree, 100, 66, 34, 620, 44, [text], size=23, color="666666")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if args.output.resolve() == args.source.resolve():
        parser.error("The output must be a separate copy")
    with ZipFile(args.source) as original:
        parts = {i.filename: original.read(i.filename) for i in original.infolist()}
        metadata = {i.filename: i for i in original.infolist()}
    baseline = dict(parts)
    slides = {n: E.fromstring(parts["ppt/slides/slide%d.xml" % n]) for n in range(1, 31)}
    note_roots = {n: E.fromstring(parts["ppt/notesSlides/notesSlide%d.xml" % n]) for n in range(1, 31)}
    changes = []

    def replace(n, sid, lines, size=None, bullet=False):
        sh = shape_by_id(slides[n], sid)
        before = "\n".join(t.text or "" for t in sh.findall(".//a:t", NS))
        set_text(sh, lines, size=size, bullet=bullet)
        changes.append({"source_slide": n, "shape_id": sid, "before": before, "after": "\n".join(lines)})

    replace(10, 192, ["Initial experiment: an LLM as player"])
    replace(11, 198, ["Initial prototype: Ministral as player"])
    replace(13, 231, ["Initial Ministral 3:14B experiment: no action history"], 14)
    replace(14, 241, ["Initial Ministral 3:14B experiment: history of 16 actions"], 14)
    replace(19, 317, ["Initial Ministral experiment: a longer action history increases exploration, but also parser rejections."], 16)
    for n in [10, 11, 13, 14, 16, 17, 18, 19, 20, 21]:
        old = "\n".join(t.text or "" for t in note_roots[n].findall(".//a:t", NS))
        preface = "Historical prototype: these observations concern the original autonomous-player experiments, not the current Gemma commentator. They are not a new benchmark of Gemma or a general claim about all LLMs.\n"
        if n == 13:
            old = "In this prototype, the model sometimes repeats an ineffective command. No previous actions were supplied in this condition. The following slides examine an explicit action history; this is a context-management experiment, not evidence that every language model is intrinsically unable to remember or count."
        notes_text(note_roots[n], preface + old)
    replace(23, 350, ["From the festival prototype to Gemma"])
    replace(23, 351, [
        "A fixed walkthrough advances the game; the model does not choose commands.",
        "The current build uses Gemma 4 E4B for English commentary, with extended reasoning enabled.",
        "Complete game outputs are buffered. The commentator notices a detail in the scene, using the interview as optional background."], 18, True)
    notes_text(note_roots[23], "The photographs show the earlier festival installation. The current build replaces Ministral with Gemma 4 E4B. A scripted walkthrough remains responsible for progress. Do not attribute determinism to the Z-machine itself: repeatability depends on the game and execution conditions, not merely the interpreter. The model reads only completed events and already executed commands.")
    replace(24, 359, [
        "Amy Briggs's interview with Jason Scott for Get Lamp supplies the documentary material.",
        "Each of 25 interview clips is indexed by an embedding of its English subtitle transcript.",
        "The generated comment is embedded in the same space. It becomes the bridge from the game to the archive; the game output itself is not vectorised."], 18, True)
    notes_text(note_roots[24], "There are two different documentary resources. Eight concise factual cards (136 words) are supplied to the commentary model. Separately, the catalogue contains one precomputed vector per interview clip, derived from the full cleaned subtitle text. The cards are not a vector store. At runtime only a visible final comment is embedded; reasoning, recent-comment memory and raw game output are not directly embedded. The existing embedding model and clip segmentation have not been replaced in this iteration.")
    replace(27, 396, [
        "The commentary is the semantic bridge between the game and Briggs's retrospective testimony.",
        "Early prompts repeatedly returned to hardware limits, commercial pressure or genre. Reworking the prompt and its context reduced these patterns in the sampled readings.",
        "A high similarity score can still yield a weak association. The sequence offers an interpretation; generated comments are not historical evidence."], 18, True)
    notes_text(note_roots[27], "The 17-scene prompt replay reduced forced technical explanations in our qualitative reading, but did not increase raw top-match diversity: both conditions yielded seven distinct first-ranked clips. This is not a controlled causal estimate: the original run was unseeded and the replay used seed 42 with regenerated memory. Making the interview optional can also weaken an explicit documentary link. Improving commentary fluency or novelty does not, by itself, establish more pertinent game/video pairings.")
    replace(28, 402, [
        "The new prompt reduces forced explanations, but can also weaken the explicit link to Briggs. Paraphrases, repetition and unsupported interpretations remain.",
        "Cosine similarity ranks associations; it does not validate them. Recent-clip exclusions can also change which video is selected.",
        "Next: assess game–comment–clip triples with human readers; compare shorter interview segments, embedding models and context positions."], 18, True)
    notes_text(note_roots[28], "The current measured prompts fit inside the configured 4,096-token window; we should not present a presumed Ministral context limit as the cause of every weak association. No model weights were trained or fine-tuned. Neither a new embedding model nor shorter subtitle chunks was evaluated in the latest tuning experiment. A shared blind human rubric for game scene, generated comment and selected clip would assess the actual mediation objective more directly than keyword counts or clip diversity. The headless run does not validate real playback timing or audience comprehension.")

    # Replace the old installation diagram with the actual data/control flow.
    tree = blank_content(slides[25])
    add_title(tree, "Current installation: Gemma as commentator")
    boxes = [
        (101, 80, 90, 170, 62, ["Scripted walkthrough", "Fixed game commands"], "FFFFFF"),
        (102, 80, 188, 170, 64, ["Frotz + Plundered Hearts", "Complete text outputs"], "F5F2B8"),
        (103, 300, 90, 190, 70, ["Bounded context", ">150 game words + 2 comments", "8 testimony cards (136 words)"], "FFFFFF"),
        (104, 300, 188, 190, 64, ["Gemma 4 E4B", "English commentary"], "F8DDE6"),
        (105, 530, 188, 150, 64, ["C64-style terminal", "Comment on display"], "FFFFFF"),
        (106, 80, 298, 170, 64, ["25 interview clips", "Subtitle embeddings", "Computed beforehand"], "D9EAF5"),
        (107, 300, 298, 190, 64, ["EmbeddingGemma 300M", "Latest comment → vector", "Cosine similarity + exclusions"], "F8DDE6"),
        (108, 530, 298, 150, 64, ["Godot viewer", "Next interview clip"], "FFFFFF")]
    for sid, x, y, w, h, lines, fill in boxes:
        textbox(tree, sid, x, y, w, h, lines, 14, fill, "AAAAAA", "ctr")
    for sid, coords in enumerate([(165,152,165,188),(250,220,300,220),(395,160,395,188),(490,220,530,220),
                                   (395,252,395,298),(250,330,300,330),(490,330,530,330)], 120):
        arrow(tree, sid, *coords)
    textbox(tree, 140, 80, 370, 610, 28, ["Video selection waits for the estimated clip end + 3 s. Godot sends no playback feedback."], 12)
    notes_text(note_roots[25], "Current source: src/faketerm.py, commit da55b03 plus the recording tools. Gemma generates a visible comment, not a game action. EmbeddingGemma runs on CPU; the commentary model uses the GPU. A comment may be displayed while an interview plays, after a complete game output. Only the latest pending comment is embedded after the estimated end of the current clip. Godot polls timestamped text requests. There is no Godot-to-Python acknowledgement, no online training and no autonomous tool-using agent loop.")
    # Make the existing photograph explicit about its date/version.
    tree26 = slides[26].find("p:cSld/p:spTree", NS)
    textbox(tree26, 500, 66, 15, 620, 28, ["Installation view: earlier festival version (Ministral)"], 14, color="666666")
    notes_text(note_roots[26], "Archive image of the festival version. It illustrates the physical arrangement, not the current Gemma wording or current measurement session.")

    def new_slide(n, title, body, note):
        slide = deepcopy(slides[27])
        tree = blank_content(slide)
        add_title(tree, title)
        if body:
            # Reuse the original theme's body placeholder and bullet geometry.
            sh = deepcopy(shape_by_id(slides[27], 396))
            set_text(sh, body, 18, bullet=True)
            tree.append(sh)
        slides[n] = slide
        note_roots[n] = deepcopy(note_roots[27])
        notes_text(note_roots[n], note)
        return tree

    tree = new_slide(31, "Choosing a local commentator", None,
        "Sources: documentation/llm-comparison-2026-10-19/results.json and review.json. The table is the original preloaded-model benchmark: 12 scenes, two rounds per model, temperature 0.3, num_ctx 4096, num_predict 256, no reasoning. It is not the current reasoning-enabled latency. The 1,250-word follow-up repeated the protocol: medians 16.906 s, 0.844 s and 0.718 s respectively. Increasing summary length changed associations, without establishing greater relevance. The original prompt still produced recurring general themes in every model. Gemma E4B is an effective-size designation; local metadata reports 7.5B total parameters. GPU peaks include desktop and temporary allocations. EmbeddingGemma remained on CPU.")
    textbox(tree, 101, 75, 100, 600, 42, ["Original comparison: 12 scenes × 2 rounds per model", "English · 256-token output cap · reasoning disabled · models preloaded"], 15)
    rows = [("Model", "Median request", "Sampled GPU peak"),
            ("Ministral 3:14B", "14.44 s", "7,054 MiB"),
            ("Qwen 3.5 4B", "0.77 s", "4,554 MiB"),
            ("Gemma 4 E4B", "0.71 s", "5,444 MiB")]
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            textbox(tree, 110+i*3+j, [75,330,490][j], 155+i*38, [255,160,190][j], 37,
                    [value], 17, "F8DDE6" if i in [0,3] else "FFFFFF", "DDDDDD")
    textbox(tree, 140, 75, 325, 610, 58, ["Gemma retained for responsiveness and its readings in this sample.",
            "The model change alone did not remove recurring themes."], 17)

    new_slide(32, "From recurring themes to situated reading", [
        "Observed problem: the model repeatedly linked fictional events to technical limits, even when the scene supplied no evidence for that explanation.",
        "Prompt revision: notice one fresh detail; develop it across two sentences. Remove the upcoming game command. Treat Briggs's testimony as optional background.",
        "Memory and rhythm: supply the two previous comments to discourage reused ideas; lower the game-history trigger from 500 to 150 words, always at a complete prompt."],
        "This was an iterative collaboration with the artist: listening to the installation exposed a recurring technical-constraint motif; commentary logs made it inspectable. The 17-record replay in commentary-prompt-review-2026-10-19 kept the 752-word context while changing the prompt and replaying rolling memory. In the assistant's qualitative reading, seven earlier comments attributed scenes to technical constraints, while no new comment did. This is an exploratory observation, not a blinded experiment. Both versions still generated seven distinct raw top clips. Memory compression was discussed but not deployed: production supplies two full comments, not compressed summaries. The 500-word threshold was judged too infrequent by the artist; 150 is an artistic pacing choice rather than a universally optimal threshold.")
    new_slide(33, "Prompt and context design", [
        "We compared 752- and 1,250-word narratives, then prose and thematic cards. The retained reference is eight factual entries totalling 136 words.",
        "Across 371 later responses, higher temperatures (0.6 / 0.9) brought no consistent qualitative gain. The practical preset remains 0.3, with reasoning enabled and a 2,560-token output cap.",
        "No model weights were fine-tuned. “Lost in the middle” motivated a question about context structure; this installation has not isolated a positional-attention effect."],
        "The 371-response study compares four contexts and three temperatures in 96 screened responses, then five rolling sequences of 55 buffers. Prose and detailed cards contain the same factual sentences in the same order, while the brief version also changes content and length. Thus the selected preset does not establish a causal advantage of list formatting or information position. Current limits: context 4096 tokens, output cap 2560 tokens shared by reasoning and visible answer, two sentences and at most 90 visible words. Earlier reasoning tests did not show a general thematic benefit; a small later anti-repetition test motivated re-enabling reasoning. Short 512/768-token reasoning budgets produced empty or truncated answers, so the production ceiling was raised. This is prompt/context engineering and inference-parameter adjustment, not weight fine-tuning. Reading: Liu et al., Lost in the Middle: How Language Models Use Long Contexts, arXiv:2307.03172 (2023; TACL publication 2024), https://arxiv.org/abs/2307.03172. That paper studies different models and retrieval tasks. It motivates a hypothesis, not a diagnosis of this Gemma installation. A proper test would move identical facts between positions while holding all other inputs fixed.")
    tree = slides[33].find("p:cSld/p:spTree", NS)
    textbox(tree, 150, 75, 365, 610, 30, ["Context-position hypothesis: Liu et al., Lost in the Middle (2023/2024), arXiv:2307.03172."], 11)
    new_slide(34, "Two rhythms, bounded memory", [
        "The game yields control at a complete prompt. Once the buffer exceeds 150 words, Gemma reads the completed outputs and executed commands since its previous successful comment.",
        "The new comment appears after the current game text, even while Amy speaks. The game continues and accumulates another buffer.",
        "After the current clip's estimated duration + 3 seconds, only the latest pending comment is vectorised to choose the next video. Intermediate comments remain visible but do not each trigger a clip."],
        "No dialogue is sent from Godot to Python. Duration-based gating uses the catalogue's duration_sec plus a 3-second margin; these are estimates, not confirmed playback boundaries. Failed/empty commentary does not clear the game buffer. A failed embedding can be retried without another commentary call. The buffer, two-comment memory and video timing survive walkthrough restarts. The player does not interrupt a game paragraph to insert a comment. This is scheduling around completed outputs, not a promise of asynchronous inference: the local model call still occupies the Python game loop while events are serviced.")

    stats = json.loads((HERE / "session-summary.json").read_text(encoding="utf-8"))
    projection = json.loads((HERE / "projection.json").read_text(encoding="utf-8"))
    tree = new_slide(35, "Evidence from the current build", None,
        "Source: presentation-update-2026-10-19/session-headless and session-summary.json. One actual full walkthrough, no fixed inference seed, real Frotz and Ollama, --once --no-renderer --no-viewer --fast. All 258 scripted game commands completed; blank acknowledgements are counted separately. The first request includes model loading and is slower than warm requests. The GPU peak is sampled every five seconds and includes the desktop. Four video request files were emitted; no viewer was running and playback is not confirmed. Acceleration drastically changes the wall-clock relation between game and video, so do not infer exhibition clip frequency from this run. The earlier tuning medians at right come from separate frozen-output rolling replays; they are not a before/after measure of this unseeded live run. Complete prompts, visible answers, timing counts, vectors and rankings are archived. Internal reasoning text is not archived or projected.")
    textbox(tree, 101, 75, 96, 290, 35, ["New instrumented walkthrough"], 19, color="A43A57")
    lines = ["%s game commands · %s comments" % (stats["game_commands"], stats["successful_comments"]),
             "Median inference: %.2f s" % stats["median_inference_sec"],
             "First request: %.2f s (includes loading)" % stats["first_inference_sec"],
             "Max prompt + output: %s / 4,096 tokens" % format(stats["max_prompt_plus_output_tokens"], ","),
             "%s video requests; playback not confirmed" % len(stats["video_requests"]),
             "Sampled GPU peak: %s MiB" % format(int(stats["sampled_gpu_peak_mib"]), ",")]
    textbox(tree, 102, 75, 145, 300, 180, lines, 17)
    textbox(tree, 103, 400, 96, 280, 45, ["Earlier tuning study", "55 rolling buffers per condition"], 18, color="A43A57")
    for i, (label, value) in enumerate([("752-word narrative · 0.3", "7.05 s"), ("136-word cards · 0.3", "4.84 s"), ("136-word cards · 0.9", "7.08 s")]):
        textbox(tree, 110+i*2, 400, 160+i*43, 205, 42, [label], 16, "F8DDE6" if i == 1 else "FFFFFF", "DDDDDD")
        textbox(tree, 111+i*2, 605, 160+i*43, 80, 42, [value], 16, "F8DDE6" if i == 1 else "FFFFFF", "DDDDDD")
    textbox(tree, 125, 400, 302, 285, 52, ["371 responses overall; qualitative review.", "These are separate measurement protocols."], 14)
    textbox(tree, 130, 75, 365, 610, 29, ["RTX 4060 (8 GiB) · Ollama 0.40.6 · Gemma 4 E4B on GPU · EmbeddingGemma 300M on CPU"], 11)

    first = projection["points"][25]
    tree = new_slide(36, "Reading the link, not just its score", None,
        "This example is the first actual request in the new walkthrough, not an invented or counterfactual pairing. It was chosen to expose a potentially weak association. The full game history, visible comment, complete 25-clip ranking and chosen file are in the session log. A cosine of about 0.269 is not a probability and does not quantify historical truth or artistic quality. The two leading scores differ by roughly 0.002. There is no implemented minimum-score or margin threshold. The first selection has no recent-video exclusion; later choices do. The model's reading of captivity may be productive, but the transition to an office-building memory still needs a human reading of both texts. The mechanism ranks associations; it cannot itself explain why a transition is meaningful.")
    textbox(tree, 101, 75, 100, 610, 30, ["First displayed comment → first actual clip request"], 18, color="A43A57")
    textbox(tree, 102, 75, 145, 610, 88, [first["text"]], 18, "F8DDE6")
    rank = first["raw_ranking"]
    textbox(tree, 103, 75, 252, 610, 70, ["1. %s — cosine %.4f" % (rank[0]["title"], rank[0]["cosine"]),
                                                    "2. %s — cosine %.4f" % (rank[1]["title"], rank[1]["cosine"])], 17)
    textbox(tree, 104, 75, 330, 610, 55, ["The transition may surprise us. Its interpretive relevance still needs to be assessed.",
                                      "A similarity score is neither a probability nor historical evidence."], 17)

    tree = new_slide(37, "Projecting the texts into two dimensions", None,
        "Projection data: projection.json; rendered with NumPy SVD and Matplotlib. All 25 stored subtitle embeddings and all 55 visible commentary embeddings are L2-normalised, jointly centred and projected by PCA. The existing catalogue is not overwritten. All 25 clip transcripts were re-embedded with the current CPU model to check consistency; the minimum stored/recomputed cosine is in session-summary.json. All comments are embedded offline for this diagram; only four comments actually generated video requests in the accelerated session. The four arrows represent those recorded requests, including live exclusions. Similarity is calculated in the original embedding space, never from projected distances. The first two axes preserve only the displayed percentage of total variance; an apparently distant point may still be a strong match in the original space. PCA axes have no intrinsic thematic labels. The interactive HTML lets an audience inspect every text and the full-space cosine ranking. The game text, factual cards and hidden reasoning are not directly embedded and do not appear as points.")
    pic = node(tree, P, "pic")
    nv = node(pic, P, "nvPicPr")
    node(nv, P, "cNvPr", id=101, name="Embedding PCA map")
    node(nv, P, "cNvPicPr")
    node(nv, P, "nvPr")
    fill = node(pic, P, "blipFill")
    blip = node(fill, A, "blip")
    blip.set(tag(R, "embed"), "rId3")
    node(node(fill, A, "stretch"), A, "fillRect")
    prop = node(pic, P, "spPr")
    transform = node(prop, A, "xfrm")
    image_width, image_height = struct.unpack(">II", (HERE / "embedding-map.png").read_bytes()[16:24])
    image_h = min(288, 622*image_height/image_width)
    image_w = image_h*image_width/image_height
    node(transform, A, "off", x=round((69+(622-image_w)/2)*12700), y=round(84*12700))
    node(transform, A, "ext", cx=round(image_w*12700), cy=round(image_h*12700))
    node(node(prop, A, "prstGeom", prst="rect"), A, "avLst")
    textbox(tree, 102, 75, 373, 610, 25, ["2D retains %.1f%% of variance. Selection uses cosine similarity in %s dimensions." %
             (projection["retained_variance_percent"], projection["dimensions"])], 12)

    # Append the one new academic reference without changing the historical bibliography.
    bib = shape_by_id(slides[30], 414)
    old_lines = ["".join(t.text or "" for t in p.findall(".//a:t", NS)) for p in bib.findall("p:txBody/a:p", NS)]
    # Rebuilding the entire bibliography would discard its existing italic runs; append one paragraph instead.
    body = bib.find("p:txBody", NS)
    p = node(body, A, "p")
    run = node(p, A, "r")
    style = node(run, A, "rPr", lang="en-GB", sz=900)
    node(style, A, "latin", typeface="Arial")
    node(run, A, "t").text = "Liu, N. F., et al. (2023/2024). Lost in the Middle: How Language Models Use Long Contexts. https://arxiv.org/abs/2307.03172"

    changed_originals = [10,11,13,14,19,23,24,25,26,27,28,30]
    for n in changed_originals + list(range(31,38)):
        parts["ppt/slides/slide%d.xml" % n] = xml(slides[n])
    for n in [10,11,13,14,16,17,18,19,20,21,23,24,25,26,27,28] + list(range(31,38)):
        parts["ppt/notesSlides/notesSlide%d.xml" % n] = xml(note_roots[n])
    presentation = E.fromstring(parts["ppt/presentation.xml"])
    rels = E.fromstring(parts["ppt/_rels/presentation.xml.rels"])
    content_types = E.fromstring(parts["[Content_Types].xml"])
    ids = presentation.find("p:sldIdLst", NS)
    original_ids = list(ids)
    new_ids = {}
    rid_max = max(int(r.get("Id")[3:]) for r in rels)
    slide_id_max = max(int(s.get("id")) for s in ids)
    for n in range(31,38):
        rid = "rId%d" % (rid_max+n-30)
        rel = node(rels, REL, "Relationship", Id=rid, Type=R+"/slide", Target="slides/slide%d.xml" % n)
        slide_id = E.Element(tag(P, "sldId"), {"id":str(slide_id_max+n-30),tag(R,"id"):rid})
        new_ids[n] = slide_id
        for kind, directory in [("slide", "slides"), ("notesSlide", "notesSlides")]:
            node(content_types, CT, "Override", PartName="/ppt/%s/%s%d.xml" % (directory,kind,n),
                 ContentType="application/vnd.openxmlformats-officedocument.presentationml.%s+xml" % kind)
        slide_rels = E.fromstring(parts["ppt/slides/_rels/slide27.xml.rels"])
        for rel in slide_rels:
            if rel.get("Type") == R+"/notesSlide":
                rel.set("Target", "../notesSlides/notesSlide%d.xml" % n)
        if n == 37:
            node(slide_rels, REL, "Relationship", Id="rId3", Type=R+"/image", Target="../media/gemma-embedding-map.png")
        parts["ppt/slides/_rels/slide%d.xml.rels" % n] = package_xml(slide_rels)
        note_rels = E.fromstring(parts["ppt/notesSlides/_rels/notesSlide27.xml.rels"])
        for rel in note_rels:
            if rel.get("Type") == R+"/slide":
                rel.set("Target", "../slides/slide%d.xml" % n)
        parts["ppt/notesSlides/_rels/notesSlide%d.xml.rels" % n] = package_xml(note_rels)
    order = list(range(1,25)) + [31,32,33,25,34,37,26,27,35,36,28,29,30]
    for item in list(ids):
        ids.remove(item)
    for n in order:
        ids.append(original_ids[n-1] if n <= 30 else new_ids[n])
    parts["ppt/presentation.xml"] = xml(presentation)
    parts["ppt/_rels/presentation.xml.rels"] = package_xml(rels)
    parts["[Content_Types].xml"] = package_xml(content_types)
    parts["ppt/media/gemma-embedding-map.png"] = (HERE / "embedding-map.png").read_bytes()
    app = E.fromstring(parts["docProps/app.xml"])
    for element in app.iter():
        if element.tag.rsplit("}", 1)[-1] == "Slides":
            element.text = str(len(order))
    parts["docProps/app.xml"] = xml(app)
    for name, data in parts.items():
        if name.endswith((".xml", ".rels")):
            E.fromstring(data)
    temporary = args.output.with_suffix(".tmp.pptx")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as out:
        for name, data in parts.items():
            out.writestr(metadata[name] if name in metadata else name, data)
    temporary.replace(args.output)
    assert source_hash == hashlib.sha256(args.source.read_bytes()).hexdigest()
    preserved_media = [name for name in baseline if name.startswith("ppt/media/") and parts[name] == baseline[name]]
    summary = {"source": str(args.source), "output": str(args.output), "source_sha256": source_hash,
               "source_unchanged": True, "slide_count_before": 30, "slide_count_after": len(order),
               "new_slides": [{"position": order.index(n)+1, "part_number": n} for n in range(31,38)],
               "original_media_preserved_byte_for_byte": len(preserved_media),
               "changed_parts": [name for name in parts if name not in baseline or baseline[name] != parts[name]],
               "text_changes": changes}
    (HERE / "presentation-changes.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k: v for k,v in summary.items() if k not in ["text_changes","changed_parts"]},ensure_ascii=True))


if __name__ == "__main__":
    main()
