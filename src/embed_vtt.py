#!/usr/bin/env python3
"""Embed subtitle text from godot-viewer/video/*.txt and write JSON."""

import argparse
import hashlib
import json
import os
import re
import sys

import ollama

DEFAULT_MODEL = "embeddinggemma:300m" # "qwen3-embedding"
DEFAULT_TITLE_MODEL = "ministral-3:14b"
DEFAULT_INPUT_DIR = os.path.join("godot-viewer", "video")
DEFAULT_OUTPUT_PATH = os.path.join("assets", "abriggs-itw-embeddings.json")
DEFAULT_PASSAGE_PATH = os.path.join("assets", "abriggs-itw-passages.json")

TIMECODE_RE = re.compile(
    r"^\s*\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}\s*,\s*\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}\s*$"
)
WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text):
    return WHITESPACE_RE.sub(" ", text).strip()


def clean_line(line):
    line = line.strip()
    if not line:
        return ""
    line = line.replace("A\u00ff", "")
    line = line.replace("\u00ff", "")
    return line


def extract_plain_text(path):
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    parts = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if TIMECODE_RE.match(stripped):
            continue
        cleaned = clean_line(stripped)
        if cleaned:
            parts.append(cleaned)
    return normalize_text(" ".join(parts))


def list_text_files(input_dir):
    if not os.path.isdir(input_dir):
        return []
    entries = [
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if name.lower().endswith(".txt") and not name.lower().endswith("-fr.txt")
    ]
    return sorted(entries)


def build_title(text, model):
    prompt = (
        "Write an archive tape label for this transcript, summarizing the main topic. "
        "Use title case, no quotes, 3-7 words, ENGLISH, PLAIN TEXT, NO MARKDOWN.\n\n"
        f"Transcript:\n{text}"
    )
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return (response.message.content or "").strip()


def embed_files(paths, model, title_model):
    results = []
    total = len(paths)
    for idx, path in enumerate(paths, start=1):
        text = extract_plain_text(path)
        if not text:
            print(f"Skipping empty text: {path}")
            continue
        response = ollama.embeddings(model=model, prompt=text)
        sequence_title = build_title(text, title_model)
        filename = os.path.basename(path)
        if filename.lower().endswith(".txt"):
            filename = filename[:-4] + ".ogv"
        results.append(
            {
                "filename": filename,
                "embedding": response.get("embedding"),
                "sequence_title": sequence_title,
            }
        )
        print(f"Embedded {idx}/{total}: {os.path.basename(path)}")
    return results


def subtitle_cues(path):
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read()
    matches = list(re.finditer(
        r"(?m)^\s*(\d+:\d+:\d+[.,]\d+)\s*,\s*(\d+:\d+:\d+[.,]\d+)\s*$", raw))
    cues = []
    for index, match in enumerate(matches):
        text = normalize_text(raw[match.end():matches[index+1].start() if index+1 < len(matches) else len(raw)])
        if text:
            cues.append({"text": text, "start": match.group(1), "end": match.group(2)})
    return cues


def contextual_passages(cues, target_words=70):
    if target_words < 30:
        raise ValueError("Passage target must be at least 30 words to retain context.")
    passages = []
    current = []
    for cue in cues:
        current.append(cue)
        if sum(len(c["text"].split()) for c in current) >= target_words:
            passages.append({"text": " ".join(c["text"] for c in current),
                             "start": current[0]["start"], "end": current[-1]["end"]})
            current = current[-1:]
    if current and (not passages or passages[-1]["end"] != current[-1]["end"]):
        tail = " ".join(c["text"] for c in current)
        if passages and len(tail.split()) < 30:
            passages[-1]["text"] += " " + " ".join(c["text"] for c in current[1:])
            passages[-1]["end"] = current[-1]["end"]
        else:
            passages.append({"text": tail, "start": current[0]["start"], "end": current[-1]["end"]})
    return passages


def embed_passage_catalogue(input_dir, base_path, model, target_words=70, num_gpu=0):
    with open(base_path, "r", encoding="utf-8") as handle:
        base = json.load(handle)
    client = ollama.Client(timeout=180)
    results = []
    for entry in base:
        path = os.path.join(input_dir, os.path.splitext(entry["filename"])[0] + ".txt")
        cues = subtitle_cues(path)
        if not cues:
            raise ValueError("No timestamped subtitle cues: %s" % path)
        units = contextual_passages(cues, target_words)
        texts = [extract_plain_text(path)] + [unit["text"] for unit in units]
        options = {"num_gpu": num_gpu, "num_ctx": 2048}
        response = client.embed(model=model, input=texts, truncate=False, options=options)
        vectors = response.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError("Incomplete embeddings: %s" % path)
        passages = [dict(unit, id="%s:%03d" % (os.path.splitext(entry["filename"])[0], index), embedding=vector)
                    for index, (unit, vector) in enumerate(zip(units, vectors[1:]), 1)]
        with open(path, "rb") as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        results.append({"filename": entry["filename"], "sequence_title": entry.get("sequence_title"),
                        "duration_sec": entry.get("duration_sec"), "embedding": vectors[0],
                        "embedding_model": model, "embedding_format": "raw", "embedding_options": options,
                        "subtitle_sha256": digest, "passage_target_words": target_words, "passages": passages})
        print("Embedded %s: %s passages" % (entry["filename"], len(passages)))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Embed subtitle text from godot-viewer/video/*.txt."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing subtitle .txt files",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path",
    )
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Ollama embedding model")
    parser.add_argument("--passages", action="store_true", help="Build grouped contextual passages instead of whole-clip vectors only")
    parser.add_argument("--base-catalog", default=DEFAULT_OUTPUT_PATH, help="Catalogue supplying clip filenames, titles and durations")
    parser.add_argument("--passage-words", type=int, default=70, help="Target words per passage, sharing one complete subtitle cue")
    parser.add_argument("--num-gpu", type=int, default=0, help="Embedding GPU layers for passage generation; 0 keeps it on CPU")
    parser.add_argument(
        "-t",
        "--title-model",
        default=DEFAULT_TITLE_MODEL,
        help="Ollama model for sequence titles",
    )
    args = parser.parse_args()

    if args.passages:
        results = embed_passage_catalogue(args.input_dir, args.base_catalog, args.model,
                                         target_words=args.passage_words, num_gpu=args.num_gpu)
    else:
        files = list_text_files(args.input_dir)
        if not files:
            print(f"No .txt files found in {args.input_dir}", file=sys.stderr)
            return 1
        results = embed_files(files, args.model, args.title_model)
    if not results:
        print("No embeddings generated.", file=sys.stderr)
        return 1

    output = args.output or (DEFAULT_PASSAGE_PATH if args.passages else DEFAULT_OUTPUT_PATH)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=True, indent=2)
    os.replace(output + ".tmp", output)

    print(f"Wrote JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
