#!/usr/bin/env python3
"""Embed subtitle text from godot-viewer/video/*.txt and write JSON."""

import argparse
import json
import os
import re
import sys

import ollama

DEFAULT_MODEL = "embeddinggemma:300m" # "qwen3-embedding"
DEFAULT_TITLE_MODEL = "ministral-3:14b"
DEFAULT_INPUT_DIR = os.path.join("godot-viewer", "video")
DEFAULT_OUTPUT_PATH = os.path.join("assets", "abriggs-itw-embeddings.json")

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
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path",
    )
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help="Ollama embedding model")
    parser.add_argument(
        "-t",
        "--title-model",
        default=DEFAULT_TITLE_MODEL,
        help="Ollama model for sequence titles",
    )
    args = parser.parse_args()

    files = list_text_files(args.input_dir)
    if not files:
        print(f"No .txt files found in {args.input_dir}", file=sys.stderr)
        return 1

    results = embed_files(files, args.model, args.title_model)
    if not results:
        print("No embeddings generated.", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=True, indent=2)

    print(f"Wrote JSON: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
