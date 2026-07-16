#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


THINK_PATTERN = re.compile(r"^AI thinks\s*:\s*'(.*)'$")
SUGGEST_PATTERN = re.compile(r"^AI suggests:\s*'(.*)'$")
PARSER_REJECTION_PATTERNS = (
    "i don't know the word",
    "that's not a verb i recognize",
    "that sentence isn't one i recognize",
    "i don't understand that sentence",
    "what do you want to",
    "what do you want to ask",
    "what do you want to tie",
    "there were too many nouns",
)
ROOM_SCORE_PATTERN = re.compile(r".*\bScore:\s*\d+\s+Moves:\s*\d+\s*>?\s*$")
LOG_HEADER_PREFIXES = (
    "Log file:",
    "Markdown archive:",
    "Suggestions archive:",
    "Model:",
    "History steps:",
)
SNIPPET_LENGTH = 32


@dataclass
class CsvRow:
    situation: str
    commande_suggeree: str
    rejected: str
    reponse_parser: str


@dataclass
class LogMetadata:
    log_file: str = ""
    model: str = ""
    history_steps: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a CSV table from interactive-fiction logs with the latest game "
            "situation, the AI suggestion, and the parser response."
        )
    )
    parser.add_argument(
        "--input",
        default="logs-analysis/logs",
        help="Directory containing .txt logs, or a single .txt log file.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory where CSV files will be written. Defaults to each source log directory.",
    )
    return parser.parse_args()


def collect_input_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.name.endswith("_suggests.txt"):
            return []
        return [input_path]

    return sorted(
        path
        for path in input_path.glob("*.txt")
        if not path.name.endswith("_suggests.txt")
    )


def strip_log_preamble(lines: list[str]) -> list[str]:
    metadata_line_count = 0
    index = 0

    while index < len(lines):
        stripped = lines[index].strip()
        if any(stripped.startswith(prefix) for prefix in LOG_HEADER_PREFIXES):
            metadata_line_count += 1
            index += 1
            continue
        break

    if metadata_line_count == 0:
        return lines

    while index < len(lines) and not lines[index].strip():
        index += 1

    return lines[index:]


def extract_metadata(lines: list[str]) -> LogMetadata:
    metadata = LogMetadata()

    for line in lines[:20]:
        stripped = line.strip()
        if stripped.startswith("Log file:"):
            metadata.log_file = stripped
        elif stripped.startswith("Model:"):
            metadata.model = stripped
        elif stripped.startswith("History steps:"):
            metadata.history_steps = stripped

    return metadata


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prepare_display_text(text: str) -> str:
    cleaned_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(">"):
            continue
        if stripped == "Using normal formatting.":
            continue
        if stripped.startswith("Loading "):
            continue
        if stripped == "PLUNDERED HEARTS":
            continue
        if stripped == "Infocom interactive fiction":
            continue
        if stripped.startswith("Copyright "):
            continue
        if stripped.startswith("Release "):
            continue
        if ROOM_SCORE_PATTERN.match(stripped):
            continue
        cleaned_lines.append(stripped)

    if cleaned_lines:
        return normalize_text("\n".join(cleaned_lines))

    return ""


def first_characters(text: str, length: int = SNIPPET_LENGTH) -> str:
    return prepare_display_text(text)[:length]


def is_parser_rejection_result(result: str) -> bool:
    result_lower = result.lower()
    return any(pattern in result_lower for pattern in PARSER_REJECTION_PATTERNS)


def format_parser_response(result: str) -> str:
    prepared_result = prepare_display_text(result)
    if is_parser_rejection_result(result):
        return prepared_result
    return prepared_result[:SNIPPET_LENGTH]


def build_output_path(source_path: Path, output_dir: Path | None) -> Path:
    filename = f"{source_path.stem}.csv"
    if output_dir is None:
        return source_path.with_suffix(".csv")
    return output_dir / filename


def parse_log_file(path: Path) -> tuple[LogMetadata, list[CsvRow]]:
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    metadata = extract_metadata(raw_lines)
    lines = strip_log_preamble(raw_lines)
    rows: list[CsvRow] = []
    current_situation_lines: list[str] = []
    index = 0

    while index < len(lines):
        current_line = lines[index].strip()
        if not THINK_PATTERN.match(current_line):
            current_situation_lines.append(lines[index])
            index += 1
            continue

        situation = "\n".join(current_situation_lines).strip()
        index += 1

        while index < len(lines) and not lines[index].strip():
            index += 1

        if index >= len(lines):
            break

        suggest_match = SUGGEST_PATTERN.match(lines[index].strip())
        if not suggest_match:
            current_situation_lines = []
            index += 1
            continue

        suggested_action = suggest_match.group(1).strip()
        index += 1

        while index < len(lines) and not lines[index].strip():
            index += 1

        if index < len(lines) and not THINK_PATTERN.match(lines[index].strip()):
            index += 1

        result_lines: list[str] = []
        while index < len(lines) and not THINK_PATTERN.match(lines[index].strip()):
            result_lines.append(lines[index])
            index += 1

        result = "\n".join(result_lines).strip()
        rejected = "Y" if is_parser_rejection_result(result) else "N"
        rows.append(
            CsvRow(
                situation=first_characters(situation),
                commande_suggeree=suggested_action,
                rejected=rejected,
                reponse_parser=format_parser_response(result),
            )
        )
        current_situation_lines = result_lines

    return metadata, rows


def write_csv(metadata: LogMetadata, rows: list[CsvRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["situation", "commande_suggeree", "rejected", "reponse_parser"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "situation": metadata.log_file,
                "commande_suggeree": metadata.model,
                "rejected": metadata.history_steps,
                "reponse_parser": "",
            }
        )
        for row in rows:
            writer.writerow(
                {
                    "situation": row.situation,
                    "commande_suggeree": row.commande_suggeree,
                    "rejected": row.rejected,
                    "reponse_parser": row.reponse_parser,
                }
            )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None

    input_paths = collect_input_paths(input_path)
    total_rows = 0
    for path in input_paths:
        metadata, rows = parse_log_file(path)
        output_path = build_output_path(path, output_dir)
        write_csv(metadata, rows, output_path)
        total_rows += len(rows)
        print(f"{path.name}: {len(rows)} row(s) -> {output_path}")

    print(f"Parsed {total_rows} row(s) from {len(input_paths)} log(s)")


if __name__ == "__main__":
    main()
