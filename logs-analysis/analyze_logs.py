#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import numpy as np
import ollama


THINK_PATTERN = re.compile(r"^AI thinks\s*:\s*'(.*)'$")
SUGGEST_PATTERN = re.compile(r"^AI suggests:\s*'(.*)'$")
TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
COMMAND_TOKEN_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9'-]*$")
LEXICAL_BACKEND = "lexical-fallback"
OLLAMA_EMBEDDING_MODELS = ("embeddinggemma", "qwen3-embedding")
OLLAMA_BATCH_SIZE = 64

FAILURE_PATTERNS = (
    "and how do you propose to do that",
    "you can't",
    "you cannot",
    "i don't know the word",
    "that's not a verb i recognize",
    "that sentence isn't one i recognize",
    "you aren't holding",
    "you're not holding",
    "you're not carrying",
    "you don't have",
    "you can't see any",
    "there's no",
    "there is no",
    "nothing happens",
    "you can't go that way",
    "you can't do that",
    "i don't understand that sentence",
)

PARSER_REJECTION_PATTERNS = (
    "i don't know the word",
    "that's not a verb i recognize",
    "that sentence isn't one i recognize",
    "i don't understand that sentence",
)

STOPWORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "them",
    "there",
    "this",
    "to",
    "was",
    "with",
    "you",
    "your",
}


@dataclass
class Step:
    step_number: int
    thought: str
    suggested_action: str
    action: str
    result: str
    normalized_action: str
    is_failure: bool
    is_parser_rejection: bool
    is_syntactically_valid: bool
    is_parser_friendly: bool
    thought_action_similarity: float
    semantic_similarities: dict[str, float] = field(default_factory=dict)


@dataclass
class ParsedLog:
    path: str
    filename: str
    metadata: dict[str, Any]
    step_count: int
    steps: list[Step]
    metrics: dict[str, Any]


class SemanticEngine:
    name: str

    def prepare_texts(self, texts: list[str]) -> None:
        return None

    def similarity(self, left: str, right: str) -> float:
        raise NotImplementedError


class LexicalSemanticEngine(SemanticEngine):
    name = LEXICAL_BACKEND

    def similarity(self, left: str, right: str) -> float:
        left_counter = Counter(self._tokenize(left))
        right_counter = Counter(self._tokenize(right))

        if not left_counter or not right_counter:
            return 0.0

        intersection = set(left_counter) & set(right_counter)
        numerator = sum(left_counter[token] * right_counter[token] for token in intersection)
        left_norm = math.sqrt(sum(value * value for value in left_counter.values()))
        right_norm = math.sqrt(sum(value * value for value in right_counter.values()))

        if left_norm == 0 or right_norm == 0:
            return 0.0

        return numerator / (left_norm * right_norm)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        tokens = [token.lower() for token in TOKEN_PATTERN.findall(text)]
        return [token for token in tokens if token not in STOPWORDS]


class OllamaEmbeddingSemanticEngine(SemanticEngine):
    def __init__(self, model_name: str):
        self.name = model_name
        self.model_name = model_name
        self._cache: dict[str, list[float]] = {}

    def prepare_texts(self, texts: list[str]) -> None:
        missing = [text for text in dict.fromkeys(texts) if text and text not in self._cache]
        if not missing:
            return

        for start in range(0, len(missing), OLLAMA_BATCH_SIZE):
            batch = missing[start:start + OLLAMA_BATCH_SIZE]
            response = ollama.embed(model=self.model_name, input=batch)
            for text, embedding in zip(batch, response.embeddings):
                self._cache[text] = embedding

    def similarity(self, left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        self.prepare_texts([left, right])
        return cosine_similarity(self._cache[left], self._cache[right])

    def get_cached_embedding(self, text: str) -> list[float] | None:
        return self._cache.get(text)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0

    numerator = sum(left_value * right_value for left_value, right_value in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return numerator / (left_norm * right_norm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze interactive-fiction LLM logs and export behavioral metrics as JSON."
    )
    parser.add_argument(
        "--input",
        default="logs-analysis/logs",
        help="Directory containing .txt logs, or a single .txt log file.",
    )
    parser.add_argument(
        "--output",
        default="logs-analysis/output/log_metrics.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--no-steps",
        action="store_true",
        help="Exclude the parsed step list from the JSON export.",
    )
    parser.add_argument(
        "--skip-ollama-embeddings",
        action="store_true",
        help="Only compute the lexical semantic profile.",
    )
    return parser.parse_args()


def normalize_action(action: str) -> str:
    return " ".join(action.upper().split())


def tokenize_command(action: str) -> list[str]:
    return [token for token in normalize_action(action).split(" ") if token]


def is_failure_result(result: str) -> bool:
    result_lower = result.lower()
    return any(pattern in result_lower for pattern in FAILURE_PATTERNS)


def is_parser_rejection_result(result: str) -> bool:
    result_lower = result.lower()
    return any(pattern in result_lower for pattern in PARSER_REJECTION_PATTERNS)


def is_syntactically_valid_command(action: str) -> bool:
    tokens = tokenize_command(action)
    if not tokens or len(tokens) > 6:
        return False
    return all(COMMAND_TOKEN_PATTERN.fullmatch(token) for token in tokens)


def is_parser_friendly_command(action: str) -> bool:
    tokens = tokenize_command(action)
    if not tokens or len(tokens) > 6:
        return False
    if len(tokens) <= 2:
        return True
    return any(
        token in {"AT", "IN", "INSIDE", "INTO", "ON", "UNDER", "THROUGH", "WITH", "FROM", "TO"}
        for token in tokens[1:]
    )


def parse_log_file(path: Path, lexical_engine: LexicalSemanticEngine) -> ParsedLog:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    metadata = extract_metadata(lines, path)
    steps: list[Step] = []
    index = 0

    while index < len(lines):
        current = lines[index].strip()
        thought_match = THINK_PATTERN.match(current)
        if not thought_match:
            index += 1
            continue

        thought = thought_match.group(1).strip()
        index += 1

        while index < len(lines) and not lines[index].strip():
            index += 1

        if index >= len(lines):
            break

        suggest_match = SUGGEST_PATTERN.match(lines[index].strip())
        if not suggest_match:
            index += 1
            continue

        suggested_action = suggest_match.group(1).strip()
        index += 1

        while index < len(lines) and not lines[index].strip():
            index += 1

        action = suggested_action
        if index < len(lines) and not THINK_PATTERN.match(lines[index].strip()):
            action = lines[index].strip()
            index += 1

        result_lines: list[str] = []
        while index < len(lines) and not THINK_PATTERN.match(lines[index].strip()):
            result_lines.append(lines[index])
            index += 1

        result = "\n".join(result_lines).strip()
        normalized_action = normalize_action(action)
        lexical_similarity = round(lexical_engine.similarity(thought, action), 6)

        steps.append(
            Step(
                step_number=len(steps) + 1,
                thought=thought,
                suggested_action=suggested_action,
                action=action,
                result=result,
                normalized_action=normalized_action,
                is_failure=is_failure_result(result),
                is_parser_rejection=is_parser_rejection_result(result),
                is_syntactically_valid=is_syntactically_valid_command(action),
                is_parser_friendly=is_parser_friendly_command(action),
                thought_action_similarity=lexical_similarity,
                semantic_similarities={LEXICAL_BACKEND: lexical_similarity},
            )
        )

    metrics = compute_metrics(steps, {LEXICAL_BACKEND: lexical_engine})
    return ParsedLog(
        path=str(path),
        filename=path.name,
        metadata=metadata,
        step_count=len(steps),
        steps=steps,
        metrics=metrics,
    )


def extract_metadata(lines: list[str], path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {"source": str(path)}

    for line in lines[:20]:
        stripped = line.strip()
        if stripped.startswith("Model:"):
            metadata["model"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("History steps:"):
            value = stripped.split(":", 1)[1].strip()
            metadata["history_steps"] = int(value) if value.isdigit() else value
        elif stripped.startswith("Log file:"):
            metadata["log_file"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Suggestions archive:"):
            metadata["suggestions_archive"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Markdown archive:"):
            metadata["markdown_archive"] = stripped.split(":", 1)[1].strip()

    return metadata


def enrich_logs_with_semantics(logs: list[ParsedLog], engines: dict[str, SemanticEngine]) -> dict[str, dict[str, str]]:
    statuses: dict[str, dict[str, str]] = {
        engine_name: {"status": "ready"}
        for engine_name in engines
    }

    unique_texts = collect_unique_texts(logs)

    for engine_name, engine in engines.items():
        if engine_name == LEXICAL_BACKEND:
            continue

        try:
            engine.prepare_texts(unique_texts)
            for log in logs:
                for step in log.steps:
                    similarity = round(engine.similarity(step.thought, step.action), 6)
                    step.semantic_similarities[engine_name] = similarity
        except Exception as error:
            statuses[engine_name] = {
                "status": "unavailable",
                "error": f"{type(error).__name__}: {error}",
            }

    for log in logs:
        log.metrics = compute_metrics(log.steps, engines, statuses)

    return statuses


def collect_unique_texts(logs: list[ParsedLog]) -> list[str]:
    ordered: dict[str, None] = {}
    for log in logs:
        for step in log.steps:
            if step.thought:
                ordered.setdefault(step.thought, None)
            if step.action:
                ordered.setdefault(step.action, None)
    return list(ordered)


def export_embedding_artifacts(
    logs: list[ParsedLog],
    semantic_engines: dict[str, SemanticEngine],
    semantic_statuses: dict[str, dict[str, str]],
    output_path: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    arrays_by_model: dict[str, list[list[float]]] = {}
    dimensions: dict[str, int] = {}

    for engine_name, engine in semantic_engines.items():
        if engine_name == LEXICAL_BACKEND:
            continue
        if semantic_statuses.get(engine_name, {}).get("status") != "ready":
            continue
        if isinstance(engine, OllamaEmbeddingSemanticEngine):
            arrays_by_model[engine_name] = []

    if not arrays_by_model:
        return {}

    for log in logs:
        for step in log.steps:
            shared_metadata = {
                "log_filename": log.filename,
                "step_number": step.step_number,
                "is_failure": step.is_failure,
                "is_parser_rejection": step.is_parser_rejection,
            }
            for kind, text in (("thought", step.thought), ("action", step.action)):
                record = {
                    "record_index": len(records),
                    "kind": kind,
                    "text": text,
                    **shared_metadata,
                }
                records.append(record)

                for engine_name, engine in semantic_engines.items():
                    if engine_name not in arrays_by_model:
                        continue
                    embedding = engine.get_cached_embedding(text) if isinstance(engine, OllamaEmbeddingSemanticEngine) else None
                    if embedding is None:
                        raise RuntimeError(f"Missing cached embedding for model={engine_name} text={text!r}")
                    arrays_by_model[engine_name].append(embedding)

    records_path = output_path.with_name(f"{output_path.stem}_embedding_records.json")
    vectors_path = output_path.with_name(f"{output_path.stem}_embedding_vectors.npz")

    npz_payload: dict[str, np.ndarray] = {}
    for engine_name, rows in arrays_by_model.items():
        matrix = np.asarray(rows, dtype=np.float32)
        npz_payload[engine_name] = matrix
        dimensions[engine_name] = int(matrix.shape[1]) if matrix.ndim == 2 else 0

    records_path.write_text(
        json.dumps(
            {
                "record_count": len(records),
                "models": sorted(arrays_by_model),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    np.savez_compressed(vectors_path, **npz_payload)

    return {
        "embedding_records_json": str(records_path),
        "embedding_vectors_npz": str(vectors_path),
        "embedding_record_count": len(records),
        "embedding_dimensions": dimensions,
    }


def compute_metrics(
    steps: list[Step],
    semantic_engines: dict[str, SemanticEngine],
    semantic_statuses: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    total_actions = len(steps)
    semantic_statuses = semantic_statuses or {
        engine_name: {"status": "ready"} for engine_name in semantic_engines
    }

    if total_actions == 0:
        return empty_metrics(semantic_statuses)

    actions = [step.normalized_action for step in steps]
    failures = [step.is_failure for step in steps]
    parser_rejections = [step.is_parser_rejection for step in steps]
    counts = Counter(actions)

    seen_actions: set[str] = set()
    repeated_actions = 0
    new_action_steps = 0
    consecutive_repeats = 0

    for position, action in enumerate(actions):
        if action in seen_actions:
            repeated_actions += 1
        else:
            new_action_steps += 1
            seen_actions.add(action)

        if position > 0 and action == actions[position - 1]:
            consecutive_repeats += 1

    failure_retry_count = count_failure_retries(actions, failures)
    cognitive_inertia = compute_cognitive_inertia(actions, failures)
    loops = detect_failed_loops(actions, failures)
    dead_ends = detect_dead_end_zones(actions, failures)
    obsession = compute_obsession_score(actions)
    semantic_profiles = build_semantic_profiles(steps, semantic_engines, semantic_statuses)
    lexical_profile = semantic_profiles[LEXICAL_BACKEND]

    top_actions = [
        {
            "action": action,
            "count": count,
            "share": round(count / total_actions, 6),
        }
        for action, count in counts.most_common(10)
    ]

    return {
        "semantic_backend": LEXICAL_BACKEND,
        "semantic_profiles": semantic_profiles,
        "total_actions": total_actions,
        "unique_actions": len(counts),
        "total_failures": sum(failures),
        "parser_rejections": sum(parser_rejections),
        "repeated_actions": repeated_actions,
        "new_action_steps": new_action_steps,
        "consecutive_repeats": consecutive_repeats,
        "repetition_rate": round(repeated_actions / total_actions, 6),
        "consecutive_repeat_rate": round(consecutive_repeats / total_actions, 6),
        "action_diversity": round(len(counts) / total_actions, 6),
        "exploration_ratio": round(new_action_steps / total_actions, 6),
        "failure_retry_rate": round(failure_retry_count / sum(failures), 6) if any(failures) else 0.0,
        "failure_retry_count": failure_retry_count,
        "thought_action_divergence": lexical_profile["thought_action_divergence"],
        "average_thought_action_similarity": lexical_profile["average_thought_action_similarity"],
        "semantic_drift_over_time": lexical_profile["semantic_drift_over_time"],
        "syntactic_validity_rate": round(sum(step.is_syntactically_valid for step in steps) / total_actions, 6),
        "parser_friendly_rate": round(sum(step.is_parser_friendly for step in steps) / total_actions, 6),
        "parser_rejection_rate": round(sum(parser_rejections) / total_actions, 6),
        "obsession_score": obsession["score"],
        "obsession_action": obsession["action"],
        "obsession_action_occurrences": obsession["occurrences"],
        "obsession_max_consecutive_repeats": obsession["max_consecutive_repeats"],
        "cognitive_inertia": round(cognitive_inertia, 6),
        "loop_count": len(loops),
        "loops": loops,
        "dead_end_zone_count": len(dead_ends),
        "dead_end_zones": dead_ends,
        "top_actions": top_actions,
    }


def build_semantic_profiles(
    steps: list[Step],
    semantic_engines: dict[str, SemanticEngine],
    semantic_statuses: dict[str, dict[str, str]],
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}

    for engine_name, engine in semantic_engines.items():
        status = semantic_statuses.get(engine_name, {"status": "ready"})
        if status.get("status") != "ready":
            profiles[engine_name] = {
                "status": status.get("status"),
                "error": status.get("error"),
                "average_thought_action_similarity": None,
                "thought_action_divergence": None,
                "semantic_drift_over_time": None,
            }
            continue

        similarities = [
            step.semantic_similarities.get(engine_name)
            for step in steps
            if step.semantic_similarities.get(engine_name) is not None
        ]
        drifts = [
            round(1.0 - engine.similarity(steps[index - 1].thought, steps[index].thought), 6)
            for index in range(1, len(steps))
        ]
        average_similarity = round(mean(similarities), 6) if similarities else 0.0

        profiles[engine_name] = {
            "status": "ready",
            "average_thought_action_similarity": average_similarity,
            "thought_action_divergence": round(1.0 - average_similarity, 6),
            "semantic_drift_over_time": round(mean(drifts), 6) if drifts else 0.0,
        }

    return profiles


def empty_metrics(semantic_statuses: dict[str, dict[str, str]]) -> dict[str, Any]:
    semantic_profiles = {}
    for engine_name, status in semantic_statuses.items():
        semantic_profiles[engine_name] = {
            "status": status.get("status"),
            "error": status.get("error"),
            "average_thought_action_similarity": None if status.get("status") != "ready" else 0.0,
            "thought_action_divergence": None if status.get("status") != "ready" else 0.0,
            "semantic_drift_over_time": None if status.get("status") != "ready" else 0.0,
        }

    return {
        "semantic_backend": LEXICAL_BACKEND,
        "semantic_profiles": semantic_profiles,
        "total_actions": 0,
        "unique_actions": 0,
        "total_failures": 0,
        "parser_rejections": 0,
        "repeated_actions": 0,
        "new_action_steps": 0,
        "consecutive_repeats": 0,
        "repetition_rate": 0.0,
        "consecutive_repeat_rate": 0.0,
        "action_diversity": 0.0,
        "exploration_ratio": 0.0,
        "failure_retry_rate": 0.0,
        "failure_retry_count": 0,
        "thought_action_divergence": 0.0,
        "average_thought_action_similarity": 0.0,
        "semantic_drift_over_time": 0.0,
        "syntactic_validity_rate": 0.0,
        "parser_friendly_rate": 0.0,
        "parser_rejection_rate": 0.0,
        "obsession_score": 0.0,
        "obsession_action": None,
        "obsession_action_occurrences": 0,
        "obsession_max_consecutive_repeats": 0,
        "cognitive_inertia": 0.0,
        "loop_count": 0,
        "loops": [],
        "dead_end_zone_count": 0,
        "dead_end_zones": [],
        "top_actions": [],
    }


def count_failure_retries(actions: list[str], failures: list[bool]) -> int:
    positions_by_action: dict[str, list[int]] = defaultdict(list)
    for index, action in enumerate(actions):
        positions_by_action[action].append(index)

    retries = 0
    for index, (action, failed) in enumerate(zip(actions, failures)):
        if not failed:
            continue
        if any(next_index > index for next_index in positions_by_action[action]):
            retries += 1
    return retries


def compute_cognitive_inertia(actions: list[str], failures: list[bool]) -> float:
    if not actions:
        return 0.0

    inertia_values: list[int] = []
    for index, failed in enumerate(failures):
        if not failed:
            continue

        repeats = 0
        next_index = index + 1
        while next_index < len(actions) and actions[next_index] == actions[index]:
            repeats += 1
            next_index += 1
        inertia_values.append(repeats)

    return mean(inertia_values) if inertia_values else 0.0


def detect_failed_loops(actions: list[str], failures: list[bool], max_cycle_length: int = 4) -> list[dict[str, Any]]:
    loops: list[dict[str, Any]] = []
    index = 0

    while index < len(actions):
        best_loop: dict[str, Any] | None = None
        max_size = min(max_cycle_length, (len(actions) - index) // 2)

        for size in range(max_size, 0, -1):
            pattern = actions[index:index + size]
            repeats = 1
            end_index = index + size

            while end_index + size <= len(actions) and actions[end_index:end_index + size] == pattern:
                repeats += 1
                end_index += size

            if repeats < 2:
                continue

            failure_slice = failures[index:end_index]
            failure_rate = sum(failure_slice) / len(failure_slice)
            if failure_rate < 0.75:
                continue

            best_loop = {
                "start_step": index + 1,
                "end_step": end_index,
                "cycle_length": size,
                "repeats": repeats,
                "failure_rate": round(failure_rate, 6),
                "pattern": pattern,
            }
            break

        if best_loop is not None:
            loops.append(best_loop)
            index = best_loop["end_step"]
        else:
            index += 1

    return loops


def detect_dead_end_zones(actions: list[str], failures: list[bool], window_size: int = 6) -> list[dict[str, Any]]:
    if len(actions) < window_size:
        return []

    candidate_windows: list[tuple[int, int]] = []
    for start in range(0, len(actions) - window_size + 1):
        end = start + window_size
        window_actions = actions[start:end]
        window_failures = failures[start:end]
        unique_ratio = len(set(window_actions)) / window_size
        failure_rate = sum(window_failures) / window_size
        consecutive_repeat_count = sum(
            1 for idx in range(start + 1, end) if actions[idx] == actions[idx - 1]
        )

        if unique_ratio <= 0.5 and failure_rate >= 0.5:
            candidate_windows.append((start, end))
        elif failure_rate >= 0.67 and consecutive_repeat_count >= 2:
            candidate_windows.append((start, end))

    merged_windows = merge_windows(candidate_windows)
    dead_ends: list[dict[str, Any]] = []

    for start, end in merged_windows:
        segment_actions = actions[start:end]
        segment_failures = failures[start:end]
        action_counts = Counter(segment_actions)
        dead_ends.append(
            {
                "start_step": start + 1,
                "end_step": end,
                "length": end - start,
                "failure_rate": round(sum(segment_failures) / len(segment_failures), 6),
                "unique_action_ratio": round(len(set(segment_actions)) / len(segment_actions), 6),
                "dominant_actions": [
                    {"action": action, "count": count}
                    for action, count in action_counts.most_common(5)
                ],
            }
        )

    return dead_ends


def merge_windows(windows: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    sorted_windows = sorted(windows)
    if not sorted_windows:
        return []

    merged = [sorted_windows[0]]
    for start, end in sorted_windows[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def compute_obsession_score(actions: list[str]) -> dict[str, Any]:
    if not actions:
        return {
            "action": None,
            "score": 0.0,
            "occurrences": 0,
            "max_consecutive_repeats": 0,
        }

    counts = Counter(actions)
    max_runs: dict[str, int] = defaultdict(int)
    current_action = actions[0]
    current_run = 1

    for action in actions[1:]:
        if action == current_action:
            current_run += 1
        else:
            max_runs[current_action] = max(max_runs[current_action], current_run)
            current_action = action
            current_run = 1
    max_runs[current_action] = max(max_runs[current_action], current_run)

    best_action = None
    best_score = -1.0
    best_occurrences = 0
    best_run = 0

    for action, occurrences in counts.items():
        max_run = max_runs.get(action, 1)
        if occurrences < 2 and max_run < 2:
            continue
        score = max_run / occurrences
        if score > best_score or (score == best_score and occurrences > best_occurrences):
            best_action = action
            best_score = score
            best_occurrences = occurrences
            best_run = max_run

    if best_action is None:
        return {
            "action": None,
            "score": 0.0,
            "occurrences": 0,
            "max_consecutive_repeats": 0,
        }

    return {
        "action": best_action,
        "score": round(best_score, 6),
        "occurrences": best_occurrences,
        "max_consecutive_repeats": best_run,
    }


def collect_input_paths(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.glob("*.txt") if not path.name.endswith("_suggests.txt"))


def build_aggregate(
    logs: list[ParsedLog],
    semantic_statuses: dict[str, dict[str, str]],
) -> dict[str, Any]:
    total_actions = sum(log.metrics["total_actions"] for log in logs)
    total_failures = sum(log.metrics["total_failures"] for log in logs)
    total_parser_rejections = sum(log.metrics["parser_rejections"] for log in logs)

    all_actions: list[str] = []
    all_loops = 0
    all_dead_ends = 0
    repeated_actions = 0
    new_action_steps = 0
    consecutive_repeats = 0
    failure_retry_count = 0
    syntactic_valid_actions = 0
    parser_friendly_actions = 0
    global_counts: Counter[str] = Counter()
    global_max_runs: dict[str, int] = defaultdict(int)

    semantic_weighted: dict[str, dict[str, float]] = {}
    for backend_name in semantic_statuses:
        semantic_weighted[backend_name] = {
            "similarity_sum": 0.0,
            "drift_sum": 0.0,
            "action_count": 0,
            "transition_count": 0,
        }

    for log in logs:
        steps = log.steps
        actions = [step.normalized_action for step in steps]
        all_actions.extend(actions)
        repeated_actions += log.metrics["repeated_actions"]
        new_action_steps += log.metrics["new_action_steps"]
        consecutive_repeats += log.metrics["consecutive_repeats"]
        failure_retry_count += log.metrics["failure_retry_count"]
        syntactic_valid_actions += sum(step.is_syntactically_valid for step in steps)
        parser_friendly_actions += sum(step.is_parser_friendly for step in steps)
        all_loops += log.metrics["loop_count"]
        all_dead_ends += log.metrics["dead_end_zone_count"]

        for backend_name, profile in log.metrics["semantic_profiles"].items():
            if profile.get("status") != "ready":
                continue
            semantic_weighted[backend_name]["similarity_sum"] += (
                profile["average_thought_action_similarity"] * log.metrics["total_actions"]
            )
            semantic_weighted[backend_name]["action_count"] += log.metrics["total_actions"]
            if log.metrics["total_actions"] > 1:
                transitions = log.metrics["total_actions"] - 1
                semantic_weighted[backend_name]["drift_sum"] += (
                    profile["semantic_drift_over_time"] * transitions
                )
                semantic_weighted[backend_name]["transition_count"] += transitions

        current_action = None
        current_run = 0
        for action in actions:
            global_counts[action] += 1
            if action == current_action:
                current_run += 1
            else:
                if current_action is not None:
                    global_max_runs[current_action] = max(global_max_runs[current_action], current_run)
                current_action = action
                current_run = 1
        if current_action is not None:
            global_max_runs[current_action] = max(global_max_runs[current_action], current_run)

    obsession = compute_global_obsession(global_counts, global_max_runs)
    aggregate_semantic_profiles = {}
    for backend_name, status in semantic_statuses.items():
        if status.get("status") != "ready":
            aggregate_semantic_profiles[backend_name] = {
                "status": status.get("status"),
                "error": status.get("error"),
                "average_thought_action_similarity": None,
                "thought_action_divergence": None,
                "semantic_drift_over_time": None,
            }
            continue

        weighted = semantic_weighted[backend_name]
        avg_similarity = (
            round(weighted["similarity_sum"] / weighted["action_count"], 6)
            if weighted["action_count"]
            else 0.0
        )
        avg_drift = (
            round(weighted["drift_sum"] / weighted["transition_count"], 6)
            if weighted["transition_count"]
            else 0.0
        )
        aggregate_semantic_profiles[backend_name] = {
            "status": "ready",
            "average_thought_action_similarity": avg_similarity,
            "thought_action_divergence": round(1.0 - avg_similarity, 6),
            "semantic_drift_over_time": avg_drift,
        }

    lexical_profile = aggregate_semantic_profiles[LEXICAL_BACKEND]

    return {
        "log_count": len(logs),
        "semantic_profiles": aggregate_semantic_profiles,
        "total_actions": total_actions,
        "unique_actions_global": len(set(all_actions)),
        "total_failures": total_failures,
        "parser_rejections": total_parser_rejections,
        "repeated_actions": repeated_actions,
        "new_action_steps": new_action_steps,
        "consecutive_repeats": consecutive_repeats,
        "repetition_rate": round(repeated_actions / total_actions, 6) if total_actions else 0.0,
        "consecutive_repeat_rate": round(consecutive_repeats / total_actions, 6) if total_actions else 0.0,
        "action_diversity": round(len(set(all_actions)) / total_actions, 6) if total_actions else 0.0,
        "exploration_ratio": round(new_action_steps / total_actions, 6) if total_actions else 0.0,
        "failure_retry_rate": round(failure_retry_count / total_failures, 6) if total_failures else 0.0,
        "failure_retry_count": failure_retry_count,
        "average_thought_action_similarity": lexical_profile["average_thought_action_similarity"],
        "thought_action_divergence": lexical_profile["thought_action_divergence"],
        "semantic_drift_over_time": lexical_profile["semantic_drift_over_time"],
        "syntactic_validity_rate": round(syntactic_valid_actions / total_actions, 6) if total_actions else 0.0,
        "parser_friendly_rate": round(parser_friendly_actions / total_actions, 6) if total_actions else 0.0,
        "parser_rejection_rate": round(total_parser_rejections / total_actions, 6) if total_actions else 0.0,
        "obsession_score": obsession["score"],
        "obsession_action": obsession["action"],
        "obsession_action_occurrences": obsession["occurrences"],
        "obsession_max_consecutive_repeats": obsession["max_consecutive_repeats"],
        "loop_count": all_loops,
        "dead_end_zone_count": all_dead_ends,
        "top_actions": [
            {"action": action, "count": count, "share": round(count / total_actions, 6)}
            for action, count in global_counts.most_common(15)
        ],
    }


def compute_global_obsession(global_counts: Counter[str], global_max_runs: dict[str, int]) -> dict[str, Any]:
    if not global_counts:
        return {
            "action": None,
            "score": 0.0,
            "occurrences": 0,
            "max_consecutive_repeats": 0,
        }

    best_action = None
    best_score = -1.0
    best_occurrences = 0
    best_run = 0

    for action, occurrences in global_counts.items():
        max_run = global_max_runs.get(action, 1)
        if occurrences < 2 and max_run < 2:
            continue
        score = max_run / occurrences
        if score > best_score or (score == best_score and occurrences > best_occurrences):
            best_action = action
            best_score = score
            best_occurrences = occurrences
            best_run = max_run

    if best_action is None:
        return {
            "action": None,
            "score": 0.0,
            "occurrences": 0,
            "max_consecutive_repeats": 0,
        }

    return {
        "action": best_action,
        "score": round(best_score, 6),
        "occurrences": best_occurrences,
        "max_consecutive_repeats": best_run,
    }


def serialize_log(log: ParsedLog, include_steps: bool) -> dict[str, Any]:
    payload = {
        "path": log.path,
        "filename": log.filename,
        "metadata": log.metadata,
        "step_count": log.step_count,
        "metrics": log.metrics,
    }
    if include_steps:
        payload["steps"] = [asdict(step) for step in log.steps]
    return payload


def build_semantic_engines(skip_ollama_embeddings: bool) -> dict[str, SemanticEngine]:
    engines: dict[str, SemanticEngine] = {
        LEXICAL_BACKEND: LexicalSemanticEngine(),
    }
    if not skip_ollama_embeddings:
        for model_name in OLLAMA_EMBEDDING_MODELS:
            engines[model_name] = OllamaEmbeddingSemanticEngine(model_name)
    return engines


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_engines = build_semantic_engines(args.skip_ollama_embeddings)
    lexical_engine = semantic_engines[LEXICAL_BACKEND]

    input_paths = collect_input_paths(input_path)
    logs = [parse_log_file(path, lexical_engine) for path in input_paths]
    semantic_statuses = enrich_logs_with_semantics(logs, semantic_engines)
    artifacts = export_embedding_artifacts(logs, semantic_engines, semantic_statuses, output_path)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output": str(output_path),
        "semantic_backend": LEXICAL_BACKEND,
        "semantic_backends": semantic_statuses,
        "artifacts": artifacts,
        "analysis_notes": {
            "failure_detection": "Keyword-based heuristic on game results.",
            "command_validity": "Approximation based on parser-shaped command tokens and command length.",
            "semantic_metrics": (
                "Lexical cosine fallback plus Ollama embeddings for embeddinggemma and qwen3-embedding "
                "when available. Raw embedding vectors are exported as a compressed NPZ artifact."
            ),
            "loop_detection": "Repeated action cycles with a failure rate of at least 75%.",
            "dead_end_detection": "Sliding windows with high failure rates and low action novelty.",
        },
        "logs": [serialize_log(log, include_steps=not args.no_steps) for log in logs],
        "aggregate": build_aggregate(logs, semantic_statuses),
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Analyzed {len(logs)} log(s) from {input_path}")
    print(f"Semantic backends: {', '.join(payload['semantic_backends'])}")
    print(f"Total actions: {payload['aggregate']['total_actions']}")
    print(f"Total failures: {payload['aggregate']['total_failures']}")
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    main()
