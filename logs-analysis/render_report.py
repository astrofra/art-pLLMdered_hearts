#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import numpy as np
import umap
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

warnings.filterwarnings(
    "ignore",
    message=r"n_jobs value 1 overridden to 1 by setting random_state\. Use no seed for parallelism\.",
    category=UserWarning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a Markdown report and figures from log_metrics.json."
    )
    parser.add_argument(
        "--input",
        default="logs-analysis/output/log_metrics.json",
        help="Input JSON produced by analyze_logs.py.",
    )
    parser.add_argument(
        "--output",
        default="logs-analysis/output/report.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--figures-dir",
        default="logs-analysis/output/figures",
        help="Directory where figures will be written.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "item"


def save_figure(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path.name


def markdown_figure_path(output_path: Path, figures_dir: Path, filename: str) -> str:
    return str((figures_dir / filename).relative_to(output_path.parent))


def build_log_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for log in data["logs"]:
        metrics = log["metrics"]
        rows.append(
            {
                "filename": log["filename"],
                "steps": log["step_count"],
                "repetition_rate": metrics["repetition_rate"],
                "exploration_ratio": metrics["exploration_ratio"],
                "failure_retry_rate": metrics["failure_retry_rate"],
                "parser_rejection_rate": metrics["parser_rejection_rate"],
                "loop_count": metrics["loop_count"],
                "dead_end_zone_count": metrics["dead_end_zone_count"],
            }
        )
    return rows


def render_global_metrics(log_rows: list[dict[str, Any]], figures_dir: Path) -> str:
    labels = [row["filename"].replace(".txt", "") for row in log_rows]
    metrics = [
        ("repetition_rate", "Repetition"),
        ("exploration_ratio", "Exploration"),
        ("failure_retry_rate", "Failure retry"),
        ("parser_rejection_rate", "Parser rejection"),
    ]
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12, 5.5))
    palette = ["#0f766e", "#d97706", "#b91c1c", "#4338ca"]

    for index, ((key, title), color) in enumerate(zip(metrics, palette)):
        values = [row[key] for row in log_rows]
        ax.bar(x + (index - 1.5) * width, values, width=width, label=title, color=color)

    ax.set_title("Behavioral metrics by log")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.25)

    return save_figure(fig, figures_dir / "global_metrics.png")


def render_top_actions(data: dict[str, Any], figures_dir: Path) -> str:
    top_actions = data["aggregate"]["top_actions"][:12]
    labels = [item["action"] for item in reversed(top_actions)]
    values = [item["count"] for item in reversed(top_actions)]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(labels, values, color="#1d4ed8")
    ax.set_title("Most frequent actions across all logs")
    ax.set_xlabel("Occurrences")
    ax.grid(axis="x", alpha=0.25)

    return save_figure(fig, figures_dir / "top_actions.png")


def render_semantic_profiles(data: dict[str, Any], figures_dir: Path) -> str:
    profiles = data["aggregate"]["semantic_profiles"]
    ready_profiles = {
        backend: profile
        for backend, profile in profiles.items()
        if profile.get("status") == "ready"
    }
    backends = list(ready_profiles)
    x = np.arange(len(backends))
    width = 0.25
    metrics = [
        ("average_thought_action_similarity", "Avg similarity", "#15803d"),
        ("thought_action_divergence", "Divergence", "#c2410c"),
        ("semantic_drift_over_time", "Drift", "#7c3aed"),
    ]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for index, (key, label, color) in enumerate(metrics):
        values = [ready_profiles[backend][key] for backend in backends]
        ax.bar(x + (index - 1) * width, values, width=width, label=label, color=color)

    ax.set_title("Semantic profile comparison by backend")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(backends, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    return save_figure(fig, figures_dir / "semantic_profiles.png")


def collect_similarity_distributions(data: dict[str, Any]) -> dict[str, list[float]]:
    distributions: dict[str, list[float]] = {}
    for log in data["logs"]:
        for step in log["steps"]:
            for backend, value in step.get("semantic_similarities", {}).items():
                distributions.setdefault(backend, []).append(value)
    return distributions


def render_similarity_distributions(data: dict[str, Any], figures_dir: Path) -> str:
    distributions = collect_similarity_distributions(data)
    labels = list(distributions)
    series = [distributions[label] for label in labels]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    box = ax.boxplot(series, tick_labels=labels, patch_artist=True)
    colors = ["#4f46e5", "#059669", "#d97706", "#dc2626"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title("Thought/action similarity distributions")
    ax.set_ylabel("Cosine similarity")
    ax.grid(axis="y", alpha=0.25)

    return save_figure(fig, figures_dir / "similarity_distributions.png")


def find_dead_end_spans(log: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (zone["start_step"], zone["end_step"])
        for zone in log["metrics"]["dead_end_zones"]
    ]


def find_loop_spans(log: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (loop["start_step"], loop["end_step"])
        for loop in log["metrics"]["loops"]
    ]


def compute_repeat_flags(steps: list[dict[str, Any]]) -> list[bool]:
    seen: set[str] = set()
    flags = []
    for step in steps:
        action = step["normalized_action"]
        flags.append(action in seen)
        seen.add(action)
    return flags


def render_log_timeline(log: dict[str, Any], figures_dir: Path) -> str:
    steps = log["steps"]
    step_numbers = np.array([step["step_number"] for step in steps], dtype=int)
    failures = np.array([1 if step["is_failure"] else 0 for step in steps], dtype=float)
    parser_rejections = np.array([1 if step["is_parser_rejection"] else 0 for step in steps], dtype=float)
    repeats = np.array([1 if flag else 0 for flag in compute_repeat_flags(steps)], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
    top_ax, bottom_ax = axes

    for start, end in find_dead_end_spans(log):
        top_ax.axvspan(start, end, color="#fca5a5", alpha=0.25)
        bottom_ax.axvspan(start, end, color="#fca5a5", alpha=0.12)
    for start, end in find_loop_spans(log):
        top_ax.axvspan(start, end, color="#bfdbfe", alpha=0.25)

    top_ax.scatter(step_numbers, failures * 1.0, color="#b91c1c", s=30, label="Failure")
    top_ax.scatter(step_numbers, repeats * 0.65, color="#1d4ed8", s=24, marker="^", label="Repeated action")
    top_ax.scatter(step_numbers, parser_rejections * 1.3, color="#d97706", s=28, marker="x", label="Parser rejection")
    top_ax.set_yticks([0.65, 1.0, 1.3])
    top_ax.set_yticklabels(["Repeat", "Fail", "Reject"])
    top_ax.set_ylim(0.3, 1.55)
    top_ax.set_title(f"Timeline: {log['filename']}")
    top_ax.grid(axis="x", alpha=0.15)
    top_ax.legend(loc="upper right", ncol=3, fontsize=8)

    plotted = 0
    palette = {
        "lexical-fallback": "#4f46e5",
        "embeddinggemma": "#059669",
        "qwen3-embedding": "#d97706",
    }
    for backend in log["metrics"]["semantic_profiles"]:
        values = [
            step.get("semantic_similarities", {}).get(backend)
            for step in steps
        ]
        if not any(value is not None for value in values):
            continue
        plotted += 1
        y = [float("nan") if value is None else value for value in values]
        bottom_ax.plot(step_numbers, y, label=backend, linewidth=1.7, color=palette.get(backend))

    bottom_ax.set_ylabel("Thought/action similarity")
    bottom_ax.set_xlabel("Step")
    bottom_ax.set_ylim(0, 1)
    bottom_ax.grid(alpha=0.25)
    if plotted:
        bottom_ax.legend(loc="lower right", fontsize=8)

    filename = f"timeline_{slugify(log['filename'].replace('.txt', ''))}.png"
    return save_figure(fig, figures_dir / filename)


def project_2d(matrix: np.ndarray) -> np.ndarray:
    if len(matrix) == 0:
        return np.zeros((0, 2), dtype=float)
    if len(matrix) == 1:
        return np.array([[0.0, 0.0]], dtype=float)
    if len(matrix) == 2:
        return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)

    try:
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(20, len(matrix) - 1),
            min_dist=0.2,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(matrix)
    except Exception:
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(matrix)


def render_umap_figures(data: dict[str, Any], figures_dir: Path) -> list[tuple[str, str]]:
    artifacts = data.get("artifacts") or {}
    records_path = artifacts.get("embedding_records_json")
    vectors_path = artifacts.get("embedding_vectors_npz")
    if not records_path or not vectors_path:
        return []

    records_data = load_json(Path(records_path))
    npz = np.load(vectors_path)
    records = records_data["records"]
    results: list[tuple[str, str]] = []

    for model_name in records_data["models"]:
        if model_name not in npz:
            continue
        matrix = npz[model_name]
        points = project_2d(matrix)
        kinds = [record["kind"] for record in records]
        logs = [record["log_filename"].replace(".txt", "") for record in records]

        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
        kind_ax, log_ax = axes

        for kind, color, marker in (
            ("thought", "#0f766e", "o"),
            ("action", "#b91c1c", "^"),
        ):
            idx = [index for index, value in enumerate(kinds) if value == kind]
            kind_ax.scatter(points[idx, 0], points[idx, 1], s=18, alpha=0.45, c=color, marker=marker, label=kind)
        kind_ax.set_title(f"{model_name}: thoughts vs actions")
        kind_ax.set_xlabel("UMAP-1")
        kind_ax.set_ylabel("UMAP-2")
        kind_ax.legend()

        unique_logs = sorted(set(logs))
        cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(unique_logs))
        for index, log_name in enumerate(unique_logs):
            idx = [row for row, value in enumerate(logs) if value == log_name]
            log_ax.scatter(points[idx, 0], points[idx, 1], s=16, alpha=0.45, color=cmap(index), label=log_name)
        log_ax.set_title(f"{model_name}: projected by source log")
        log_ax.set_xlabel("UMAP-1")
        log_ax.set_ylabel("UMAP-2")
        log_ax.legend(fontsize=7, ncol=2)

        filename = f"umap_{slugify(model_name)}.png"
        results.append((model_name, save_figure(fig, figures_dir / filename)))

    return results


def render_overview_table(log_rows: list[dict[str, Any]]) -> str:
    header = (
        "| Log | Steps | Repetition | Exploration | Retry after failure | Parser rejection | Loops | Dead-ends |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows = []
    for row in log_rows:
        rows.append(
            "| {filename} | {steps} | {repetition_rate:.3f} | {exploration_ratio:.3f} | "
            "{failure_retry_rate:.3f} | {parser_rejection_rate:.3f} | {loop_count} | {dead_end_zone_count} |".format(
                **row
            )
        )
    return "\n".join([header, *rows])


def render_semantic_table(data: dict[str, Any]) -> str:
    profiles = data["aggregate"]["semantic_profiles"]
    header = (
        "| Backend | Status | Avg similarity | Divergence | Drift |\n"
        "| --- | --- | ---: | ---: | ---: |"
    )
    rows = []
    for backend, profile in profiles.items():
        if profile.get("status") != "ready":
            rows.append(f"| {backend} | {profile.get('status')} | - | - | - |")
            continue
        rows.append(
            f"| {backend} | ready | {profile['average_thought_action_similarity']:.3f} | "
            f"{profile['thought_action_divergence']:.3f} | {profile['semantic_drift_over_time']:.3f} |"
        )
    return "\n".join([header, *rows])


def build_key_findings(data: dict[str, Any], log_rows: list[dict[str, Any]]) -> list[str]:
    aggregate = data["aggregate"]
    most_repetitive = max(log_rows, key=lambda row: row["repetition_rate"])
    most_fragile = max(log_rows, key=lambda row: row["parser_rejection_rate"])
    most_retrying = max(log_rows, key=lambda row: row["failure_retry_rate"])
    top_action = aggregate["top_actions"][0]

    findings = [
        (
            f"The corpus covers {aggregate['log_count']} logs and {aggregate['total_actions']} actions, "
            f"with a global repetition rate of {aggregate['repetition_rate']:.3f} and "
            f"an exploration ratio of {aggregate['exploration_ratio']:.3f}."
        ),
        (
            f"The most repetitive run is `{most_repetitive['filename']}` at {most_repetitive['repetition_rate']:.3f}, "
            f"while `{most_retrying['filename']}` shows the highest retry-after-failure rate at "
            f"{most_retrying['failure_retry_rate']:.3f}."
        ),
        (
            f"`{most_fragile['filename']}` has the highest parser rejection rate at "
            f"{most_fragile['parser_rejection_rate']:.3f}, and the dominant action globally is "
            f"`{top_action['action']}` with {top_action['count']} occurrences."
        ),
    ]

    semantic_profiles = data["aggregate"]["semantic_profiles"]
    ready_profiles = {
        backend: profile
        for backend, profile in semantic_profiles.items()
        if profile.get("status") == "ready"
    }
    if ready_profiles:
        best_alignment = max(
            ready_profiles.items(),
            key=lambda item: item[1]["average_thought_action_similarity"],
        )
        lowest_drift = min(
            ready_profiles.items(),
            key=lambda item: item[1]["semantic_drift_over_time"],
        )
        findings.append(
            f"Among semantic backends, `{best_alignment[0]}` yields the highest mean thought/action alignment "
            f"({best_alignment[1]['average_thought_action_similarity']:.3f}), and `{lowest_drift[0]}` gives "
            f"the lowest drift over time ({lowest_drift[1]['semantic_drift_over_time']:.3f})."
        )

    return findings


def build_log_summary(log: dict[str, Any]) -> str:
    metrics = log["metrics"]
    top_action = metrics["top_actions"][0] if metrics["top_actions"] else None
    fragments = [
        f"`{log['filename']}` contains {log['step_count']} parsed steps.",
        (
            f"Repetition is {metrics['repetition_rate']:.3f}, exploration is {metrics['exploration_ratio']:.3f}, "
            f"and retry-after-failure is {metrics['failure_retry_rate']:.3f}."
        ),
        (
            f"It contains {metrics['loop_count']} detected loops and {metrics['dead_end_zone_count']} dead-end zones."
        ),
    ]
    if top_action:
        fragments.append(
            f"The most frequent action is `{top_action['action']}` ({top_action['count']} times)."
        )
    return " ".join(fragments)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    figures_dir = Path(args.figures_dir)
    figures_dir_for_markdown = figures_dir

    data = load_json(input_path)
    log_rows = build_log_rows(data)

    global_metrics_figure = render_global_metrics(log_rows, figures_dir)
    top_actions_figure = render_top_actions(data, figures_dir)
    semantic_profiles_figure = render_semantic_profiles(data, figures_dir)
    similarity_distributions_figure = render_similarity_distributions(data, figures_dir)
    umap_figures = render_umap_figures(data, figures_dir)

    timeline_figures = []
    for log in data["logs"]:
        timeline_figures.append((log["filename"], render_log_timeline(log, figures_dir)))
    timeline_map = dict(timeline_figures)

    findings = build_key_findings(data, log_rows)

    lines: list[str] = []
    lines.append("# LLM Behavioral Metrics Report")
    lines.append("")
    lines.append(f"Generated from `{input_path.name}` on `{data['generated_at']}`.")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    for finding in findings:
        lines.append(f"- {finding}")
    lines.append("")
    lines.append("## Corpus Overview")
    lines.append("")
    lines.append(render_overview_table(log_rows))
    lines.append("")
    lines.append(f"![Behavioral metrics overview]({markdown_figure_path(output_path, figures_dir_for_markdown, global_metrics_figure)})")
    lines.append("")
    lines.append(f"![Top actions]({markdown_figure_path(output_path, figures_dir_for_markdown, top_actions_figure)})")
    lines.append("")
    lines.append("## Semantic Analysis")
    lines.append("")
    lines.append(render_semantic_table(data))
    lines.append("")
    lines.append(
        "The semantic comparison uses cosine similarity for the lexical fallback and for the two Ollama "
        "embedding models. UMAP projections below are computed independently for each embedding space, "
        "so axes are only meaningful within a given model."
    )
    lines.append("")
    lines.append(f"![Semantic profiles]({markdown_figure_path(output_path, figures_dir_for_markdown, semantic_profiles_figure)})")
    lines.append("")
    lines.append(f"![Similarity distributions]({markdown_figure_path(output_path, figures_dir_for_markdown, similarity_distributions_figure)})")
    lines.append("")

    if umap_figures:
        lines.append("## Embedding Space Projections")
        lines.append("")
        for model_name, filename in umap_figures:
            lines.append(f"### {model_name}")
            lines.append("")
            lines.append(
                f"![UMAP projection for {model_name}]({markdown_figure_path(output_path, figures_dir_for_markdown, filename)})"
            )
            lines.append("")

    lines.append("## Per-Log Timelines")
    lines.append("")
    lines.append(
        "Dead-end spans are shaded in red, loop spans in blue, and the lower panel tracks thought/action similarity "
        "through time for each semantic backend."
    )
    lines.append("")
    for log in data["logs"]:
        lines.append(f"### {log['filename']}")
        lines.append("")
        lines.append(textwrap.fill(build_log_summary(log), width=100))
        lines.append("")
        figure_name = timeline_map[log["filename"]]
        lines.append(f"![Timeline for {log['filename']}]({markdown_figure_path(output_path, figures_dir_for_markdown, figure_name)})")
        lines.append("")

    if data.get("artifacts"):
        lines.append("## Artifacts")
        lines.append("")
        artifacts = data["artifacts"]
        if artifacts.get("embedding_records_json"):
            lines.append(f"- Embedding metadata: `{artifacts['embedding_records_json']}`")
        if artifacts.get("embedding_vectors_npz"):
            lines.append(f"- Embedding matrices: `{artifacts['embedding_vectors_npz']}`")
        if artifacts.get("embedding_dimensions"):
            dims = ", ".join(
                f"{name}={dim}"
                for name, dim in artifacts["embedding_dimensions"].items()
            )
            lines.append(f"- Embedding dimensions: {dims}")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    print(f"Report written to {output_path}")
    print(f"Figures written to {figures_dir}")


if __name__ == "__main__":
    main()
