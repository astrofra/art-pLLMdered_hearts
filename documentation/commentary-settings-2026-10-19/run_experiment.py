"""Compare sampling and interview contexts, then replay finalists in order."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ollama
import faketerm

CONTEXT_FILES = {
    "original": "abriggs-itw-750-words.txt",
    "prose": "abriggs-itw-facts-prose.txt",
    "cards": "abriggs-itw-facts-cards.txt",
    "brief": "abriggs-itw-facts-brief.txt",
}
TEMPERATURES = (0.3, 0.6, 0.9)
SEEDS = (42, 43)


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare_contexts():
    facts = json.loads((DIRECTORY / "facts.json").read_text(encoding="utf-8"))
    formats = {
        "prose": " ".join(fact["text"] for fact in facts),
        "cards": "\n\n".join(fact["title"] + ":\n" + fact["text"] for fact in facts),
        "brief": "\n".join(fact["title"] + ": " + fact["brief"] for fact in facts),
    }
    for key, text in formats.items():
        (ROOT / "assets" / CONTEXT_FILES[key]).write_text(text + "\n", encoding="utf-8")
    contexts = {}
    for key, name in CONTEXT_FILES.items():
        text = (ROOT / "assets" / name).read_text(encoding="utf-8").strip()
        contexts[key] = {"file": "assets/" + name, "text": text, "words": len(text.split()),
                         "sha256": hashlib.sha256(text.encode()).hexdigest()}
    assert " ".join(fact["text"] for fact in facts) == contexts["prose"]["text"]
    return contexts


def full_walkthrough_buffers():
    rows = json.loads((ROOT / "documentation/commentary-buffer-analysis/walkthrough.json").read_text(encoding="utf-8"))
    samples = []
    history = []
    for row in rows[:-1]:
        history.append(row["game_text"])
        if sum(len(text.split()) for text in history) > 150:
            samples.append({"id": "step_%03d" % row["step"], "step": row["step"],
                            "game_history": "\n".join(history)})
            history = []
        history.append(">> " + row["command"])
    return samples


def sample(client, context, game_history, recent, temperature, seed):
    prompt = faketerm.build_prompt(game_history, interview_text=context, recent_comments=recent)
    options = {"temperature": temperature, "num_ctx": 4096, "num_predict": 2560, "seed": seed}
    record = {"prompt": prompt, "recent_comments": list(recent), "options": options}
    start = time.monotonic()
    try:
        response = client.chat(model="gemma4:e4b", messages=[{"role": "user", "content": prompt}],
                               think=True, options=options, keep_alive="30m")
        text = faketerm.clean_output(response.message.content or "").strip()
        record.update(commentary=text, done_reason=response.done_reason,
                      prompt_tokens=response.prompt_eval_count, output_tokens=response.eval_count,
                      thinking_characters=len(response.message.thinking or ""),
                      prompt_eval_sec=(response.prompt_eval_duration or 0) / 1e9,
                      generation_sec=(response.eval_duration or 0) / 1e9)
    except Exception as error:
        record.update(commentary="", error=str(error))
    record["elapsed_sec"] = round(time.monotonic() - start, 3)
    record["words"] = len(record["commentary"].split())
    tokens = lambda text: re.findall(r"[a-z0-9']+", text.casefold())
    record["longest_shared_words"] = max((
        difflib.SequenceMatcher(None, tokens(record["commentary"]), tokens(old), autojunk=False)
        .find_longest_match().size for old in recent), default=0)
    return record


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["condition"], []).append(row)
    return [{"condition": name, "responses": len(group),
             "complete_nonempty": sum(bool(row["commentary"]) and row.get("done_reason") == "stop" for row in group),
             "errors": sum("error" in row for row in group),
             "median_sec": round(statistics.median(row["elapsed_sec"] for row in group), 3),
             "max_words": max(row["words"] for row in group),
             "max_shared_words": max(row["longest_shared_words"] for row in group),
             "technical_word_flags": sum(bool(re.search(r"\b(memory|technical|hardware|commodore|storage)\b", row["commentary"], re.I)) for row in group),
             "genre_word_flags": sum(bool(re.search(r"\b(romance|romantic|swashbuckling|genre)\b", row["commentary"], re.I)) for row in group)}
            for name, group in sorted(groups.items())]


def announce(row, completed, total):
    print(json.dumps({"progress": "%s/%s" % (completed, total),
                      **{key: row.get(key) for key in ("condition", "case_id", "seed", "elapsed_sec", "done_reason", "commentary")}},
                     ensure_ascii=False), flush=True)


def run_screen(client, contexts):
    path = DIRECTORY / "screen.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        source = json.loads((ROOT / "documentation/commentary-prompt-review-2026-10-19/results.json").read_text(encoding="utf-8"))
        cases = [{"id": "session_%02d" % index, "game_history": source["results"][index - 1]["game_history"],
                  "recent_comments": source["results"][index - 1]["recent_comments"]}
                 for index in (1, 8, 11, 17)]
        models = client.list().models
        data = {"protocol": {"model": "gemma4:e4b", "model_digest": next(model.digest for model in models if model.model == "gemma4:e4b"),
                             "think": True, "num_ctx": 4096, "num_predict": 2560, "seeds": SEEDS,
                             "temperatures": TEMPERATURES, "ordering": "Condition order rotated by case and reversed on the second seed.",
                             "memory": "The same fixed two prior comments for each case and all conditions; no memory at the opening.",
                             "quality": "Human-readable paired outputs; lexical flags do not score quality.",
                             "template_sha256": hashlib.sha256(faketerm.build_prompt("", interview_text="").encode()).hexdigest()},
                "contexts": contexts, "cases": cases, "warmups": [], "results": []}
    assert data["contexts"] == contexts
    if not data["warmups"]:
        for name, context in contexts.items():
            data["warmups"].append(sample(client, context["text"], data["cases"][0]["game_history"], [], 0.6, 41))
            save_json(path, data)
            print("Warmup " + name, flush=True)
    conditions = [(name, temperature) for name in contexts for temperature in TEMPERATURES]
    finished = {(row["condition"], row["case_id"], row["seed"]) for row in data["results"]}
    total = len(conditions) * len(data["cases"]) * len(SEEDS)
    for repeat, seed in enumerate(SEEDS):
        for case_index, case in enumerate(data["cases"]):
            shift = ((repeat * len(data["cases"]) + case_index) * 5) % len(conditions)
            order = conditions[shift:] + conditions[:shift]
            if repeat:
                order = list(reversed(order))
            for name, temperature in order:
                label = name + "_t" + str(temperature)
                if (label, case["id"], seed) in finished:
                    continue
                row = sample(client, contexts[name]["text"], case["game_history"], case["recent_comments"], temperature, seed)
                row.update(condition=label, context=name, temperature=temperature, case_id=case["id"], seed=seed)
                data["results"].append(row)
                data["summary"] = summarize(data["results"])
                save_json(path, data)
                announce(row, len(data["results"]), total)
    print(json.dumps(data["summary"]), flush=True)


def run_sequence(client, contexts, condition, seed):
    name, temperature_text = condition.split(":")
    temperature = float(temperature_text)
    label = name + "_t" + str(temperature)
    path = DIRECTORY / ("sequence_" + label + "_s" + str(seed) + ".json")
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {"protocol": {"model": "gemma4:e4b", "think": True, "num_ctx": 4096,
                             "num_predict": 2560, "condition": label, "seed": seed,
                             "memory": "Last two successful responses from this sequence, oldest first.",
                             "scope": "Complete captured walkthrough buffered at 150 words; no viewer requests."},
                "context": contexts[name], "cases": full_walkthrough_buffers(), "results": []}
    recent = [row["commentary"] for row in data["results"] if row["commentary"]][-2:]
    for case in data["cases"][len(data["results"]):]:
        row = sample(client, contexts[name]["text"], case["game_history"], recent, temperature, seed)
        row.update(condition=label, context=name, temperature=temperature, case_id=case["id"], seed=seed)
        data["results"].append(row)
        if row["commentary"]:
            recent = (recent + [row["commentary"]])[-2:]
        data["summary"] = summarize(data["results"])
        save_json(path, data)
        announce(row, len(data["results"]), len(data["cases"]))


def run_finalists(client, contexts, conditions, seed):
    screen = json.loads((DIRECTORY / "screen.json").read_text(encoding="utf-8"))
    assert screen["protocol"]["template_sha256"] == hashlib.sha256(
        faketerm.build_prompt("", interview_text="").encode()).hexdigest()
    cases = full_walkthrough_buffers()
    datasets = []
    for condition in conditions:
        name, temperature_text = condition.split(":")
        temperature = float(temperature_text)
        label = name + "_t" + str(temperature)
        path = DIRECTORY / ("sequence_" + label + "_s" + str(seed) + ".json")
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = {"protocol": {"model": "gemma4:e4b", "model_digest": screen["protocol"]["model_digest"],
                                 "think": True, "num_ctx": 4096, "num_predict": 2560,
                                 "condition": label, "seed": seed,
                                 "ordering": "Four conditions interleaved at every buffer, with rotating order.",
                                 "memory": "Last two successful comments from this condition only.",
                                 "scope": "Complete captured walkthrough at 150 words, no viewer requests."},
                    "context": contexts[name], "cases": cases, "results": []}
        assert data["context"] == contexts[name] and data["cases"] == cases
        datasets.append((name, temperature, label, path, data))
    total = len(cases) * len(datasets)
    completed = sum(len(item[4]["results"]) for item in datasets)
    for case_index, case in enumerate(cases):
        shift = case_index % len(datasets)
        for name, temperature, label, path, data in datasets[shift:] + datasets[:shift]:
            if case_index < len(data["results"]):
                continue
            recent = [row["commentary"] for row in data["results"] if row["commentary"]][-2:]
            row = sample(client, contexts[name]["text"], case["game_history"], recent, temperature, seed)
            row.update(condition=label, context=name, temperature=temperature, case_id=case["id"], seed=seed)
            data["results"].append(row)
            data["summary"] = summarize(data["results"])
            save_json(path, data)
            completed += 1
            announce(row, completed, total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "screen", "sequence", "finalists"), default="screen")
    parser.add_argument("--condition", help="Context and temperature, e.g. cards:0.6")
    parser.add_argument("--conditions", nargs="+", default=["original:0.3", "cards:0.6", "brief:0.3", "brief:0.9"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    contexts = prepare_contexts()
    print(json.dumps({name: context["words"] for name, context in contexts.items()}), flush=True)
    if args.stage == "prepare":
        return
    client = ollama.Client(host=faketerm.OLLAMA_HOST, timeout=faketerm.LLM_TIMEOUT)
    if args.stage == "screen":
        run_screen(client, contexts)
    elif args.stage == "finalists":
        run_finalists(client, contexts, args.conditions, args.seed)
    else:
        if not args.condition:
            parser.error("--condition is required for sequence replay")
        run_sequence(client, contexts, args.condition, args.seed)


if __name__ == "__main__":
    main()
