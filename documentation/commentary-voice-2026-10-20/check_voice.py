"""Replay eight histories from the user's session with the current first-person prompt.

Offline commentary check: no game commands or video requests are sent.
Uses the project's existing dependencies and stores no hidden reasoning text.
"""

import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import faketerm
import ollama


def main():
    source = HERE / "source-session.jsonl"
    original = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    client = ollama.Client(host=faketerm.OLLAMA_HOST, timeout=faketerm.LLM_TIMEOUT)
    catalog = faketerm.load_video_embeddings(faketerm.VIDEO_EMBEDDINGS_PATH)
    context = faketerm.load_itw_redux()
    options = {"temperature": faketerm.LLM_TEMPERATURE, "num_ctx": faketerm.LLM_NUM_CTX,
               "num_predict": faketerm.LLM_NUM_PREDICT, "seed": 42}
    data = {"source_session": str(source.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "context": context, "context_words": len(context.split()),
            "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
            "model": faketerm.LLM_MODEL, "think": True, "options": options,
            "method": "Eight frozen histories from the user's last session; rolling memory contains only this replay's last two successful comments. One seed; exploratory reading, not a blind quality benchmark.",
            "results": []}
    recent = []
    for number in [1, 2, 4, 8, 9, 12, 14, 15]:
        old = original[number-1]
        prompt = faketerm.build_prompt(old["game_history"], interview_text=context, recent_comments=recent)
        started = time.monotonic()
        response = client.chat(model=faketerm.LLM_MODEL, messages=[{"role": "user", "content": prompt}],
                               think=True, options=options)
        comment = faketerm.clean_output(response.message.content or "").strip()
        row = {"source_comment": number, "history": old["game_history"], "old_comment": old["commentary"],
               "recent_comments": list(recent), "prompt": prompt, "commentary": comment,
               "elapsed_sec": round(time.monotonic()-started, 3), "prompt_tokens": response.prompt_eval_count,
               "output_tokens": response.eval_count, "done_reason": response.done_reason,
               "thinking_characters": len(response.message.thinking or "")}
        vector, norm = faketerm.embed_commentary_text(comment)
        ranked = faketerm.rank_video_matches(vector, norm, catalog)
        row["top_matches"] = [{"score": score, "filename": entry["filename"],
                                "passage": entry.get("matched_passage")} for score, entry in ranked[:3]]
        data["results"].append(row)
        (HERE / "results.json").write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        if comment:
            recent = (recent+[comment])[-2:]
        print(number, row["elapsed_sec"], row["done_reason"], comment, flush=True)


if __name__ == "__main__":
    main()
