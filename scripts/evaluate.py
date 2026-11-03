"""Evaluate the base model and the LoRA adapter on the 100 held-out documents.

    python scripts/evaluate.py                  # writes results/before_after.json
    python scripts/evaluate.py --limit 10       # quick check

Decoding is greedy: temperature 0. The same prompts, the same test items, the
same deterministic verifier for both models. No LLM judge.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

from extraction import aggregate, bootstrap_delta, build_prompt, parse_prediction, score_item
from mlx_lm import generate, load
from mlx_lm.sample_utils import greedy_sampler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
ADAPTER_PATH = ROOT / "adapters"
MAX_TOKENS = 160


def load_test_rows(limit: int | None) -> list[dict]:
    rows = [json.loads(line) for line in (DATA_DIR / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    truths = [json.loads(line) for line in (DATA_DIR / "test_truth.jsonl").read_text(encoding="utf-8").splitlines()]
    for row, truth in zip(rows, truths, strict=True):
        row["truth"] = truth
    return rows[:limit] if limit else rows


def run_model(label: str, rows: list[dict], adapter_path: str | None) -> list[dict]:
    from mlx_lm import load as mlx_load

    model, tokenizer = mlx_load(MODEL, adapter_path=adapter_path)
    answers = []
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        prompt = build_prompt(tokenizer, row["messages"])
        text = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, sampler=greedy_sampler, verbose=False)
        prediction = parse_prediction(text)
        scores = score_item(prediction, row["truth"])
        answers.append({"id": row["id"], "answer": text.strip(), "parsed": prediction, **scores})
        if index % 20 == 0:
            print(f"  {label} {index}/{len(rows)} ({time.perf_counter() - started:.1f}s)", flush=True)
    print(f"  {label} done in {time.perf_counter() - started:.1f}s", flush=True)
    return answers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N test items")
    args = parser.parse_args()

    if not ADAPTER_PATH.exists():
        print(f"{ADAPTER_PATH} does not exist. Run `make train` first.", file=sys.stderr)
        raise SystemExit(2)

    rows = load_test_rows(args.limit)
    started = time.perf_counter()
    print(f"{len(rows)} test documents; base then adapter", flush=True)
    base = run_model("base", rows, None)
    adapter = run_model("adapter", rows, str(ADAPTER_PATH))

    base_metrics = aggregate(base)
    adapter_metrics = aggregate(adapter)
    delta = bootstrap_delta(base, adapter)

    result = {
        "task": "structured field extraction from synthetic invoice text",
        "model": MODEL,
        "adapter": str(ADAPTER_PATH.relative_to(ROOT)),
        "test_items": len(rows),
        "decoding": {"temperature": 0.0, "max_tokens": MAX_TOKENS, "seed": 20261103},
        "verifier": "scripts/extraction.py (deterministic; no LLM judge)",
        "environment": {
            "date": str(date.today()),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "base": base_metrics,
        "adapter": adapter_metrics,
        "delta": delta,
        "rows": [
            {
                "id": base_row["id"],
                "truth": row["truth"],
                "base_answer": base_row["answer"],
                "base_parsed": base_row["parsed"],
                "base_exact_match": base_row["exact_match"],
                "adapter_answer": adapter_row["answer"],
                "adapter_parsed": adapter_row["parsed"],
                "adapter_exact_match": adapter_row["exact_match"],
            }
            for base_row, adapter_row, row in zip(base, adapter, rows, strict=True)
        ],
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / "before_after.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    for label, metrics in (("base", base_metrics), ("adapter", adapter_metrics)):
        print(
            f"{label:8s} exact-match {metrics['exact_match']:.3f}  "
            f"field-F1 {metrics['field_f1']:.3f}  "
            f"invalid JSON {metrics['invalid_json']}/{metrics['items']}"
        )
    print(
        f"delta    exact-match {delta['exact_match_delta']:+.3f} "
        f"CI95 {delta['exact_match_ci95']}  "
        f"field-F1 {delta['field_f1_delta']:+.3f} CI95 {delta['field_f1_ci95']}"
    )
    print(f"wrote {output.relative_to(ROOT)} ({result['wall_seconds']}s)")


if __name__ == "__main__":
    main()
