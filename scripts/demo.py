"""Show base vs adapter on 10 held-out documents.

    python scripts/demo.py          # 10 items
    python scripts/demo.py --limit 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from extraction import build_prompt, parse_prediction, score_item
from mlx_lm import generate, load
from mlx_lm.sample_utils import greedy_sampler

ROOT = Path(__file__).resolve().parents[1]
MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
ADAPTER_PATH = ROOT / "adapters"
MAX_TOKENS = 160


def load_rows(limit: int) -> list[dict]:
    rows = [json.loads(line) for line in (ROOT / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    truths = [json.loads(line) for line in (ROOT / "data" / "test_truth.jsonl").read_text(encoding="utf-8").splitlines()]
    for row, truth in zip(rows, truths, strict=True):
        row["truth"] = truth
    return rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if not ADAPTER_PATH.exists():
        print(f"{ADAPTER_PATH} does not exist. Run `make train` first.", file=sys.stderr)
        raise SystemExit(2)

    rows = load_rows(args.limit)
    print(f"{len(rows)} held-out documents\n")
    for label, adapter_path in (("base", None), ("adapter", str(ADAPTER_PATH))):
        model, tokenizer = load(MODEL, adapter_path=adapter_path)
        correct = 0
        print(f"--- {label} ---")
        for row in rows:
            prompt = build_prompt(tokenizer, row["messages"])
            text = generate(model, tokenizer, prompt=prompt, max_tokens=MAX_TOKENS, sampler=greedy_sampler, verbose=False)
            prediction = parse_prediction(text)
            scores = score_item(prediction, row["truth"])
            correct += scores["exact_match"]
            shown = json.dumps(prediction, ensure_ascii=False) if prediction else text.strip()[:110]
            print(f"{row['id']} {'PASS' if scores['exact_match'] else 'FAIL'} {shown}")
        print(f"{label}: exact match {correct}/{len(rows)}\n")


if __name__ == "__main__":
    main()
