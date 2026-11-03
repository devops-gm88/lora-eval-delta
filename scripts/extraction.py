"""Parsing, normalisation and scoring for the invoice-extraction task.

Pure functions: no model, no MLX. The verifier is deterministic.
"""
from __future__ import annotations

import json
import random
import re

FIELDS = ("vendor", "date", "total", "currency")
CURRENCIES = ("USD", "EUR", "CAD", "GBP")

CURRENCY_ALIASES = {
    "$": "USD", "US$": "USD", "USD": "USD",
    "\u20ac": "EUR", "EUR": "EUR",
    "C$": "CAD", "CAD": "CAD",
    "\u00a3": "GBP", "GBP": "GBP",
}

DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%B %d, %Y", "%b %d, %Y", "%d %b %Y", "%b %d %Y")


def parse_prediction(text: str) -> dict | None:
    """Return the first JSON object in the answer, or None.

    Accepts a bare object, a fenced block or an object embedded in prose. A
    model that never emits a JSON object scores zero fields; it is not repaired.
    """
    if not text:
        return None
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None


def _normalise_date(value) -> str | None:
    from datetime import datetime

    if not isinstance(value, str):
        return None
    text = value.strip().strip(".,;")
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    # The model may wrap the date in a label: "Date: 14/03/2026".
    match = re.search(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{4}|[A-Za-z]{3,9} \d{1,2},? \d{4}|\d{1,2} [A-Za-z]{3,9} \d{4})", text)
    if match:
        for fmt in DATE_FORMATS:
            try:
                return datetime.strptime(match.group(1), fmt).date().isoformat()
            except ValueError:
                continue
    return None


def _normalise_total(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if not isinstance(value, str):
        return None
    text = value.strip().replace(",", "").replace(" ", "")
    text = re.sub(r"[^0-9.\-]", "", text)
    if text.count(".") > 1:
        # "1.234.56" is unreadable; treat as a miss rather than guess.
        return None
    try:
        return round(float(text), 2)
    except ValueError:
        return None


def _normalise_currency(value) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    return CURRENCY_ALIASES.get(text) if text in CURRENCY_ALIASES else (text if text in CURRENCIES else None)


def normalise(field: str, value) -> str | float | None:
    if field == "vendor":
        if not isinstance(value, str):
            return None
        text = re.sub(r"\s+", " ", value).strip().lower()
        return text.rstrip(" .,;:") or None
    if field == "date":
        return _normalise_date(value)
    if field == "total":
        return _normalise_total(value)
    if field == "currency":
        return _normalise_currency(value)
    raise ValueError(field)


def score_item(prediction: dict | None, truth: dict) -> dict:
    """Score one prediction. A wrong value counts once as a false positive and
    once as a false negative; a missing field is a false negative; extra keys are
    false positives."""
    if prediction is None:
        return {
            "exact_match": False,
            "field_correct": {field: False for field in FIELDS},
            "field_present": {field: False for field in FIELDS},
            "tp": 0, "fp": 0, "fn": len(FIELDS),
            "invalid_json": True,
        }
    correct, present = {}, {}
    tp = fp = fn = 0
    for field in FIELDS:
        present[field] = field in prediction
        matched = present[field] and normalise(field, prediction[field]) == normalise(field, truth[field])
        correct[field] = matched
        if matched:
            tp += 1
        elif present[field]:
            fp += 1
            fn += 1
        else:
            fn += 1
    extra = [key for key in prediction if key not in FIELDS]
    fp += len(extra)
    return {
        "exact_match": all(correct.values()) and not extra,
        "field_correct": correct,
        "field_present": present,
        "tp": tp, "fp": fp, "fn": fn,
        "invalid_json": False,
        "extra_keys": extra,
    }


def aggregate(scores: list[dict]) -> dict:
    tp = sum(score["tp"] for score in scores)
    fp = sum(score["fp"] for score in scores)
    fn = sum(score["fn"] for score in scores)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "items": len(scores),
        "exact_match": round(sum(score["exact_match"] for score in scores) / len(scores), 4),
        "field_precision": round(precision, 4),
        "field_recall": round(recall, 4),
        "field_f1": round(f1, 4),
        "invalid_json": sum(score["invalid_json"] for score in scores),
        "per_field_accuracy": {
            field: round(sum(score["field_correct"][field] for score in scores) / len(scores), 4)
            for field in FIELDS
        },
    }


def bootstrap_delta(base: list[dict], adapter: list[dict], resamples: int = 2000, seed: int = 20261103) -> dict:
    """Paired percentile bootstrap of the adapter-minus-base delta on the same items."""
    n = len(base)
    rng = random.Random(seed)

    def mean(values):
        return sum(values) / len(values)

    em_deltas, f1_deltas = [], []
    for _ in range(resamples):
        indices = [rng.randrange(n) for _ in range(n)]
        base_em = mean([base[i]["exact_match"] for i in indices])
        adapter_em = mean([adapter[i]["exact_match"] for i in indices])
        em_deltas.append(adapter_em - base_em)
        base_f1 = _micro_f1([base[i] for i in indices])["field_f1"]
        adapter_f1 = _micro_f1([adapter[i] for i in indices])["field_f1"]
        f1_deltas.append(adapter_f1 - base_f1)

    def interval(values):
        ordered = sorted(values)
        low = ordered[int(0.025 * len(ordered))]
        high = ordered[int(0.975 * len(ordered)) - 1]
        return [round(low, 4), round(high, 4)]

    point_em = mean([item["exact_match"] for item in adapter]) - mean([item["exact_match"] for item in base])
    point_f1 = _micro_f1(adapter)["field_f1"] - _micro_f1(base)["field_f1"]
    return {
        "exact_match_delta": round(point_em, 4),
        "exact_match_ci95": interval(em_deltas),
        "field_f1_delta": round(point_f1, 4),
        "field_f1_ci95": interval(f1_deltas),
        "resamples": resamples,
        "seed": seed,
    }


def _micro_f1(scores: list[dict]) -> dict:
    tp = sum(score["tp"] for score in scores)
    fp = sum(score["fp"] for score in scores)
    fn = sum(score["fn"] for score in scores)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"field_f1": round(f1, 4)}


def build_prompt(tokenizer, messages: list[dict]) -> str:
    """The assistant turn is the target; the prompt is everything before it."""
    return tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=False)
