"""Tests for the deterministic parts: data generation and the verifier."""
from __future__ import annotations

import json

import generate_data
import pytest
from extraction import aggregate, bootstrap_delta, parse_prediction, score_item


def test_splits_have_the_documented_sizes():
    splits = generate_data.build()
    assert {name: len(rows) for name, rows in splits.items()} == {"train": 600, "valid": 100, "test": 100}


def test_generation_is_deterministic():
    first = generate_data.build()
    second = generate_data.build()
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_committed_data_matches_the_seed():
    manifest = json.loads((generate_data.DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert generate_data.check(manifest) == 0


def test_every_document_has_one_json_truth():
    truths = [json.loads(line) for line in (generate_data.DATA_DIR / "test_truth.jsonl").read_text(encoding="utf-8").splitlines()]
    documents = [json.loads(line) for line in (generate_data.DATA_DIR / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(truths) == len(documents) == 100
    for document, truth in zip(documents, truths, strict=True):
        roles = [message["role"] for message in document["messages"]]
        assert roles == ["system", "user", "assistant"]
        target = json.loads(document["messages"][-1]["content"])
        assert target == truth
        assert target["currency"] in ("USD", "EUR", "CAD", "GBP")
        assert len(target["date"]) == 10 and target["date"][4] == "-"
        assert round(target["total"], 2) == target["total"]


def test_parse_prediction_accepts_fences_and_prose():
    payload = '{"vendor": "Alder Ridge Supply", "date": "2026-03-14", "total": 1234.56, "currency": "USD"}'
    assert parse_prediction(payload)["vendor"] == "Alder Ridge Supply"
    assert parse_prediction(f"Here it is:\n```json\n{payload}\n```")["total"] == 1234.56
    assert parse_prediction("no json here") is None
    assert parse_prediction("") is None


def test_scoring_credits_exact_and_counts_partial():
    truth = {"vendor": "Alder Ridge Supply", "date": "2026-03-14", "total": 1234.56, "currency": "USD"}
    perfect = score_item(dict(truth), truth)
    assert perfect["exact_match"] and perfect["tp"] == 4

    wrong_total = score_item({**truth, "total": 123.45}, truth)
    assert not wrong_total["exact_match"]
    assert wrong_total["tp"] == 3 and wrong_total["fp"] == 1 and wrong_total["fn"] == 1

    missing = score_item({"vendor": truth["vendor"]}, truth)
    assert missing["tp"] == 1 and missing["fn"] == 3 and not missing["invalid_json"]

    extra = score_item({**truth, "note": "x"}, truth)
    assert not extra["exact_match"] and extra["extra_keys"] == ["note"]

    assert score_item(None, truth)["invalid_json"]


def test_aggregate_micro_f1_matches_hand_count():
    truth = {"vendor": "A", "date": "2026-03-14", "total": 1.0, "currency": "USD"}
    scores = [score_item(dict(truth), truth), score_item({"vendor": "B", "date": "x", "total": 2.0, "currency": "CAD"}, truth)]
    metrics = aggregate(scores)
    # 4 correct fields, 4 wrong present fields: precision 0.5, recall 0.5, F1 0.5
    assert metrics["field_f1"] == 0.5
    assert metrics["exact_match"] == 0.5
    assert metrics["per_field_accuracy"]["vendor"] == 0.5


def test_bootstrap_is_deterministic_and_brackets_the_delta():
    truth = {"vendor": "A", "date": "2026-03-14", "total": 1.0, "currency": "USD"}
    base = [score_item({"vendor": "B", "date": "2026-03-14", "total": 1.0, "currency": "USD"}, truth) for _ in range(20)]
    adapter = [score_item(dict(truth), truth) for _ in range(20)]
    first = bootstrap_delta(base, adapter, resamples=200)
    second = bootstrap_delta(base, adapter, resamples=200)
    assert first == second
    assert first["exact_match_delta"] == 1.0
    assert first["exact_match_ci95"] == [1.0, 1.0]
