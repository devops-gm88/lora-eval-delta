# lora-eval-delta

A LoRA fine-tune of **Qwen2.5-0.5B-Instruct** on a synthetic structured-extraction
task, evaluated before and after on the same 100 held-out documents with a
**deterministic verifier**. The point is the measured delta, not the adapter.

**Result, 2026-11-03:** exact match **35 / 100 → 100 / 100**. Field-level F1
**0.540 → 1.000**. Paired bootstrap 95% CI on the exact-match delta:
**[+0.55, +0.74]**.

## Before and after

Base and adapter, same 100 test items, same prompts, greedy decoding
(temperature 0), same verifier. Source: `results/before_after.json`.

| Metric | Base | Adapter | Delta | 95% CI (paired bootstrap) |
|---|---|---|---|---|
| Exact match | 0.350 (35/100) | **1.000 (100/100)** | **+0.650** | [+0.55, +0.74] |
| Field-level F1 | 0.540 | **1.000** | **+0.460** | [+0.371, +0.564] |
| Invalid JSON | 61 / 100 | 0 / 100 | — | — |

Per field, exact-match accuracy:

| Field | Base | Adapter |
|---|---|---|
| vendor | 0.390 | 1.000 |
| date | 0.390 | 1.000 |
| total | 0.380 | 1.000 |
| currency | 0.360 | 1.000 |

The bootstrap is 2,000 paired resamples of the 100 test items, seed 20261103
(`scripts/extraction.py`). One item is one percentage point; the interval is
reported for that reason.

### Read the delta honestly

The base model fails in two different ways, and the delta mixes them. In 61 of
100 items it never finished a valid JSON object inside the 160-token budget: it
wrote a code fence, or prose, or a truncated object. Its field precision when it
did emit JSON was 0.9325, so it can copy values; it could not reliably commit to
the output format. The adapter fixes the format **and** the extraction. This
experiment does not separate the two effects.

The task is in-distribution: test documents come from the same generator family
as training documents. A near-perfect score is expected here and says nothing
about real invoices. There is no human-labelled holdout.

## The task

Input: a messy synthetic invoice as text, with vendor labels such as `From:` or
`Bill from:`, dates in six formats, currency symbols and codes, and distractor
amounts (line items, subtotal, tax). Output: one JSON object:

```json
{"vendor": "Alder Ridge Supply", "date": "2026-03-14", "total": 1234.56, "currency": "USD"}
```

Verifier: `scripts/extraction.py`. Vendor is compared case-insensitively after
whitespace collapse; date is parsed and must be the same calendar day; total is
compared as a decimal rounded to 2 places; currency must be one of the four
codes. Wrong values, missing fields and extra keys are scored as errors; a model
that never emits a JSON object scores zero fields. No LLM judge, no fuzzy match.
The full data description is in `DATASET_CARD.md`.

## Training

| Setting | Value |
|---|---|
| Base model | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` |
| Toolchain | Apple MLX `mlx-lm==0.32.2` |
| Fine-tune type | LoRA, 16 layers, rank 8, scale 20 |
| Iterations | 600 |
| Batch size | 4 |
| Learning rate | 1e-4 (AdamW) |
| Max sequence | 1024 tokens |
| Prompt masking | on (loss on the answer only) |
| Seed | 20261103 |
| Trainable parameters | 2.933M / 494.033M (0.594%) |

Exact command: `make train`, which runs
`.venv/bin/mlx_lm.lora -c config/lora_config.yaml` and tees the log to
`logs/train.log`. The config is committed at `config/lora_config.yaml`. The raw
training log is committed at `logs/train.log`; the loss curve is
`results/loss.csv` and `results/loss_curve.png` (train and validation loss over
the 600 iterations, parsed from that log).

## Reproduction

Prerequisites: macOS on Apple Silicon, Python 3.14 (the MLX wheels exist for
3.14; verified on this machine). The base model downloads from Hugging Face on
first use (about 280 MB) into `.cache/`, which is gitignored. No paid API.

```bash
make setup        # python3 -m venv .venv; install requirements
make data         # regenerate 600/100/100 documents from the seed and check hashes
make train        # LoRA 600 iterations; writes logs/train.log, results/loss.csv, results/loss_curve.png
make eval         # base vs adapter on 100 test items; writes results/before_after.json
make demo         # base vs adapter side by side on 10 held-out documents
make test         # unit tests (8)
```

`make train` takes about 32 minutes on this machine; `make eval` takes about
2.5 minutes. The adapter weights are **not committed** (`adapters/` is
gitignored); `make train` rebuilds them deterministically.

## Layout

```
config/lora_config.yaml   the training configuration
scripts/generate_data.py  seeded dataset generator
scripts/extraction.py     parsing, normalisation, scoring, bootstrap
scripts/evaluate.py       base vs adapter, writes before_after.json
scripts/demo.py           10 items, side by side
scripts/plot_loss.py      log -> results/loss.csv and results/loss_curve.png
data/                     600/100/100 JSONL + truth + hash manifest
adapters/                 LoRA weights (gitignored, rebuilt by make train)
results/                  before_after.json, loss.csv, loss_curve.png
tests/                    8 tests
```

## Limits

- One model, one task family, one seed, 100 test items.
- The training data and the test data share a generator. Good scores here are a
  check of the harness, not evidence of real-world extraction quality.
- The base model was given the same 160-token budget as the adapter. A larger
  budget would have changed how often it finished valid JSON.
- No quantisation-aware evaluation was done for the adapter vs base beyond using
  the same 4-bit weights for both.
