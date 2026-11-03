# Dataset card: synthetic invoice field extraction

## What it is

700 synthetic invoice-like documents, split 600 train / 100 valid / 100 test.
Each document is a block of messy text. The target is one JSON object:

```json
{"vendor": "Alder Ridge Supply", "date": "2026-03-14", "total": 1234.56, "currency": "USD"}
```

The task measured in the README is: given the document, produce that object.

## How it was made

`scripts/generate_data.py`, seed `20261103`. The generator composes vendor names
from invented word lists, dates from a fixed range (2024-2026), totals, and line
items. It then renders them through one of four layouts: a block invoice, a
letter, a table, or a "Bill from" note.

Messiness, by design:

- **Vendor labels vary**: `Vendor:`, `From:`, `Supplier:`, `Bill from:`.
- **Date formats vary**: `2026-03-14`, `14/03/2026`, `March 14, 2026`,
  `14 Mar 2026`, `2026/03/14`, `Mar 14 2026`.
- **Currency forms vary**: `$`, `USD`, `EUR`, `€`, `C$`, `CAD`, `£`, `GBP`,
  as a prefix, suffix or code.
- **Totals vary**: `1,234.56`, `1234.56`, five different total labels.
- **Distractor amounts**: every document prints line-item amounts, a subtotal,
  tax and sometimes shipping. The total is the labelled total, not the sum.
- **Ambiguity is removed on purpose**: a `dd/mm/yyyy` date never has a day
  below 13, and the total is always printed with a `,` thousands separator or
  none, never a decimal comma. The document therefore has exactly one correct
  answer. This keeps the verifier deterministic rather than lucky.

No real company, person, address, invoice or currency amount is in the data.
The generator is the only input; there is no scraped text.

## Splits

| Split | Documents | Use |
|---|---|---|
| train | 600 | LoRA training |
| valid | 100 | validation loss during training |
| test | 100 | before/after evaluation; never trained on |

The splits are generated from disjoint index ranges. `data/manifest.json`
commits the SHA-256 of each split, and `make data` re-checks it.

## Label quality

The labels are exact by construction: the generator writes the document and the
answer from the same values. There is no annotator, so there is no human label
noise. The hard part is extraction, not labelling.

## Intended use and limits

- Intended: a small, reproducible supervised fine-tuning and evaluation harness.
- Not intended: a claim about real invoice accuracy. Real invoices have layouts
  this generator does not produce.
- 100 test items means one item is one percentage point. The README reports a
  bootstrap interval for that reason.
- One model, one task, one seed. Nothing here generalises on its own.
