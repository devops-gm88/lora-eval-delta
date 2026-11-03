"""Generate the synthetic invoice-extraction dataset.

Deterministic: the same seed and counts produce byte-identical JSONL.
    python scripts/generate_data.py            # writes data/{train,valid,test}.jsonl
    python scripts/generate_data.py --check    # verify committed data matches the seed

No real company, person or document. Every vendor is invented.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SEED = 20261103
SPLITS = {"train": 600, "valid": 100, "test": 100}

SYSTEM = (
    "You extract structured fields from messy invoice text. "
    "Reply with one JSON object and nothing else. "
    "Keys: vendor, date, total, currency. "
    "date is YYYY-MM-DD. total is a number with two decimals. "
    "currency is one of USD, EUR, CAD, GBP."
)

PREFIXES = [
    "Alder", "Birch", "Cobalt", "Dawn", "Elm", "Foundry", "Granite", "Harbor",
    "Iron", "Juniper", "Kestrel", "Larkspur", "Mica", "Northline", "Orchard",
    "Pinecrest", "Quarry", "Ridge", "Sable", "Tamarack", "Union", "Vantage",
    "Willow", "Yarrow", "Zephyr", "Beacon", "Copper", "Drift", "Ember", "Fathom",
]
SUFFIXES = [
    "Supply", "Logistics", "Parts", "Foods", "Industrial", "Trading",
    "Materials", "Systems", "Works", "Distribution",
]
ITEMS = [
    "hex bolts M8", "packing tape", "cold-rolled steel sheet", "safety gloves",
    "pallet wrap", "hydraulic hose", "circuit board", "toner cartridge",
    "copper wire spool", "steel strapping", "bearing assembly", "gasket set",
]
CURRENCIES = ["USD", "EUR", "CAD", "GBP"]
CURRENCY_FORMS = {
    "USD": ["${amount}", "USD {amount}", "{amount} USD"],
    "EUR": ["\u20ac{amount}", "EUR {amount}", "{amount} EUR"],
    "CAD": ["C${amount}", "CAD {amount}", "{amount} CAD"],
    "GBP": ["\u00a3{amount}", "GBP {amount}", "{amount} GBP"],
}
TOTAL_LABELS = ["Total:", "Total due:", "Amount due:", "Grand total:", "Balance due:", "Invoice total:"]
DATE_FORMATS = ["iso", "dmy", "month_name", "short_month", "ymd_slash", "dmy_short"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def vendor_name(rng: random.Random) -> str:
    return f"{rng.choice(PREFIXES)} {rng.choice(SUFFIXES)}"


def format_date(day: date, style: str) -> str:
    if style == "iso":
        return day.isoformat()
    if style == "dmy":
        return f"{day.day:02d}/{day.month:02d}/{day.year}"
    if style == "month_name":
        return f"{MONTHS[day.month - 1]} {day.day}, {day.year}"
    if style == "short_month":
        return f"{day.day:02d} {MONTH_ABBR[day.month - 1]} {day.year}"
    if style == "ymd_slash":
        return f"{day.year}/{day.month:02d}/{day.day:02d}"
    if style == "dmy_short":
        return f"{MONTH_ABBR[day.month - 1]} {day.day} {day.year}"
    raise ValueError(style)


def format_amount(total: float, rng: random.Random) -> str:
    whole, cents = f"{total:.2f}".split(".")
    if total >= 1000 and rng.random() < 0.5:
        return f"{int(whole):,}.{cents}"
    return f"{total:.2f}"


def make_document(rng: random.Random, index: int) -> dict:
    vendor = vendor_name(rng)
    currency = rng.choice(CURRENCIES)
    day = date(2024, 1, 1) + timedelta(days=rng.randrange(0, 900))
    total = round(rng.uniform(10, 9800), 2)
    date_style = rng.choice(DATE_FORMATS)
    if date_style == "dmy" and day.day <= 12:
        # "03/04/2026" has two readings. Move the day into the unambiguous range
        # so the document has exactly one correct date.
        day = day.replace(day=rng.randrange(13, 29))
    date_text = format_date(day, date_style)
    number = f"INV-{day.year}-{1000 + index:04d}"
    total_text = rng.choice(CURRENCY_FORMS[currency]).format(amount=format_amount(total, rng))

    items = []
    for _ in range(rng.randrange(1, 5)):
        quantity = rng.randrange(1, 9)
        unit = round(rng.uniform(5, 320), 2)
        items.append((rng.choice(ITEMS), quantity, unit))
    subtotal = round(sum(quantity * unit for _, quantity, unit in items), 2)
    tax_rate = rng.choice([0.05, 0.08, 0.13, 0.20])
    tax = round(subtotal * tax_rate, 2)
    # The printed total is the truth. Line-item arithmetic is approximate on
    # purpose: these are messy documents, not a spreadsheet.
    line = "\n".join(f"  {desc:<26} x{quantity:<2} {unit:>8.2f} {quantity * unit:>9.2f}"
                     for desc, quantity, unit in items)
    layout = rng.choice(["block", "letter", "table", "bill_from"])
    invoice_word = rng.choice(["Invoice", "Statement", "Bill"])

    if layout == "letter":
        body = (
            f"{vendor}\n"
            f"{rng.randrange(10, 990)} {rng.choice(['Mill', 'Foundry', 'Harbor', 'Cedar'])} "
            f"{rng.choice(['Street', 'Road', 'Avenue'])}\n"
            f"Reference {number}\n\n"
            f"From: {vendor}\n"
            f"Document date: {date_text}\n\n"
            f"{invoice_word} detail:\n{line}\n\n"
            f"Subtotal: {subtotal:.2f}\n"
            f"Tax ({int(tax_rate * 100)}%): {tax:.2f}\n"
            f"{rng.choice(TOTAL_LABELS)} {total_text}\n\n"
            f"Payment within 30 days. Questions: accounts@example.invalid\n"
        )
    elif layout == "table":
        body = (
            f"{invoice_word.upper()} {number}\n"
            f"Supplier: {vendor}\n"
            f"Date of issue: {date_text}\n\n"
            f"Description                Qty    Unit    Amount\n"
            f"{line}\n\n"
            f"Subtotal                  {subtotal:.2f}\n"
            f"Tax                       {tax:.2f}\n"
            f"{rng.choice(TOTAL_LABELS):<24} {total_text}\n"
        )
    elif layout == "bill_from":
        body = (
            f"Bill from: {vendor}\n"
            f"Invoice no. {number}\n"
            f"{rng.choice(['Billing date', 'Date', 'Issued'])}: {date_text}\n\n"
            f"Items supplied:\n{line}\n\n"
            f"Net amount: {subtotal:.2f}\n"
            f"VAT: {tax:.2f}\n"
            f"{rng.choice(TOTAL_LABELS)} {total_text}\n"
            f"PO ref {rng.randrange(10000, 99999)}. Thank you for your business.\n"
        )
    else:
        body = (
            f"{vendor}\n"
            f"Vendor invoice {number}\n"
            f"Date: {date_text}\n\n"
            f"{line}\n\n"
            f"Subtotal: {subtotal:.2f}\n"
            f"Tax: {tax:.2f}\n"
            f"Shipping: {rng.choice([0, 15, 25, 40]):.2f}\n"
            f"{rng.choice(TOTAL_LABELS)} {total_text}\n"
        )

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Extract the fields from this document:\n\n{body.strip()}"},
        {
            "role": "assistant",
            "content": json.dumps(
                {"vendor": vendor, "date": day.isoformat(), "total": total, "currency": currency},
                ensure_ascii=False,
            ),
        },
    ]
    return {
        "id": f"D{index:04d}",
        "messages": messages,
        "truth": {"vendor": vendor, "date": day.isoformat(), "total": total, "currency": currency},
    }


def build() -> dict[str, list[dict]]:
    rng = random.Random(SEED)
    # One RNG stream per split so changing one count does not rewrite the others.
    splits = {}
    for name, count in SPLITS.items():
        rows = []
        start = {"train": 0, "valid": 600, "test": 700}[name]
        for index in range(start, start + count):
            rows.append(make_document(rng, index))
        splits[name] = rows
    return splits


def write(splits: dict[str, list[dict]]) -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    manifest = {"seed": SEED, "splits": {}, "system_prompt": SYSTEM}
    for name, rows in splits.items():
        payload = "".join(
            json.dumps({"id": row["id"], "messages": row["messages"]}, ensure_ascii=False) + "\n"
            for row in rows
        )
        path = DATA_DIR / f"{name}.jsonl"
        path.write_text(payload, encoding="utf-8")
        truth = "".join(json.dumps(row["truth"], ensure_ascii=False) + "\n" for row in rows)
        (DATA_DIR / f"{name}_truth.jsonl").write_text(truth, encoding="utf-8")
        manifest["splits"][name] = {
            "documents": len(rows),
            "sha256": hashlib.sha256(payload.encode()).hexdigest(),
        }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def check(manifest: dict) -> int:
    for name in SPLITS:
        payload = (DATA_DIR / f"{name}.jsonl").read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        expected = manifest["splits"][name]["sha256"]
        if digest != expected:
            print(f"MISMATCH {name}: {digest} != {expected}")
            return 1
        print(f"ok {name} {digest[:16]}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify the committed data matches this seed")
    args = parser.parse_args()
    if args.check:
        manifest = json.loads((DATA_DIR / "manifest.json").read_text())
        raise SystemExit(check(manifest))
    manifest = write(build())
    for name, info in manifest["splits"].items():
        print(f"{name}: {info['documents']} documents, sha256 {info['sha256'][:16]}")
    print("wrote data/{train,valid,test}.jsonl and data/manifest.json")


if __name__ == "__main__":
    main()
