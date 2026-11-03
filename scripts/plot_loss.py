"""Turn an mlx_lm.lora log into results/loss.csv and results/loss_curve.png."""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
# mlx-lm 0.32 prints a table:
#     10    0.261 ▼       59      1.8k
#    100    val 0.204     1.2s
# Older versions print "Iter 10: Train loss 0.261". Both are accepted.
TRAIN_PATTERN = re.compile(r"^\s*(\d+)\s+([\d.]+)\s*[\u25bc\u25b2]?\s")
VAL_PATTERN = re.compile(r"^\s*(\d+)\s+val\s+([\d.]+)")
LEGACY_PATTERN = re.compile(r"Iter (\d+): (Train|Val) loss ([\d.]+)")


def parse(log_path: Path) -> list[tuple[int, str, float]]:
    raw = log_path.read_bytes().decode("utf-8", errors="replace")
    raw = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw)
    points = []
    for line in re.split(r"[\r\n]+", raw):
        line = line.strip()
        legacy = LEGACY_PATTERN.search(line)
        if legacy:
            points.append((int(legacy.group(1)), legacy.group(2).lower(), float(legacy.group(3))))
            continue
        val = VAL_PATTERN.match(line)
        if val:
            points.append((int(val.group(1)), "val", float(val.group(2))))
            continue
        train = TRAIN_PATTERN.match(line)
        if train:
            points.append((int(train.group(1)), "train", float(train.group(2))))
    return points


def main() -> None:
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "train.log"
    if not log_path.exists():
        print(f"{log_path} does not exist. Run `make train` first.", file=sys.stderr)
        raise SystemExit(2)
    points = parse(log_path)
    if not points:
        print(f"no loss lines in {log_path}", file=sys.stderr)
        raise SystemExit(2)
    RESULTS.mkdir(exist_ok=True)
    csv_path = RESULTS / "loss.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "split", "loss"])
        writer.writerows(points)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5), dpi=150)
    for split, colour in (("train", "#1f6feb"), ("val", "#d29922")):
        series = [(iteration, loss) for iteration, name, loss in points if name == split]
        if series:
            axis.plot([x for x, _ in series], [y for _, y in series], label=split, color=colour, linewidth=1.6)
    axis.set_xlabel("iteration")
    axis.set_ylabel("loss")
    axis.set_title("LoRA fine-tune: Qwen2.5-0.5B-Instruct on synthetic invoice extraction")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    png_path = RESULTS / "loss_curve.png"
    figure.savefig(png_path)
    print(f"wrote {csv_path.relative_to(ROOT)} ({len(points)} points) and {png_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
