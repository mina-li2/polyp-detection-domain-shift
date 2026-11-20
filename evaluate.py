"""
evaluate.py
-----------
Tests every trained model on:
  1. Kvasir test set   (same hospital it trained on)
  2. CVC-ClinicDB test (a different hospital it has NEVER seen)

The gap between those two numbers is the headline of your project.

Outputs (in results/):
  results.csv          table of precision / recall / mAP for each model x dataset
  domain_shift.png     bar chart: same-hospital vs new-hospital performance
  cvc_examples_<exp>.jpg  sample CVC frames, green = doctor's box, red = model's box

Usage:
    python evaluate.py
"""

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

EXPERIMENTS = ["baseline", "strongaug", "finetune"]
DATASETS = {
    "Kvasir (same hospital)": "data/kvasir.yaml",
    "CVC-ClinicDB (new hospital)": "data/cvc_test.yaml",
}
OUT = Path("results")


def evaluate_all():
    rows = []
    for exp in EXPERIMENTS:
        weights = Path(f"runs/{exp}/weights/best.pt")
        if not weights.exists():
            print(f"skipping {exp}: no weights at {weights}")
            continue
        model = YOLO(str(weights))
        for ds_name, yaml in DATASETS.items():
            m = model.val(data=yaml, split="test", project=str(Path("runs/eval").resolve()),
                          name=f"{exp}_{Path(yaml).stem}", exist_ok=True,
                          plots=False, verbose=False)
            rows.append({
                "experiment": exp,
                "dataset": ds_name,
                "precision": round(float(m.box.mp), 3),
                "recall (sensitivity)": round(float(m.box.mr), 3),
                "mAP50": round(float(m.box.map50), 3),
                "mAP50-95": round(float(m.box.map), 3),
            })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame):
    pivot = df.pivot(index="experiment", columns="dataset", values="mAP50")
    pivot = pivot.reindex([e for e in EXPERIMENTS if e in pivot.index])
    cols = list(DATASETS.keys())
    x = np.arange(len(pivot))
    width = 0.38
    colors = ["#3B6EA8", "#D9822B"]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, col in enumerate(cols):
        vals = pivot[col].values
        bars = ax.bar(x + (i - 0.5) * width, vals, width, label=col, color=colors[i])
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9)
    ax.set_xticks(x, pivot.index)
    ax.set_ylabel("mAP@0.5")
    ax.set_ylim(0, 1.05)
    ax.set_title("Polyp detection: same hospital vs a new hospital")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "domain_shift.png", dpi=200)
    print(f"saved {OUT / 'domain_shift.png'}")


def example_grid(exp: str, n: int = 8, seed: int = 0):
    """Draw doctor boxes (green) vs model boxes (red) on random CVC test frames."""
    weights = Path(f"runs/{exp}/weights/best.pt")
    if not weights.exists():
        return
    model = YOLO(str(weights))
    img_dir = Path("data/yolo/cvc/images/test")
    imgs = sorted(img_dir.iterdir())
    random.Random(seed).shuffle(imgs)

    tiles = []
    for img_path in imgs[:n]:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        lbl = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
        for line in lbl.read_text().splitlines():
            _, cx, cy, bw, bh = map(float, line.split())
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        for box in model.predict(str(img_path), conf=0.25, verbose=False)[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 230), 2)
            cv2.putText(img, f"{float(box.conf):.2f}", (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 230), 1)
        tiles.append(cv2.resize(img, (384, 288)))

    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    out = OUT / f"cvc_examples_{exp}.jpg"
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"saved {out}  (green = ground truth, red = model)")


def main():
    OUT.mkdir(exist_ok=True)
    df = evaluate_all()
    if df.empty:
        raise SystemExit("No trained models found. Run train.py first.")
    df.to_csv(OUT / "results.csv", index=False)
    print("\n" + df.to_string(index=False) + "\n")
    plot(df)
    for exp in EXPERIMENTS:
        example_grid(exp)


if __name__ == "__main__":
    main()
