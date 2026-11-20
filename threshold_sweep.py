"""
threshold_sweep.py
------------------
Missing a polyp is worse than a false alarm. This script shows the trade-off:
for each confidence threshold, how many real polyps are caught (recall) and how
many false alarms appear per frame, on the CVC-ClinicDB "new hospital" test set.

No training. It runs each model once, then tries every threshold.

Outputs (in results/):
  threshold_sweep.csv   recall + false alarms per frame at every threshold
  recall_vs_false_alarms.png   the trade-off curve for each model

Usage:
    python threshold_sweep.py
    python threshold_sweep.py --target-recall 0.95
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

EXPERIMENTS = ["baseline", "strongaug", "finetune"]
COLORS = {"baseline": "#3B6EA8", "strongaug": "#8A8A8A", "finetune": "#D9822B"}
IMG_DIR = Path("data/yolo/cvc/images/test")
THRESHOLDS = np.round(np.arange(0.05, 0.91, 0.05), 2)
OUT = Path("results")


def load_gt(img_path, w, h):
    lbl = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    boxes = []
    for line in lbl.read_text().splitlines():
        _, cx, cy, bw, bh = map(float, line.split())
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(boxes).reshape(-1, 4)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0


def count(preds, gt, t):
    """Greedy match predictions (most confident first) to doctor boxes at IoU >= 0.5."""
    keep = sorted([p for p in preds if p[4] >= t], key=lambda p: -p[4])
    matched, tp = set(), 0
    for p in keep:
        best, best_j = 0, -1
        for j, g in enumerate(gt):
            if j not in matched:
                o = iou(p, g)
                if o > best:
                    best, best_j = o, j
        if best >= 0.5:
            matched.add(best_j)
            tp += 1
    return tp, len(keep) - tp


def sweep(exp, images):
    weights = Path(f"runs/{exp}/weights/best.pt")
    if not weights.exists():
        print(f"skipping {exp}: no weights")
        return []
    model = YOLO(str(weights))
    data = []
    for img in images:
        r = model.predict(str(img), conf=0.01, verbose=False)[0]
        h, w = r.orig_shape
        preds = [b.xyxy[0].tolist() + [float(b.conf)] for b in r.boxes]
        data.append((preds, load_gt(img, w, h)))

    total_gt = sum(len(g) for _, g in data)
    rows = []
    for t in THRESHOLDS:
        tp = fp = 0
        for preds, gt in data:
            a, b = count(preds, gt, t)
            tp, fp = tp + a, fp + b
        rows.append({"experiment": exp, "threshold": t,
                     "recall": round(tp / total_gt, 3),
                     "missed polyps": total_gt - tp,
                     "false alarms": fp,
                     "false alarms per frame": round(fp / len(data), 3)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-recall", type=float, default=0.95)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    images = sorted(IMG_DIR.iterdir())
    df = pd.DataFrame([r for exp in EXPERIMENTS for r in sweep(exp, images)])
    if df.empty:
        raise SystemExit("No trained models found.")
    df.to_csv(OUT / "threshold_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for exp, g in df.groupby("experiment", sort=False):
        ax.plot(g["false alarms per frame"], g["recall"], marker="o", ms=3,
                label=exp, color=COLORS.get(exp))
        for _, row in g[g["threshold"].isin([0.1, 0.25, 0.5])].iterrows():
            ax.annotate(f"{row['threshold']:.2f}", (row["false alarms per frame"], row["recall"]),
                        textcoords="offset points", xytext=(4, -10), fontsize=7, color=COLORS.get(exp))
    ax.axhline(args.target_recall, ls="--", lw=1, color="#C0392B")
    ax.text(ax.get_xlim()[1], args.target_recall, f" target recall {args.target_recall:.0%}",
            va="bottom", ha="right", fontsize=8, color="#C0392B")
    ax.set_xlabel("False alarms per frame (lower is better)")
    ax.set_ylabel("Recall: share of polyps caught (higher is better)")
    ax.set_title("Catching more polyps costs more false alarms (new hospital)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "recall_vs_false_alarms.png", dpi=200)

    print(f"\nCVC test set: {len(images)} frames")
    print(f"Threshold 0.25 (the usual default) vs the threshold needed for {args.target_recall:.0%} recall:\n")
    for exp, g in df.groupby("experiment", sort=False):
        d = g[g["threshold"] == 0.25].iloc[0]
        hit = g[g["recall"] >= args.target_recall]
        print(f"{exp:10s} @0.25: recall {d['recall']:.3f}, missed {d['missed polyps']}, "
              f"false alarms/frame {d['false alarms per frame']:.2f}")
        if len(hit):
            b = hit.sort_values("threshold").iloc[-1]  # highest threshold that still meets target
            print(f"{'':10s} to reach {args.target_recall:.0%} recall: threshold {b['threshold']:.2f}, "
                  f"missed {b['missed polyps']}, false alarms/frame {b['false alarms per frame']:.2f}")
        else:
            print(f"{'':10s} never reaches {args.target_recall:.0%} recall "
                  f"(best: {g['recall'].max():.3f} at threshold {g.loc[g['recall'].idxmax(), 'threshold']:.2f})")
    print(f"\nSaved {OUT/'threshold_sweep.csv'} and {OUT/'recall_vs_false_alarms.png'}")


if __name__ == "__main__":
    main()
