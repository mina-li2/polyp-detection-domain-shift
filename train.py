"""
train.py
--------
Three experiments, each one answers a question for your write-up:

  baseline  : Train on Kvasir (hospital A) with default settings.
              -> "How good is a standard detector?"
  strongaug : Same data, but heavy colour/brightness/blur-style augmentation so the
              model can't rely on one hospital's camera colours.
              -> "Can augmentation alone survive the move to a new hospital?"
  finetune  : Start from the baseline model and continue training with a SMALL
              slice of CVC frames (hospital B) mixed in.
              -> "How much new-hospital data does it take to recover?"

Usage:
    python train.py --exp baseline
    python train.py --exp strongaug
    python train.py --exp finetune
Weights end up in runs/<exp>/weights/best.pt
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

COMMON = dict(
    imgsz=640,
    batch=16,
    patience=20,      # stop early if val mAP stops improving
    project=str(Path("runs").resolve()),  # absolute path so weights land in runs/<exp>/
    exist_ok=True,
    seed=42,
    plots=True,
)

STRONG_AUG = dict(
    hsv_h=0.03,       # hue shift: different scopes render tissue slightly differently
    hsv_s=0.8,        # saturation
    hsv_v=0.6,        # brightness: lighting varies a lot between scopes
    degrees=20,       # the scope can be at any rotation
    flipud=0.5,       # no real "up" inside a colon
    fliplr=0.5,
    scale=0.5,
    mixup=0.1,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["baseline", "strongaug", "finetune"], required=True)
    ap.add_argument("--model", default="yolo11s.pt",
                    help="yolo11n.pt is faster, yolo11s.pt is a bit more accurate")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    if args.exp == "baseline":
        model = YOLO(args.model)
        model.train(data="data/kvasir.yaml", name="baseline",
                    epochs=args.epochs or 100, **COMMON)

    elif args.exp == "strongaug":
        model = YOLO(args.model)
        model.train(data="data/kvasir.yaml", name="strongaug",
                    epochs=args.epochs or 100, **COMMON, **STRONG_AUG)

    elif args.exp == "finetune":
        base = Path("runs/baseline/weights/best.pt")
        if not base.exists():
            raise SystemExit("Run the baseline experiment first (python train.py --exp baseline)")
        model = YOLO(str(base))
        model.train(data="data/finetune.yaml", name="finetune",
                    epochs=args.epochs or 30, lr0=0.002, **COMMON)

    print(f"Finished. Best weights: runs/{args.exp}/weights/best.pt")


if __name__ == "__main__":
    main()
