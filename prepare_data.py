"""
prepare_data.py
---------------
Converts the two polyp datasets (which come with segmentation MASKS) into
YOLO detection format (one bounding box per polyp).

Why masks -> boxes?  Both datasets give a black/white mask showing exactly
which pixels are polyp. YOLO detection wants boxes, so for every separate
white blob in the mask we draw the tightest box around it.

Output layout:
    data/yolo/kvasir/images/{train,val,test}   + matching labels/
    data/yolo/cvc/images/{finetune,test}       + matching labels/
    data/kvasir.yaml      -> train/val/test all from Kvasir (hospital A)
    data/cvc_test.yaml    -> CVC-ClinicDB test frames (hospital B, never trained on)
    data/finetune.yaml    -> Kvasir train + a small slice of CVC (domain-shift fix)

Usage:
    python prepare_data.py --kvasir raw/Kvasir-SEG --cvc raw/CVC-ClinicDB
"""

import argparse
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MIN_BLOB_FRACTION = 0.001  # ignore mask specks smaller than 0.1% of the image


# ---------------------------------------------------------------- helpers
def find_dir(root: Path, candidates):
    """Find the first sub-folder (searching recursively) whose name matches one of the candidates."""
    candidates = [c.lower() for c in candidates]
    for p in sorted(root.rglob("*")):
        if p.is_dir() and p.name.lower() in candidates:
            return p
    raise FileNotFoundError(f"Couldn't find any of {candidates} inside {root}")


def list_images(folder: Path):
    return {p.stem: p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS}


def mask_to_yolo_boxes(mask_path: Path):
    """Read a mask, split it into separate blobs, return YOLO boxes (cx, cy, w, h normalised)."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask {mask_path}")
    h, w = mask.shape
    # JPG masks (Kvasir) have fuzzy edges, so threshold to pure black/white
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []
    for i in range(1, n):  # label 0 is background
        x, y, bw, bh, area = stats[i]
        if area < MIN_BLOB_FRACTION * h * w:
            continue
        boxes.append(((x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
    return boxes


def write_sample(img_path: Path, mask_path: Path, out_root: Path, split: str, prefix: str):
    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    name = f"{prefix}_{img_path.stem}"
    if img_path.suffix.lower() in {".tif", ".tiff", ".bmp"}:
        img = cv2.imread(str(img_path))
        cv2.imwrite(str(img_out / f"{name}.png"), img)
    else:
        shutil.copy(img_path, img_out / f"{name}{img_path.suffix.lower()}")

    boxes = mask_to_yolo_boxes(mask_path)
    with open(lbl_out / f"{name}.txt", "w") as f:
        for cx, cy, bw, bh in boxes:
            f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    return len(boxes)


def pair_up(img_dir: Path, mask_dir: Path):
    imgs, masks = list_images(img_dir), list_images(mask_dir)
    common = sorted(set(imgs) & set(masks))
    if not common:
        raise RuntimeError(f"No matching image/mask filenames between {img_dir} and {mask_dir}")
    missing = len(imgs) - len(common)
    if missing:
        print(f"  warning: {missing} images had no mask and were skipped")
    return [(imgs[k], masks[k]) for k in common]


def frame_number(path: Path):
    m = re.search(r"\d+", path.stem)
    return int(m.group()) if m else 0


# ---------------------------------------------------------------- datasets
def prepare_kvasir(kvasir_root: Path, out_root: Path, seed: int):
    print("Kvasir-SEG (hospital A, used for training)")
    pairs = pair_up(find_dir(kvasir_root, ["images"]), find_dir(kvasir_root, ["masks"]))
    random.Random(seed).shuffle(pairs)
    n = len(pairs)
    n_train, n_val = int(0.8 * n), int(0.1 * n)
    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }
    for split, items in splits.items():
        boxes = sum(write_sample(i, m, out_root, split, "kvasir") for i, m in items)
        print(f"  {split:6s}: {len(items):4d} images, {boxes:4d} polyp boxes")


def prepare_cvc(cvc_root: Path, out_root: Path, finetune_fraction: float):
    print("CVC-ClinicDB (hospital B, the 'new hospital' test)")
    pairs = pair_up(
        find_dir(cvc_root, ["original", "images"]),
        find_dir(cvc_root, ["ground truth", "groundtruth", "masks"]),
    )
    # CVC frames come from 29 colonoscopy VIDEOS, numbered in order. Neighbouring
    # frames are near-duplicates, so a random split would leak test images into
    # training. Instead we take one contiguous block of frames for fine-tuning
    # and keep the rest as a clean test set.
    pairs.sort(key=lambda p: frame_number(p[0]))
    n_ft = int(finetune_fraction * len(pairs))
    splits = {"finetune": pairs[:n_ft], "test": pairs[n_ft:]}
    for split, items in splits.items():
        boxes = sum(write_sample(i, m, out_root, split, "cvc") for i, m in items)
        print(f"  {split:8s}: {len(items):4d} images, {boxes:4d} polyp boxes")


def write_yamls(data_dir: Path):
    yolo = (data_dir / "yolo").resolve()
    names = "names:\n  0: polyp\n"
    (data_dir / "kvasir.yaml").write_text(
        f"path: {yolo / 'kvasir'}\ntrain: images/train\nval: images/val\ntest: images/test\n{names}"
    )
    (data_dir / "cvc_test.yaml").write_text(
        f"path: {yolo / 'cvc'}\ntrain: images/finetune\nval: images/test\ntest: images/test\n{names}"
    )
    (data_dir / "finetune.yaml").write_text(
        f"path: {yolo}\n"
        f"train:\n  - kvasir/images/train\n  - cvc/images/finetune\n"
        f"val: kvasir/images/val\ntest: kvasir/images/test\n{names}"
    )
    print(f"Wrote kvasir.yaml, cvc_test.yaml, finetune.yaml to {data_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kvasir", required=True, type=Path, help="unzipped Kvasir-SEG folder")
    ap.add_argument("--cvc", required=True, type=Path, help="unzipped CVC-ClinicDB folder")
    ap.add_argument("--out", default=Path("data"), type=Path)
    ap.add_argument("--finetune-fraction", default=0.2, type=float,
                    help="share of CVC frames given to the fine-tuning experiment")
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    yolo_root = args.out / "yolo"
    if yolo_root.exists():
        shutil.rmtree(yolo_root)  # start clean so reruns don't double up files

    prepare_kvasir(args.kvasir, yolo_root / "kvasir", args.seed)
    prepare_cvc(args.cvc, yolo_root / "cvc", args.finetune_fraction)
    write_yamls(args.out)
    print("Done. Next: python train.py --exp baseline")


if __name__ == "__main__":
    main()
