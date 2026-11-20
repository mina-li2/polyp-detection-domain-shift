"""
demo.py
-------
Runs the detector on a colonoscopy video (or a folder of frames) and writes an
annotated video with an FPS counter, to show it is fast enough for real time.

Usage:
    python demo.py --weights runs/finetune/weights/best.pt --source my_video.mp4
    python demo.py --weights runs/baseline/weights/best.pt --source data/yolo/cvc/images/test

If you have no video, pointing it at the CVC test folder works: those frames come
from real colonoscopy videos, so played in order they look like a short clip.
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def frames_from(source: Path):
    if source.is_dir():
        files = sorted(p for p in source.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
        for f in files:
            yield cv2.imread(str(f))
    else:
        cap = cv2.VideoCapture(str(source))
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--out", default="results/demo.mp4")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--fps", type=int, default=15, help="playback fps of the output video")
    args = ap.parse_args()

    model = YOLO(args.weights)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer, times = None, []

    for frame in frames_from(args.source):
        if frame is None:
            continue
        t0 = time.perf_counter()
        result = model.predict(frame, conf=args.conf, verbose=False)[0]
        times.append(time.perf_counter() - t0)

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"polyp {float(box.conf):.2f}", (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        live_fps = 1 / max(sum(times[-20:]) / len(times[-20:]), 1e-6)
        cv2.putText(frame, f"{live_fps:.0f} FPS", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        writer.write(frame)

    if writer is None:
        raise SystemExit(f"No frames read from {args.source}")
    writer.release()
    avg = len(times) / sum(times)
    print(f"Processed {len(times)} frames, average {avg:.1f} FPS. Saved {args.out}")
    print("Colonoscopy video is usually 25-30 FPS, so anything above that counts as real time.")


if __name__ == "__main__":
    main()
