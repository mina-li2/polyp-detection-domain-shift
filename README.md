# Polyp Detection Across Hospitals

Colorectal polyp detection with YOLO11, trained on colonoscopy images from one hospital and tested on a hospital the model never saw during training.

Colorectal cancer usually starts as a polyp, and polyps missed during colonoscopy are a known problem. Detection models often report high accuracy on their own test sets, but images from a different hospital (different scopes, lighting and patients) can look different enough to hurt performance. This project measures how large that drop is, tests two common fixes, and picks a confidence threshold based on what matters clinically: not missing polyps.

## Data

| Dataset | Role | Images | Source |
|---|---|---|---|
| Kvasir-SEG | Training hospital (Norway) | 1,000 | https://datasets.simula.no/kvasir-seg/ |
| CVC-ClinicDB | Unseen hospital (Barcelona) | 612 frames from 29 videos | https://polyp.grand-challenge.org/CVCClinicDB/ |

Both datasets come with pixel-level polyp masks. I converted each mask to bounding boxes, one box per separate polyp region, ignoring specks smaller than 0.1% of the image.

- **Kvasir:** random 80/10/10 train/val/test split.
- **CVC-ClinicDB:** split by frame order, not at random. The first 122 frames are used only for the fine-tuning experiment, and the remaining 490 frames (512 polyps) are the held-out test set. CVC frames come from videos, so neighbouring frames are near-duplicates. A random split would put near-copies of test frames into training and inflate the scores.

The datasets are not included in this repo because their licences don't allow redistribution.

## Experiments

All models are YOLO11s at 640 px, trained on a Colab T4 GPU.

1. **Baseline:** trained on Kvasir only (50 epochs).
2. **Strong augmentation:** Kvasir only, with heavy colour, brightness, rotation and flip augmentation, to test whether the model can be made robust to a new hospital's camera without any new data (50 epochs).
3. **Fine-tuning:** the baseline model trained further on Kvasir plus the 122 CVC fine-tuning frames (30 epochs).

Each model takes 8–13 minutes to train. Inference runs at about 5.5 ms per frame, well above colonoscopy video rates.

## Results

![Same hospital vs new hospital](results/domain_shift.png)

| Model | Kvasir mAP50 | CVC mAP50 | Kvasir mAP50-95 | CVC mAP50-95 | CVC precision | CVC recall |
|---|---|---|---|---|---|---|
| Baseline | 0.949 | 0.903 | 0.771 | 0.611 | 0.860 | **0.852** |
| Strong augmentation | 0.927 | 0.885 | 0.709 | 0.441 | 0.856 | 0.827 |
| Fine-tuned (+122 CVC frames) | 0.922 | 0.907 | 0.756 | 0.635 | **0.908** | 0.810 |

**At the new hospital, the model still finds most polyps, but its boxes get much less accurate.** mAP50 dropped by about 0.05, while mAP50-95, which rewards tight boxes, dropped by 0.16. Recall fell from 0.91 to 0.85.

**Augmentation alone did not help.** It lowered every metric, and box accuracy at the new hospital fell to 0.44. The augmentation was likely too aggressive for 50 epochs.

**Fine-tuning reduced false alarms but missed more polyps.** Precision rose from 0.86 to 0.91, and recall fell from 0.85 to 0.81. For screening, that is the wrong trade-off. A false alarm costs the doctor a second look, while a missed polyp can develop into cancer.

## Choosing a screening threshold

Since sensitivity matters more than precision here, I swept the confidence threshold on the 490 unseen CVC frames.

![Recall vs false alarms](results/recall_vs_false_alarms.png)

| Baseline threshold | Recall | Polyps missed (of 512) | False alarms per frame |
|---|---|---|---|
| 0.25 (default) | 0.861 | 71 | 0.18 |
| **0.10 (chosen)** | **0.902** | **50** | **0.30** |
| 0.05 | 0.918 | 42 | 0.42 |

Lowering the threshold from 0.25 to 0.10 catches 21 more polyps for about one extra false alarm every 8 frames. Going down to 0.05 catches only 8 more but adds 40% more false alarms, so I chose 0.10.

No model reached 95% recall at any threshold. About 8% of polyps were never matched by any box, which a lower threshold cannot fix. The baseline had the highest recall of the three models at every threshold tested.

## Failure cases

Example frames are in `results/cvc_examples_*.jpg` (green = ground truth, red = prediction).

- Yellow stool or debris detected as a polyp
- A large box over fluid and glare, only partly covering the real polyp
- Oversized boxes at the new hospital, consistent with the mAP50-95 drop
- Clear, rounded, well-lit polyps are detected with high confidence (0.8–0.9)

## Conclusion

A detector scoring 0.95 at its own hospital missed noticeably more polyps at a new one. Lowering the threshold recovered some of them, but neither augmentation nor a small amount of fine-tuning improved recall, which is the number that matters most for screening.

## Limitations

- Results are per frame, and 0.30 false alarms per frame still means several false boxes per second in live video. A deployed system would need smoothing across frames.
- Two datasets with about 1,600 images in total.
- Test sets are small, so differences of 1–2 points may not be meaningful.
- Not clinically validated.

## Future work

- **Per-polyp detection:** count a polyp as found if it is detected in any frame of its video sequence, which is closer to what matters in a real procedure.
- **Data efficiency:** fine-tune with different amounts of new-hospital data to see how much is needed to recover.
- Select fine-tuned models by recall instead of overall mAP.
- Test on more centres (ETIS-Larib, CVC-ColonDB, PolypGen).

## Reproducing

The easiest way is to open `polyp_detection_colab.ipynb` in Google Colab with a T4 GPU. To run the scripts directly:

```
pip install -r requirements.txt
python prepare_data.py --kvasir raw/Kvasir-SEG --cvc raw/CVC-ClinicDB
python train.py --exp baseline --epochs 50
python train.py --exp strongaug --epochs 50
python train.py --exp finetune
python evaluate.py
python threshold_sweep.py
python demo.py --weights runs/baseline/weights/best.pt --source data/yolo/cvc/images/test
```

Kvasir-SEG (`kvasir-seg.zip`) has to be downloaded from the dataset page and unzipped into `raw/`. CVC-ClinicDB can be downloaded from the Hugging Face mirror `berkaytrhn/cvc-clinicdb`, as in the notebook.

## Files

| File | Purpose |
|---|---|
| `polyp_detection_colab.ipynb` | Full pipeline in Colab |
| `prepare_data.py` | Mask-to-box conversion, leakage-safe splits, dataset configs |
| `train.py` | The three experiments |
| `evaluate.py` | Same-hospital vs new-hospital metrics, chart, example images |
| `threshold_sweep.py` | Recall vs false alarms at every confidence threshold |
| `demo.py` | Annotated video with FPS counter |

## Citations

- Jha D. et al. *Kvasir-SEG: A Segmented Polyp Dataset.* MultiMedia Modeling (MMM), 2020.
- Bernal J. et al. *WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians.* Computerized Medical Imaging and Graphics 43, 99–111, 2015.
- Ultralytics YOLO: https://github.com/ultralytics/ultralytics
