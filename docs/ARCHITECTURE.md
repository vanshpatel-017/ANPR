# ANPR Junigadi — Architecture

## System objective

Detect vehicle number plates in unconstrained images and decode them into normalised plate text with confidence-aware post-analysis.

## Pipeline stages

| # | Notebook | What it does |
|---|---|---|
| 0 | `00_project_setup_and_paths` | Runtime checks, canonical path contracts, directory initialisation |
| 1 | `01_dataset_recreation_and_cleaning` | Reconstruct class-wise plate folders from raw captures; clean invalid/tiny/document images; produce 70/10/20 train/val/test splits |
| 2 | `02_detection_training_stages` | Multi-stage YOLOv8 training: external pretraining → bootstrap finetuning → optional domain finetuning |
| 3 | `03_bootstrap_autolabel_and_audit` | Generate pseudo-labels with current detector; export visual audit overlays; merge human-corrected labels; retrain |
| 4 | `04_ocr_dataset_creation` | Convert detection boxes to plate crops; generate OCR supervision CSV (`image`, `text`) |
| 5 | `05_ocr_training_and_benchmarks` | Evaluate TrOCR predictions; extract near-miss and state-code error slices; EasyOCR baseline comparison |
| 6 | `06_end_to_end_anpr_inference` | Run YOLO + TrOCR in sequence; save plate crops and prediction/confidence CSV |
| 7 | `07_error_analysis_and_hard_example_mining` | Build review queues from confidence bands; create hard-example retraining files |

## Data and artifact boundaries

- Raw dataset, derived dataset, and production checkpoints are private internal assets.
- This repository includes pipeline code and reproducibility contracts only.
- Public platform reference: https://www.junigadi.com/
