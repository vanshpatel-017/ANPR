# ANPR Junigadi

Production-style repository for the internal ANPR workflow used in JuniGadi operations.

This codebase documents the actual engineering pipeline used in the project:

1. Raw dataset reconstruction and cleaning
2. Multi-stage license plate detector training (YOLOv8)
3. Bootstrap auto-label + human audit loop
4. OCR dataset generation from detected crops
5. OCR evaluation (TrOCR-focused) and error slicing
6. End-to-end detector+OCR inference
7. Hard-example mining for retraining

## Scope and access constraints

- **Dataset:** private internal JuniGadi data (not published in this repository)
- **Production checkpoints:** private internal model assets (not published in this repository)
- **Public product context:** https://www.junigadi.com/ and https://www.junigadi.com/e-challan

This repository is intentionally designed to be reproducible for authorized internal users while remaining safe for public GitHub exposure.

## Repository layout

```text
.
├── notebooks/
│   ├── 00_project_setup_and_paths.ipynb
│   ├── 01_dataset_recreation_and_cleaning.ipynb
│   ├── 02_detection_training_stages.ipynb
│   ├── 03_bootstrap_autolabel_and_audit.ipynb
│   ├── 04_ocr_dataset_creation.ipynb
│   ├── 05_ocr_training_and_benchmarks.ipynb
│   ├── 06_end_to_end_anpr_inference.ipynb
│   └── 07_error_analysis_and_hard_example_mining.ipynb
├── docs/
│   └── ARCHITECTURE.md
├── scripts/
│   ├── check_env.py
│   ├── validate_dataset_structure.py
│   ├── run_inference.py
│   └── evaluate_predictions.py
├── src/anpr_junigadi/
│   ├── data/
│   │   └── plate_utils.py
│   ├── detection/
│   │   └── yolo_detector.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── ocr/
│   │   ├── parseq_reader.py
│   │   ├── trocr_reader.py
│   │   └── postprocess.py
│   ├── utils/
│   │   ├── files.py
│   │   └── image.py
│   ├── cli.py
│   ├── config.py
│   └── pipeline.py
├── .env.example
├── .gitignore
├── pyproject.toml
└── requirements.txt
```

## Notebook execution order

Run notebooks in order from `00` to `07`. Each notebook produces concrete deliverables consumed by the next stage.

Detailed notebook contracts are documented in `notebooks/README.md`.

## Environment setup

1. Create and activate a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -e .
   ```
3. Copy `.env.example` to `.env` and set internal path values.
4. Verify local paths and package availability with `scripts/check_env.py`.
5. Start from `notebooks/00_project_setup_and_paths.ipynb`.

## Run CLI inference

Run the end-to-end detection and OCR pipeline via CLI:

```bash
python scripts/run_inference.py \
  --input <image-or-folder-path> \
  --detector <yolov8-detector-weights.pt> \
  --ocr-checkpoint <ocr-checkpoint-path> \
  --ocr-type <trocr|parseq> \
  --output <output-directory>
```

You can evaluate the predictions CSV using:

```bash
python scripts/evaluate_predictions.py --csv <output-directory>/predictions.csv
```

## Reproducibility notes

- No private image data, labels, or checkpoints are committed.
- Notebook code includes placeholder paths and explicit contract points where internal assets are expected.
- The pipeline logic and experiment flow are preserved for engineering transparency.

## License

This repository is released under the MIT License. See `LICENSE`.

