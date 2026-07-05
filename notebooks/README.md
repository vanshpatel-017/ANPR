# Notebook Runbook

Execute notebooks in strict numeric order. Each notebook produces artifacts used by later stages.

## Ordered sequence

1. `00_project_setup_and_paths.ipynb`  
	Runtime checks, canonical path contracts, directory initialization.

2. `01_dataset_recreation_and_cleaning.ipynb`  
	Recreate dataset from raw folders, clean invalid samples, build train/val/test splits.

3. `02_detection_training_stages.ipynb`  
	Stage-wise YOLOv8 training (pretraining → bootstrap finetune → optional domain finetune).

4. `03_bootstrap_autolabel_and_audit.ipynb`  
	Pseudo-label generation, visual audit export, merge corrected labels.

5. `04_ocr_dataset_creation.ipynb`  
	Crop plate ROIs from detection labels and build OCR supervision CSV.

6. `05_ocr_training_and_benchmarks.ipynb`  
	TrOCR evaluation and targeted error-slice generation.

7. `06_end_to_end_anpr_inference.ipynb`  
	Detector+OCR inference with confidence logging.

8. `07_error_analysis_and_hard_example_mining.ipynb`  
	Confidence-band review queue and hard-example retraining file creation.

## Operational notes

- Keep internal dataset/model paths in `.env` or local path cells.
- Do not commit private data or model checkpoints.
- Run notebooks top-to-bottom to preserve expected artifact contracts.
