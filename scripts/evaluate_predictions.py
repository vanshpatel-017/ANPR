from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anpr_junigadi.data.plate_utils import extract_plate_from_filename
from anpr_junigadi.evaluation.metrics import evaluate_ocr_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ANPR predictions CSV.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--prediction-column", default="prediction")
    parser.add_argument(
        "--ground-truth-column",
        default=None,
        help="If omitted, ground truth is extracted from image filename prefix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)

    if args.ground_truth_column:
        ground_truth = df[args.ground_truth_column].astype(str).tolist()
    else:
        ground_truth = [extract_plate_from_filename(path) for path in df["image_path"]]

    predictions = df[args.prediction_column].fillna("").astype(str).tolist()
    metrics = evaluate_ocr_pairs(list(zip(ground_truth, predictions)))

    print(f"Total: {metrics.total}")
    print(f"Correct: {metrics.correct}")
    print(f"Exact accuracy: {metrics.exact_accuracy:.2%}")
    print(f"Average char accuracy: {metrics.average_char_accuracy:.2%}")
    print(f"Average edit distance: {metrics.average_edit_distance:.3f}")


if __name__ == "__main__":
    main()
