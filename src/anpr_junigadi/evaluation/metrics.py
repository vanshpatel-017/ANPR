from __future__ import annotations

from dataclasses import dataclass

from anpr_junigadi.data.plate_utils import normalize_plate_text


@dataclass(frozen=True)
class OCRMetrics:
    total: int
    correct: int
    exact_accuracy: float
    average_char_accuracy: float
    average_edit_distance: float


def levenshtein(a: str, b: str) -> int:
    """Compute Levenshtein edit distance."""
    a = normalize_plate_text(a)
    b = normalize_plate_text(b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            insert = curr[j - 1] + 1
            delete = prev[j] + 1
            replace = prev[j - 1] + (ca != cb)
            curr.append(min(insert, delete, replace))
        prev = curr
    return prev[-1]


def char_accuracy(ground_truth: str, prediction: str) -> float:
    """Position-wise character accuracy normalized by the longer string."""
    gt = normalize_plate_text(ground_truth)
    pred = normalize_plate_text(prediction)
    denom = max(len(gt), len(pred), 1)
    matches = sum(1 for a, b in zip(gt, pred) if a == b)
    return matches / denom


def evaluate_ocr_pairs(pairs: list[tuple[str, str]]) -> OCRMetrics:
    """Evaluate OCR predictions against ground truth pairs."""
    total = len(pairs)
    if total == 0:
        return OCRMetrics(0, 0, 0.0, 0.0, 0.0)

    correct = 0
    char_scores = []
    edit_distances = []

    for ground_truth, prediction in pairs:
        gt = normalize_plate_text(ground_truth)
        pred = normalize_plate_text(prediction)
        correct += int(gt == pred)
        char_scores.append(char_accuracy(gt, pred))
        edit_distances.append(levenshtein(gt, pred))

    return OCRMetrics(
        total=total,
        correct=correct,
        exact_accuracy=correct / total,
        average_char_accuracy=sum(char_scores) / total,
        average_edit_distance=sum(edit_distances) / total,
    )
