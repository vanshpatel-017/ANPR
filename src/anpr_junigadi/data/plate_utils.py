from __future__ import annotations

import re
from pathlib import Path


PLATE_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Practical Indian plate pattern used throughout the notebooks.
# It accepts common forms like GJ01AB1234, GJ1A1234, DL3CBA6560.
INDIAN_PLATE_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{3,4}$")


def normalize_plate_text(text: str | None) -> str:
    """Uppercase and keep only alphanumeric plate characters."""
    if text is None:
        return ""
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


def is_valid_indian_plate(text: str | None) -> bool:
    """Return True if text matches the practical Indian plate regex."""
    return bool(INDIAN_PLATE_RE.match(normalize_plate_text(text)))


def extract_plate_from_filename(path: str | Path) -> str:
    """Extract plate label from filenames like GJ01AB1234_001.jpg.

    The original notebooks consistently used the prefix before the first
    underscore as the ground-truth plate label.
    """
    stem = Path(path).stem
    return normalize_plate_text(stem.split("_")[0])


def group_by_plate(paths: list[str | Path]) -> dict[str, list[Path]]:
    """Group image paths by plate label extracted from filename."""
    groups: dict[str, list[Path]] = {}
    for path in paths:
        label = extract_plate_from_filename(path)
        groups.setdefault(label, []).append(Path(path))
    return groups
