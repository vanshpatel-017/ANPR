"""
validate_dataset_structure.py

Validates a YOLO-format dataset directory:
  - Checks that train/val/test splits exist
  - Reports image counts per split
  - Checks images/labels pairing (missing or extra label files)
  - Validates that image filenames start with a valid Indian plate pattern

Usage:
    python scripts/validate_dataset_structure.py <dataset_root>
"""

import re
import sys
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
PLATE_RE = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}', re.IGNORECASE)
SPLITS = ('train', 'val', 'test')


def validate_split(split_dir: Path) -> dict:
    images_dir = split_dir / 'images'
    labels_dir = split_dir / 'labels'

    if not images_dir.exists():
        return {'error': f'images/ not found under {split_dir}'}

    image_stems = {p.stem for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    label_stems = {p.stem for p in labels_dir.iterdir() if p.suffix == '.txt'} if labels_dir.exists() else set()

    missing_labels = image_stems - label_stems
    extra_labels   = label_stems - image_stems

    invalid_names = [s for s in image_stems if not PLATE_RE.match(s)]

    return {
        'images': len(image_stems),
        'labels': len(label_stems),
        'missing_labels': len(missing_labels),
        'extra_labels': len(extra_labels),
        'invalid_name_samples': sorted(invalid_names)[:10],
    }


def main(root: str) -> None:
    dataset = Path(root)
    if not dataset.exists():
        print(f'ERROR: path does not exist: {dataset}')
        sys.exit(1)

    found_any = False
    for split in SPLITS:
        split_dir = dataset / split
        if not split_dir.exists():
            print(f'[{split}] MISSING')
            continue
        found_any = True
        info = validate_split(split_dir)
        if 'error' in info:
            print(f'[{split}] {info["error"]}')
            continue
        print(f'[{split}]  images={info["images"]}  labels={info["labels"]}'
              f'  missing_labels={info["missing_labels"]}'
              f'  extra_labels={info["extra_labels"]}')
        if info['invalid_name_samples']:
            print(f'         invalid_name_samples={info["invalid_name_samples"]}')

    if not found_any:
        print('No recognised split directories found. Expected: train/, val/, test/')
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python validate_dataset_structure.py <dataset_root>')
        sys.exit(1)
    main(sys.argv[1])
