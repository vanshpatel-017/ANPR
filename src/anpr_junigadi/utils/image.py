from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    """Read an image with Windows-unicode-safe OpenCV loading.

    Returns a BGR image, matching OpenCV conventions.
    """
    path = Path(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    """Save an image with Windows-unicode-safe OpenCV writing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Could not encode image for saving: {path}")
    encoded.tofile(str(path))


def crop_xyxy(
    image: np.ndarray,
    xyxy: tuple[float, float, float, float],
    padding: int = 0,
) -> np.ndarray:
    """Crop an image using x1, y1, x2, y2 coordinates with optional padding."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1 = max(0, int(round(x1)) - padding)
    y1 = max(0, int(round(y1)) - padding)
    x2 = min(w, int(round(x2)) + padding)
    y2 = min(h, int(round(y2)) + padding)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop box: {xyxy}")
    return image[y1:y2, x1:x2].copy()


def draw_box(
    image: np.ndarray,
    xyxy: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw a detection box and label on an image."""
    out = image.copy()
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(
            out,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def bgr_to_pil(image: np.ndarray):
    """Convert an OpenCV BGR image to a PIL RGB image."""
    from PIL import Image

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
