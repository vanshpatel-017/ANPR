from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PlateDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str = "license_plate"


class YoloPlateDetector:
    """Thin wrapper around Ultralytics YOLO for license plate detection."""

    def __init__(
        self,
        weights: str | Path,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        device: str | int | None = None,
    ) -> None:
        self.weights = Path(weights)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.device = device

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install ultralytics to use YoloPlateDetector.") from exc

        self.model = YOLO(str(self.weights))

    def detect(self, image: str | Path | np.ndarray) -> list[PlateDetection]:
        """Return license plate detections sorted by confidence descending."""
        results = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        detections: list[PlateDetection] = []
        result = results[0]
        names = getattr(result, "names", {}) or {}

        if result.boxes is None:
            return []

        for box in result.boxes:
            xyxy_values = box.xyxy[0].detach().cpu().numpy().astype(float).tolist()
            class_id = int(box.cls[0].detach().cpu().item()) if box.cls is not None else 0
            confidence = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
            detections.append(
                PlateDetection(
                    xyxy=tuple(xyxy_values),  # type: ignore[arg-type]
                    confidence=confidence,
                    class_id=class_id,
                    class_name=str(names.get(class_id, "license_plate")),
                )
            )

        return sorted(detections, key=lambda item: item.confidence, reverse=True)
