from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from anpr_junigadi.detection.yolo_detector import PlateDetection, YoloPlateDetector
from anpr_junigadi.utils.files import collect_images, ensure_dir, safe_stem
from anpr_junigadi.utils.image import crop_xyxy, draw_box, read_image, save_image

if TYPE_CHECKING:
    from anpr_junigadi.ocr.parseq_reader import ParseqPlateReader
    from anpr_junigadi.ocr.trocr_reader import TrocrPlateReader


@dataclass
class ANPRResult:
    image_path: str
    prediction: str
    raw_prediction: str
    ocr_confidence: float
    yolo_confidence: float
    x1: float | None
    y1: float | None
    x2: float | None
    y2: float | None
    crop_path: str
    error: str = ""


class ANPRPipeline:
    """End-to-end license plate detection + OCR pipeline."""

    def __init__(
        self,
        detector: YoloPlateDetector,
        ocr_reader: ParseqPlateReader | TrocrPlateReader,
        crop_padding: int = 2,
    ) -> None:
        self.detector = detector
        self.ocr_reader = ocr_reader
        self.crop_padding = crop_padding

    def process_image(
        self,
        image_path: str | Path,
        output_dir: str | Path | None = None,
        save_crop: bool = True,
        save_annotated: bool = True,
    ) -> ANPRResult:
        image_path = Path(image_path)
        output_dir = Path(output_dir) if output_dir else None

        try:
            image = read_image(image_path)
            detections = self.detector.detect(image)
            if not detections:
                return ANPRResult(
                    image_path=str(image_path),
                    prediction="",
                    raw_prediction="",
                    ocr_confidence=0.0,
                    yolo_confidence=0.0,
                    x1=None,
                    y1=None,
                    x2=None,
                    y2=None,
                    crop_path="",
                    error="no_plate_detected",
                )

            detection = detections[0]
            crop = crop_xyxy(image, detection.xyxy, padding=self.crop_padding)

            crop_path = ""
            if output_dir and save_crop:
                crop_file = ensure_dir(output_dir / "crops") / f"{safe_stem(image_path)}_plate.jpg"
                save_image(crop_file, crop)
                crop_path = str(crop_file)

            ocr = self.ocr_reader.predict(crop)

            if output_dir and save_annotated:
                label = f"{ocr.text} ({detection.confidence:.2f})"
                annotated = draw_box(image, detection.xyxy, label=label)
                save_image(ensure_dir(output_dir / "annotated") / image_path.name, annotated)

            return _result_from_prediction(image_path, detection, ocr, crop_path)

        except Exception as exc:  # noqa: BLE001 - keep batch inference running
            return ANPRResult(
                image_path=str(image_path),
                prediction="",
                raw_prediction="",
                ocr_confidence=0.0,
                yolo_confidence=0.0,
                x1=None,
                y1=None,
                x2=None,
                y2=None,
                crop_path="",
                error=str(exc),
            )

    def process_folder(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        save_crop: bool = True,
        save_annotated: bool = True,
    ) -> Path:
        """Run ANPR on an image or folder and save predictions.csv."""
        output_dir = ensure_dir(output_dir)
        image_paths = collect_images(input_path)
        results = [
            self.process_image(
                image_path,
                output_dir=output_dir,
                save_crop=save_crop,
                save_annotated=save_annotated,
            )
            for image_path in tqdm(image_paths, desc="ANPR inference")
        ]

        csv_path = output_dir / "predictions.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else [])
            if results:
                writer.writeheader()
                writer.writerows(asdict(item) for item in results)

        return csv_path


def _result_from_prediction(
    image_path: Path,
    detection: PlateDetection,
    ocr: object,
    crop_path: str,
) -> ANPRResult:
    x1, y1, x2, y2 = detection.xyxy
    return ANPRResult(
        image_path=str(image_path),
        prediction=getattr(ocr, "text", ""),
        raw_prediction=getattr(ocr, "raw_text", ""),
        ocr_confidence=float(getattr(ocr, "confidence", 0.0)),
        yolo_confidence=detection.confidence,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        crop_path=crop_path,
        error="",
    )
