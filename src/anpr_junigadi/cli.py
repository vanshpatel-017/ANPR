from __future__ import annotations

import argparse
from pathlib import Path

from anpr_junigadi.detection.yolo_detector import YoloPlateDetector
from anpr_junigadi.ocr.parseq_reader import ParseqPlateReader
from anpr_junigadi.ocr.trocr_reader import TrocrPlateReader
from anpr_junigadi.pipeline import ANPRPipeline


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the JuniGadi ANPR pipeline")
    parser.add_argument("--input", required=True, type=Path, help="Image file or folder")
    parser.add_argument("--detector", required=True, type=Path, help="Detector weights")
    parser.add_argument("--ocr-checkpoint", required=True, type=Path, help="OCR checkpoint")
    parser.add_argument("--ocr-type", default="trocr", choices=["trocr", "parseq"], help="OCR engine type")
    parser.add_argument("--parseq-repo", default=None, type=Path, help="Optional PARSeq repository path (only for parseq OCR)")
    parser.add_argument("--output", default=Path("outputs/inference"), type=Path)
    parser.add_argument("--conf", default=0.25, type=float, help="Detection confidence threshold")
    parser.add_argument("--iou", default=0.45, type=float, help="Detection IoU threshold")
    parser.add_argument("--imgsz", default=640, type=int, help="Detector image size")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or GPU index")
    parser.add_argument("--no-postprocess", action="store_true")
    parser.add_argument("--no-crops", action="store_true")
    parser.add_argument("--no-annotated", action="store_true")
    args = parser.parse_args(argv)

    detector = YoloPlateDetector(
        weights=args.detector,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        image_size=args.imgsz,
        device=None if args.device == "auto" else args.device,
    )

    if args.ocr_type == "parseq":
        reader = ParseqPlateReader(
            checkpoint=args.ocr_checkpoint,
            parseq_repo=args.parseq_repo,
            device=args.device,
            apply_postprocess=not args.no_postprocess,
        )
    else:
        reader = TrocrPlateReader(
            checkpoint=args.ocr_checkpoint,
            device=args.device,
            apply_postprocess=not args.no_postprocess,
        )

    pipeline = ANPRPipeline(detector=detector, ocr_reader=reader)
    csv_path = pipeline.process_folder(
        input_path=args.input,
        output_dir=args.output,
        save_crop=not args.no_crops,
        save_annotated=not args.no_annotated,
    )
    print(f"Saved predictions to: {csv_path}")


if __name__ == "__main__":
    main()
