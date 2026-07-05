from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anpr_junigadi.ocr.postprocess import conservative_plate_postprocess
from anpr_junigadi.utils.image import bgr_to_pil, read_image


@dataclass(frozen=True)
class OCRPrediction:
    text: str
    confidence: float
    raw_text: str


class ParseqPlateReader:
    """PARSeq-based OCR wrapper for cropped license plates.

    This wrapper expects the PARSeq repository to be installed or available via
    `parseq_repo`. It supports the common PARSeq Lightning checkpoint workflow.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        parseq_repo: str | Path | None = None,
        device: str = "auto",
        apply_postprocess: bool = True,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.parseq_repo = Path(parseq_repo) if parseq_repo else None
        self.apply_postprocess = apply_postprocess

        if self.parseq_repo:
            sys.path.insert(0, str(self.parseq_repo.resolve()))

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for PARSeq OCR.") from exc

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.torch = torch
        self.model, self.transform = self._load_model()

    def _load_model(self) -> tuple[Any, Any]:
        try:
            from strhub.data.module import SceneTextDataModule
        except ImportError as exc:
            raise RuntimeError(
                "Could not import PARSeq/strhub. Clone PARSeq into external/parseq "
                "and run `pip install -e external/parseq`."
            ) from exc

        model = None
        load_error: Exception | None = None

        try:
            from strhub.models.parseq.system import PARSeq

            model = PARSeq.load_from_checkpoint(
                str(self.checkpoint),
                map_location=self.device,
            )
        except Exception as exc:  # noqa: BLE001
            load_error = exc

        if model is None:
            try:
                from strhub.models.utils import load_from_checkpoint

                model = load_from_checkpoint(str(self.checkpoint))
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(f"Could not load PARSeq checkpoint: {self.checkpoint}") from (
                    load_error or exc
                )

        model = model.eval().to(self.device)

        img_size = getattr(getattr(model, "hparams", object()), "img_size", (32, 128))
        transform = SceneTextDataModule.get_transform(img_size)
        return model, transform

    def predict(self, image: str | Path | np.ndarray) -> OCRPrediction:
        """Predict text from a cropped plate image."""
        if isinstance(image, (str, Path)):
            image_array = read_image(image)
        else:
            image_array = image

        pil_image = bgr_to_pil(image_array)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with self.torch.no_grad():
            logits = self.model(tensor)
            probabilities = logits.softmax(-1)
            labels, token_confidences = self.model.tokenizer.decode(probabilities)

        raw_text = labels[0] if labels else ""
        confidence = _sequence_confidence(token_confidences[0]) if token_confidences else 0.0
        text = conservative_plate_postprocess(raw_text) if self.apply_postprocess else raw_text

        return OCRPrediction(text=text, confidence=confidence, raw_text=raw_text)


def _sequence_confidence(token_confidences: Any) -> float:
    """Compute a stable mean confidence from PARSeq token probabilities."""
    try:
        if hasattr(token_confidences, "detach"):
            token_confidences = token_confidences.detach().cpu()
        if len(token_confidences) == 0:
            return 0.0
        return float(token_confidences.float().mean().item())
    except Exception:
        return 0.0
