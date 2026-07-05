from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from anpr_junigadi.ocr.postprocess import conservative_plate_postprocess


@dataclass(frozen=True)
class OCRPrediction:
    text: str
    confidence: float
    raw_text: str


class TrocrPlateReader:
    """TrOCR-based OCR wrapper for cropped license plates."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
        apply_postprocess: bool = True,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.apply_postprocess = apply_postprocess

        try:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise RuntimeError("PyTorch and transformers are required for TrOCR OCR.") from exc

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.torch = torch
        self.processor = TrOCRProcessor.from_pretrained(str(self.checkpoint))
        self.model = VisionEncoderDecoderModel.from_pretrained(str(self.checkpoint)).to(self.device)
        self.model.eval()

    def predict(self, image: str | Path | np.ndarray) -> OCRPrediction:
        """Predict text from a cropped plate image."""
        if isinstance(image, (str, Path)):
            from anpr_junigadi.utils.image import read_image
            image_array = read_image(image)
        else:
            image_array = image

        pil_image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        pixel_values = self.processor(images=pil_image, return_tensors='pt').pixel_values.to(self.device)

        with self.torch.no_grad():
            outputs = self.model.generate(
                pixel_values, max_length=16,
                output_scores=True, return_dict_in_generate=True
            )

        raw_text = self.processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].upper().replace(' ', '')
        
        # Calculate sequence confidence
        probs = [self.torch.softmax(s, dim=-1).max().item() for s in outputs.scores]
        confidence = sum(probs) / len(probs) if probs else 0.0
        
        text = conservative_plate_postprocess(raw_text) if self.apply_postprocess else raw_text

        return OCRPrediction(text=text, confidence=confidence, raw_text=raw_text)
