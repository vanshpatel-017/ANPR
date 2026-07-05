from __future__ import annotations

from anpr_junigadi.data.plate_utils import normalize_plate_text


LETTER_CONFUSIONS = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "8": "B",
}

DIGIT_CONFUSIONS = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}


def conservative_plate_postprocess(text: str) -> str:
    """Clean OCR text and apply only conservative Indian-plate corrections.

    This is intentionally less aggressive than the fuzzy closed-set correction
    used in the original notebooks. It does not use a list of known plates,
    because that would inflate real-world accuracy.
    """
    text = normalize_plate_text(text)
    if len(text) < 6:
        return text

    chars = list(text)

    # First two characters are state letters in standard Indian plates.
    for idx in range(min(2, len(chars))):
        chars[idx] = LETTER_CONFUSIONS.get(chars[idx], chars[idx])

    # Last four characters are usually digits.
    for idx in range(max(0, len(chars) - 4), len(chars)):
        chars[idx] = DIGIT_CONFUSIONS.get(chars[idx], chars[idx])

    return "".join(chars)
