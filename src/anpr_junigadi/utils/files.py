from __future__ import annotations

from pathlib import Path
from typing import Iterable


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def collect_images(path: str | Path, recursive: bool = True) -> list[Path]:
    """Collect image files from a single image path or a directory."""
    path = Path(path)
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    iterator: Iterable[Path] = path.rglob("*") if recursive else path.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def safe_stem(path: str | Path) -> str:
    """Return a filesystem-friendly stem."""
    return Path(path).stem.replace(" ", "_")
