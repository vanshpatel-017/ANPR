from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class Paths:
    data_root: Path
    model_root: Path



def load_paths() -> Paths:
    data_root = Path(os.getenv("ANPR_DATA_ROOT", "C:/path/to/private/data"))
    model_root = Path(os.getenv("ANPR_MODEL_ROOT", "C:/path/to/private/models"))
    return Paths(data_root=data_root, model_root=model_root)
