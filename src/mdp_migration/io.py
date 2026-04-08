from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from scipy.io import loadmat


def load_trace_data(path: str | Path) -> dict[str, np.ndarray]:
    data = loadmat(path)
    return {key: value for key, value in data.items() if not key.startswith("__")}


def scalar(value: np.ndarray) -> float | int:
    return value.reshape(-1)[0].item()


def save_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
