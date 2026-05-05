from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .types import TrajectorySample


def save_sample_json(sample: TrajectorySample, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sample.to_json(), indent=2), encoding="utf-8")


def load_sample_json(path: str | Path) -> TrajectorySample:
    path = Path(path)
    return TrajectorySample.from_json(json.loads(path.read_text(encoding="utf-8")))


def save_dataset_npz(samples: list[TrajectorySample], path: str | Path, extra_metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    states = np.stack([s.states for s in samples], axis=0) if samples else np.empty((0, 0, 7))
    controls = np.stack([s.controls for s in samples], axis=0) if samples else np.empty((0, 0, 2))
    env = np.stack([s.env57 for s in samples], axis=0) if samples else np.empty((0, 57))
    success_mask = np.asarray([bool(s.metadata.get("success", False)) for s in samples], dtype=bool)
    metadata = {
        **extra_metadata,
        "samples": [s.metadata for s in samples],
        "scenarios": [s.scenario.to_json() for s in samples],
        "paths": [np.asarray(s.path, dtype=float).tolist() for s in samples],
    }
    np.savez_compressed(
        path,
        states=states,
        controls=controls,
        env=env,
        success_mask=success_mask,
        metadata=json.dumps(metadata),
    )
