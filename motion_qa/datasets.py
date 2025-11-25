# motion_qa/datasets.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import numpy as np
import torch
from torch.utils.data import Dataset


class MotionQADataset(Dataset):
    """
    Simple dataset for motion + QA pairs.

    Expects a metadata JSON file with a list of items:
    [
      {
        "id": "clip_0001",
        "motion_file": "clip_0001.npy",
        "questions": [
          {
            "q": "How many times does the person sit down?",
            "a": "2",
            "type": "count_sit_events"
          },
          ...
        ]
      },
      ...
    ]

    And a directory containing corresponding .npy files with shape (T, J, 3).
    """

    def __init__(
        self,
        metadata_path: str | Path,
        motion_dir: str | Path,
        pick_first_question: bool = True,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.motion_dir = Path(motion_dir)
        self.pick_first_question = pick_first_question

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.items: List[Dict[str, Any]] = json.load(f)

        if not isinstance(self.items, list):
            raise ValueError("Metadata JSON must be a list of items")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]

        motion_file = self.motion_dir / item["motion_file"]
        if not motion_file.exists():
            raise FileNotFoundError(f"Motion file not found: {motion_file}")

        # Load motion as (T, J, 3) tensor
        motion_np = np.load(motion_file)  # type: ignore[arg-type]
        motion = torch.from_numpy(motion_np).float()

        questions = item.get("questions", [])
        if not questions:
            question: Optional[str] = None
            answer: Optional[str] = None
            q_type: Optional[str] = None
        else:
            if self.pick_first_question:
                qa = questions[0]
            else:
                # you can later randomize or sample multiple QAs
                qa = questions[0]
            question = qa.get("q")
            answer = qa.get("a")
            q_type = qa.get("type")

        return {
            "id": item.get("id"),
            "motion": motion,         # (T, J, 3) float tensor
            "question": question,    # str | None
            "answer": answer,        # str | None (ground truth, if available)
            "q_type": q_type,        # e.g. "count_sit_events"
            "raw_item": item,        # full metadata row
        }


def load_single_motion(motion_file: str | Path) -> torch.Tensor:
    """
    Convenience helper to load a single motion file without the dataset.

    Parameters
    ----------
    motion_file: path to .npy file with shape (T, J, 3)

    Returns
    -------
    motion: torch.Tensor (T, J, 3)
    """
    motion_path = Path(motion_file)
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_path}")
    motion_np = np.load(motion_path)  # type: ignore[arg-type]
    return torch.from_numpy(motion_np).float()


if __name__ == "__main__":
    # Tiny self-test (will fail if you don't have metadata yet, that's ok).
    sample_meta = Path("data/babel_subset/metadata.json")
    sample_motion_dir = Path("data/babel_subset/motions")

    if sample_meta.exists():
        ds = MotionQADataset(sample_meta, sample_motion_dir)
        print("Dataset length:", len(ds))
        item0 = ds[0]
        print("First item keys:", item0.keys())
        print("Motion shape:", item0["motion"].shape)
        print("Question:", item0["question"])
    else:
        print("No sample metadata found, skipping test.")
