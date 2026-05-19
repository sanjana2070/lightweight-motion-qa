# motion_qa/datasets.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import numpy as np
import torch
from torch.utils.data import Dataset

from motion_qa.config import AIST_GENRE_MAP


class MotionQADataset(Dataset):
    """
    Dataset for motion + QA pairs. Compatible with AIST++, Let's Dance,
    and any preprocessed dataset that writes the standard metadata.json format:

    [
      {
        "id": "clip_0001",
        "motion_file": "clip_0001.npy",   # (T, 17, 3) COCO-17 pose
        "style": "BR",                     # AIST++ genre code (optional)
        "video_path": "...",               # original video path (optional)
        "questions": [
          {"q": "What dance style is this?", "a": "Breaking", "type": "classify_dance_style"},
          ...
        ]
      },
      ...
    ]
    """

    def __init__(
        self,
        metadata_path: str | Path,
        motion_dir: str | Path,
    ) -> None:
        self.metadata_path = Path(metadata_path)
        self.motion_dir = Path(motion_dir)

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

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

        motion = torch.from_numpy(np.load(motion_file)).float()  # (T, J, 3)

        questions = item.get("questions", [])
        if questions:
            qa = questions[0]
            question: Optional[str] = qa.get("q")
            answer: Optional[str] = qa.get("a")
            q_type: Optional[str] = qa.get("type")
        else:
            question = answer = q_type = None

        # Resolve AIST++ short code to full genre name if present
        style_code = item.get("style", "")
        genre = AIST_GENRE_MAP.get(style_code, style_code)

        return {
            "id": item.get("id"),
            "motion": motion,
            "genre": genre,
            "video_path": item.get("video_path"),
            "question": question,
            "answer": answer,
            "q_type": q_type,
            "raw_item": item,
        }


def load_single_motion(motion_file: str | Path) -> torch.Tensor:
    """Load a single .npy motion file as a (T, J, 3) float tensor."""
    path = Path(motion_file)
    if not path.exists():
        raise FileNotFoundError(f"Motion file not found: {path}")
    return torch.from_numpy(np.load(path)).float()
