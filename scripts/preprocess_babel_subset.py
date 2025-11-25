# scripts/preprocess_babel_subset.py

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

import json
import numpy as np
import torch

from motion_qa.features import compute_features_with_events
from motion_qa.modules import (
    dominant_direction,
    count_sit_events,
)

# ---------------------------------------------------------------------
# CONFIG: update these paths to match your local setup
# ---------------------------------------------------------------------

# Where your BABEL JSONs live (even if we don't fully use them yet)
BABEL_DIR = Path("data") / "babel"   # contains train.json, val.json, etc.

# Where your AMASS CMU SMPL+H G data lives.
AMASS_DIR = Path("data") / "CMU"
# If you have an extra level like data/CMU/smplh/CMU, change to:
# AMASS_DIR = Path("data") / "CMU" / "smplh"

# Where WE will write the project-ready subset
OUTPUT_ROOT = Path("data") / "babel_subset"
OUTPUT_MOTION_DIR = OUTPUT_ROOT / "motions"
OUTPUT_META_PATH = OUTPUT_ROOT / "metadata.json"

# How many clips to process:
#   - None or <= 0  -> use ALL .npz files
#   - positive int  -> use only the first N files
MAX_CLIPS: Optional[int] = None   # change to e.g. 50 if you want a cap


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def collect_amass_files(max_clips: Optional[int] = MAX_CLIPS) -> List[Path]:
    """
    Collect AMASS .npz files from AMASS_DIR.

    If max_clips is None or <= 0, return all files.
    Otherwise, return only the first max_clips files.
    """
    if not AMASS_DIR.exists():
        raise FileNotFoundError(f"AMASS_DIR not found: {AMASS_DIR}")

    all_npz = sorted(AMASS_DIR.rglob("*.npz"))
    if not all_npz:
        raise FileNotFoundError(f"No .npz files found under {AMASS_DIR}")

    if max_clips is None or max_clips <= 0:
        selected = all_npz
        print(f"[info] Found {len(all_npz)} .npz files, selecting ALL of them.")
    else:
        selected = all_npz[:max_clips]
        print(f"[info] Found {len(all_npz)} .npz files, selecting first {len(selected)}.")

    return selected


def load_joints_from_npz(path: Path) -> np.ndarray:
    """
    Load a motion from an AMASS .npz file and return a (T, J, 3) numpy array.

    Lightweight version:
      - If 'joints' is present: use it directly as (T, J, 3).
      - Else if 'trans' is present: treat it as a single root joint trajectory
        and create a fake skeleton of shape (T, 1, 3).
    """
    data = np.load(path)
    keys = list(data.keys())

    if "joints" in data:
        joints = data["joints"]  # expected shape: (T, J, 3)
        if joints.ndim != 3 or joints.shape[-1] != 3:
            raise ValueError(
                f"'joints' in {path} must have shape (T, J, 3), got {joints.shape}"
            )
        return joints.astype(np.float32)

    if "trans" in data:
        trans = data["trans"]  # (T, 3)
        if trans.ndim != 2 or trans.shape[-1] != 3:
            raise ValueError(
                f"'trans' in {path} must have shape (T, 3), got {trans.shape}"
            )
        joints = trans[:, None, :]  # (T, 1, 3)
        return joints.astype(np.float32)

    raise ValueError(
        f"No 'joints' or 'trans' key found in {path}. "
        f"Available keys: {keys}. "
        "You may need to adapt load_joints_from_npz() to your file format."
    )


def make_questions_for_clip(
    motion_torch: torch.Tensor,
) -> List[Dict[str, Any]]:
    """
    Given a motion tensor, compute labels using our modules
    and produce corresponding Q/A pairs.

    We focus on:
      - dominant_direction
      - count_sit_events
    """
    features = compute_features_with_events(motion_torch, fps=30.0, hip_index=0)

    # Dominant direction
    dir_result = dominant_direction(motion_torch, features, params={})
    dir_value = dir_result.get("value", "forward")

    # Sit event count
    sit_result = count_sit_events(
        motion_torch,
        features,
        params={"fps": 30.0, "hip_index": 0},
    )
    sit_value = sit_result.get("value", 0)

    qa_list: List[Dict[str, Any]] = []

    qa_list.append({
        "q": "Does the person move more forward or sideways?",
        "a": str(dir_value),
        "type": "dominant_direction",
    })

    qa_list.append({
        "q": "How many times does the person sit down?",
        "a": str(sit_value),
        "type": "count_sit_events",
    })

    return qa_list


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    OUTPUT_MOTION_DIR.mkdir(parents=True, exist_ok=True)

    try:
        selected_files = collect_amass_files(MAX_CLIPS)
    except Exception as e:
        print(f"[error] Could not collect AMASS files: {e}")
        return

    metadata: List[Dict[str, Any]] = []
    clip_index = 0

    for src_path in selected_files:
        print(f"[info] Processing {src_path}")

        try:
            joints_np = load_joints_from_npz(src_path)  # (T, J, 3)
        except Exception as e:
            print(f"[warn] Skipping {src_path} due to error in load_joints_from_npz: {e}")
            continue

        T, J, _ = joints_np.shape
        print(f"       shape: T={T}, J={J}")

        clip_id = f"babel_clip_{clip_index:04d}"
        dst_motion_path = OUTPUT_MOTION_DIR / f"{clip_id}.npy"
        np.save(dst_motion_path, joints_np)

        motion_torch = torch.from_numpy(joints_np).float()
        qa_list = make_questions_for_clip(motion_torch)

        meta_item = {
            "id": clip_id,
            "motion_file": dst_motion_path.name,
            "questions": qa_list,
        }
        metadata.append(meta_item)
        clip_index += 1

    if not metadata:
        print("[error] No clips were successfully processed. "
              "Check that your .npz files contain 'trans' (or 'joints'), "
              "and adapt load_joints_from_npz() if needed.")
        return

    OUTPUT_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[done] Wrote metadata for {len(metadata)} clips to {OUTPUT_META_PATH}")


if __name__ == "__main__":
    main()
