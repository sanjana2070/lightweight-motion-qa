# motion_qa/features.py

from __future__ import annotations

from typing import Dict, Any

import torch


def compute_basic_features(
    motion: torch.Tensor,
    fps: float = 30.0,
) -> Dict[str, Any]:
    """
    Compute basic motion statistics from a skeleton sequence.

    Parameters
    ----------
    motion : torch.Tensor
        Shape (T, J, 3) representing a sequence of 3D joint positions.
    fps : float
        Frames per second.

    Returns
    -------
    features : dict
        duration_sec, root_displacement (3,), per_joint_path_length (J,),
        per_joint_speed_mean (J,), per_joint_speed_max (J,), fps
    """
    if motion.ndim != 3 or motion.shape[-1] != 3:
        raise ValueError(
            f"Expected motion of shape (T, J, 3), got {tuple(motion.shape)}"
        )

    T, J, _ = motion.shape

    root = motion[:, 0, :]          # (T, 3) — joint 0 is pelvis/root in COCO-17
    root_displacement = root[-1] - root[0]  # (3,)

    diffs = motion[1:] - motion[:-1]                  # (T-1, J, 3)
    dists = torch.linalg.norm(diffs, dim=-1)           # (T-1, J)
    speeds = dists * fps

    return {
        "duration_sec": float(T / fps),
        "root_displacement": root_displacement,
        "per_joint_path_length": dists.sum(dim=0),    # (J,)
        "per_joint_speed_mean": speeds.mean(dim=0),   # (J,)
        "per_joint_speed_max": speeds.max(dim=0).values,  # (J,)
        "fps": fps,
    }
