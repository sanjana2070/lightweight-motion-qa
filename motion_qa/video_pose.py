# motion_qa/video_pose.py
# Thin shim — delegates to hf_pose.extract_pose_from_video (ViTPose, 17 joints).
# Kept for backwards compatibility so existing call sites need no changes.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from motion_qa.hf_pose import extract_pose_from_video


def extract_root_trajectory_from_video(
    video_path: str | Path,
    max_frames: Optional[int] = 600,
) -> tuple[np.ndarray, float]:
    """
    Extract full 17-joint COCO pose trajectory from a video.

    Delegates to hf_pose.extract_pose_from_video (ViTPose + YOLOv8 person detector).

    Returns
    -------
    motion : np.ndarray, shape (T, 17, 3), dtype float32
        Third channel = (x_norm, y_norm, confidence).
    fps : float
        Frames per second from the video container.
    """
    return extract_pose_from_video(video_path, max_frames=max_frames)
