# motion_qa/video_pose.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def extract_root_trajectory_from_video(
    video_path: str | Path,
    max_frames: Optional[int] = 600,
) -> np.ndarray:
    """
    Extract a simple root joint trajectory (T, 1, 3) from a video
    using MediaPipe Pose.

    - Uses the midpoint of left_hip and right_hip as the "root".
    - Coordinates are MediaPipe's normalized (x, y, z) in image space.
    - Returns an array of shape (T, 1, 3), dtype float32.

    Parameters
    ----------
    video_path : str or Path
        Path to the input video file.
    max_frames : int or None
        Optional cap on the number of frames to process.

    Raises
    ------
    ValueError
        If no pose is detected in any frame.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    # Lazy import so that non-video code paths don't require mediapipe
    import mediapipe as mp

    mp_pose = mp.solutions.pose

    frames_root: list[list[float]] = []
    frame_idx = 0

    # Mediapipe landmark indices for hips
    LEFT_HIP = 23
    RIGHT_HIP = 24

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames is not None and frame_idx > max_frames:
                break

            # Convert BGR (OpenCV) -> RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                lm = results.pose_landmarks.landmark

                lh = lm[LEFT_HIP]
                rh = lm[RIGHT_HIP]

                # MediaPipe gives normalized x,y,z
                x = (lh.x + rh.x) / 2.0
                y = (lh.y + rh.y) / 2.0
                z = (lh.z + rh.z) / 2.0

                frames_root.append([x, y, z])
            else:
                # If no pose is detected for this frame, you can either:
                # - skip it, or
                # - repeat the last valid position.
                # Here we skip, so T may be slightly smaller than frame count.
                continue

    cap.release()

    if not frames_root:
        raise ValueError(
            f"No pose landmarks detected in video {video_path}. "
            "Make sure the person is visible and the video is not empty."
        )

    arr = np.asarray(frames_root, dtype=np.float32)  # (T, 3)
    motion = arr[:, None, :]  # (T, 1, 3)
    return motion
