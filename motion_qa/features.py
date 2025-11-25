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
        Frames per second (used to convert frame-to-frame distances into speeds).

    Returns
    -------
    features : dict
        A dictionary containing:
          - duration_sec
          - root_displacement (3,)
          - per_joint_path_length (J,)
          - per_joint_speed_mean (J,)
          - per_joint_speed_max (J,)
    """
    if motion.ndim != 3 or motion.shape[-1] != 3:
        raise ValueError(
            f"Expected motion of shape (T, J, 3), got {tuple(motion.shape)}"
        )

    T, J, _ = motion.shape

    # Root joint: assume index 0 for now (commonly pelvis).
    root = motion[:, 0, :]  # (T, 3)
    root_displacement = root[-1] - root[0]  # (3,)

    # Frame-to-frame differences
    diffs = motion[1:] - motion[:-1]  # (T-1, J, 3)
    dists = torch.linalg.norm(diffs, dim=-1)  # (T-1, J)
    speeds = dists * fps                      # approximate per-joint speed

    per_joint_path_length = dists.sum(dim=0)           # (J,)
    per_joint_speed_mean = speeds.mean(dim=0)          # (J,)
    per_joint_speed_max = speeds.max(dim=0).values     # (J,)

    features: Dict[str, Any] = {
        "duration_sec": float(T / fps),
        "root_displacement": root_displacement,          # (3,)
        "per_joint_path_length": per_joint_path_length,  # (J,)
        "per_joint_speed_mean": per_joint_speed_mean,    # (J,)
        "per_joint_speed_max": per_joint_speed_max,      # (J,)
        "fps": fps,
    }
    return features


def detect_sit_events(
    motion: torch.Tensor,
    fps: float = 30.0,
    hip_index: int = 0,
    threshold_ratio: float = 0.85,
    min_event_duration_sec: float = 0.2,
) -> int:
    """
    Very simple heuristic sit/squat event detector based on hip height.

    Idea:
    - Take hip joint (pelvis) y-coordinate over time.
    - Estimate a "standing" height (e.g., 90th percentile).
    - Consider the person "sitting" when hip height < threshold_ratio * standing_height.
    - Count transitions from not-sit -> sit that last at least `min_event_duration_sec`.

    Parameters
    ----------
    motion : torch.Tensor
        Shape (T, J, 3)
    fps : float
        Frames per second.
    hip_index : int
        Index of the hip/pelvis joint in the skeleton. Adjust based on your skeleton!
    threshold_ratio : float
        Fraction of the standing height used as sit threshold.
    min_event_duration_sec : float
        Minimum duration (seconds) for a sit event to be counted.

    Returns
    -------
    count : int
        Number of sit/squat events detected.
    """
    if motion.ndim != 3 or motion.shape[-1] != 3:
        raise ValueError(
            f"Expected motion of shape (T, J, 3), got {tuple(motion.shape)}"
        )

    T, J, _ = motion.shape
    if not (0 <= hip_index < J):
        raise ValueError(f"hip_index {hip_index} is out of range for J={J}")

    hip_heights = motion[:, hip_index, 1]  # assuming y-axis is "up"
    standing_height = hip_heights.quantile(0.9)
    if standing_height <= 0:
        # degenerate, but avoid divide-by-zero
        return 0

    threshold = threshold_ratio * standing_height
    is_sitting = hip_heights < threshold  # (T,)

    # Count transitions into "sitting" that last at least min_event_duration_sec
    min_frames = int(min_event_duration_sec * fps)
    count = 0
    in_event = False
    event_start = 0

    for t in range(T):
        if is_sitting[t]:
            if not in_event:
                # start of a new potential sit event
                in_event = True
                event_start = t
        else:
            if in_event:
                # leaving sitting state, check duration
                duration = t - event_start
                if duration >= min_frames:
                    count += 1
                in_event = False

    # Handle case where clip ends while still sitting
    if in_event:
        duration = T - event_start
        if duration >= min_frames:
            count += 1

    return int(count)


def compute_features_with_events(
    motion: torch.Tensor,
    fps: float = 30.0,
    hip_index: int = 0,
) -> Dict[str, Any]:
    """
    Convenience function to compute both basic stats and sit-event count.

    Returns the same dict as compute_basic_features, plus:
      - sit_event_count : int
    """
    features = compute_basic_features(motion, fps=fps)
    sit_count = detect_sit_events(
        motion,
        fps=fps,
        hip_index=hip_index,
    )
    features["sit_event_count"] = sit_count
    features["hip_index"] = hip_index
    return features


if __name__ == "__main__":
    # Small self test with a fake motion sequence
    T, J = 100, 17
    fake_motion = torch.zeros(T, J, 3)
    fake_motion[:, 0, 1] = 1.0  # hip at height 1.0
    # simulate a sit from t=30..50
    fake_motion[30:50, 0, 1] = 0.5

    feats = compute_features_with_events(fake_motion, fps=30.0, hip_index=0)
    print("Duration (sec):", feats["duration_sec"])
    print("Sit events detected:", feats["sit_event_count"])
