# motion_qa/modules.py

from __future__ import annotations

from typing import Dict, Any, Optional

import torch

from .features import compute_basic_features, compute_features_with_events


# NOTE: These joint groups are example indices.
# You MUST adapt them to match your skeleton layout once you have more joints!
JOINT_GROUPS: Dict[str, list[int]] = {
    "left_arm": [5, 6, 7],
    "right_arm": [8, 9, 10],
    "left_leg": [11, 12, 13],
    "right_leg": [14, 15, 16],
}


# ---------------------------------------------------------------------
# Limb activity
# ---------------------------------------------------------------------
def most_active_limb(
    motion: torch.Tensor,
    features: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Determine which limb moves the most based on per-joint path length.

    Assumes:
      - features["per_joint_path_length"] is a (J,) tensor or array giving
        the total path length for each joint over the clip.
      - JOINT_GROUPS maps limb names to joint indices.

    Returns
    -------
    result : dict
        {
          "type": "categorical",
          "value": "<limb_name>",
          "details": {
              "left_arm": float,
              "right_arm": float,
              "left_leg": float,
              "right_leg": float
          }
        }
    """
    if features is None:
        features = compute_basic_features(motion)

    path = features["per_joint_path_length"]  # (J,)
    if not isinstance(path, torch.Tensor):
        path = torch.as_tensor(path)

    limb_activity: Dict[str, float] = {}
    for name, idxs in JOINT_GROUPS.items():
        # Filter out indices that might be out of range
        valid_idxs = [i for i in idxs if i < path.shape[0]]
        if not valid_idxs:
            limb_activity[name] = 0.0
        else:
            limb_activity[name] = float(path[valid_idxs].sum().item())

    best_limb = max(limb_activity, key=lambda k: limb_activity[k])

    return {
        "type": "categorical",
        "value": best_limb,
        "details": limb_activity,
    }


# ---------------------------------------------------------------------
# Global direction & displacement
# ---------------------------------------------------------------------
def dominant_direction(
    motion: torch.Tensor,
    features: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Determine whether the motion is mostly forward/backward vs left/right.

    Assumes:
    - x-axis: left/right
    - z-axis: forward/backward
    (You may need to adapt this based on your coordinate system.)

    Returns
    -------
    result : dict
        {
          "type": "categorical",
          "value": "forward" | "backward" | "left" | "right" | "stationary",
          "details": {
              "lr": float,  # left(+)/right(-) displacement
              "fb": float   # forward(+)/backward(-) displacement
          }
        }
    """
    if features is None:
        features = compute_basic_features(motion)

    disp = features["root_displacement"]  # (3,)
    if not isinstance(disp, torch.Tensor):
        disp = torch.as_tensor(disp)

    # Assumed mapping: x = left/right, z = forward/back.
    lr = float(disp[0].item())
    fb = float(disp[2].item()) if disp.numel() > 2 else 0.0

    tol = params.get("tolerance", 1e-3) if params else 1e-3
    if abs(lr) < tol and abs(fb) < tol:
        direction = "stationary"
    elif abs(fb) >= abs(lr):
        direction = "forward" if fb > 0 else "backward"
    else:
        direction = "left" if lr > 0 else "right"

    return {
        "type": "categorical",
        "value": direction,
        "details": {"lr": lr, "fb": fb},
    }


def global_displacement(
    motion: torch.Tensor,
    features: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute how far the root moves from start to end of the clip.

    Assumes joint 0 is the root. Works even if J=1 (root only).
    """
    if features is None:
        features = compute_basic_features(motion)

    # Root trajectory: (T, 3)
    root = motion[:, 0, :]
    start = root[0]
    end = root[-1]

    disp_vec = end - start  # (3,)
    dist = float(torch.linalg.norm(disp_vec))

    return {
        "type": "scalar",
        "value": dist,
        "details": {
            "dx": float(disp_vec[0]),
            "dy": float(disp_vec[1]),
            "dz": float(disp_vec[2]),
        },
    }


def displacement_category(
    motion: torch.Tensor,
    features: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Categorize how much the person moves:
      - 'stationary', 'small', 'medium', 'large'

    Based on total root displacement from start to end.
    """
    if features is None:
        features = compute_basic_features(motion)

    dist_info = global_displacement(motion, features, params or {})
    dist = float(dist_info["value"])

    if dist < 0.2:
        cat = "stationary"
    elif dist < 1.0:
        cat = "small"
    elif dist < 3.0:
        cat = "medium"
    else:
        cat = "large"

    return {
        "type": "categorical",
        "value": cat,
        "details": {"distance": dist},
    }


# ---------------------------------------------------------------------
# Sit / squat events
# ---------------------------------------------------------------------
def count_sit_events(
    motion: torch.Tensor,
    features: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Count how many sit/squat-like events occur in the motion.

    Uses the `sit_event_count` returned by compute_features_with_events().
    You can override hip_index or fps via params.

    Parameters
    ----------
    params : dict (optional)
        {
          "fps": float,
          "hip_index": int
        }

    Returns
    -------
    result : dict
        {
          "type": "count",
          "value": int
        }
    """
    params = params or {}
    fps = params.get("fps", 30.0)
    hip_index = params.get("hip_index", 0)

    if features is None or "sit_event_count" not in features:
        features = compute_features_with_events(
            motion,
            fps=fps,
            hip_index=hip_index,
        )

    sit_count = int(features["sit_event_count"])

    return {
        "type": "count",
        "value": sit_count,
        "details": {
            "fps": fps,
            "hip_index": hip_index,
        },
    }


# ---------------------------------------------------------------------
# Clip duration
# ---------------------------------------------------------------------
def clip_duration(
    motion: torch.Tensor,
    features: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute the duration of the motion clip in seconds.

    Parameters
    ----------
    params : dict (optional)
        {
          "fps": float   # default 30.0
        }

    Returns
    -------
    result : dict
        {
          "type": "scalar",
          "value": float,  # duration in seconds
          "details": {
              "T": int,   # number of frames
              "fps": float
          }
        }
    """
    params = params or {}
    fps = float(params.get("fps", 30.0))

    T = motion.shape[0]
    duration = T / fps

    return {
        "type": "scalar",
        "value": float(duration),
        "details": {
            "T": int(T),
            "fps": fps,
        },
    }


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Quick self-test with fake motion
    T, J = 120, 17
    fake = torch.zeros(T, J, 3)

    # root moves forward along z-axis
    fake[:, 0, 2] = torch.linspace(0.0, 2.0, T)

    # left_leg moves a bit more than others
    fake[:, 11, 1] = 0.1 * torch.sin(torch.linspace(0, 10, T))
    fake[:, 12, 1] = 0.1 * torch.sin(torch.linspace(0, 10, T))
    fake[:, 13, 1] = 0.1 * torch.sin(torch.linspace(0, 10, T))

    # simulate two sits (hip drops)
    fake[:, 0, 1] = 1.0
    fake[30:40, 0, 1] = 0.5
    fake[80:95, 0, 1] = 0.5

    print("Testing dominant_direction:")
    print(dominant_direction(fake))

    print("\nTesting most_active_limb:")
    print(most_active_limb(fake))

    print("\nTesting count_sit_events:")
    print(count_sit_events(fake, params={"fps": 30.0, "hip_index": 0}))

    print("\nTesting global_displacement:")
    print(global_displacement(fake))

    print("\nTesting displacement_category:")
    print(displacement_category(fake))

    print("\nTesting clip_duration:")
    print(clip_duration(fake, params={"fps": 30.0}))
