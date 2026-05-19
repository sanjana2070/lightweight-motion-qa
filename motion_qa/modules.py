# motion_qa/modules.py

from __future__ import annotations

from typing import Dict, Any, Optional

import torch

from .features import compute_basic_features


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
# Dance-specific modules (Hip-Hop / House)
# ---------------------------------------------------------------------

def detect_freeze(
    motion: torch.Tensor,
    features: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Detect freeze events — moments where all joints are nearly stationary.

    A freeze is detected when the summed per-joint velocity across all joints
    stays below `velocity_threshold` for at least `min_duration_sec` seconds.

    Returns
    -------
    dict with:
      - value : int  — number of freeze events
      - details.events : list of {start_frame, end_frame, duration_sec}
    """
    params = params or {}
    fps = float(params.get("fps", 30.0))
    velocity_threshold = float(params.get("velocity_threshold", 0.02))
    min_frames = max(1, int(params.get("min_duration_sec", 0.25) * fps))

    T, J, _ = motion.shape
    if T < 2:
        return {"type": "event_list", "value": 0, "details": {"events": []}}

    # Per-frame total velocity: sum of L2 norms across all joints
    diff = motion[1:] - motion[:-1]  # (T-1, J, 3)
    vel = diff.norm(dim=-1).sum(dim=-1)  # (T-1,)

    is_frozen = (vel < velocity_threshold).tolist()

    events: list[Dict[str, Any]] = []
    in_freeze = False
    start = 0
    for i, frozen in enumerate(is_frozen):
        if frozen and not in_freeze:
            in_freeze = True
            start = i
        elif not frozen and in_freeze:
            in_freeze = False
            length = i - start
            if length >= min_frames:
                events.append({
                    "start_frame": start,
                    "end_frame": i,
                    "duration_sec": round(length / fps, 3),
                })
    # Handle freeze that runs to the end
    if in_freeze:
        length = len(is_frozen) - start
        if length >= min_frames:
            events.append({
                "start_frame": start,
                "end_frame": len(is_frozen),
                "duration_sec": round(length / fps, 3),
            })

    return {
        "type": "event_list",
        "value": len(events),
        "details": {"events": events},
    }


def detect_jacking(
    motion: torch.Tensor,
    features: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Detect House dance jacking — rhythmic hip oscillation at ~1.9–2.1 Hz.

    Uses FFT on the left-hip (joint 11) y-coordinate. If the dominant
    frequency falls in the jacking range the result is True.

    Returns
    -------
    dict with:
      - value : bool  — True if jacking rhythm detected
      - details.dominant_freq_hz : float
      - details.power_ratio : float  — fraction of total power at dominant freq
    """
    params = params or {}
    fps = float(params.get("fps", 30.0))
    jack_lo = float(params.get("jack_freq_lo", 1.7))
    jack_hi = float(params.get("jack_freq_hi", 2.3))

    T = motion.shape[0]
    hip_y = motion[:, 11, 1].detach().cpu().numpy()  # left hip y
    hip_y = hip_y - hip_y.mean()  # remove DC

    if T < 16:
        return {
            "type": "categorical",
            "value": False,
            "details": {"dominant_freq_hz": 0.0, "power_ratio": 0.0},
        }

    import numpy as _np
    freqs = _np.fft.rfftfreq(T, d=1.0 / fps)
    spectrum = _np.abs(_np.fft.rfft(hip_y)) ** 2
    total_power = float(spectrum.sum()) or 1.0

    dominant_idx = int(spectrum.argmax())
    dominant_freq = float(freqs[dominant_idx])
    dominant_power = float(spectrum[dominant_idx])

    is_jacking = jack_lo <= dominant_freq <= jack_hi

    return {
        "type": "categorical",
        "value": bool(is_jacking),
        "details": {
            "dominant_freq_hz": round(dominant_freq, 3),
            "power_ratio": round(dominant_power / total_power, 4),
        },
    }


def compute_rhythm_regularity(
    motion: torch.Tensor,
    features: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Measure how rhythmically regular the movement is via autocorrelation
    of the root-joint speed signal.

    A score close to 1.0 means highly periodic (regular, on-beat movement).
    A score close to 0.0 means irregular / arrhythmic.

    Returns
    -------
    dict with:
      - value : float [0, 1]  — regularity score
      - details.peak_period_frames : int  — estimated cycle length in frames
    """
    params = params or {}
    T = motion.shape[0]

    if T < 4:
        return {
            "type": "scalar",
            "value": 0.0,
            "details": {"peak_period_frames": 0},
        }

    import numpy as _np
    root = motion[:, 0, :].detach().cpu().numpy()  # (T, 3)
    speed = _np.linalg.norm(_np.diff(root, axis=0), axis=1)  # (T-1,)
    speed = speed - speed.mean()
    if speed.std() < 1e-9:
        return {"type": "scalar", "value": 0.0, "details": {"peak_period_frames": 0}}

    # Normalised autocorrelation
    n = len(speed)
    acf = _np.correlate(speed, speed, mode="full")[n - 1:]  # (n,)
    acf /= acf[0]  # normalize

    # Find first significant peak after lag=2 (ignore zero-lag and its neighbours)
    search = acf[2:]
    if len(search) == 0:
        return {"type": "scalar", "value": 0.0, "details": {"peak_period_frames": 0}}

    peak_lag = int(search.argmax()) + 2
    peak_value = float(search.max())

    # Clamp to [0, 1]
    score = max(0.0, min(1.0, peak_value))

    return {
        "type": "scalar",
        "value": round(score, 4),
        "details": {"peak_period_frames": peak_lag},
    }


# ---------------------------------------------------------------------
# Dance style classification (X-CLIP, zero-shot)
# ---------------------------------------------------------------------

def classify_dance_style(
    motion: torch.Tensor,
    features: Optional[Dict[str, Any]] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Classify the dance genre/style of a video clip using X-CLIP (zero-shot).

    Requires `video_path` in params — the pose tensor is not used here;
    X-CLIP operates directly on raw video frames.

    Parameters
    ----------
    params : dict
        {
          "video_path": str  — path to the video file (required)
          "num_frames": int  — frames to sample (default 8)
        }

    Returns
    -------
    dict with:
      - type      : "categorical"
      - value     : str   — best matching genre (e.g. "Breaking")
      - details   : {confidence: float, scores: {genre: float, ...}}
    """
    params = params or {}
    video_path = params.get("video_path")
    num_frames = int(params.get("num_frames", 8))

    if not video_path:
        return {
            "type": "categorical",
            "value": "unknown",
            "details": {"error": "video_path not provided in params"},
        }

    from motion_qa.hf_video import classify_from_video
    from motion_qa.config import AIST_GENRE_LABELS

    result = classify_from_video(video_path, AIST_GENRE_LABELS, num_frames=num_frames)
    return {
        "type": "categorical",
        "value": result["best_label"],
        "details": {
            "confidence": result["confidence"],
            "scores": result["scores"],
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
