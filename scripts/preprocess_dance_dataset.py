# scripts/preprocess_dance_dataset.py
#
# HOW TO USE:
# 1. Put your video clips in:
#      data/raw_videos/hip_hop/  (e.g. toprock_01.mp4, freeze_01.mp4)
#      data/raw_videos/house/    (e.g. jacking_01.mp4, footwork_02.mp4)
#
# 2. (Optional) For clips with multiple moves, place an annotation JSON
#    alongside each video with the same stem name:
#      data/raw_videos/hip_hop/mixed_01.json
#    Format:
#      {"move_labels": [{"label": "toprock", "start_frame": 0, "end_frame": 90}, ...]}
#
# 3. Run:
#      python -m scripts.preprocess_dance_dataset
#
# Output:
#   data/dance_dataset/motions/{clip_id}.npy   — (T, 17, 3) pose arrays
#   data/dance_dataset/metadata.json           — dataset manifest with Q&A pairs

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from motion_qa.hf_pose import extract_pose_from_video
from motion_qa import modules

RAW_VIDEO_ROOT = Path("data/raw_videos")
OUTPUT_ROOT = Path("data/dance_dataset")
MOTION_DIR = OUTPUT_ROOT / "motions"

STYLES = ["hip_hop", "house"]


def _infer_move_from_filename(stem: str) -> str | None:
    """
    Extract a move label from the filename stem if it starts with a known move name.
    e.g. "toprock_01" → "toprock", "jacking_003" → "jacking"
    """
    known_moves = {
        # hip-hop
        "toprock", "freeze", "6step", "sixstep", "windmill", "headspin",
        "running_man", "roger_rabbit", "cabbage_patch", "brooklyn_stomp",
        # house
        "jacking", "footwork", "lofting", "stalking",
        "salsa_hop", "pas_de_bourée", "shuffle", "crossroads", "scribble_foot", "farmer",
    }
    lower = stem.lower().replace("-", "_")
    for move in sorted(known_moves, key=len, reverse=True):  # longest match first
        if lower.startswith(move):
            return move
    return None


def _load_annotation(json_path: Path) -> dict | None:
    if not json_path.exists():
        return None
    try:
        with open(json_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  [warn] Could not read annotation {json_path}: {e}")
        return None


def _make_questions(
    motion_t: torch.Tensor,
    style: str,
    fps: float,
    move_labels: list[dict],
) -> list[dict]:
    """Generate Q&A pairs for a dance clip."""
    questions: list[dict] = []

    # --- move classification questions ---
    for seg in move_labels:
        label = seg.get("label", "")
        if not label:
            continue
        if style == "hip_hop":
            questions.append({
                "q": f"Does this clip contain a {label.replace('_', ' ')}?",
                "a": "yes",
                "type": "hip_hop_has_move",
            })
        else:
            questions.append({
                "q": f"Is there {label.replace('_', ' ')} in this clip?",
                "a": "yes",
                "type": "house_has_move",
            })

    if len(move_labels) > 1:
        questions.append({
            "q": "How many distinct moves are performed?",
            "a": str(len(move_labels)),
            "type": "move_count",
        })

    if move_labels:
        first_label = move_labels[0].get("label", "unknown")
        questions.append({
            "q": "What move is performed at the start of the clip?",
            "a": first_label.replace("_", " "),
            "type": "move_classification",
        })

    # --- module-based questions ---
    params = {"fps": fps}
    feats = None

    freeze_res = modules.detect_freeze(motion_t, params=params)
    freeze_count = freeze_res.get("value", 0)
    questions.append({
        "q": "Does the performer freeze at any point?",
        "a": "yes" if freeze_count > 0 else "no",
        "type": "detect_freeze",
    })

    if style == "house":
        jack_res = modules.detect_jacking(motion_t, params=params)
        questions.append({
            "q": "Is jacking present in this clip?",
            "a": "yes" if jack_res.get("value") else "no",
            "type": "detect_jacking",
        })

    rr = modules.compute_rhythm_regularity(motion_t, params=params)
    score = rr.get("value", 0.0)
    rhythm_label = "highly regular" if score > 0.6 else "moderately regular" if score > 0.3 else "irregular"
    questions.append({
        "q": "How rhythmically regular is the movement?",
        "a": rhythm_label,
        "type": "compute_rhythm_regularity",
    })

    # Duration
    dur_res = modules.clip_duration(motion_t, params=params)
    dur = dur_res.get("value", 0.0)
    questions.append({
        "q": "How long is this clip in seconds?",
        "a": f"{dur:.1f}",
        "type": "clip_duration",
    })

    return questions


def process_video(
    video_path: Path,
    style: str,
    clip_id: str,
) -> dict | None:
    """
    Extract pose from a single video and return a metadata item.
    Returns None if extraction fails.
    """
    print(f"  Processing {video_path.name} …")

    try:
        motion_np, fps = extract_pose_from_video(str(video_path))
    except Exception as e:
        print(f"  [error] Pose extraction failed: {e}")
        return None

    # Save motion array
    out_path = MOTION_DIR / f"{clip_id}.npy"
    np.save(out_path, motion_np)

    motion_t = torch.from_numpy(motion_np).float()
    T = motion_np.shape[0]
    duration_sec = T / fps

    # Load annotation JSON if it exists
    annotation = _load_annotation(video_path.with_suffix(".json"))

    if annotation and annotation.get("move_labels"):
        move_labels = annotation["move_labels"]
    else:
        # Fall back to filename-based label
        inferred = _infer_move_from_filename(video_path.stem)
        if inferred:
            move_labels = [{"label": inferred, "start_frame": 0, "end_frame": T}]
        else:
            move_labels = []

    questions = _make_questions(motion_t, style, fps, move_labels)

    return {
        "id": clip_id,
        "motion_file": f"{clip_id}.npy",
        "style": style,
        "source": str(video_path),
        "fps": round(fps, 2),
        "duration_sec": round(duration_sec, 3),
        "move_labels": move_labels,
        "questions": questions,
    }


def main() -> None:
    if not RAW_VIDEO_ROOT.exists():
        print(f"[error] Raw video directory not found: {RAW_VIDEO_ROOT}")
        print("Create it and add your video clips in subdirectories: hip_hop/ and house/")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MOTION_DIR.mkdir(parents=True, exist_ok=True)

    metadata: list[dict] = []
    counters: dict[str, int] = {}

    for style in STYLES:
        style_dir = RAW_VIDEO_ROOT / style
        if not style_dir.exists():
            print(f"[info] No {style} directory found at {style_dir}, skipping.")
            continue

        videos = sorted(style_dir.glob("*.mp4")) + sorted(style_dir.glob("*.avi"))
        if not videos:
            print(f"[info] No .mp4/.avi files found in {style_dir}")
            continue

        print(f"\n[{style}] Found {len(videos)} video(s).")

        for video_path in videos:
            style_prefix = "hh" if style == "hip_hop" else "hs"
            counters[style] = counters.get(style, 0) + 1
            clip_id = f"{style_prefix}_{counters[style]:04d}"

            item = process_video(video_path, style, clip_id)
            if item:
                metadata.append(item)
                print(f"    → {clip_id}: {item['duration_sec']:.1f}s, "
                      f"{len(item['move_labels'])} move label(s), "
                      f"{len(item['questions'])} Q&A pairs.")

    if not metadata:
        print("\n[warn] No clips were processed. Check that videos exist in data/raw_videos/.")
        return

    meta_path = OUTPUT_ROOT / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[done] Processed {len(metadata)} clip(s).")
    print(f"       Metadata → {meta_path}")
    print(f"       Motion arrays → {MOTION_DIR}/")


if __name__ == "__main__":
    main()
