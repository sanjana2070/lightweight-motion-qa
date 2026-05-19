# scripts/preprocess_aist.py
#
# Preprocesses AIST++ videos into the project's standard dataset format.
#
# AIST++ folder structure expected at data/aist_raw/:
#   data/aist_raw/
#     videos/
#       mBR_sFM_cAll_d04_mBR0_ch01.mp4   (genre code in filename prefix)
#       mHO_sFM_cAll_d04_mHO0_ch01.mp4
#       ...
#     keypoints2d/                         (optional — 2D COCO keypoints)
#       mBR_sFM_cAll_d04_mBR0_ch01.pkl
#       ...
#
# Filename convention:  m{GENRE}_{rest}.mp4
#   GENRE codes: BR PO LO MH LH HO WA KR JS JB
#
# HOW TO USE:
#   1. Download AIST++ videos from https://google.github.io/aistplusplus_dataset/
#   2. Place them in data/aist_raw/videos/
#   3. Run:  python -m scripts.preprocess_aist
#
# Output:
#   data/aist/motions/{clip_id}.npy   — (T, 17, 3) COCO-17 pose arrays
#   data/aist/metadata.json           — dataset manifest with Q&A pairs

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch

from motion_qa.config import AIST_GENRE_MAP
from motion_qa.hf_pose import extract_pose_from_video
from motion_qa import modules

RAW_ROOT = Path("data/aist_raw/videos")
OUTPUT_ROOT = Path("data/aist")
MOTION_DIR = OUTPUT_ROOT / "motions"

# Regex to extract genre code from AIST++ filenames like "mBR_sFM_..."
_GENRE_RE = re.compile(r"^m([A-Z]{2})_", re.IGNORECASE)


def _genre_from_filename(stem: str) -> str | None:
    match = _GENRE_RE.match(stem)
    if match:
        code = match.group(1).upper()
        return AIST_GENRE_MAP.get(code)
    return None


def _make_questions(motion_t: torch.Tensor, genre: str, fps: float) -> list[dict]:
    questions: list[dict] = []

    # Genre label question
    questions.append({
        "q": "What dance style is this?",
        "a": genre,
        "type": "classify_dance_style",
    })

    # Freeze detection
    freeze_res = modules.detect_freeze(motion_t, params={"fps": fps})
    questions.append({
        "q": "Does the performer freeze at any point?",
        "a": "yes" if freeze_res["value"] > 0 else "no",
        "type": "detect_freeze",
    })

    # Rhythm regularity
    rr = modules.compute_rhythm_regularity(motion_t, params={"fps": fps})
    score = float(rr["value"])
    rhythm_label = (
        "highly regular" if score > 0.6
        else "moderately regular" if score > 0.3
        else "irregular"
    )
    questions.append({
        "q": "How rhythmically regular is the movement?",
        "a": rhythm_label,
        "type": "compute_rhythm_regularity",
    })

    # House-specific: jacking
    if genre == "House":
        jack_res = modules.detect_jacking(motion_t, params={"fps": fps})
        questions.append({
            "q": "Is jacking present in this clip?",
            "a": "yes" if jack_res["value"] else "no",
            "type": "detect_jacking",
        })

    # Displacement
    dur_res = modules.clip_duration(motion_t, params={"fps": fps})
    questions.append({
        "q": "How long is this clip in seconds?",
        "a": f"{float(dur_res['value']):.1f}",
        "type": "clip_duration",
    })

    return questions


def process_video(video_path: Path, clip_id: str, genre: str) -> dict | None:
    print(f"  Processing {video_path.name} ({genre}) …")

    try:
        motion_np, fps = extract_pose_from_video(str(video_path))
    except Exception as e:
        print(f"  [error] Pose extraction failed: {e}")
        return None

    out_path = MOTION_DIR / f"{clip_id}.npy"
    np.save(out_path, motion_np)

    motion_t = torch.from_numpy(motion_np).float()
    T = motion_np.shape[0]
    duration_sec = T / fps

    questions = _make_questions(motion_t, genre, fps)

    return {
        "id": clip_id,
        "motion_file": f"{clip_id}.npy",
        "style": genre,
        "video_path": str(video_path),
        "fps": round(fps, 2),
        "duration_sec": round(duration_sec, 3),
        "questions": questions,
    }


def main() -> None:
    if not RAW_ROOT.exists():
        print(f"[error] Raw video directory not found: {RAW_ROOT}")
        print("Download AIST++ videos and place them in data/aist_raw/videos/")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MOTION_DIR.mkdir(parents=True, exist_ok=True)

    videos = sorted(RAW_ROOT.glob("*.mp4")) + sorted(RAW_ROOT.glob("*.avi"))
    if not videos:
        print(f"[info] No video files found in {RAW_ROOT}")
        return

    print(f"[info] Found {len(videos)} video(s) in {RAW_ROOT}")

    metadata: list[dict] = []
    counters: dict[str, int] = {}

    for video_path in videos:
        genre = _genre_from_filename(video_path.stem)
        if genre is None:
            print(f"  [skip] Cannot determine genre from filename: {video_path.name}")
            continue

        counters[genre] = counters.get(genre, 0) + 1
        clip_id = f"{genre.lower().replace(' ', '_').replace('-', '_')}_{counters[genre]:04d}"

        item = process_video(video_path, clip_id, genre)
        if item:
            metadata.append(item)
            print(f"    → {clip_id}: {item['duration_sec']:.1f}s, "
                  f"{len(item['questions'])} Q&A pairs.")

    if not metadata:
        print("\n[warn] No clips processed.")
        return

    meta_path = OUTPUT_ROOT / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n[done] Processed {len(metadata)} clip(s).")
    print(f"       Metadata → {meta_path}")
    print(f"       Motion arrays → {MOTION_DIR}/")

    # Genre summary
    from collections import Counter
    genre_counts = Counter(item["style"] for item in metadata)
    print("\n[genre breakdown]")
    for genre, count in sorted(genre_counts.items()):
        print(f"  {genre}: {count} clip(s)")


if __name__ == "__main__":
    main()
