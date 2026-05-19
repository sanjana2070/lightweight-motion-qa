# motion_qa/hf_video.py
# X-CLIP video-text classification for zero-shot dance genre recognition.
# Model: microsoft/xclip-base-patch32 (MIT license).
# First run downloads ~600 MB of weights from HuggingFace Hub.
#
# Inference: sample N frames uniformly from the video, score them against
# a list of text label descriptions, return the best match + per-label scores.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

from motion_qa.config import XCLIP_MODEL_ID

_processor = None
_model = None


def _load_model():
    global _processor, _model
    if _model is not None:
        return _processor, _model

    from transformers import XCLIPProcessor, XCLIPModel

    print(f"[hf_video] Loading X-CLIP model {XCLIP_MODEL_ID}…")
    _processor = XCLIPProcessor.from_pretrained(XCLIP_MODEL_ID)
    _model = XCLIPModel.from_pretrained(XCLIP_MODEL_ID)
    _model.eval()
    if torch.cuda.is_available():
        _model = _model.cuda()
    print("[hf_video] X-CLIP loaded.")
    return _processor, _model


def _sample_frames(video_path: str, num_frames: int = 8) -> list[Image.Image]:
    """Sample num_frames evenly spaced frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = max(total, 1)
    indices = set(
        int(i * total / num_frames) for i in range(num_frames)
    )

    frames: list[Image.Image] = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
        frame_idx += 1

    cap.release()

    if not frames:
        raise ValueError(f"No frames could be read from {video_path}")

    # Pad to exactly num_frames by repeating the last frame if needed
    while len(frames) < num_frames:
        frames.append(frames[-1])

    return frames[:num_frames]


def classify_from_video(
    video_path: str | Path,
    labels: list[str],
    num_frames: int = 8,
) -> dict:
    """
    Zero-shot dance style classification using X-CLIP.

    Parameters
    ----------
    video_path : str or Path
        Path to the input video file.
    labels : list[str]
        Dance genre/style names to classify against (e.g. AIST_GENRE_LABELS).
    num_frames : int
        Number of frames to sample from the video (8 is X-CLIP's default).

    Returns
    -------
    dict with:
      - best_label : str       — top predicted genre
      - confidence : float     — softmax probability for best_label [0, 1]
      - scores : dict[str, float]  — softmax probability per label
    """
    video_path = str(video_path)
    processor, model = _load_model()

    frames = _sample_frames(video_path, num_frames=num_frames)

    # X-CLIP expects text as short descriptions
    text_inputs = [f"a person performing {label.lower()} dance" for label in labels]

    inputs = processor(
        text=text_inputs,
        videos=frames,
        return_tensors="pt",
        padding=True,
    )
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # logits_per_video: (1, num_labels)
    probs = outputs.logits_per_video.softmax(dim=-1)[0].cpu().tolist()

    scores = {label: round(float(p), 4) for label, p in zip(labels, probs)}
    best_label = max(scores, key=scores.__getitem__)

    return {
        "best_label": best_label,
        "confidence": scores[best_label],
        "scores": scores,
    }
