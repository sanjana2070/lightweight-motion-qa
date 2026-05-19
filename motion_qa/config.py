# motion_qa/config.py

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# Flag: True → use local HuggingFace LLM for planning + answering
#       False → use fast heuristic keyword matching + rule-based formatting
USE_LLM: bool = os.getenv("USE_LLM", "false").lower() == "true"

# HuggingFace model for local LLM (planner + answerer)
HF_MODEL_ID: str = os.getenv("HF_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")

# HuggingFace model for pose estimation
POSE_MODEL_ID: str = os.getenv("POSE_MODEL_ID", "usyd-community/vitpose-base-simple")

# HuggingFace model for video-level dance style classification (X-CLIP)
XCLIP_MODEL_ID: str = os.getenv("XCLIP_MODEL_ID", "microsoft/xclip-base-patch32")

# AIST++ genre labels (10 genres) used for zero-shot X-CLIP classification
AIST_GENRE_LABELS: list[str] = [
    "Breaking",
    "Popping",
    "Locking",
    "Middle Hip-Hop",
    "LA-style Hip-Hop",
    "House",
    "Waacking",
    "Krump",
    "Street Jazz",
    "Ballet Jazz",
]

# Short code → full name (matches AIST++ filename prefixes)
AIST_GENRE_MAP: dict[str, str] = {
    "BR": "Breaking",
    "PO": "Popping",
    "LO": "Locking",
    "MH": "Middle Hip-Hop",
    "LH": "LA-style Hip-Hop",
    "HO": "House",
    "WA": "Waacking",
    "KR": "Krump",
    "JS": "Street Jazz",
    "JB": "Ballet Jazz",
}

print(f"[config] USE_LLM={USE_LLM}, HF_MODEL_ID={HF_MODEL_ID}")
