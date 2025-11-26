# motion_qa/config.py

from __future__ import annotations

import os
from dotenv import load_dotenv

# Load .env file (if present) into environment
load_dotenv()

# --- OpenAI config ---

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[config] WARNING: OPENAI_API_KEY is not set. LLM calls will fail.")

# Models for planner + answerer (can be the same)
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "gpt-4.1-mini")
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4.1-mini")

# Flag to control whether we *try* to use LLM
# When True, code uses LLM planner + answerer, with heuristic fallback on errors.
USE_LLM = os.getenv("USE_LLM", "false").lower() == "true"

print(
    f"[config] USE_LLM={USE_LLM}, "
    f"PLANNER_MODEL={PLANNER_MODEL}, ANSWER_MODEL={ANSWER_MODEL}"
)
