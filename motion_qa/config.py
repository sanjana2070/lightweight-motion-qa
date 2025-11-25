# motion_qa/config.py

from __future__ import annotations

import os
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# --- Main toggle flag -------------------------------------------------
# If True  -> use LLM-based planner + answerer (requires API key + quota)
# If False -> use heuristic planner + rule-based answerer (no API calls)
USE_LLM: bool = os.getenv("USE_LLM", "false").lower() == "true"

# Optional: you can configure model names via env, or ignore these for now
PLANNER_MODEL: str = os.getenv("PLANNER_MODEL", "gpt-4o-mini")
ANSWER_MODEL: str = os.getenv("ANSWER_MODEL", "gpt-4o-mini")
