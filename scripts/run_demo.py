# scripts/run_demo.py

from __future__ import annotations

from pathlib import Path

import torch

from motion_qa.datasets import MotionQADataset
from motion_qa.features import compute_features_with_events
from motion_qa import modules, config  # config has USE_LLM flag

# We import both LLM and non-LLM versions; choose at runtime via config.USE_LLM
from motion_qa.planner import plan_from_question, plan_from_question_llm
from motion_qa.answerer import format_answer, answer_with_llm

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_ROOT = Path("data") / "babel_subset"
MOTION_DIR = DATA_ROOT / "motions"
META_PATH = DATA_ROOT / "metadata.json"


def main() -> None:
    # 1) Check that the preprocessed subset exists
    if not META_PATH.exists():
        print(f"[error] Metadata file not found: {META_PATH}")
        print("Hint: run `python -m scripts.preprocess_babel_subset` first.")
        return

    if not MOTION_DIR.exists():
        print(f"[error] Motion directory not found: {MOTION_DIR}")
        print("Hint: run `python -m scripts.preprocess_babel_subset` first.")
        return

    # 2) Load dataset
    ds = MotionQADataset(META_PATH, MOTION_DIR)
    print(f"[info] Dataset size: {len(ds)}")

    if len(ds) == 0:
        print("[error] Dataset is empty. Check your preprocess_babel_subset step.")
        return

    # For now, just take the first item
    item_index = 0
    item = ds[item_index]

    motion: torch.Tensor = item["motion"]  # (T, J, 3)
    question = item["question"]
    clip_id = item["id"]

    print(f"[info] Using item index: {item_index}")
    print(f"[info] Clip ID: {clip_id}")
    print(f"[info] Motion shape: {tuple(motion.shape)} (T, J, 3)")
    print(f"[info] Question: {question}")

    # 3) Compute features (including sit-event count)
    features = compute_features_with_events(
        motion,
        fps=30.0,
        hip_index=0,  # adjust if your skeleton uses a different hip index
    )

    # 4) Map tool names to module functions
    module_map = {
        "count_sit_events": modules.count_sit_events,
        "dominant_direction": modules.dominant_direction,
        "most_active_limb": modules.most_active_limb,
    }

    # 5) Choose planner based on config.USE_LLM
    if config.USE_LLM:
        print("[config] USE_LLM=True -> using LLM-based planner + answerer.")
        plan = plan_from_question_llm(
            question or "",
            list(module_map.keys()),
            model=config.PLANNER_MODEL,
        )
    else:
        print("[config] USE_LLM=False -> using heuristic planner + rule-based answerer.")
        plan = plan_from_question(
            question or "",
            list(module_map.keys()),
        )

    tool_name = plan["tool"]
    params = plan.get("params", {})

    print(f"[info] Planner chose tool: {tool_name} with params={params}")

    module_fn = module_map.get(tool_name)
    if module_fn is None:
        print(f"[error] No module implemented for tool_name={tool_name}")
        return

    # 6) Run the chosen module
    raw_answer = module_fn(motion, features, params=params)
    print("[info] Raw module output:", raw_answer)

    # 7) Turn raw answer into human-readable text
    print("\n" + "-" * 60)
    if config.USE_LLM:
        final_text = answer_with_llm(
            question,
            tool_name,
            raw_answer,
            model=config.ANSWER_MODEL,
        )
    else:
        final_text = format_answer(question, tool_name, raw_answer)

    print(final_text)
    print("-" * 60)


if __name__ == "__main__":
    main()
