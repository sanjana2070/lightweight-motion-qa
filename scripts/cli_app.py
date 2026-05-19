# scripts/cli_app.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import torch

from motion_qa.datasets import MotionQADataset
from motion_qa.features import compute_basic_features
from motion_qa import config
from motion_qa.registry import MODULE_MAP
from motion_qa.viz import plot_root_trajectory_2d
from motion_qa.planner import plan_from_question, plan_from_question_llm
from motion_qa.answerer import format_answer, answer_with_llm

DATA_ROOT = Path("data") / "aist"
MOTION_DIR = DATA_ROOT / "motions"
META_PATH = DATA_ROOT / "metadata.json"

# Extra pre-written questions that work well with your current modules
PREDEFINED_QUESTIONS: List[str] = [
    # dominant_direction / count_sit_events (existing)
    "Does the person move more forward or sideways?",
    "Does the person move more left or right?",
    "How many times does the person sit down?",

    # new ones for global_displacement / displacement_category / clip_duration
    "How far does the person travel from start to end?",
    "Does the person mostly stay in place, or move a lot?",
    "How long is this motion clip in seconds?",
]


def load_dataset() -> MotionQADataset:
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {META_PATH}\n"
            "Hint: run `python -m scripts.preprocess_aist` first."
        )
    if not MOTION_DIR.exists():
        raise FileNotFoundError(
            f"Motion directory not found: {MOTION_DIR}\n"
            "Hint: run `python -m scripts.preprocess_aist` first."
        )

    ds = MotionQADataset(META_PATH, MOTION_DIR)
    if len(ds) == 0:
        raise RuntimeError(
            "Dataset is empty. Check your preprocess_babel_subset step."
        )
    return ds


def choose_clip_index(num_clips: int) -> Optional[int]:
    """
    Ask the user which clip index to use.
    Returns an integer index or None if the user quits.
    """
    while True:
        print()
        print(f"[clips] There are {num_clips} clips (0 to {num_clips - 1}).")
        user_in = input(
            "Enter a clip index to inspect (or 'q' to quit): "
        ).strip()

        if user_in.lower() in {"q", "quit", "exit"}:
            return None

        if not user_in.isdigit():
            print("[warn] Please enter a valid integer index.")
            continue

        idx = int(user_in)
        if 0 <= idx < num_clips:
            return idx
        else:
            print("[warn] Index out of range.")


def summarize_motion(motion: torch.Tensor) -> str:
    """
    Return a short text summary of the clip: length & rough displacement.
    """
    T, J, _ = motion.shape
    root = motion[:, 0, :]
    start = root[0]
    end = root[-1]
    disp_vec = end - start
    lr = float(disp_vec[0])
    fb = float(disp_vec[2])
    return (
        f"T = {T} frames, J = {J} joint(s), "
        f"net LR displacement ≈ {lr:.2f}, FB displacement ≈ {fb:.2f}"
    )


def choose_question(default_question: str) -> str:
    """
    Let the user either:
      - use the default metadata question,
      - choose one of the pre-written questions, or
      - type a custom question.
    """
    print()
    print("[question] Default question from metadata:")
    print(f"  \"{default_question}\"")
    print()
    print("Question options:")
    print("  1) Use this metadata question")
    print("  2) Choose from pre-written questions")
    print("  3) Type my own question")

    while True:
        choice = input("Choose 1, 2, or 3: ").strip()

        if choice == "1":
            return default_question

        elif choice == "2":
            # Show pre-written questions
            print("\n[pre-written questions]")
            for i, q in enumerate(PREDEFINED_QUESTIONS):
                print(f"  {i}: {q}")
            while True:
                idx_str = input(
                    "Enter question index (or 'b' to go back): "
                ).strip()
                if idx_str.lower() in {"b", "back"}:
                    break
                if not idx_str.isdigit():
                    print("[warn] Please enter a valid integer index.")
                    continue
                idx = int(idx_str)
                if 0 <= idx < len(PREDEFINED_QUESTIONS):
                    return PREDEFINED_QUESTIONS[idx]
                else:
                    print("[warn] Index out of range.")
            # If user backs out, show main question menu again
            print()
            print("Question options:")
            print("  1) Use metadata question")
            print("  2) Choose from pre-written questions")
            print("  3) Type my own question")

        elif choice == "3":
            custom = input("Enter your question about this motion: ").strip()
            if custom:
                return custom
            else:
                print("[warn] Question cannot be empty.")
        else:
            print("[warn] Please enter 1, 2, or 3.")


def run_qa_for_item(ds: MotionQADataset, idx: int) -> None:
    """
    Run the full pipeline (planner -> module -> answerer) for a given item index.
    """
    item = ds[idx]
    motion: torch.Tensor = item["motion"]
    default_question = (
        item.get("question")
        or "Does the person move more forward or sideways?"
    )
    clip_id = item["id"]

    print()
    print("=" * 70)
    print(f"[clip] Index: {idx}")
    print(f"[clip] ID: {clip_id}")
    print(f"[clip] Motion shape: {tuple(motion.shape)} (T, J, 3)")
    print(f"[clip] Summary: {summarize_motion(motion)}")

    # Optional visualization
    view_choice = input(
        "View a 2D plot of the root trajectory? [y/n]: "
    ).strip().lower()
    if view_choice in {"y", "yes"}:
        plot_root_trajectory_2d(motion, clip_id)

    # Let the user pick or type a question
    question = choose_question(default_question)

    print()
    print(f"[info] Final question: {question!r}")

    # Compute features
    features = compute_basic_features(motion, fps=30.0)

    # Planner: LLM or heuristic based on config.USE_LLM
    if config.USE_LLM:
        print("[config] USE_LLM=True -> using LLM-based planner + answerer.")
        plan = plan_from_question_llm(question or "", list(MODULE_MAP.keys()))
    else:
        print("[config] USE_LLM=False -> using heuristic planner + rule-based answerer.")
        plan = plan_from_question(question or "", list(MODULE_MAP.keys()))

    tool_name = plan["tool"]
    # Include video_path from raw_item so classify_dance_style can read frames
    video_path = item.get("raw_item", {}).get("video_path", "")
    params = {**plan.get("params", {}), "video_path": video_path, "fps": 30.0}

    print(f"[info] Planner chose tool: {tool_name} with params={params}")

    module_fn = MODULE_MAP.get(tool_name)
    if module_fn is None:
        print(f"[error] No module implemented for tool_name={tool_name}")
        return

    # Run module
    raw_answer = module_fn(motion, features, params=params)
    print("[info] Raw module output:", raw_answer)

    # Format answer
    print("\n" + "-" * 60)
    if config.USE_LLM:
        final_text = answer_with_llm(question, tool_name, raw_answer)
    else:
        final_text = format_answer(question, tool_name, raw_answer)

    print(final_text)
    print("-" * 60)
    print("=" * 70)
    print()


def main() -> None:
    print("[setup] Loading MotionQADataset...")
    ds = load_dataset()
    print(f"[setup] Loaded dataset with {len(ds)} clips.")

    if config.USE_LLM:
        print("[config] USE_LLM=True (LLM planner + answerer enabled, requires API + quota).")
    else:
        print("[config] USE_LLM=False (offline heuristic + rule-based answers).")

    while True:
        idx = choose_clip_index(len(ds))
        if idx is None:
            print("\n[bye] Exiting Motion QA CLI.")
            break

        run_qa_for_item(ds, idx)


if __name__ == "__main__":
    main()
