# scripts/app_web.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import gradio as gr

from motion_qa.video_pose import extract_root_trajectory_from_video
from motion_qa.features import compute_features_with_events
from motion_qa import modules, config
from motion_qa.planner import plan_from_question, plan_from_question_llm
from motion_qa.answerer import format_answer, answer_with_llm


def _resolve_video_path(video: Any) -> str | None:
    """
    Try to extract a filesystem path from whatever Gradio's Video component
    passes into the function.

    Depending on gradio version / config, `video` might be:
      - a string filepath
      - a dict like {"name": "...", "data": "..."} (older file-like behavior)
      - or something else file-like

    We do a best-effort resolution here.
    """
    if video is None:
        return None

    # Common case: gradio already gives you a string path
    if isinstance(video, (str, Path)):
        return str(video)

    # Sometimes File/Video can be a dict with name/data
    if isinstance(video, dict):
        # Try common keys
        candidate = video.get("name") or video.get("data")
        if isinstance(candidate, (str, Path)):
            return str(candidate)

    # Fallback: last resort, try casting to str
    try:
        return str(video)
    except Exception:
        return None


def qa_on_video(video: Any, question: str) -> str:
    """
    Main function used by the web UI.

    Parameters
    ----------
    video : Any
        Value from the Gradio Video component (type depends on Gradio version).
    question : str
        User's natural language question.

    Returns
    -------
    str
        Answer text (or an error message).
    """
    video_path = _resolve_video_path(video)
    if not video_path:
        return "Please upload a video (could not resolve video path from the upload)."

    question = (question or "").strip()
    if not question:
        return "Please enter a question about the motion."

    try:
        # 1) Extract motion from video -> (T, 1, 3)
        motion_np = extract_root_trajectory_from_video(video_path)
        motion = torch.from_numpy(motion_np)  # (T, 1, 3)

        # 2) Compute features (same as with AMASS)
        features = compute_features_with_events(
            motion,
            fps=30.0,   # approximate; adjust if you know the real FPS
            hip_index=0,
        )

        # 3) Map tool names to module functions
        module_map = {
            "count_sit_events": modules.count_sit_events,
            "dominant_direction": modules.dominant_direction,
            "most_active_limb": modules.most_active_limb,
            "global_displacement": modules.global_displacement,
            "displacement_category": modules.displacement_category,
            "clip_duration": modules.clip_duration,
        }
        tool_names = list(module_map.keys())

        # 4) Choose tool: LLM or heuristic planner
        if config.USE_LLM:
            print("[web] USE_LLM=True -> using LLM-based planner + answerer.")
            plan = plan_from_question_llm(
                question,
                tool_names,
                model=config.PLANNER_MODEL,
            )
        else:
            print("[web] USE_LLM=False -> using heuristic planner + rule-based answerer.")
            plan = plan_from_question(
                question,
                tool_names,
            )

        tool_name = plan["tool"]
        params = plan.get("params", {})

        if tool_name not in module_map:
            return f"Planner chose unknown tool '{tool_name}'."

        # 5) Run the selected analysis tool
        module_fn = module_map[tool_name]
        raw_answer = module_fn(motion, features, params=params)

        # 6) Format final answer
        if config.USE_LLM:
            final_text = answer_with_llm(
                question,
                tool_name,
                raw_answer,
                model=config.ANSWER_MODEL,
            )
        else:
            final_text = format_answer(question, tool_name, raw_answer)

        return final_text

    except Exception as e:
        # Surface any errors nicely on the page
        return f"Error while processing video: {e}"


def main() -> None:
    """
    Launch the Gradio web app.
    """
    description = (
        "Upload a short video of a person moving (walking, sitting, etc.), "
        "then ask a question about the motion.\n\n"
        "Examples:\n"
        "- Does the person move more forward or sideways?\n"
        "- How many times does the person sit down?\n"
        "- How far does the person travel from start to end?\n"
        "- Does the person mostly stay in place, or move a lot?\n"
        "- How long is this motion clip in seconds?"
    )

    # NOTE: no 'type=' argument here to keep it compatible with your gradio version
    video_input = gr.Video(
        label="Upload a short motion video",
    )
    question_input = gr.Textbox(
        lines=2,
        label="Question about this motion",
        placeholder="e.g., Does the person move more forward or sideways?",
    )
    answer_output = gr.Textbox(
        lines=5,
        label="Answer",
    )

    iface = gr.Interface(
        fn=qa_on_video,
        inputs=[video_input, question_input],
        outputs=answer_output,
        title="Lightweight Motion QA on Video",
        description=description,
    )

    iface.launch()


if __name__ == "__main__":
    main()
