# scripts/app_web.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import gradio as gr

from motion_qa.video_pose import extract_root_trajectory_from_video
from motion_qa.features import compute_basic_features
from motion_qa import config
from motion_qa.registry import MODULE_MAP
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
        # 1) Extract pose from video -> (T, 17, 3) and real FPS
        motion_np, fps = extract_root_trajectory_from_video(video_path)
        motion = torch.from_numpy(motion_np)

        # 2) Compute features (pose-based spatial/temporal modules)
        features = compute_basic_features(motion, fps=fps)

        # 3) Choose tool: LLM or heuristic planner
        tool_names = list(MODULE_MAP.keys())
        if config.USE_LLM:
            plan = plan_from_question_llm(question, tool_names)
        else:
            plan = plan_from_question(question, tool_names)

        tool_name = plan["tool"]
        # Pass video_path in params so classify_dance_style can access raw frames
        params = {**plan.get("params", {}), "video_path": video_path, "fps": fps}

        if tool_name not in MODULE_MAP:
            return f"Planner chose unknown tool '{tool_name}'."

        # 4) Run the selected analysis tool
        raw_answer = MODULE_MAP[tool_name](motion, features, params=params)

        # 5) Format final answer
        if config.USE_LLM:
            final_text = answer_with_llm(question, tool_name, raw_answer)
        else:
            final_text = format_answer(question, tool_name, raw_answer)

        return final_text

    except Exception as e:
        return f"Error while processing video: {e}"


def main() -> None:
    """
    Launch the Gradio web app.
    """
    description = (
        "Upload a short dance video, then ask a question about the movement.\n\n"
        "Examples:\n"
        "- Does the performer freeze at any point?\n"
        "- Is jacking present in this clip?\n"
        "- How rhythmically regular is the movement?\n"
        "- What move is performed at the start?\n"
        "- Which limb is most active?\n"
        "- Does the performer travel across the floor or stay in place?\n"
        "- How long is this clip in seconds?"
    )

    with gr.Blocks(title="Dance QA") as demo:
        gr.Markdown(f"### Dance QA\n\n{description}")

        with gr.Row():
            video_input = gr.Video(
                label="Upload or record a dance video clip",
                format="mp4",
                sources=["webcam", "upload"],
            )

        question_input = gr.Textbox(
            lines=2,
            label="Question about this dance",
            placeholder="e.g., Does the performer freeze at any point?",
        )
        answer_output = gr.Textbox(
            lines=5,
            label="Answer",
        )
        submit_btn = gr.Button("Ask")

        submit_btn.click(
            fn=qa_on_video,
            inputs=[video_input, question_input],
            outputs=answer_output,
        )

    demo.launch()


if __name__ == "__main__":
    main()
