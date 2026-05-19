# motion_qa/answerer.py

from __future__ import annotations

from typing import Dict, Any
import json


def format_answer(
    question: str,
    tool_name: str,
    raw_answer: Dict[str, Any] | None,
) -> str:
    """
    Rule-based formatter: turn a tool's raw output into a human-readable answer.

    Parameters
    ----------
    question : str
        The user's question.
    tool_name : str
        Name of the tool that produced raw_answer.
    raw_answer : dict
        The dictionary returned by the module (e.g., dominant_direction, etc.).

    Returns
    -------
    str
        Human-readable multi-line string: "Q: ...\nA: ..."
    """
    raw_answer = raw_answer or {}
    details = raw_answer.get("details", {}) or {}
    value = raw_answer.get("value", None)

    # ------------- classify_dance_style -------------
    if tool_name == "classify_dance_style":
        genre = str(value) if value is not None else "unknown"
        confidence = float(details.get("confidence", 0.0))
        scores = details.get("scores", {})
        top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str = ", ".join(f"{g} ({p*100:.1f}%)" for g, p in top3)
        ans = (
            f"The dance style is most likely **{genre}** "
            f"(confidence: {confidence*100:.1f}%). "
            f"Top matches: {top3_str}."
        )
        return f"Q: {question}\nA: {ans}"

    # ------------- dominant_direction -------------
    if tool_name == "dominant_direction":
        direction = str(value) if value is not None else "unknown"
        lr = float(details.get("lr", 0.0))
        fb = float(details.get("fb", 0.0))

        if direction == "stationary":
            ans = (
                "The performer stays roughly in place "
                f"(left-right displacement={lr:.2f}, forward-back={fb:.2f})."
            )
        else:
            ans = (
                f"The performer moves mostly {direction} "
                f"(left-right displacement={lr:.2f}, forward-back={fb:.2f})."
            )

        return f"Q: {question}\nA: {ans}"

    # ------------- most_active_limb -------------
    if tool_name == "most_active_limb":
        limb = str(value) if value is not None else "unknown limb"
        limb_activity = details or {}
        limb_summary = ", ".join(
            f"{name}={float(v):.2f}" for name, v in limb_activity.items()
        )
        ans = (
            f"The most active limb is the {limb.replace('_', ' ')}. "
            f"(movement per limb: {limb_summary})"
        )
        return f"Q: {question}\nA: {ans}"

    # ------------- global_displacement -------------
    if tool_name == "global_displacement":
        dist = float(value) if value is not None else 0.0
        dx = float(details.get("dx", 0.0))
        dy = float(details.get("dy", 0.0))
        dz = float(details.get("dz", 0.0))
        ans = (
            f"The performer travels about {dist:.2f} units from start to end "
            f"(Δx={dx:.2f}, Δy={dy:.2f}, Δz={dz:.2f})."
        )
        return f"Q: {question}\nA: {ans}"

    # ------------- displacement_category -------------
    if tool_name == "displacement_category":
        category = str(value) if value is not None else "unknown"
        dist = float(details.get("distance", 0.0))
        if category == "stationary":
            ans = (
                f"The performer dances mostly in place "
                f"(total displacement ≈ {dist:.2f} units)."
            )
        else:
            ans = (
                f"The performer covers {category} floor space overall "
                f"(total displacement ≈ {dist:.2f} units)."
            )
        return f"Q: {question}\nA: {ans}"

    # ------------- clip_duration -------------
    if tool_name == "clip_duration":
        duration = float(value) if value is not None else 0.0
        T = int(details.get("T", 0))
        fps = float(details.get("fps", 30.0))
        ans = (
            f"The clip is about {duration:.2f} seconds long "
            f"({T} frames at {fps:.1f} fps)."
        )
        return f"Q: {question}\nA: {ans}"

    # ------------- detect_freeze -------------
    if tool_name == "detect_freeze":
        count = int(value) if value is not None else 0
        events = details.get("events", [])
        if count == 0:
            ans = "No clear freeze events were detected in this clip."
        else:
            total_dur = sum(e.get("duration_sec", 0) for e in events)
            ans = (
                f"The performer freezes approximately {count} time(s) "
                f"(total frozen time ≈ {total_dur:.2f}s)."
            )
        return f"Q: {question}\nA: {ans}"

    # ------------- detect_jacking -------------
    if tool_name == "detect_jacking":
        is_jacking = bool(value)
        freq = float(details.get("dominant_freq_hz", 0.0))
        ratio = float(details.get("power_ratio", 0.0))
        if is_jacking:
            ans = (
                f"Yes, jacking is detected — the hips oscillate at "
                f"≈{freq:.2f} Hz ({ratio*100:.1f}% of total movement energy)."
            )
        else:
            ans = (
                f"No clear jacking groove detected (dominant hip frequency "
                f"≈{freq:.2f} Hz, outside the 1.7–2.3 Hz jacking range)."
            )
        return f"Q: {question}\nA: {ans}"

    # ------------- compute_rhythm_regularity -------------
    if tool_name == "compute_rhythm_regularity":
        score = float(value) if value is not None else 0.0
        period = int(details.get("peak_period_frames", 0))
        if score > 0.6:
            label = "highly regular"
        elif score > 0.3:
            label = "moderately regular"
        else:
            label = "irregular"
        ans = (
            f"The movement is {label} (rhythm score = {score:.2f}). "
            f"Estimated cycle length ≈ {period} frames."
        )
        return f"Q: {question}\nA: {ans}"

    # ------------- generic fallback -------------
    # If we don't have a custom formatter, fall back to a generic JSON view.
    pretty = json.dumps(raw_answer, indent=2, default=str)
    ans = (
        "I ran the tool "
        f"'{tool_name}' and obtained the following result:\n{pretty}"
    )
    return f"Q: {question}\nA: {ans}"


def answer_with_llm(
    question: str,
    tool_name: str,
    raw_answer: Dict[str, Any] | None,
) -> str:
    """
    LLM-based answerer using a local HuggingFace model (Phi-3-mini).
    Falls back to rule-based formatter on any error.
    """
    raw_answer = raw_answer or {}

    try:
        from motion_qa.hf_llm import generate
    except ImportError as e:
        print(f"[answerer_llm] hf_llm unavailable ({e}); using rule-based fallback.")
        return format_answer(question, tool_name, raw_answer)

    tool_json = json.dumps(
        {"tool_name": tool_name, "tool_output": raw_answer},
        indent=2,
        default=str,
    )
    system_prompt = (
        "You are a helpful assistant that explains dance movement analysis results.\n"
        "The subject is always a dancer or performer. Use dance-appropriate language.\n"
        "Answer in 1–3 clear sentences using only the provided tool output.\n"
        "Do not mention the tool name unless necessary."
    )
    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Tool output (JSON):\n{tool_json}\n\n"
        "Answer the user's question clearly."
    )

    try:
        return generate(system_prompt, user_prompt, max_new_tokens=128, temperature=0.2)
    except Exception as e:
        print(f"[answerer_llm] LLM error ({e}); using rule-based fallback.")
        return format_answer(question, tool_name, raw_answer)
