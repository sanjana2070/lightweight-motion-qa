# motion_qa/answerer.py

from __future__ import annotations

from typing import Dict, Any
import json

try:
    from openai import OpenAI  # Optional; we fall back gracefully if missing
except ImportError:
    OpenAI = None  # type: ignore


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

    # ------------- dominant_direction -------------
    if tool_name == "dominant_direction":
        direction = str(value) if value is not None else "unknown"
        lr = float(details.get("lr", 0.0))
        fb = float(details.get("fb", 0.0))

        if direction == "stationary":
            ans = (
                "The person stays roughly in place "
                f"(left-right displacement={lr:.2f}, forward-back={fb:.2f})."
            )
        else:
            ans = (
                f"The person moves mostly {direction} "
                f"(left-right displacement={lr:.2f}, forward-back={fb:.2f})."
            )

        return f"Q: {question}\nA: {ans}"

    # ------------- count_sit_events -------------
    if tool_name == "count_sit_events":
        count = int(value) if value is not None else 0
        ans = f"The person sits down approximately {count} time(s) in this clip."
        return f"Q: {question}\nA: {ans}"

    # ------------- most_active_limb -------------
    if tool_name == "most_active_limb":
        limb = str(value) if value is not None else "unknown limb"
        limb_activity = details or {}
        limb_summary = ", ".join(
            f"{name}={float(v):.2f}" for name, v in limb_activity.items()
        )
        ans = (
            f"The most active limb appears to be the {limb.replace('_', ' ')}. "
            f"(total movement per limb: {limb_summary})"
        )
        return f"Q: {question}\nA: {ans}"

    # ------------- global_displacement -------------
    if tool_name == "global_displacement":
        dist = float(value) if value is not None else 0.0
        dx = float(details.get("dx", 0.0))
        dy = float(details.get("dy", 0.0))
        dz = float(details.get("dz", 0.0))
        ans = (
            f"The person moves about {dist:.2f} units from start to end "
            f"(Δx={dx:.2f}, Δy={dy:.2f}, Δz={dz:.2f})."
        )
        return f"Q: {question}\nA: {ans}"

    # ------------- displacement_category -------------
    if tool_name == "displacement_category":
        category = str(value) if value is not None else "unknown"
        dist = float(details.get("distance", 0.0))
        if category == "stationary":
            ans = (
                f"The overall motion is {category} "
                f"(total displacement ≈ {dist:.2f} units)."
            )
        else:
            ans = (
                f"The person exhibits {category} movement overall "
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
    model: str,
) -> str:
    """
    LLM-based answerer.

    Uses an LLM to turn (question + tool_name + raw_answer) into a concise,
    natural-language answer.

    If anything goes wrong (no client, API error, etc.), we fall back to
    the rule-based `format_answer`.
    """
    raw_answer = raw_answer or {}

    # If OpenAI is not available, or we don't want LLM, fall back immediately.
    if OpenAI is None:
        print("[answerer_llm] openai package not installed; falling back to rule-based answer.")
        return format_answer(question, tool_name, raw_answer)

    try:
        client = OpenAI()  # reads API key from environment
    except Exception as e:
        print(f"[answerer_llm] Error creating OpenAI client: {e}")
        return format_answer(question, tool_name, raw_answer)

    system_msg = (
        "You are a helpful assistant that explains the results of motion-analysis tools.\n"
        "You are given a user's question about a human motion clip and the numeric\n"
        "output of a specific tool (e.g., dominant_direction, count_sit_events, etc.).\n"
        "Respond with a concise, clear answer (1-3 sentences) that directly addresses\n"
        "the user's question using the tool's output. Do not mention the tool name\n"
        "unless it is necessary."
    )

    tool_json = json.dumps(
        {"tool_name": tool_name, "tool_output": raw_answer},
        indent=2,
        default=str,
    )

    user_msg = (
        f"User question:\n{question}\n\n"
        "Tool analysis output (JSON):\n"
        f"{tool_json}\n\n"
        "Using this information, answer the user's question clearly."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        text = response.choices[0].message.content.strip()
        return text

    except Exception as e:
        print(f"[answerer_llm] Error calling LLM: {e}")
        return format_answer(question, tool_name, raw_answer)
