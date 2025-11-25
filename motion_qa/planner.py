# motion_qa/planner.py

from __future__ import annotations

from typing import List, Dict, Any
import json

try:
    from openai import OpenAI  # Optional; we fall back gracefully if missing
except ImportError:
    OpenAI = None  # type: ignore


def _choose_tool(
    tools: List[str],
    preferred: str,
    params: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    """
    Helper to build a plan if `preferred` is available in `tools`.
    """
    if preferred in tools:
        return {"tool": preferred, "params": params or {}}
    return None


def plan_from_question(question: str, tools: List[str]) -> Dict[str, Any]:
    """
    Heuristic (non-LLM) planner.

    Given a natural language question and a set of available tool names,
    choose which motion-analysis tool to call and with which parameters.

    tools is a list of strings, e.g.:
      ["dominant_direction", "count_sit_events", "global_displacement", ...]
    """
    q = (question or "").lower()

    # Quick helper for this function
    def choose(name: str, params: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
        return _choose_tool(tools, name, params)

    # 1) Sit / squat related
    if any(word in q for word in ["sit", "squat", "chair", "sitting", "stand up"]):
        plan = choose("count_sit_events")
        if plan:
            return plan

    # 2) Distance / how far they travel
    if any(word in q for word in ["how far", "distance", "travel", "displacement"]):
        plan = choose("global_displacement")
        if plan:
            return plan

    # 3) How much movement overall
    if any(
        phrase in q
        for phrase in [
            "stay in place",
            "move a lot",
            "movement amount",
            "mostly stationary",
            "very dynamic",
        ]
    ):
        plan = choose("displacement_category")
        if plan:
            return plan

    # 4) Duration / time / seconds
    if any(
        word in q
        for word in ["how long", "duration", "seconds", "length of clip", "time"]
    ):
        plan = choose("clip_duration", params={"fps": 30.0})
        if plan:
            return plan

    # 5) Limb / body part activity
    if any(word in q for word in ["arm", "hand", "leg", "foot", "limb"]):
        plan = choose("most_active_limb")
        if plan:
            return plan

    # 6) Direction of movement (forward/back/left/right/sideways)
    if any(
        word in q
        for word in [
            "forward",
            "backward",
            "sideways",
            "left",
            "right",
            "direction",
            "move towards",
        ]
    ):
        plan = choose("dominant_direction")
        if plan:
            return plan

    # 7) Fallback ordering preference
    for candidate in [
        "dominant_direction",
        "global_displacement",
        "displacement_category",
        "clip_duration",
        "count_sit_events",
        "most_active_limb",
    ]:
        plan = choose(candidate)
        if plan:
            return plan

    # 8) Absolute fallback: just pick the first tool if nothing else matched
    if tools:
        return {"tool": tools[0], "params": {}}

    # Degenerate case: no tools at all
    return {"tool": "unknown", "params": {}}


def plan_from_question_llm(
    question: str,
    tools: List[str],
    model: str,
) -> Dict[str, Any]:
    """
    LLM-based planner.

    Uses an LLM to choose one tool from the provided list and (optionally) params.

    If anything goes wrong (no client, API error, bad JSON, etc.) we fall back
    to the heuristic `plan_from_question`.
    """
    # If the OpenAI client is not available, immediately fall back
    if OpenAI is None:
        print("[planner_llm] openai package not installed; falling back to heuristic planner.")
        return plan_from_question(question, tools)

    if not tools:
        print("[planner_llm] No tools provided; falling back to heuristic planner.")
        return plan_from_question(question, tools)

    try:
        client = OpenAI()  # reads API key from environment
    except Exception as e:
        print(f"[planner_llm] Error creating OpenAI client: {e}")
        return plan_from_question(question, tools)

    # Build the prompt
    tools_str = ", ".join(tools)
    system_msg = (
        "You are a planner that selects exactly one motion-analysis tool to call.\n"
        "You are given a user's question about a human motion clip and a list of\n"
        "available tools. You must respond ONLY with a JSON object of the form:\n"
        '{\n  "tool": "<one of the tool names>",\n  "params": { ... }\n}\n'
        "No extra text, no explanations."
    )

    user_msg = (
        f"User question:\n{question}\n\n"
        f"Available tools (Python identifiers):\n{tools_str}\n\n"
        "Select the most appropriate single tool from this list, and specify any\n"
        "parameters it may need (or an empty object if none). Respond ONLY with\n"
        "a JSON object."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        # Try to parse the response as JSON
        plan = json.loads(text)
        if not isinstance(plan, dict) or "tool" not in plan:
            raise ValueError("LLM response is not a valid plan dict")

        # Ensure the tool is one of the available tools
        if plan["tool"] not in tools:
            print(f"[planner_llm] LLM selected unknown tool '{plan['tool']}', falling back.")
            return plan_from_question(question, tools)

        if "params" not in plan or not isinstance(plan["params"], dict):
            plan["params"] = {}

        return plan

    except Exception as e:
        print(f"[planner_llm] Error calling LLM or parsing response: {e}")
        return plan_from_question(question, tools)
