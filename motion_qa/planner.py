# motion_qa/planner.py

from __future__ import annotations

from typing import List, Dict, Any


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
    Heuristic (non-LLM) planner for dance QA.

    Given a natural language question and a set of available tool names,
    choose which dance-analysis tool to call and with which parameters.
    """
    q = (question or "").lower()

    def choose(name: str, params: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
        return _choose_tool(tools, name, params)

    # 1) Dance style / genre classification
    if any(word in q for word in [
        "what style", "what genre", "what dance", "what type of dance",
        "which dance", "identify the dance", "classify",
        "breaking", "popping", "locking", "waacking", "krump",
        "hip hop or house", "is this hip hop", "is this house",
        "street jazz", "ballet jazz",
    ]):
        plan = choose("classify_dance_style")
        if plan:
            return plan

    # 2) Freeze / hold / stillness
    if any(word in q for word in [
        "freeze", "still", "hold", "stop moving", "pose hold", "stationary pose",
    ]):
        plan = choose("detect_freeze")
        if plan:
            return plan

    # 2) Jacking / house groove / bounce
    if any(word in q for word in [
        "jacking", "jack", "house groove", "body bounce", "bounce", "grove",
    ]):
        plan = choose("detect_jacking")
        if plan:
            return plan

    # 3) Rhythm / beat / tempo / regularity
    if any(word in q for word in [
        "rhythmic", "on beat", "regular", "rhythm", "consistent beat",
        "tempo", "in time", "off beat", "timing",
    ]):
        plan = choose("compute_rhythm_regularity")
        if plan:
            return plan

    # 4) Active limb / body part
    if any(word in q for word in [
        "arm", "hand", "leg", "foot", "feet", "limb",
        "upper body", "lower body", "most active", "leading",
    ]):
        plan = choose("most_active_limb")
        if plan:
            return plan

    # 6) Traveling / floor coverage / displacement
    if any(word in q for word in [
        "how far", "distance", "travel", "displacement", "cover",
        "across the floor", "floor space", "stage space",
    ]):
        plan = choose("global_displacement")
        if plan:
            return plan

    # 7) In-place vs. traveling
    if any(phrase in q for phrase in [
        "stay in place", "in place", "on the spot", "move a lot",
        "traveling", "stationary", "dynamic",
    ]):
        plan = choose("displacement_category")
        if plan:
            return plan

    # 8) Direction (stage direction, facing)
    if any(word in q for word in [
        "forward", "backward", "sideways", "left", "right",
        "direction", "facing", "stage left", "stage right", "upstage", "downstage",
    ]):
        plan = choose("dominant_direction")
        if plan:
            return plan

    # 9) Duration / clip length
    if any(word in q for word in [
        "how long", "duration", "seconds", "length", "time", "how many seconds",
    ]):
        plan = choose("clip_duration", params={"fps": 30.0})
        if plan:
            return plan

    # Fallback ordering (dance tools first)
    for candidate in [
        "classify_dance_style",
        "detect_freeze",
        "compute_rhythm_regularity",
        "detect_jacking",
        "most_active_limb",
        "displacement_category",
        "dominant_direction",
        "global_displacement",
        "clip_duration",
    ]:
        plan = choose(candidate)
        if plan:
            return plan

    if tools:
        return {"tool": tools[0], "params": {}}

    return {"tool": "unknown", "params": {}}


def plan_from_question_llm(
    question: str,
    tools: List[str],
) -> Dict[str, Any]:
    """
    LLM-based planner using a local HuggingFace model (Phi-3-mini).
    Falls back to the heuristic planner on any error.
    """
    try:
        from motion_qa.hf_llm import generate
        import json as _json
    except ImportError as e:
        print(f"[planner_llm] hf_llm unavailable ({e}); using heuristic fallback.")
        return plan_from_question(question, tools)

    if not tools:
        return plan_from_question(question, tools)

    system_prompt = (
        "You are a planner for a dance movement analysis system.\n"
        "You are given a user's question about a dance clip and a list of\n"
        "available analysis tools. Select exactly one tool to call.\n"
        "Respond ONLY with a JSON object of the form:\n"
        '{"tool": "<one of the tool names>", "params": {}}\n'
        "No extra text, no explanation, no markdown."
    )
    user_prompt = (
        f"User question about a dance clip:\n{question}\n\n"
        f"Available tools: {', '.join(tools)}\n\n"
        "Respond ONLY with a JSON object."
    )

    try:
        text = generate(system_prompt, user_prompt, max_new_tokens=64, temperature=0.0)
        plan = _json.loads(text)
        if not isinstance(plan, dict) or "tool" not in plan:
            raise ValueError("invalid plan structure")
        if plan["tool"] not in tools:
            print(f"[planner_llm] LLM chose unknown tool '{plan['tool']}'; using heuristic.")
            return plan_from_question(question, tools)
        plan.setdefault("params", {})
        return plan
    except Exception as e:
        print(f"[planner_llm] LLM error ({e}); using heuristic fallback.")
        return plan_from_question(question, tools)
