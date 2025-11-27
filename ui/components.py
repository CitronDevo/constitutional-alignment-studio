"""
Reusable Streamlit UI components for the Constitutional AI Gym.
"""

import streamlit as st

from src.validator import validate_adaptive_constraint, ValidationResult
from src.tracing import TraceLog


def render_acrostic_validator(text: str, compact: bool = False) -> ValidationResult:
    """
    Visualize the ADAPTIVE validator alongside the rewrite text.

    Returns the ValidationResult so callers can gate actions (e.g., Save).
    """
    # For display, keep original text but validator will clean scaffolded prefixes
    result = validate_adaptive_constraint(text or "")
    target = "ADAPTIVE"

    status_color = "green" if result.is_valid else "red"
    pad = "6px" if compact else "8px"
    st.markdown(
        f"<div style='padding:{pad};border-radius:6px;background:{status_color};color:white;font-size:14px;'>"
        f"{result.feedback}"
        "</div>",
        unsafe_allow_html=True,
    )

    # Build rows for each expected letter (always show 8 rows for clarity)
    rows = []
    breakdown = result.breakdown
    for idx in range(len(target)):
        expected_char = target[idx]
        actual_char = breakdown[idx][1] if idx < len(breakdown) else ""
        sentence = breakdown[idx][2] if idx < len(breakdown) else ""
        is_match = expected_char == actual_char
        rows.append((expected_char, sentence, is_match))

    font_size = "16px" if compact else "20px"
    cell_pad = "4px" if compact else "6px"
    for expected_char, sentence, is_match in rows:
        left, right = st.columns([1, 10])
        bg = "#16a34a" if is_match else "#dc2626"
        left.markdown(
            f"<div style='text-align:center;font-weight:700;font-size:{font_size};"
            f"padding:{cell_pad};border-radius:6px;background:{bg};color:white;'>"
            f"{expected_char}"
            "</div>",
            unsafe_allow_html=True,
        )
        right.markdown(sentence if sentence else "*No sentence provided*")

    return result


def render_chat_message(role: str, text: str) -> None:
    """Simple wrapper around st.chat_message."""
    st.chat_message(role).write(text)


def render_trace_viewer(state_key: str = "latest_trace", trace_entry: TraceLog | None = None) -> None:
    """
    Render a popover with the latest captured trace (if any).
    """
    trace: TraceLog | None = trace_entry or st.session_state.get(state_key)
    if not trace:
        return

    label = f"🔍 Trace: {trace.operation_name}" if trace_entry else "🔍 Inspect Trace"
    with st.popover(label, use_container_width=True):
        tabs = st.tabs(["System Prompt", "User Prompt", "Raw Output", "Metadata"])
        with tabs[0]:
            st.code(trace.system_prompt or "", language="text")
        with tabs[1]:
            st.code(trace.user_prompt or "", language="text")
        with tabs[2]:
            st.code(trace.raw_output or "", language="text")
        with tabs[3]:
            st.write(f"**Operation:** {trace.operation_name}")
            st.write(f"**Model:** {trace.model_name}")
            st.write(f"**Timestamp:** {trace.timestamp}")
            if trace.latency_ms is not None:
                st.write(f"**Latency:** {trace.latency_ms:.0f} ms")
