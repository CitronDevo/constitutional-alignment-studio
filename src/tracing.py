"""
Tracing utilities for capturing LLM inputs/outputs for observability.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple
import time

import streamlit as st


@dataclass
class TraceLog:
    operation_name: str
    model_name: str
    system_prompt: str
    user_prompt: str
    raw_output: str
    timestamp: str
    latency_ms: Optional[float] = None


def _log_trace(trace: TraceLog) -> None:
    """Console log of full prompts and output for debugging."""
    sep = "-" * 60
    print(sep)
    print(f"[TRACE] {trace.operation_name} | model={trace.model_name} | ts={trace.timestamp} | latency={trace.latency_ms} ms")
    print("[SYSTEM PROMPT]")
    print(trace.system_prompt)
    print("[USER PROMPT]")
    print(trace.user_prompt)
    print("[RAW OUTPUT]")
    print(trace.raw_output)
    print(sep)


class TraceManager:
    """
    Small helper to capture and persist the last LLM call for inspection.
    """

    def __init__(self, state_key: str = "latest_trace"):
        self.state_key = state_key
        if self.state_key not in st.session_state:
            st.session_state[self.state_key] = None

    def _build_trace(
        self,
        operation_name: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        raw_output: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> TraceLog:
        latency_ms = None
        if start_time is not None and end_time is not None:
            latency_ms = (end_time - start_time) * 1000

        return TraceLog(
            operation_name=operation_name,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_output=raw_output,
            timestamp=datetime.utcnow().isoformat(),
            latency_ms=latency_ms,
        )

    def capture(self, provider, system_prompt: str, user_prompt: str, operation_name: str) -> Tuple[str, TraceLog]:
        """
        Wrap provider.generate to capture prompts and output.
        """
        model_name = getattr(getattr(provider, "config", None), "model", "unknown")
        start = time.perf_counter()
        output = provider.generate(system_prompt, user_prompt)
        end = time.perf_counter()

        trace = self._build_trace(
            operation_name=operation_name,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_output=output,
            start_time=start,
            end_time=end,
        )
        st.session_state[self.state_key] = trace
        _log_trace(trace)
        return output, trace

    def save_trace(
        self,
        operation_name: str,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        raw_output: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> None:
        """
        Manually save a trace (useful for streaming flows).
        """
        trace = self._build_trace(
            operation_name=operation_name,
            model_name=model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            raw_output=raw_output,
            start_time=start_time,
            end_time=end_time,
        )
        st.session_state[self.state_key] = trace
        _log_trace(trace)
