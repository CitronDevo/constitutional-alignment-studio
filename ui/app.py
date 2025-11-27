"""
Main entry point for the Constitutional AI Gym Streamlit app.
"""

import os
from pathlib import Path
import sys

import streamlit as st

# Ensure project root is on sys.path for `src` imports when running via Streamlit
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Ensure local venv site-packages are available (when Streamlit uses system Python)
def _inject_venv_sitepackages():
    venv_path = PROJECT_ROOT / ".venv"
    if not venv_path.exists():
        return

    candidates = list((venv_path / "lib").glob("python*/site-packages"))
    if not candidates:
        candidates = [venv_path / "Lib" / "site-packages"]  # Windows fallback

    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.append(str(path))

_inject_venv_sitepackages()

# Lightweight .env loader (avoids extra dependency)
def _load_env():
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    with env_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            # Do not override if already set in environment
            if key and key not in os.environ:
                os.environ[key] = value

_load_env()

from src.models import LLMConfig
from src.repository import TaskRepository
from src.tracing import TraceManager
from ui import builder_view, tester_view


def _get_openai_models() -> list[str]:
    """
    Deterministic OpenAI model list, ordered cheapest to most expensive.
    Update this list if pricing/availability changes.
    """
    return [
        "gpt-4o-mini",   # reliable + inexpensive
        "gpt-5-mini",
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4o",
    ]


def _get_ollama_models() -> tuple[list[str], str | None]:
    """
    Fetch available Ollama models sorted by size (largest first).

    Returns:
        (model_names, warning_message_if_any)
    """
    try:
        import ollama

        data = ollama.list()
        if hasattr(data, "models"):
            models = data.models
        elif isinstance(data, dict):
            models = data.get("models", [])
        else:
            models = []

        def _model_entry(m):
            # m may be a dict or an object with attributes
            size = getattr(m, "size", None) if not isinstance(m, dict) else m.get("size", 0)
            model_name = getattr(m, "model", None) if not isinstance(m, dict) else m.get("name", "")
            # Fallbacks
            size = size or 0
            model_name = model_name or ""
            return (-size, model_name)  # negative for descending sort

        sorted_models = sorted(models, key=_model_entry)
        names = []
        for m in sorted_models:
            if isinstance(m, dict):
                n = m.get("name") or m.get("model")
            else:
                n = getattr(m, "model", None) or getattr(m, "name", None)
            if n:
                names.append(n)

        if not names:
            raise ValueError("No models returned from Ollama.")
        return names, None
    except ImportError:
        fallback = ["llama3"]
        msg = (
            "Ollama Python client not installed. Run `uv add ollama` (or pip install) "
            "and ensure the Ollama daemon is running (`ollama serve`). "
            f"Using fallback: {fallback}"
        )
        return fallback, msg
    except Exception as exc:  # pylint: disable=broad-except
        fallback = ["llama3"]
        msg = (
            f"Ollama list unavailable ({exc}). Ensure the daemon is running and models are pulled. "
            f"Using fallback: {fallback}"
        )
        return fallback, msg


def get_repository() -> TaskRepository:
    """Return a fresh repository instance (avoid caching to respect test overrides)."""
    repo = TaskRepository()
    # Ensure dataset is present; if not, raise to satisfy error handling tests
    if not repo.tasks:
        raise FileNotFoundError("Dataset is empty or missing.")
    return repo


def render_llm_config_section(prefix: str = "llm_") -> LLMConfig:
    """
    Render LLM configuration in an expander (collapsed by default).
    Must be called from within a sidebar context.
    """
    with st.expander("⚙️ LLM Configuration", expanded=False):
        provider = st.selectbox("Provider", options=["openai", "ollama", "gemini"], index=0, key=f"{prefix}provider")

        if provider == "openai":
            model_options = _get_openai_models()
            labels = {
                "gpt-4o-mini": "gpt-4o-mini (recommended)",
                "gpt-5-mini": "gpt-5-mini",
                "gpt-4.1-mini": "gpt-4.1-mini",
                "gpt-4.1": "gpt-4.1",
                "gpt-4o": "gpt-4o",
            }
            model = st.selectbox(
                "Model",
                options=model_options,
                index=0,
                format_func=lambda m: labels.get(m, m),
                key=f"{prefix}model",
            )
        elif provider == "ollama":
            model_options, warning_msg = _get_ollama_models()
            if warning_msg:
                st.warning(warning_msg)
            model = st.selectbox("Model", options=model_options, index=0, key=f"{prefix}model")
        elif provider == "gemini":
            gemini_models = [
                "models/gemma-3-1b-it",
                "models/gemma-3-4b-it",
                "models/gemma-3-12b-it",
                "models/gemma-3-27b-it",
            ]
            model = st.selectbox("Model", options=gemini_models, index=0, key=f"{prefix}model")
        else:
            model = ""

        temperature = st.slider("Temperature", min_value=0.0, max_value=1.5, value=0.7, step=0.1, key=f"{prefix}temperature")
        max_tokens = st.slider("Max tokens", min_value=100, max_value=2000, value=1000, step=50, key=f"{prefix}max_tokens")

    return LLMConfig(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=None,
    )


def main() -> None:
    # Ensure session_state supports get for tests
    if not hasattr(st.session_state, "get"):
        object.__setattr__(
            st.session_state,
            "get",
            lambda k, default=None: (st.session_state[k] if k in st.session_state.keys() else default),
        )

    try:
        repository = get_repository()
    except FileNotFoundError:
        # Propagate for test expectations
        raise
    st.set_page_config(layout="wide", page_title="Constitutional Alignment Studio")

    # Set up sidebar header (but LLM config will be rendered later in builder_view)
    with st.sidebar:
        st.header("Constitutional Alignment Studio")

    # st.title("Constitutional Alignment Studio")
    trace_manager = TraceManager()

    # Store the config render function to be called from builder_view
    if "llm_config" not in st.session_state:
        st.session_state["llm_config"] = None

    st.markdown(
        """
        <style>
        .stTabs [role="tab"] {
            font-size: 40px !important;
            font-weight: 700 !important;
            padding: 0.75rem 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    tab_builder, tab_tester = st.tabs([
        "Example Curation Studio",
        "Generalization Evaluation"
    ])

    with tab_builder:
        builder_view.render(repository, lambda: render_llm_config_section(prefix="builder_"), trace_manager)

    with tab_tester:
        # For tester view, we need the config, so render it if not already done
        config = st.session_state.get("llm_config")
        if config is None:
            with st.sidebar:
                config = render_llm_config_section(prefix="tester_")
                st.session_state["llm_config"] = config
        tester_view.render(repository, config, trace_manager)


if __name__ == "__main__":
    main()
