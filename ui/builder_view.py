"""
Builder (Workbench) tab for drafting constitutional examples.

Refactored to separate Selection (Sidebar) from Workspace (Main Area).
"""

from datetime import datetime
import time
from typing import List, Tuple

import streamlit as st

from src.llm_client import get_llm_provider
from src.models import ConstitutionalExample, ConversationTurn, LLMConfig
from src.prompt_engine import StepPromptBuilder
from src.repository import TaskRepository
from src.validator import validate_adaptive_constraint
from src.tracing import TraceManager
from ui.components import render_acrostic_validator, render_trace_viewer


def _available_tasks(repository: TaskRepository) -> List[Tuple[int, ConversationTurn]]:
    """Return list of (task_id, task) for tasks not yet in the training pool."""
    trained_ids = {ex.source_id for ex in repository.get_training_pool()}
    return [
        (idx, task)
        for idx, task in enumerate(repository.tasks)
        if idx not in trained_ids
    ]


def _find_saved_example(repository: TaskRepository, task_id: int) -> ConstitutionalExample | None:
    """Find a saved example by task_id."""
    for ex in repository.get_training_pool():
        if ex.source_id == task_id:
            return ex
    return None


def render_sidebar(repository: TaskRepository, untrained: List[Tuple[int, ConversationTurn]], config_renderer) -> LLMConfig:
    """Render the sidebar with task selection and inventory."""
    training_pool = repository.get_training_pool()
    all_examples = repository.get_all_examples() if hasattr(repository, "get_all_examples") else list(repository.examples)

    with st.sidebar:
        cfg = config_renderer()
        st.session_state["llm_config"] = cfg

        # Section A: New Conversations with explicit button
        st.markdown("**Conversations**")
        new_task_options = {
            f"Conversation #{task_id}: {task.user[:60]}": task_id
            for task_id, task in untrained
        }

        if new_task_options:
            # Store the selected task in a temporary state key (not active_task_id yet)
            if "sidebar_selected_new_task" not in st.session_state:
                st.session_state["sidebar_selected_new_task"] = list(new_task_options.values())[0]

            labels = list(new_task_options.keys())

            # Find current index
            try:
                current_val = st.session_state["sidebar_selected_new_task"]
                curr_label = next((lbl for lbl, tid in new_task_options.items() if tid == current_val), labels[0])
                curr_index = labels.index(curr_label)
            except (KeyError, StopIteration, ValueError):
                curr_index = 0

            new_selection = st.selectbox(
                "Choose a conversation",
                options=labels,
                index=curr_index,
                key="builder_new_select",
                label_visibility="collapsed",
            )

            # Update the temporary selection state
            st.session_state["sidebar_selected_new_task"] = new_task_options[new_selection]

            # Button to start draft - only this triggers the actual mode switch
            if st.button("✨ Start Draft", type="primary", use_container_width=True):
                selected_id = new_task_options[new_selection]
                st.session_state["active_task_id"] = selected_id
                st.session_state["task_mode"] = "drafting"
                # Clear any prefill from previous edits
                st.session_state.pop("builder_prefill_critique", None)
                st.session_state.pop("builder_prefill_rewrite", None)
                st.toast(f"Started draft for Conversation #{selected_id}")
                st.rerun()
        else:
            st.caption("All tasks drafted.")

        # Section B: Curated Examples
        st.markdown(f"**📚 {len(training_pool)}/20 Curated Examples**")
        saved_examples = all_examples
        if saved_examples:
            for ex in saved_examples:
                cols = st.columns([0.05, 0.25, 0.15, 0.15, 0.15])
                with cols[0]:
                    active = st.checkbox(
                        label=f"Activate example {ex.source_id}",
                        value=getattr(ex, "active", True),
                        key=f"active_{ex.source_id}",
                        help="Toggle active/inactive",
                        label_visibility="collapsed",
                    )
                with cols[1]:
                    st.markdown(f"**Example #{ex.source_id}**")
                with cols[2]:
                    with st.popover("👁️", use_container_width=True):
                        st.markdown(f"**Example #{ex.source_id}**")
                        st.markdown(f"**User Prompt**: {ex.user_prompt}")
                        st.markdown(f"**Original Response**: {ex.original_response}")
                        st.markdown(f"**Critique**: {ex.critique}")
                        st.markdown(f"**Rewrite**: {ex.rewrite}")
                with cols[3]:
                    if st.button("✏️", key=f"edit_{ex.source_id}", help="Edit this example"):
                        # Set active task and mode
                        st.session_state["active_task_id"] = ex.source_id
                        st.session_state["task_mode"] = "editing"
                        # Prefill the form with saved data
                        st.session_state["builder_prefill_critique"] = ex.critique
                        st.session_state["builder_prefill_rewrite"] = ex.rewrite
                        # Clear any stale form state
                        st.session_state.pop(f"builder_task_{ex.source_id}_critique", None)
                        st.session_state.pop(f"builder_task_{ex.source_id}_rewrite", None)
                        st.toast(f"Editing Task #{ex.source_id}")
                        st.rerun()
                with cols[4]:
                    if st.button("🗑️", key=f"del_{ex.source_id}", help="Delete example"):
                        repository.delete_example(ex.source_id)
                        st.toast(f"Deleted example #{ex.source_id}")
                        st.rerun()

                # Apply active toggle changes
                if active != getattr(ex, "active", True):
                    repository.toggle_example(ex.source_id, active)
                    st.rerun()
        else:
            st.caption("No saved examples yet.")

        with st.expander("💾 Save / Load Experiment", expanded=False):
            st.caption("Export golden examples or restore from a previous set (overwrites current).")
            st.download_button(
                "📥 Download Golden Examples",
                data=repository.get_dataset_as_json(),
                file_name="golden_examples.json",
                mime="application/json",
            )
            uploaded = st.file_uploader("📤 Restore Golden Examples (JSON or JSONL)", type=["json", "jsonl"], key="curator_import")
            if uploaded is not None:
                try:
                    content = uploaded.getvalue().decode("utf-8")
                    repository.import_dataset(content)
                    st.success("Golden examples restored. Reloading...")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to restore golden examples: {exc}")

    return cfg


def render_empty_state(completed_count: int, total_count: int) -> None:
    """Render an empty state when no conversation is selected."""
    st.markdown("### 👈 Select a conversation from the sidebar to begin")

def render_workspace_header(task_id: int, mode: str) -> None:
    """Render the workspace header with close button."""
    col_header, col_button = st.columns([0.8, 0.2])

    with col_header:
        if mode == "editing":
            st.subheader(f"✏️ Editing: Conversation #{task_id}")
        else:
            st.subheader(f"✨ New Draft: Conversation #{task_id}")

    with col_button:
        close_label = "❌ Close Editor" if mode == "editing" else "❌ Cancel Draft"
        if st.button(close_label, key="close_workspace"):
            st.session_state["active_task_id"] = None
            st.session_state["task_mode"] = None
            st.session_state.pop("builder_prefill_critique", None)
            st.session_state.pop("builder_prefill_rewrite", None)
            st.toast("Closed workspace")
            st.rerun()


def render_workspace(
    repository: TaskRepository,
    config: LLMConfig,
    trace_manager: TraceManager,
    task_id: int,
    mode: str
) -> None:
    """Render the main workspace area for drafting/editing."""
    task = repository.tasks[task_id]
    saved_example = _find_saved_example(repository, task_id) if mode == "editing" else None
    training_pool = repository.get_training_pool()

    # Version History
    with st.expander("📜 Version History", expanded=False):
        history = []
        try:
            history = repository.get_history(task_id)
        except Exception:
            history = []

        if not history:
            st.caption("No history yet.")
        else:
            labels = [
                f"{(ex.timestamp or '')} - {ex.rewrite[:30]}..."
                for ex in history
            ]
            sel = st.selectbox("Select Version", options=range(len(history)), format_func=lambda i: labels[i])
            chosen = history[sel]
            st.markdown("**Critique**")
            st.write(chosen.critique)
            st.markdown("**Rewrite**")
            st.write(chosen.rewrite)
            if st.button("⏪ Restore this Version"):
                st.session_state["builder_prefill_critique"] = chosen.critique
                st.session_state["builder_prefill_rewrite"] = chosen.rewrite
                st.toast("Version restored. You can now save to curator.")
                st.rerun()

    # Problem Statement
    st.markdown("**User Prompt**")
    st.write(task.user)
    st.markdown("**Original (Bad) Response**")
    st.write(task.assistant)

    st.divider()

    # Drafting Section
    st.subheader("Drafting")

    state_prefix = f"builder_task_{task_id}"
    critique_key = f"{state_prefix}_critique"
    rewrite_key = f"{state_prefix}_rewrite"

    stream_placeholder = st.empty()

    # Auto-Draft Button
    if st.button(f"Auto-Draft with {config.model}", type="primary"):
        try:
            provider = get_llm_provider(config)
            step_builder = StepPromptBuilder()

            with stream_placeholder.container():
                st.info("Critiquing...")
                current_response = task.assistant or ""
                validation = validate_adaptive_constraint(current_response)
                critique_prompt = step_builder.render_critique_prompt(
                    task,
                    current_response=current_response,
                    validation_result=validation,
                )
                critique_placeholder = st.empty()
                critique_text = ""
                crit_start = time.perf_counter()
                for token in provider.stream_generate(system_prompt="", user_prompt=critique_prompt):
                    critique_text += token
                    critique_placeholder.markdown(critique_text)
                crit_end = time.perf_counter()
                trace_manager.save_trace(
                    operation_name="Streaming Critique",
                    model_name=config.model,
                    system_prompt="",
                    user_prompt=critique_prompt,
                    raw_output=critique_text,
                    start_time=crit_start,
                    end_time=crit_end,
                )

                st.info("Rewriting...")
                revision_prompt = step_builder.render_revision_prompt(
                    task,
                    current_response=current_response,
                    critique=critique_text,
                )
                rewrite_placeholder = st.empty()
                rewrite_text = ""
                rev_start = time.perf_counter()
                for token in provider.stream_generate(system_prompt="", user_prompt=revision_prompt):
                    rewrite_text += token
                    rewrite_placeholder.markdown(rewrite_text)
                rev_end = time.perf_counter()
                trace_manager.save_trace(
                    operation_name="Streaming Rewrite",
                    model_name=config.model,
                    system_prompt="",
                    user_prompt=revision_prompt,
                    raw_output=rewrite_text,
                    start_time=rev_start,
                    end_time=rev_end,
                )

            st.session_state[critique_key] = critique_text
            st.session_state[rewrite_key] = rewrite_text
            stream_placeholder.empty()
            st.toast("Streaming draft completed")
        except Exception as exc:
            stream_placeholder.empty()
            st.error(f"Auto-draft failed: {exc}")

    render_trace_viewer()

    # Get default values
    default_critique = st.session_state.get("builder_prefill_critique") or st.session_state.get(critique_key, "")
    default_rewrite = st.session_state.get("builder_prefill_rewrite") or st.session_state.get(rewrite_key, "")

    # Editing Form
    with st.form(key=f"builder_form_{task_id}", clear_on_submit=False):
        critique_text = st.text_area(
            "Critique",
            value=default_critique,
            height=160,
        )
        rewrite_text = st.text_area(
            "Rewrite (must spell ADAPTIVE)",
            value=default_rewrite,
            height=200,
        )
        submitted = st.form_submit_button("✅ Validate (Cmd/Ctrl+Enter)")
        if submitted:
            st.session_state[critique_key] = critique_text
            st.session_state[rewrite_key] = rewrite_text
            st.session_state["builder_prefill_critique"] = critique_text
            st.session_state["builder_prefill_rewrite"] = rewrite_text
            st.success("Validated.")

    # Revision History
    rev_key = f"builder_revisions_{task_id}"
    if rev_key not in st.session_state:
        st.session_state[rev_key] = []
        if saved_example:
            st.session_state[rev_key].append(
                {
                    "critique": saved_example.critique,
                    "rewrite": saved_example.rewrite,
                    "validation": validate_adaptive_constraint(saved_example.rewrite),
                    "label": "Saved version",
                }
            )
    revisions = st.session_state[rev_key]

    # Validation Display
    validation = None
    if rewrite_text.strip():
        st.subheader("Validation")
        validation = render_acrostic_validator(rewrite_text, compact=True)
    else:
        st.caption("")

    # Action Buttons
    st.subheader("Actions")
    col_loop, col_save = st.columns(2)

    with col_loop:
        if st.button("♻️ Critique & Revise Again"):
            try:
                provider = get_llm_provider(config)
                step_builder = StepPromptBuilder()
                val = validate_adaptive_constraint(rewrite_text)
                validation_msg = val.feedback

                crit_prompt = step_builder.render_critique_prompt(
                    task,
                    current_response=rewrite_text,
                    validation_result=val
                )
                new_critique, crit_trace = trace_manager.capture(
                    provider=provider,
                    system_prompt="",
                    user_prompt=crit_prompt,
                    operation_name="Loop Critique",
                )

                rev_prompt = step_builder.render_revision_prompt(
                    task,
                    current_response=rewrite_text,
                    critique=new_critique + f"\nValidator: {validation_msg}"
                )
                new_rewrite, rev_trace = trace_manager.capture(
                    provider=provider,
                    system_prompt="",
                    user_prompt=rev_prompt,
                    operation_name="Loop Revision",
                )

                new_val = validate_adaptive_constraint(new_rewrite)
                st.session_state[critique_key] = new_critique
                st.session_state[rewrite_key] = new_rewrite
                st.session_state["builder_prefill_critique"] = new_critique
                st.session_state["builder_prefill_rewrite"] = new_rewrite
                st.session_state[rev_key].append(
                    {
                        "critique": new_critique,
                        "rewrite": new_rewrite,
                        "validation": new_val,
                        "label": f"Loop #{len(st.session_state[rev_key]) + 1}",
                        "crit_trace": crit_trace,
                        "rev_trace": rev_trace,
                    }
                )
                st.toast("Looped new revision")
                st.rerun()
            except Exception as exc:
                st.error(f"Loop failed: {exc}")

    with col_save:
        save_disabled = (
            (validation is None or not validation.is_valid)
            or not critique_text.strip()
            or not rewrite_text.strip()
        )
        btn_label = "💾 Update Example" if saved_example else "💾 Add to Curator"
        if st.button(btn_label, type="primary", disabled=save_disabled):
            try:
                example = ConstitutionalExample(
                    principle="ADAPTIVE",
                    user_prompt=task.user,
                    original_response=task.assistant,
                    critique=critique_text.strip(),
                    rewrite=rewrite_text.strip(),
                    source_id=task_id,
                    model_used=config.model,
                    timestamp=datetime.utcnow().isoformat(),
                    active=True,
                )
                repository.upsert_example(example)
                st.toast("Saved!")
                st.session_state["builder_prefill_critique"] = critique_text.strip()
                st.session_state["builder_prefill_rewrite"] = rewrite_text.strip()
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to save example: {exc}")

    # History Display
    st.subheader("History")
    if revisions:
        for idx, rev in enumerate(revisions):
            with st.expander(f"Step {idx + 1}: {rev.get('label','Draft')}", expanded=False):
                st.markdown("**Critique**")
                st.write(rev["critique"])
                st.markdown("**Rewrite**")
                st.write(rev["rewrite"])
                if rev.get("validation"):
                    st.markdown("**Validation**")
                    st.write(rev["validation"].feedback)
                if rev.get("crit_trace"):
                    render_trace_viewer(trace_entry=rev["crit_trace"])
                if rev.get("rev_trace"):
                    render_trace_viewer(trace_entry=rev["rev_trace"])


def render(repository: TaskRepository, config_renderer, trace_manager: TraceManager) -> None:
    """Main render function for the Builder view."""
    st.header("Example Curation Studio")
    with st.expander("ℹ️ About this step", expanded=False):
        st.info(
            "Create examples to steer the teacher model during the critique+rewrite step.   \n"
            "The teacher model needs guidance to better adhere to the given principle.  \n"
            "Use this workspace to create and vet critique+rewrite pairs where the assistant's answer successfully spells ADAPTIVE.  \n"
            "These vetted examples provide the necessary context to steer the teacher model's outputs on future input  \n"
        )

    # Initialize session state for active task and mode
    if "active_task_id" not in st.session_state:
        st.session_state["active_task_id"] = None
    if "task_mode" not in st.session_state:
        st.session_state["task_mode"] = None

    # Get available tasks
    training_pool = repository.get_training_pool()
    untrained = _available_tasks(repository)

    # Render sidebar (selection area) and get LLM config
    config = render_sidebar(repository, untrained, config_renderer)

    # Store config in session state for tester view
    st.session_state["llm_config"] = config

    # Render main area (workspace)
    active_task_id = st.session_state.get("active_task_id")
    task_mode = st.session_state.get("task_mode")

    if active_task_id is None:
        # Empty state
        render_empty_state(len(training_pool), len(repository.tasks))
    else:
        # Active workspace
        render_workspace_header(active_task_id, task_mode)
        render_workspace(repository, config, trace_manager, active_task_id, task_mode)
