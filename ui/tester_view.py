"""
Generalization Evaluation tab for alignment checks.
"""

from typing import Dict, List, Tuple
import re
import json
import streamlit as st

from src.llm_client import get_llm_provider
from src.models import ConversationTurn, LLMConfig
from src.prompt_engine import PromptBuilder
from src.repository import TaskRepository
from src.validator import validate_adaptive_constraint
from src.tracing import TraceManager, TraceLog
from ui.components import render_acrostic_validator, render_trace_viewer


def _available_test_tasks(repository: TaskRepository) -> List[Tuple[int, ConversationTurn]]:
    """Return list of (task_id, task) eligible for testing (not in training pool)."""
    trained_ids = {ex.source_id for ex in repository.get_training_pool()}
    return [
        (idx, task)
        for idx, task in enumerate(repository.tasks)
        if idx not in trained_ids
    ]


def _parse_response(provider, raw: str) -> tuple[str | None, str]:
    """Extract critique and rewrite; fallback to None/raw."""
    try:
        parsed = provider.parse_critique_rewrite(raw)
        return parsed.critique, parsed.rewrite
    except Exception:
        return None, raw


def run_comparison(task: ConversationTurn, training_examples, config: LLMConfig, provider=None, prompt_builder=None, trace_manager: TraceManager | None = None) -> Dict:
    """
    Run zero-shot vs few-shot comparison for a single task.

    Returns dict with raw outputs, rewrites, and validation results.
    """
    provider = provider or get_llm_provider(config)
    prompt_builder = prompt_builder or PromptBuilder()

    zero_sys, zero_user = prompt_builder.render_full_prompt(task, examples=[])
    few_sys, few_user = prompt_builder.render_full_prompt(task, examples=training_examples)

    if trace_manager:
        zr = trace_manager.capture(
            provider=provider,
            system_prompt=zero_sys,
            user_prompt=zero_user,
            operation_name="Zero-shot Comparison",
        )
        fr = trace_manager.capture(
            provider=provider,
            system_prompt=few_sys,
            user_prompt=few_user,
            operation_name="Few-shot Comparison",
        )
        # Backward/defensive handling if capture ever returns only output
        zero_raw, zero_trace = zr if isinstance(zr, tuple) and len(zr) == 2 else (zr, None)
        few_raw, few_trace = fr if isinstance(fr, tuple) and len(fr) == 2 else (fr, None)
    else:
        zero_raw = provider.generate(zero_sys, zero_user)
        few_raw = provider.generate(few_sys, few_user)
        zero_trace = None
        few_trace = None

    # Debug logging for empty outputs
    print(f"[DEBUG] run_comparison task_id={task.user[:30]}... zero_raw_len={len(zero_raw) if zero_raw else 0} few_raw_len={len(few_raw) if few_raw else 0}")

    zero_critique, zero_rewrite = _parse_response(provider, zero_raw)
    few_critique, few_rewrite = _parse_response(provider, few_raw)

    zero_validation = validate_adaptive_constraint(zero_rewrite) if zero_rewrite else None
    few_validation = validate_adaptive_constraint(few_rewrite) if few_rewrite else None

    return {
        "zero_raw": zero_raw,
        "few_raw": few_raw,
        "zero_critique": zero_critique,
        "few_critique": few_critique,
        "zero_rewrite": zero_rewrite,
        "few_rewrite": few_rewrite,
        "zero_validation": zero_validation,
        "few_validation": few_validation,
        "zero_trace": zero_trace,
        "few_trace": few_trace,
    }


def _status_label(z_valid: bool, f_valid: bool) -> str:
    if not z_valid and f_valid:
        return "Improved ✅"
    if z_valid and f_valid:
        return "Pass → Pass"
    if z_valid and not f_valid:
        return "Regressed ❌"
    return "Fail → Fail"


def render(repository: TaskRepository, config: LLMConfig, trace_manager: TraceManager) -> None:
    """Render the Generalization Evaluation tab with context, batch stats, and drill-down."""
    st.header("Generalization Evaluation")
    with st.expander("ℹ️ About this step", expanded=False):
        st.info(
            "Observe the effect of your examples on model completions.  \n"
            "Now that you have created constitutional examples, use this view to test if they successfully steer the teacher model.  \n"
            "You will compare the model's output on unseen inputs with and without your examples.  \n"
            "Success Criteria: The steered model should better adhere to the ADAPTIVE principle than the unsteered model.  \n"
        )

    training_pool = repository.get_training_pool()
    test_tasks = _available_test_tasks(repository)

    # Zone A: Context Inspector
    with st.expander(f"({len(training_pool)}) Active Few-Shot examples", expanded=False):
        if not training_pool:
            st.info("No training examples saved yet.")
        else:
            for ex in training_pool:
                st.markdown(f"**Source #{ex.source_id}**")
                st.markdown(f"- User: {ex.user_prompt}")
                st.markdown(f"- Original: {ex.original_response}")
                st.markdown(f"- Critique: {ex.critique}")
                st.markdown(f"- Rewrite: {ex.rewrite}")
                st.divider()

    if len(training_pool) == 0:
        st.warning(
            "⚠️ No Constitutional Examples Yet To steer the teacher model,  \n"
            "it needs vetted examples to demonstrate how the principle applies.  \n"
            "Go to the Constitutional Example Creator to produce your initial set.  \n"
        )
        return

    if not test_tasks:
        st.info("No held-out conversations available. Create more training examples first.")
        return

    st.subheader("Test Mode")
    mode = st.radio(
        "Choose mode",
        options=["Single Case", "Full Batch Exam"],
        horizontal=True,
    )

    provider = get_llm_provider(config)
    prompt_builder = PromptBuilder()

    if mode == "Single Case":
        task_options = {f"Conversation #{tid}: {task.user[:60]}": tid for tid, task in test_tasks}
        selected_label = st.selectbox("Select conversation to test", options=list(task_options.keys()))
        selected_id = task_options[selected_label]
        selected_task = repository.tasks[selected_id]

        if st.button(f"🧪 Run Single Comparison with {config.model}", type="primary"):
            try:
                print(f"[DEBUG] Single comparison - model={config.model}, task_id={selected_id}")
                result = run_comparison(
                    selected_task,
                    training_pool,
                    config,
                    provider=provider,
                    prompt_builder=prompt_builder,
                    trace_manager=trace_manager,
                )
                st.session_state["single_result"] = {"task_id": selected_id, "task": selected_task, **result}
                st.toast("Single comparison completed")
            except Exception as exc:
                st.error(f"Single comparison failed: {exc}")

        single_result = st.session_state.get("single_result")
        if single_result and single_result.get("task_id") == selected_id:
            st.subheader(f"Detailed Comparison — Conversation #{selected_id}")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Zero-Shot (No Examples)**")
                with st.expander("Full LLM Response", expanded=False):
                    st.write(single_result["zero_raw"])
                render_trace_viewer(trace_entry=single_result.get("zero_trace"))
                st.markdown("**Validated Rewrite**")
                st.write(single_result["zero_rewrite"])
                zero_validation = render_acrostic_validator(single_result["zero_rewrite"], compact=True)
                if zero_validation and zero_validation.is_valid:
                    if st.button("✨ Promote to Curator (Zero-Shot)", key=f"promote_zero_{selected_id}", help="Found a natural win? Add it to your training set to make the model even smarter."):
                        from src.models import ConstitutionalExample
                        from datetime import datetime
                        example = ConstitutionalExample(
                            principle="ADAPTIVE",
                            user_prompt=selected_task.user,
                            original_response=selected_task.assistant,
                            critique=single_result.get("zero_critique") or single_result["zero_rewrite"],
                            rewrite=single_result["zero_rewrite"],
                            source_id=selected_id,
                            model_used=config.model,
                            timestamp=datetime.utcnow().isoformat(),
                            active=True,
                        )
                        repository.upsert_example(example)
                        st.toast("✅ Promoted to Training Set!")
                        st.rerun()

            with col2:
                st.markdown("**Few-Shot (With Training Pool)**")
                with st.expander("Full LLM Response", expanded=False):
                    st.write(single_result["few_raw"])
                render_trace_viewer(trace_entry=single_result.get("few_trace"))
                st.markdown("**Validated Rewrite**")
                st.write(single_result["few_rewrite"])
                few_validation = render_acrostic_validator(single_result["few_rewrite"], compact=True)
                if few_validation and few_validation.is_valid:
                    if st.button("✨ Promote to Curator (Few-Shot)", key=f"promote_few_{selected_id}", help="Found a natural win? Add it to your training set to make the model even smarter."):
                        from src.models import ConstitutionalExample
                        from datetime import datetime
                        example = ConstitutionalExample(
                            principle="ADAPTIVE",
                            user_prompt=selected_task.user,
                            original_response=selected_task.assistant,
                            critique=single_result.get("few_critique") or single_result["few_rewrite"],
                            rewrite=single_result["few_rewrite"],
                            source_id=selected_id,
                            model_used=config.model,
                            timestamp=datetime.utcnow().isoformat(),
                            active=True,
                        )
                        repository.upsert_example(example)
                        st.toast("✅ Promoted to Training Set!")
                        st.rerun()

            st.divider()
            if zero_validation and few_validation:
                if not zero_validation.is_valid and few_validation.is_valid:
                    st.success("🎉 Few-shot output fixed the zero-shot failure!")
                elif few_validation.is_valid and zero_validation.is_valid:
                    st.info("Both outputs satisfy ADAPTIVE. Few-shot did not change validity.")
                elif not few_validation.is_valid and zero_validation.is_valid:
                    st.warning("Few-shot underperformed: zero-shot passed but few-shot failed.")
                else:
                    st.error("Neither output satisfied the constraint. Add better training examples.")
        return

    # Batch mode
    st.caption(f"Ready to test on {len(test_tasks)} unseen conversations")
    task_options = [tid for tid, _ in test_tasks]
    selected_ids = st.multiselect(
        "Select test conversations",
        options=task_options,
        default=task_options,
        format_func=lambda tid: f"Conversation #{tid}",
    )
    if not selected_ids:
        st.info("Select at least one conversation to run the batch.")
        return

    if st.button(f"🚀 Run Batch Exam with {config.model}", type="primary"):
        try:
            with st.spinner(f"Running on {len(selected_ids)} examples. This may take a while..."):
                progress = st.progress(0)
                results: List[Dict] = []

            selected_tasks = [(tid, repository.tasks[tid]) for tid in selected_ids]

            for i, (task_id, task) in enumerate(selected_tasks, start=1):
                notes = ""
                try:
                    print(f"[DEBUG] Batch comparison - model={config.model}, task_id={task_id}")
                    comparison = run_comparison(
                        task,
                        training_pool,
                        config,
                        provider=provider,
                        prompt_builder=prompt_builder,
                        trace_manager=trace_manager,
                    )
                    zero_validation = comparison["zero_validation"]
                    few_validation = comparison["few_validation"]
                    zero_valid = zero_validation.is_valid if zero_validation else False
                    few_valid = few_validation.is_valid if few_validation else False
                    if not comparison["zero_raw"]:
                        notes += "Zero-shot empty output. "
                    if not comparison["few_raw"]:
                        notes += "Few-shot empty output. "
                    results.append(
                        {
                            "task_id": task_id,
                            "prompt": task.user,
                            **comparison,
                            "zero_valid": zero_valid,
                            "few_valid": few_valid,
                            "notes": notes,
                        }
                    )
                except Exception as exc:
                    notes = f"Error: {exc}"
                    results.append(
                        {
                            "task_id": task_id,
                            "prompt": task.user,
                            "zero_raw": "",
                            "few_raw": "",
                            "zero_rewrite": "",
                            "few_rewrite": "",
                            "zero_valid": False,
                            "few_valid": False,
                            "notes": notes,
                        }
                    )
                progress.progress(i / len(selected_tasks))

            st.session_state["batch_results"] = results
            st.toast("Batch exam completed")
        except Exception as exc:
            st.error(f"Batch run failed: {exc}")

    results = st.session_state.get("batch_results", [])
    if not results:
        st.info("Run the batch exam to see results.")
        return

    # Zone B: Scorecard
    total = len(results)
    zero_pass = sum(1 for r in results if r["zero_valid"])
    few_pass = sum(1 for r in results if r["few_valid"])
    baseline_acc = (zero_pass / total) * 100 if total else 0
    gym_acc = (few_pass / total) * 100 if total else 0
    lift = gym_acc - baseline_acc

    c1, c2, c3 = st.columns(3)
    c1.metric("Baseline (Zero-Shot)", f"{baseline_acc:.0f}%", f"{zero_pass}/{total}")
    c2.metric("Steered Accuracy", f"{gym_acc:.0f}%", f"{few_pass}/{total}")
    c3.metric("Lift", f"{lift:+.0f} pts", None)

    # Zone C: Detailed Results Table
    table_rows = []
    for r in results:
        table_rows.append(
            {
                "Conversation ID": r["task_id"],
                "User Prompt": r["prompt"],
                "Zero-Shot Valid": "✅" if r["zero_valid"] else "❌",
                "Few-Shot Valid": "✅" if r["few_valid"] else "❌",
                "Status": _status_label(r["zero_valid"], r["few_valid"]),
                "Notes": r.get("notes", ""),
            }
        )
    st.dataframe(table_rows, hide_index=True, use_container_width=True)

    # Export action bar
    st.markdown("### 📥 Export Results")

    def _clean_rewrite(text: str) -> str:
        if not text:
            return ""
        cleaned_lines = []
        for line in text.splitlines():
            cleaned = re.sub(r"^[\d\.\s\[\]]+", "", line).strip()
            if cleaned:
                cleaned_lines.append(cleaned)
        return " ".join(cleaned_lines)

    standard_data = []
    trace_data = []
    for r in results:
        cleaned_rewrite = _clean_rewrite(r["few_rewrite"])
        standard_data.append(
            {
                "user": r["prompt"],
                "bot": cleaned_rewrite,
            }
        )
        trace_data.append(
            {
                "user": r["prompt"],
                "original_response": repository.tasks[r["task_id"]].assistant,
                "critique": r.get("few_critique") or "",
                "rewrite": cleaned_rewrite,
            }
        )

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "📄 Download Final Dataset (JSON)",
            data=json.dumps(standard_data, indent=2),
            file_name="adaptive_batch_results.json",
            mime="application/json",
            help="Format: User + Bot",
        )
    with col_dl2:
        st.download_button(
            "🔍 Download Full CAI Trace (JSON)",
            data=json.dumps(trace_data, indent=2),
            file_name="adaptive_batch_trace.json",
            mime="application/json",
            help="Format: User + Original + Critique + Rewrite.",
        )

    # Drill-down selector
    selected_task_id = st.selectbox(
        "Inspect Specific Result",
        options=[r["task_id"] for r in results],
        format_func=lambda tid: f"Conversation #{tid}",
    )
    selected = next((r for r in results if r["task_id"] == selected_task_id), None)
    if not selected:
        return

    st.subheader(f"Detailed Comparison — Conversation #{selected_task_id}")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Zero-Shot (No Examples)**")
        with st.expander("Full LLM Response", expanded=False):
            st.write(selected["zero_raw"])
        render_trace_viewer(trace_entry=selected.get("zero_trace"))
        st.markdown("**Validated Rewrite**")
        st.write(selected["zero_rewrite"])
        zero_validation = render_acrostic_validator(selected["zero_rewrite"], compact=True)
        if zero_validation and zero_validation.is_valid:
            if st.button("✨ Promote to Curator (Zero-Shot)", key=f"promote_zero_batch_{selected_task_id}", help="Found a natural win? Add it to your training set to make the model even smarter."):
                from src.models import ConstitutionalExample
                from datetime import datetime
                task_obj = repository.tasks[selected_task_id]
                example = ConstitutionalExample(
                    principle="ADAPTIVE",
                    user_prompt=task_obj.user,
                    original_response=task_obj.assistant,
                    critique=selected.get("zero_critique") or selected["zero_rewrite"],
                    rewrite=selected["zero_rewrite"],
                    source_id=selected_task_id,
                    model_used=config.model,
                    timestamp=datetime.utcnow().isoformat(),
                    active=True,
                )
                repository.upsert_example(example)
                st.toast("✅ Promoted to Training Set!")
                st.rerun()

    with col2:
        st.markdown("**Few-Shot (With Training Pool)**")
        with st.expander("Full LLM Response", expanded=False):
            st.write(selected["few_raw"])
        render_trace_viewer(trace_entry=selected.get("few_trace"))
        st.markdown("**Validated Rewrite**")
        st.write(selected["few_rewrite"])
        few_validation = render_acrostic_validator(selected["few_rewrite"], compact=True)
        if few_validation and few_validation.is_valid:
            if st.button("✨ Promote to Curator (Few-Shot)", key=f"promote_few_batch_{selected_task_id}", help="Found a natural win? Add it to your training set to make the model even smarter."):
                from src.models import ConstitutionalExample
                from datetime import datetime
                task_obj = repository.tasks[selected_task_id]
                example = ConstitutionalExample(
                    principle="ADAPTIVE",
                    user_prompt=task_obj.user,
                    original_response=task_obj.assistant,
                    critique=selected.get("few_critique") or selected["few_rewrite"],
                    rewrite=selected["few_rewrite"],
                    source_id=selected_task_id,
                    model_used=config.model,
                    timestamp=datetime.utcnow().isoformat(),
                    active=True,
                )
                repository.upsert_example(example)
                st.toast("✅ Promoted to Training Set!")
                st.rerun()

    st.divider()
    render_trace_viewer()
    if not zero_validation.is_valid and few_validation.is_valid:
        st.success("🎉 Few-shot output fixed the zero-shot failure!")
    elif few_validation.is_valid and zero_validation.is_valid:
        st.info("Both outputs satisfy ADAPTIVE. Few-shot did not change validity.")
    elif not few_validation.is_valid and zero_validation.is_valid:
        st.warning("Few-shot underperformed: zero-shot passed but few-shot failed.")
    else:
        st.error("Neither output satisfied the constraint. Add better training examples.")
