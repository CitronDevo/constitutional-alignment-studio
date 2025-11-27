# Constitutional Alignment Studio

A specialized interface for bootstrapping and evaluating Constitutional AI workflows. This tool supports the “critique + rewrite” phase (ADAPTIVE principle: first letters spell ADAPTIVE) by letting humans efficiently create, vet, and test examples.

## 🚀 Quick Start
Managed with **uv** for fast, reproducible environments.

**Prerequisites**
- Python 3.12+
- uv
- An LLM provider: OpenAI API key and/or Google Gemini key; or a local Ollama model

**Install**
```bash
git clone <repo_url>
cd adaptive_constitutional_AI
uv sync
```

**Configure**
```bash
cp .env.example .env  # then add OPENAI_API_KEY / GOOGLE_API_KEY, etc.
```

**Run**
```bash
uv run streamlit run ui/app.py
```

## 🏗️ Architecture (Two Steps)

### 1) Example Curation Studio (Drafting)
- Human-in-the-loop workflow: auto-draft → human edit → deterministic validator → save.
- Iterative loop: “Critique & Revise Again” using validator feedback.
- Dataset management: activate/deactivate, view, edit, delete, version history/restore, save/load “golden examples.”
- Data flywheel: promote wins from evaluation back into the curated set.

### 2) Generalization Evaluation (Testing)
- Zero-shot vs few-shot comparisons on unseen conversations with validators and traces.
- Batch testing with “Steered Accuracy,” lift, and drill-down views.
- Export results in two formats: final dataset (user/bot) and full CAI trace (user/original/critique/rewrite).

## 🛠️ Technical Stack
- Streamlit UI
- Provider-agnostic LLM client (OpenAI, Google Gemini, Ollama)
- Pydantic models for validation (conversations, examples, configs, results)
- Deterministic validator for the ADAPTIVE constraint
- Local JSON/JSONL storage (`data/output/golden_examples.json`, history log in `data/output/history.jsonl`)

## 📂 Key Files
- `data/input/dev.jsonl` — raw conversations
- `data/output/golden_examples.json` — curated few-shot set (head)
- `src/llm_client.py` — LLM abstraction
- `src/prompt_engine.py` — constitutional prompts/templates
- `src/repository.py` — CRUD, history, import/export
- `src/validator.py` — ADAPTIVE constraint checker
- `ui/builder_view.py` — Example Curation Studio
- `ui/tester_view.py` — Generalization Evaluation

## Deliverables Coverage
- Documented tool to create critique+rewrite examples and observe their effect.
- Exportable examples (golden examples download) and batch evaluation exports (final dataset + CAI trace).
- Prompting/UX approach embodied in the two-step workflow with validator-driven loops and active context control.
