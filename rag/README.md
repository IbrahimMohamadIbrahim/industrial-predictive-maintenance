# RAG foundation

This directory contains the curated knowledge base for the Industrial Predictive
Maintenance assistant. The assistant is intended to explain the project, the
AI4I 2020 synthetic dataset, its failure modes, and the project's demo
maintenance guidance alongside the model's prediction.

## Scope and guardrails

The assistant must:

- answer from retrieved material in `sources/` and cite the source file title;
- state when the answer is not supported by the knowledge base;
- describe AI4I as a synthetic benchmark, not evidence of real industrial
  reliability or a production safety system;
- distinguish a model prediction from a factual diagnosis;
- frame `04_demo_maintenance_guidance.md` as a project-demo inspection guide,
  not as a replacement for site procedures, qualified personnel, or safety
  controls.

The assistant must not:

- invent maintenance procedures, threshold values, or data provenance;
- claim that a predicted failure probability proves a failure mode or cause;
- provide emergency, safety-critical, or production-operating instructions
  beyond advising the user to follow their approved site procedure.

## Initial corpus

| File | Purpose | Source type |
| --- | --- | --- |
| `01_project_requirements.md` | Project scope and required RAG/agent behaviour | Local project brief |
| `02_ai4i_dataset_reference.md` | Dataset provenance, variables, limitations, and citation | UCI primary source |
| `03_ai4i_failure_modes.md` | Definitions of the five published AI4I failure labels | UCI primary source |
| `04_demo_maintenance_guidance.md` | Clearly labelled demo-only response patterns | Project-authored |
| `source_registry.yaml` | Provenance, licensing, and retrieval metadata | Metadata |

## Documents to add before a production-style demo

Only add documents your team is allowed to use and cite. The highest-value next
sources are:

1. A machine or process owner's approved maintenance manual.
2. Site-approved standard operating procedures and safety/escalation policy.
3. A documented maintenance decision policy that maps the team's risk tiers to
   permitted actions and owners.
4. Model card and evaluation report: features, threshold, metrics, limitations,
   version, and date.
5. Data dictionary and SQL/EDA findings produced by this repository.

Do not treat the synthetic AI4I failure formulas as universal equipment rules.

## Retrieval design target

The implemented retrieval layer indexes only `sources/`, preserves the file
name and heading for citations, and uses an in-memory TF-IDF vector index. It
returns the most relevant source chunks and passes them to the answer generator
with the guardrails above. A query returns a concise answer plus a `Sources`
list. If nothing relevant is retrieved, the assistant says that the project
knowledge base does not contain an answer.

## Run the RAG dashboard

From the repository root:

```bash
python -m pip install -r requirements.txt
streamlit run app/app.py
```

The assistant always performs local retrieval. To enable generated answers,
set these environment variables before starting Streamlit:

```text
GROQ_API_KEY=your_key
GROQ_MODEL=qwen/qwen3.8-27b
```

Without an API key, the dashboard remains usable in retrieval-only mode. It
shows the supporting source excerpts rather than pretending to generate an
answer. Generated answers use Groq's Python SDK. Source citations shown in the
UI always come from the local retriever, not from the model.

## Keeping the Groq API key secret

Never place a real key in Python files, notebooks, the README, or any file you
commit. Use one of these local-only options:

1. Set `GROQ_API_KEY` in your shell session.
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, insert
   your real key locally, and do not rename or commit that secrets file.

The repository's `.gitignore` excludes `.streamlit/secrets.toml` and `.env`.
The app never prints the key.

## Tests

Run the offline RAG tests with:

```bash
pytest -q tests/test_rag.py
```
