# Project requirements for the RAG assistant

## Project context

The graduation project is **Industrial Predictive Maintenance & Failure
Prevention**. Its business goal is to estimate machine-failure risk and
recommend maintenance actions. The brief identifies false negatives as costly
and requires model interpretability.

## Required AI-assistant deliverable

The project brief includes **RAG/agent** in the required final deliverables. It
also calls for an AI-assistant or advanced-integration demonstration in the
final presentation.

The RAG assistant should therefore be integrated into the deployed Streamlit
application and help a user interpret available project information. It is not
a substitute for the classifier, anomaly model, monitoring components, or
explainability views required elsewhere in the project.

## Constraints inherited from the brief

- The AI4I dataset is a synthetic benchmark. Do not claim real industrial
  reliability from it.
- Document assumptions about sensor frequency and failure timing.
- Do not leak future or failure-related information into predictive features.
- Use task-appropriate evidence rather than accuracy alone.

## Appropriate assistant questions

- What does this project predict and what are its limitations?
- Which input variables are used by the deployed model?
- What do the published AI4I failure labels mean?
- What is the difference between a high model risk score and a confirmed
  failure diagnosis?
- What project-demo follow-up inspection can be considered for a retrieved
  failure-mode description?

## Source

- `project_desc.pdf`, pages 2-6, located in the workspace parent directory.
- Read on 2026-09-16.
