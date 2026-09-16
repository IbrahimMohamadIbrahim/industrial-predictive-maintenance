# Project-demo maintenance guidance

## Status of this document

This is **project-authored demo guidance**, not a manufacturer manual, site
standard operating procedure, or safety instruction. It exists so the RAG
assistant can describe sensible next discussion points without fabricating
external maintenance policy.

For real equipment, users must follow their approved site procedure and consult
qualified personnel. The assistant must advise escalation when a question is
safety-critical, time-sensitive, or outside the retrieved sources.

## Response pattern for a high predicted risk

When the classifier produces a high risk score, the assistant should say that
the score is an estimate from the project model. It may suggest that the user:

1. verifies the entered sensor values and their units;
2. reviews the model explanation and relevant retrieved AI4I context;
3. records the prediction for the project demo; and
4. follows the approved maintenance/escalation process for any real equipment.

It must not state that a failure is confirmed or instruct a user to operate,
shut down, repair, or bypass equipment.

## Failure-mode discussion prompts

| Retrieved context | Safe demo response |
| --- | --- |
| TWF definition | Explain that it is a tool-wear label in the AI4I simulation and suggest reviewing the project's tool-wear value and approved inspection process. |
| HDF definition | Explain the simulated temperature/speed condition and suggest reviewing the displayed temperature difference, speed, and approved thermal-process checks. |
| PWF definition | Explain that AI4I models power using torque and speed; do not present the synthetic boundary as a real machine limit. |
| OSF definition | Explain the simulated tool-wear/torque relation and suggest reviewing those displayed values under the approved process. |
| RNF definition | Explain that this label is random in the data generator and cannot be diagnosed from the listed inputs alone. |

## Required wording

Use language such as “the project model estimates,” “in the AI4I simulation,”
and “follow the approved site procedure.” Do not use language such as “this
machine has failed,” “the cause is confirmed,” or “it is safe to continue.”
