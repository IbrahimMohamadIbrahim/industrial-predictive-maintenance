# Deep Learning deployment handoff

This directory is the handoff location for the Deep Learning teammate's
artifacts. The Streamlit dashboard reads `model_manifest.json` and reports
whether the MLP and autoencoder handoffs are complete. It does **not** invent
or display DL predictions until the real inference implementation is reviewed
and integrated.

## Expected structure

```text
models/deep_learning/
├── model_manifest.json             # Create from model_manifest.example.json
├── model_card.md                   # Create from model_card_template.md
├── mlp/
│   ├── model.keras | model.pt | model.joblib
│   └── preprocessing.joblib
└── autoencoder/
    ├── model.keras | model.pt
    └── preprocessing.joblib
```

The exact artifact format depends on the training framework. The teammate must
provide an inference function or a minimal reproducible example before the
deployment team loads an artifact in the application.

## Mandatory handoff items

For both models, supply:

- trained artifact and framework/version;
- preprocessing artifact or reproducible preprocessing code;
- exact feature names and order;
- training/test split description and metrics;
- a model card and the completed manifest.

For the MLP, supply the binary decision threshold. For the autoencoder, supply
the anomaly-score method and anomaly threshold. Do not use failure-mode labels
(`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) as features when predicting `Machine failure`.

## Deployment activation

1. Copy `model_manifest.example.json` to `model_manifest.json` and complete it.
2. Place artifacts at the declared relative paths.
3. Provide the inference contract to the deployment owner.
4. The dashboard will show each handoff as ready only after its metadata and
   declared files validate. The deployment owner then wires the reviewed
   inference adapter into `app/`.
