# Model Comparison — Predictive Maintenance

## 1. Overview

This project evaluates multiple approaches for industrial predictive maintenance:

1. **Classical Supervised Machine Learning**
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting

2. **Supervised Deep Learning**
   - Wide Multi-Layer Perceptron (MLP)

3. **Unsupervised Deep Learning**
   - Tabular Autoencoder for anomaly detection

The supervised models are trained to predict whether a machine will experience a failure, while the Autoencoder follows a different paradigm: it learns the normal operating behavior of the machine and identifies observations that deviate significantly from that behavior.

Because the business objective is to **minimize unexpected machine failures**, particular attention is given to **Recall, PR-AUC, False Negatives, and simulated operational cost**, rather than Accuracy alone.

---

# 2. Classical ML vs. Supervised MLP

The following models solve the same supervised binary classification task:

> **Given the current machine sensor readings, predict whether the machine will experience a failure.**

| Model | F1-Score | Recall | PR-AUC | Selected Threshold |
|---|---:|---:|---:|---:|
| **Random Forest** | **0.7313** | 0.7206 | **0.7921** | 0.345 |
| **Decision Tree** | 0.6923 | **0.7941** | 0.7100 | 0.950 |
| **Gradient Boosting** | 0.4437 | **0.9265** | 0.7086 | 0.295 |
| **Logistic Regression** | 0.3866 | 0.3382 | 0.3811 | 0.890 |
| **Supervised MLP (Wide)** | 0.6483 | 0.6912 | 0.7275 | — |

### Observations

The **Random Forest** achieved the highest F1-score and PR-AUC among the evaluated models:

- F1-score: **0.7313**
- PR-AUC: **0.7921**

The **Gradient Boosting** model achieved the highest Recall:

- Recall: **0.9265**

This means that, under the selected threshold, Gradient Boosting detected the largest proportion of actual failures. However, its F1-score was substantially lower than Random Forest, indicating a different precision/recall trade-off.

The supervised MLP achieved:

- F1-score: **0.6483**
- Recall: **0.6912**
- PR-AUC: **0.7275**

Therefore, in this experiment, the MLP did not outperform the strongest classical tree-based models on the supervised failure-prediction task.

This is an important result rather than a problem: **deep learning does not automatically outperform classical ML on structured/tabular datasets.**

---

# 3. Supervised MLP vs. Unsupervised Autoencoder

The MLP and Autoencoder should not be interpreted as two identical models solving exactly the same problem.

### Supervised MLP

The MLP learns directly from historical failure labels:

```text
Sensor Data
     ↓
    MLP
     ↓
P(Failure)