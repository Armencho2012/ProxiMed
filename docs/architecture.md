# System Architecture

> **ProxiMed — Diabetes Risk Prediction System**
> Author: Armen · GitHub: [Armencho2012](https://github.com/Armencho2012)

## Overview

ProxiMed follows a **decoupled three-module architecture** designed for clarity, testability, and separation of concerns. Each module owns a single responsibility:

| Module | Role | Dependencies |
|--------|------|--------------|
| `model.py` | Mathematical core — data loading, normalization, KNN classification, evaluation | `pandas` |
| `main.py` | Desktop dashboard — interactive patient input form and diagnostic display | `tkinter`, `ttkbootstrap`, `model.py` |
| `robot.py` | Automated pipeline — headless verification of preset patient vectors | `model.py` |

No external machine-learning libraries (scikit-learn, TensorFlow, etc.) are used. Every algorithm — from distance computation to ROC-AUC — is implemented **from scratch**.

---

## Module Details

### model.py — The Mathematical Core

The backbone of ProxiMed. Responsibilities include:

1. **Data Ingestion** — Loads the CDC BRFSS CSV (253,680 records) via `pandas.read_csv()`.
2. **Column Pruning** — Drops three low-signal columns before classification:
   - `CholCheck` — cholesterol check status (redundant with `HighChol`)
   - `AnyHealthcare` — healthcare coverage flag
   - `NoDocbcCost` — cost-barrier flag
3. **Target Partitioning** — Separates `Diabetes_binary` as the label column; remaining 18 columns become features.
4. **Min-Max Normalization** — Scales every feature to [0, 1] so no single scale dominates distance.
5. **Euclidean Distance** — Computes point-to-point distance between a patient vector and every training record.
6. **KNN Voting (k = 5)** — Selects the five nearest neighbors; majority class wins. Probability is `positive_votes / k`.
7. **Confusion Matrix Evaluation** — Derives TP, TN, FP, FN and computes:
   - **Accuracy** — overall correctness
   - **Precision** — reliability of positive flags
   - **Recall** — sensitivity to actual positives
   - **F1-Score** — harmonic mean of Precision and Recall
   - **ROC-AUC** — trapezoidal approximation across thresholds

### main.py — The Desktop Dashboard

An interactive GUI built with `tkinter` and the `ttkbootstrap` **superhero** theme.

- **Scrollable Canvas** — Wraps the entire form so all fields remain accessible on smaller screens.
- **Patient Input Form** — 18 features organized into two groups:
  - **11 Binary Radio Buttons** — HighBP, HighChol, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, DiffWalk, Sex
  - **7 Numeric Fields** — BMI, GenHlth, MentHlth, PhysHlth, Age, Education, Income
- **Dynamic Diagnostic Cards** — After prediction:
  - 🔴 **Red card (danger)** — Elevated diabetic risk detected
  - 🟢 **Green card (success)** — Healthy / low-risk profile

### robot.py — Automated Pipeline

A headless test script that bypasses the GUI entirely:

- Constructs **preset patient vectors** (dictionaries of 18 features).
- Passes each vector directly into the `CustomKNN` class.
- Prints classification results and probability scores for quick verification.
- Useful for CI sanity checks and batch testing without manual interaction.

---

## System Dataflow

```
[Patient Data Input] (main.py GUI)
       │
       ▼ (Passes Dict)
[Normalization Engine] (model.py) → Scales raw values to [0,1]
       │
       ▼ (Passes Vector)
[Distance Calculator] (model.py) → Euclidean distance against all records
       │
       ▼ (Selects k-closest)
[Majority Voting] (model.py) → Identifies nearest neighbors & probability
       │
       ▼ (Dynamic Evaluation)
[Metrics Calculator] (model.py) → Accuracy, Precision, Recall, F1, ROC-AUC
       │
       ▼ (Returns Dict)
[Dashboard Display] (main.py GUI) → Renders diagnostic card
```

---

## Data Pipeline

```
CDC BRFSS Survey (253,680 records)
  → pandas CSV ingestion
  → Column pruning (22 → 18 features)
  → Min-Max normalization to [0, 1]
  → KNN classification (k = 5)
  → Diagnostic result + probability
```

**Dataset**: [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) — sourced from the CDC Behavioral Risk Factor Surveillance System (BRFSS).

---

## Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Python 3.x | Rapid prototyping, rich ecosystem |
| Data handling | `pandas` | Efficient CSV I/O and DataFrame operations |
| GUI framework | `tkinter` | Ships with CPython — zero extra installs |
| GUI theme | `ttkbootstrap` (superhero) | Modern dark-themed widgets without web overhead |
| ML algorithms | Custom (from scratch) | Educational transparency; no black-box dependencies |

---

## Cross-References

- **[Mathematical Methodology](methodology.md)** — Feature normalization, distance formulas, and KNN mechanics.
- **[Model Evaluation](evaluation.md)** — Confusion matrix, metric definitions, and performance results.
