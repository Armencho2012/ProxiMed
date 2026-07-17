# Mathematical Methodology

> **ProxiMed — Diabetes Risk Prediction System**
> Author: Armen · GitHub: [Armencho2012](https://github.com/Armencho2012)

## Why K-Nearest Neighbors?

KNN was chosen as the classification algorithm for several reasons:

- **Non-parametric** — Makes no assumptions about the underlying data distribution, unlike logistic regression or naive Bayes.
- **Instance-based learning** — Stores the entire training set and defers computation to prediction time, avoiding a separate training phase.
- **Binary classification fit** — Naturally supports the two-class problem (diabetic vs. healthy) through majority voting.
- **Medical interpretability** — "Your profile is closest to these five patients, and three of them are diabetic" is an intuitive explanation for clinicians and patients alike.
- **Educational transparency** — Every step (distance, voting, metrics) can be implemented from scratch without opaque library calls.

---

## Feature Normalization — Min-Max Scaling

### The Problem

Raw features span vastly different scales:
- **BMI** ranges from ~10 to ~98
- **Binary flags** (e.g., HighBP) are only 0 or 1

Without normalization, high-magnitude features like BMI would **dominate** the Euclidean distance, drowning out the contribution of binary and ordinal features.

### The Formula

```
x_scaled = (x - x_min) / (x_max - x_min)
```

Every feature value is linearly mapped to the range **[0, 1]**:
- The minimum observed value maps to **0**.
- The maximum observed value maps to **1**.
- All intermediate values are proportionally distributed.

This ensures every feature contributes **equally** to distance calculations.

---

## Euclidean Distance

The similarity between two patient vectors **p** and **q** is measured using Euclidean distance across all 18 normalized features:

```
d(p, q) = sqrt( sum( (p_i - q_i)^2 ) )    for i = 1..18
```

- A **smaller** distance means the patients are more alike.
- The distance is computed between the input vector and **every record** in the training set.
- Results are sorted in ascending order to identify the closest neighbors.

---

## KNN Classification

### Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **k** | 5 | Odd number avoids ties; small enough for locality, large enough to reduce noise |
| **Voting** | Simple majority | Class with ≥ 3 of 5 votes wins |
| **Threshold** | 0.5 | If `positive_votes / k ≥ 0.5`, predict diabetic |

### Algorithm Steps

1. **Normalize** the incoming patient vector using the training set's min/max values.
2. **Compute** Euclidean distance between the patient vector and all training records.
3. **Sort** distances in ascending order.
4. **Select** the **k = 5** nearest neighbors.
5. **Vote** — count labels among the five neighbors.
6. **Output**:
   - **Predicted class** — majority label (0 = healthy, 1 = diabetic)
   - **Probability** — `positive_votes / k` (e.g., 3/5 = 0.60)

---

## Class Imbalance

The CDC BRFSS dataset exhibits significant class imbalance:

| Class | Proportion |
|-------|-----------|
| Healthy (0) | ~86% |
| Diabetic (1) | ~14% |

### Why This Matters

A trivial classifier that **always predicts healthy** would achieve ~86% accuracy — a misleading number. This is why ProxiMed computes multiple evaluation metrics beyond accuracy:

- **Precision** — Are positive predictions trustworthy?
- **Recall** — Are actual diabetic patients being caught?
- **F1-Score** — Balanced trade-off between Precision and Recall.
- **ROC-AUC** — Performance across all decision thresholds.

In a medical screening context, **Recall is paramount**: a missed diabetic diagnosis (false negative) carries far greater risk than a false alarm (false positive).

---

## Feature Table

ProxiMed uses **18 of the 22 original features** after dropping `CholCheck`, `AnyHealthcare`, and `NoDocbcCost`.

| # | Feature | Type | Range | Description |
|---|---------|------|-------|-------------|
| 1 | HighBP | Binary | 0–1 | High blood pressure diagnosis |
| 2 | HighChol | Binary | 0–1 | High cholesterol diagnosis |
| 3 | BMI | Numeric | ~10–98 | Body Mass Index |
| 4 | Smoker | Binary | 0–1 | Has smoked ≥ 100 cigarettes in lifetime |
| 5 | Stroke | Binary | 0–1 | History of stroke |
| 6 | HeartDiseaseorAttack | Binary | 0–1 | Coronary heart disease or myocardial infarction |
| 7 | PhysActivity | Binary | 0–1 | Physical activity in past 30 days |
| 8 | Fruits | Binary | 0–1 | Consumes fruit ≥ 1 time per day |
| 9 | Veggies | Binary | 0–1 | Consumes vegetables ≥ 1 time per day |
| 10 | HvyAlcoholConsump | Binary | 0–1 | Heavy alcohol consumption |
| 11 | GenHlth | Ordinal | 1–5 | Self-rated general health (1 = excellent, 5 = poor) |
| 12 | MentHlth | Numeric | 0–30 | Days of poor mental health in past 30 days |
| 13 | PhysHlth | Numeric | 0–30 | Days of poor physical health in past 30 days |
| 14 | DiffWalk | Binary | 0–1 | Serious difficulty walking or climbing stairs |
| 15 | Sex | Binary | 0–1 | Biological sex (0 = female, 1 = male) |
| 16 | Age | Ordinal | 1–13 | 13-level age category |
| 17 | Education | Ordinal | 1–6 | Education level (1 = none, 6 = college graduate) |
| 18 | Income | Ordinal | 1–8 | Annual household income bracket |

---

## Cross-References

- **[System Architecture](architecture.md)** — Module design, dataflow diagram, and technology stack.
- **[Model Evaluation](evaluation.md)** — Confusion matrix, metric formulas, and performance results.
