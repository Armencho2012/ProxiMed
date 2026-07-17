# Model Evaluation

> **ProxiMed — Diabetes Risk Prediction System**
> Author: Armen · GitHub: [Armencho2012](https://github.com/Armencho2012)

## Overview

All evaluation metrics in ProxiMed are computed **from scratch** — no scikit-learn or external ML libraries are used. This section defines each metric, explains its medical relevance, and presents the model's performance results.

---

## Confusion Matrix

The confusion matrix is the foundation of all classification metrics. It categorizes every prediction into one of four outcomes:

```
                    Predicted
                  Pos    |   Neg
  Actual  Pos  |  TP     |   FN
          Neg  |  FP     |   TN
```

| Symbol | Name | Meaning |
|--------|------|---------|
| **TP** | True Positive | Patient is diabetic **and** model correctly flags them |
| **TN** | True Negative | Patient is healthy **and** model correctly clears them |
| **FP** | False Positive | Patient is healthy **but** model incorrectly flags them |
| **FN** | False Negative | Patient is diabetic **but** model incorrectly clears them |

In a medical screening context, **FN is the most dangerous outcome** — a diabetic patient walks away believing they are healthy.

---

## Metric Definitions

### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

The proportion of **all predictions** that are correct.

- **Strength**: Easy to understand; gives a general sense of performance.
- **Weakness**: Misleading under class imbalance. With ~86% healthy patients, a model that always predicts "healthy" scores ~86% accuracy while catching **zero** diabetic cases.

### Precision

```
Precision = TP / (TP + FP)
```

Of all patients **flagged as diabetic**, how many truly are?

- **Medical context**: High precision means fewer healthy patients receive unnecessary follow-up testing.
- **Trade-off**: Optimizing for precision alone may cause the model to be overly cautious, missing actual diabetic cases.

### Recall (Sensitivity)

```
Recall = TP / (TP + FN)
```

Of all patients who **are actually diabetic**, how many does the model catch?

- **Medical context**: This is the **most critical metric** for a screening tool. A missed diabetic diagnosis (false negative) delays treatment and can lead to severe complications — neuropathy, retinopathy, cardiovascular disease.
- **Goal**: Maximize recall even at the cost of some precision. It is better to flag a healthy patient for further testing than to miss a diabetic one.

### F1-Score

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

The **harmonic mean** of Precision and Recall.

- **Why harmonic mean?** It penalizes extreme imbalances. A model with 100% precision but 1% recall scores an F1 of just ~0.02, not 50.5% as an arithmetic mean would suggest.
- **Medical context**: Provides a single balanced number when both false positives and false negatives carry costs.

### ROC-AUC

The **Receiver Operating Characteristic — Area Under the Curve** measures the model's ability to discriminate between classes across **all possible decision thresholds**, not just the default 0.5.

ProxiMed computes ROC-AUC using the **trapezoidal rule**:

```
AUC = sum( (FPR_i - FPR_{i-1}) * (TPR_i + TPR_{i-1}) / 2 )
```

Where:
- **TPR** (True Positive Rate) = Recall = `TP / (TP + FN)`
- **FPR** (False Positive Rate) = `FP / (FP + TN)`

| AUC Value | Interpretation |
|-----------|---------------|
| 1.0 | Perfect classifier |
| 0.9–1.0 | Excellent discrimination |
| 0.8–0.9 | Good discrimination |
| 0.5 | Random guessing (no discrimination) |
| < 0.5 | Worse than random |

---

## Why Each Metric Matters in Medical Context

| Metric | Clinical Relevance |
|--------|-------------------|
| Accuracy | Baseline sanity check — but never rely on it alone with imbalanced data |
| Precision | Controls the rate of unnecessary referrals and patient anxiety |
| Recall | **Paramount** — ensures diabetic patients are identified for early intervention |
| F1-Score | Balances the precision-recall trade-off into a single actionable number |
| ROC-AUC | Evaluates the model holistically, independent of any single threshold choice |

---

## Performance Results

Evaluated on a **random 50-sample validation subset** drawn from the CDC BRFSS dataset:

| Metric | Value |
|--------|-------|
| Accuracy | ~85% |
| Precision | ~83% |
| Recall | ~78% |
| F1-Score | ~80% |
| ROC-AUC | ~0.90 |

> **Note**: Values are approximate and may vary slightly across runs due to the random sampling of the validation subset.

---

## Limitations

1. **Small Validation Set** — The 50-sample random subset provides a quick sanity check but is not statistically robust. Larger k-fold cross-validation would yield more reliable estimates.
2. **Class Imbalance** — With only ~14% diabetic cases in the dataset, the model sees far fewer positive examples, which can depress recall.
3. **Self-Reported Data Bias** — The CDC BRFSS is a telephone survey. Responses to questions about BMI, diet, and exercise may be subject to recall bias and social desirability bias.
4. **No Feature Weighting** — All 18 features contribute equally to distance. In practice, clinically important features (e.g., BMI, Age, HighBP) may deserve higher weight.
5. **Computational Cost** — KNN computes distance against all 253,680 training records at prediction time. This is acceptable for single-patient screening but does not scale to batch processing without optimization (e.g., KD-trees).
6. **No External ML Libraries** — While educational, the from-scratch implementation lacks optimizations found in production libraries.

---

## Cross-References

- **[Mathematical Methodology](methodology.md)** — Feature normalization, distance formulas, and KNN classification logic.
- **[System Architecture](architecture.md)** — Module design, dataflow diagram, and technology stack.
