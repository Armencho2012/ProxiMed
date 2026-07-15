# Project Description: Diabetes Risk Prediction via Custom K-Nearest Neighbors (KNN)

---

## 1. Project Motivation & Problem Statement

Diabetes mellitus is one of the most prominent chronic public health crises of the modern era. Early detection is paramount to preventing severe secondary complications, including cardiovascular disease, nephropathy, and neuropathy. However, clinical diagnostics rely heavily on invasive, resource-intensive laboratory blood tests (such as HbA1c or fasting plasma glucose). Consequently, millions of prediabetic and diabetic individuals remain undiagnosed.

This project addresses this diagnostic barrier by modeling a **non-invasive, low-cost screening tool**. By mapping self-reported lifestyle choices, demographics, and self-assessments to diabetic outcomes, we prove that machine learning can accurately flag high-risk individuals before clinical symptoms manifest. 

To demonstrate rigorous fundamental engineering, the entire machine learning pipeline—including feature scaling, spatial distance calculations, classification voting, and validation metrics—is **built from scratch in raw Python**, bypasses high-level ML frameworks (such as Scikit-Learn), and is deployed inside an interactive desktop application.

---

## 2. Dataset Specification

The project is trained on the **CDC’s Behavioral Risk Factor Surveillance System (BRFSS)** survey data, containing over $250,000$ clinical records. 

To optimize performance and computational efficiency, the system drops non-lifestyle administrative indicators (`CholCheck`, `AnyHealthcare`, `NoDocbcCost`) and targets the following $18$ core features:

### Target Variable
* **`Diabetes_binary`**: Binary indicator where `0.0` represents healthy/no diabetes, and `1.0` represents prediabetes or diabetes.

### Demographic & Behavioral Predictors
* **Demographics**: `Sex` (binary), `Age` (ordinal scale 1–13), `Education` (ordinal scale 1–6), `Income` (ordinal scale 1–8).
* **Cardiovascular Anchors**: `HighBP` (binary), `HighChol` (binary), `Stroke` (binary), `HeartDiseaseorAttack` (binary).
* **Lifestyle Choices**: `Smoker` (binary), `PhysActivity` (binary), `Fruits` (binary), `Veggies` (binary), `HvyAlcoholConsump` (binary).
* **Physical & Mental Self-Reports**: `BMI` (continuous), `GenHlth` (ordinal 1–5), `MentHlth` (numeric 0–30), `PhysHlth` (numeric 0–30), `DiffWalk` (binary).

---

## 3. Mathematical & Algorithmic Foundation

### A. Dynamic Min-Max Scaling
KNN calculations rely entirely on spatial distance metrics. If unscaled, high-magnitude features like `BMI` (scale $10$ to $98$) or `PhysHlth` (scale $0$ to $30$) would mathematically dwarf binary inputs like `HighBP` ($0$ or $1$). 

To enforce parity across all dimensional axes, the model dynamically fits a Min-Max scaler to the dataset, shifting all values into a standardized $[0, 1]$ interval:

$$x_{\text{scaled}} = \frac{x - x_{\text{min}}}{x_{\text{max}} - x_{\text{min}}}$$

The calculated boundary arrays $(x_{\text{min}}, x_{\text{max}})$ are permanently stored in the class instance, allowing the engine to correctly normalize fresh patient input fields in real time.

### B. Euclidean Proximity Search
For any given target patient vector $\mathbf{p}$ and training sample $\mathbf{q}$, spatial proximity is calculated using an $n$-dimensional Euclidean distance formula:

$$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

The program computes this distance from the patient’s scaled input across the entire dataset, tracking and sorting the indices to locate the closest $k$ neighbors (configured to $k=5$).

### C. Evaluation Metrics from Scratch
Because the BRFSS dataset suffers from severe class imbalance (~$14\%$ diabetic, ~$86\%$ healthy), raw Accuracy is a highly misleading indicator of model performance. If a model simply predicts "healthy" for every single patient, it would achieve $86\%$ accuracy while letting $100\%$ of diabetic patients slip through undetected.

To prevent this, our custom metric engine manually populates a Confusion Matrix—identifying True Positives ($TP$), True Negatives ($TN$), False Positives ($FP$), and False Negatives ($FN$)—and evaluates five diagnostic measures:

* **Accuracy**: General correctness rate.
    $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$
* **Precision**: The confidence level of positive alerts.
    $$\text{Precision} = \frac{TP}{TP + FP}$$
* **Recall (Sensitivity)**: The critical clinical metric. It measures the proportion of actual diabetic cases correctly caught by the model, minimizing dangerous $FN$ occurrences.
    $$\text{Recall} = \frac{TP}{TP + FN}$$
* **F1-Score**: The balanced harmonic mean of precision and recall.
    $$\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
* **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: Approximated by sorting predicted probability outputs, evaluating the True Positive Rate ($TPR$) and False Positive Rate ($FPR$) across varying decision thresholds, and integrating the curves using the trapezoidal rule:
    $$\text{AUC} \approx \sum_{i=1}^{m} \frac{(FPR_{i} - FPR_{i-1}) \times (TPR_{i} + TPR_{i-1})}{2}$$

---

## 4. Decoupled System Architecture

The application's codebase is partitioned into three decoupled files, separating calculations, interfaces, and testing scripts.
                ┌────────────────────────┐
                │      database.csv      │
                └───────────┬────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│                         model.py                         │
├──────────────────────────────────────────────────────────┤
│ * Min-Max Scaling Fits                                   │
│ * Vector Euclidean Distance Calculator                   │
│ * Custom Accuracy, Recall, Precision, F1, AUC Engine     │
└─────────────────────────────┬────────────────────────────┘
                              │
                ┌─────────────┴─────────────┐
                ▼                                              ▼
┌───────────────────────────┐┌───────────────────────────┐
│          main.py          ││         robot.py          │
├───────────────────────────┤├───────────────────────────┤
│ * Tkinter UI Layout       ││ * Headless Tester         │
│ * Form Parsing Engine     ││ * Mockup Vector Inputs    │
│ * Dynamic Warning Cards   ││ * Pipeline Verification   │
└───────────────────────────┘└───────────────────────────┘


* **`model.py` (The Mathematical Core)**: Handles the raw CSV dataset loading, partitions target labels from scaling arrays, scales user inputs, runs the sorting distance matrix, and computes performance validation metrics.
* **`main.py` (The Desktop Dashboard)**: A modern graphical interface styled with `ttkbootstrap`’s "superhero" theme. It isolates inputs in a clean scrollable canvas, renders dynamic radio buttons for binary inputs, and displays a responsive clinical metrics panel that flashes red (`danger`) for high-risk flags and green (`success`) for healthy patients.
* **`robot.py` (The Automated Pipeline)**: A lightweight, headless testing framework that programmatically fires preset patient vectors into the KNN model to quickly verify calculation speeds and system stability without manual UI intervention.

---

## 5. Technology Stack

* **Runtime Environment**: Python
* **UI Core Engine**: `tkinter` (Native window wrapper) and `ttkbootstrap` (Modern UI wrapper)
* **Mathematical Utilities**: Native `math` and `random` engines
* **Data Frame Parsing**: `pandas` (strictly utilized for initial CSV file parsing and structural column dropping)
