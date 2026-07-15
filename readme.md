# Project Portfolio: Diabetes Risk Prediction via Custom K-Nearest Neighbors (KNN) and Interactive GUI

---

## 1. Project Overview

Diabetes is a highly prevalent chronic condition globally. Because clinical diagnostics require invasive blood draws, a large portion of the high-risk population remains undiagnosed until complications arise. 

This project develops an end-to-end medical screening application. It utilizes a **custom-built K-Nearest Neighbors (KNN) algorithm** to predict whether a patient has a diabetic or prediabetic risk based purely on self-reported lifestyle, demographic, and clinical status indicators. 

The software system is split into an elegant, decoupled architecture:
* **`model.py`**: A pure Python machine learning engine written without high-level ML libraries (like Scikit-Learn) to handle custom normalization, Euclidean distance calculation, nearest neighbor voting, and manual metric evaluation.
* **`main.py`**: A state-of-the-art interactive graphical user interface (GUI) developed in `tkinter` with modern `ttkbootstrap` styles to gather patient inputs via logical radio buttons/text fields and output real-time risk evaluations.
* **`robot.py`**: An automated pipeline script engineered to pass structured mockup vectors into the model for rapid testing and performance verification.

---

## 2. Dataset Architecture

The project processes data from the **CDC Behavioral Risk Factor Surveillance System (BRFSS)**. The program drops non-informative indicators (`CholCheck`, `AnyHealthcare`, `NoDocbcCost`) to focus strictly on the **18 core features** parsed below.

### Feature Mapping Table

| Feature | Input UI Control | Type | Description |
| :--- | :--- | :--- | :--- |
| **Diabetes_binary** | *Target Variable* | Binary | 0 = Healthy; 1 = Diabetic / Prediabetic |
| **HighBP** | Radio Buttons (Yes/No) | Binary | High Blood Pressure status |
| **HighChol** | Radio Buttons (Yes/No) | Binary | High Cholesterol status |
| **BMI** | Input Field | Numeric | Body Mass Index value |
| **Smoker** | Radio Buttons (Yes/No) | Binary | Smoked at least 100 cigarettes in lifetime |
| **Stroke** | Radio Buttons (Yes/No) | Binary | History of stroke |
| **HeartDiseaseorAttack** | Radio Buttons (Yes/No) | Binary | History of coronary heart disease or myocardial infarction |
| **PhysActivity** | Radio Buttons (Yes/No) | Binary | Physical activity within the past 30 days |
| **Fruits** | Radio Buttons (Yes/No) | Binary | Consumes fruit 1 or more times per day |
| **Veggies** | Radio Buttons (Yes/No) | Binary | Consumes vegetables 1 or more times per day |
| **HvyAlcoholConsump** | Radio Buttons (Yes/No) | Binary | Heavy alcohol consumption status |
| **GenHlth** | Input Field | Ordinal | Self-reported general health scale (1 = Excellent to 5 = Poor) |
| **MentHlth** | Input Field | Numeric | Number of bad mental health days in the past 30 days (0–30) |
| **PhysHlth** | Input Field | Numeric | Number of bad physical health days in the past 30 days (0–30) |
| **DiffWalk** | Radio Buttons (Yes/No) | Binary | Serious difficulty walking or climbing stairs |
| **Sex** | Radio Buttons (Yes/No) | Binary | Biological sex |
| **Age** | Input Field | Ordinal | Age category scale (1–13 representing 5-year bands) |
| **Education** | Input Field | Ordinal | Completed education level scale (1–6) |
| **Income** | Input Field | Ordinal | Household income scale (1–8) |

---

## 3. Mathematical Methodology

To ensure deep theoretical understanding, **all mathematical operations and evaluation metrics are written completely from scratch** without using Scikit-Learn, NumPy, or other modeling libraries.

### A. Feature Normalization
Because KNN relies on spatial distance, features with larger mathematical ranges (e.g., BMI ranging from 12 to 98) would disproportionately overpower binary features (e.g., Smoker ranging from 0 to 1). To resolve this, we apply Min-Max Normalization to scale every feature into a uniform [0, 1] interval:

`scaled_x = (x - min_x) / (max_x - min_x)`

The scaling parameters (min_x, max_x) are derived dynamically from the training corpus and subsequently applied to patient input vectors during prediction to preserve mathematical parity.

### B. Euclidean Distance
To determine proximity, the distance between a target patient vector `p` and a database vector `d` is computed across `n` features:

`distance = sqrt( sum( (p_i - d_i)^2 ) )`

### C. Evaluation Metrics
Instead of simple accuracy (which fails to depict model strength on highly imbalanced clinical datasets), the engine evaluates model performance using five metrics built from a confusion matrix of True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN):

* **Accuracy**: The overall proportion of correct classifications.
  `Accuracy = (TP + TN) / (TP + TN + FP + FN)`
* **Precision**: The reliability of positive risk flags.
  `Precision = TP / (TP + FP)`
* **Recall (Sensitivity)**: The ability of the model to catch diabetic cases (crucial in clinical screening to minimize FN rates).
  `Recall = TP / (TP + FN)`
* **F1-Score**: The harmonic mean of precision and recall.
  `F1-Score = 2 * (Precision * Recall) / (Precision + Recall)`
* **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: Approximated mathematically by sorting predicted probability outputs, calculating the True Positive Rate (TPR) and False Positive Rate (FPR) across multiple thresholds, and integrating the area using the trapezoidal rule:
  `AUC = sum( (FPR_i - FPR_i-1) * (TPR_i + TPR_i-1) / 2 )`

To prevent UI freezes when parsing thousands of high-dimensional records on single-threaded Python runs, metrics are computed dynamically on a randomized, representative validation subset (n = 50).

---

## 4. Software Architecture

### System Dataflow

[Patient Data Input] (main.py GUI)
│
▼ (Passes Dict)
[Normalization Engine] (model.py) ──> Scales raw values to [0,1] based on dataset mins/maxs
│
▼ (Passes Vector)
[Distance Calculator] (model.py) ───> Computes Euclidean distance against database records
│
▼ (Selects k-closest)
[Majority Voting] (model.py) ────────> Identifies nearest neighbors & calculates probability
│
▼ (Dynamic Evaluation)
[Metrics Calculator] (model.py) ─────> Computes Accuracy, Precision, Recall, F1, and ROC-AUC
│
▼ (Returns Dict)
[Dashboard Display] (main.py GUI) ───> Renders Diagnostic Card (Green=Healthy, Red=Risk)


### Script Directory

#### 1. `model.py`
This module acts as the core mathematical backend. It reads `database.csv`, separates labels from features, handles standard min-max scaling vector transformations, sorts closest neighbors to vote on classifications, and executes custom metric evaluations.

#### 2. `main.py`
A modern dashboard UI developed using `tkinter` and `ttkbootstrap`.
* **Dynamic Inputs**: Parses the model's feature headers, generating binary "Yes/No" Radiobuttons for the 11 binary variables, and input boxes for the 7 numerical attributes.
* **Scrollable Workspace**: Embeds inputs in an organized canvas frame with a custom vertical scrollbar.
* **Responsive Metrics Panel**: Visualizes diagnostic predictions dynamically. A positive classification highlights the metrics card in vibrant red (`danger` theme), while a healthy classification flashes in bright green (`success` theme).

#### 3. `robot.py`
An automated test script designed for headless validation. It programmatically maps predefined list vectors to the features list, queries the `CustomKNN` object, and prints raw output predictions to verify terminal stability.

---

## 5. Technology Stack

* **Programming Language**: Python
* **Data Processing**: Pandas (for initial CSV reading and column isolation)
* **Mathematical Libraries**: Native `math` and `random` engines
* **Graphical User Interface**: `tkinter` (Native Python GUI framework)
* **UI Customization Styling**: `ttkbootstrap` (Modern widget styling library)
