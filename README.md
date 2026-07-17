![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Last Commit](https://img.shields.io/github/last-commit/Armencho2012/ProxiMed)
![Stars](https://img.shields.io/github/stars/Armencho2012/ProxiMed?style=social)
![Forks](https://img.shields.io/github/forks/Armencho2012/ProxiMed?style=social)
![Issues](https://img.shields.io/github/issues/Armencho2012/ProxiMed)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

# 🏥 ProxiMed

**Diabetes Risk Prediction using Machine Learning — Built Entirely from Scratch**

---

ProxiMed is a machine learning–powered diabetes risk prediction system that uses a **custom K-Nearest Neighbors (KNN) algorithm written entirely from scratch in pure Python** — no scikit-learn, no TensorFlow, no ML libraries. Trained on **253,680+ real CDC health records** from the Behavioral Risk Factor Surveillance System (BRFSS), it delivers real-time risk assessments through a sleek, interactive desktop GUI. Every component of the ML pipeline — from min-max normalization and Euclidean distance computation to majority voting and evaluation metrics — is hand-built, making ProxiMed both a powerful predictive analytics tool and a transparent educational resource for understanding how classification algorithms work under the hood.

---

## 📸 Screenshots

<!-- Add your screenshots here -->

![ProxiMed GUI](assets/gui.png)

---

## ✨ Features

- 🧮 **100% Custom KNN Algorithm** — No scikit-learn dependency; every mathematical operation is implemented from scratch
- 🏥 **Real-Time Diabetes Risk Prediction** — Instant classification from 18 health indicators for early detection
- 🖥️ **Modern Interactive Desktop GUI** — Built with tkinter + ttkbootstrap (superhero theme), featuring scrollable patient forms and dynamic diagnostic cards
- 📊 **Built-in Model Evaluation Engine** — Accuracy, Precision, Recall, F1 Score, and ROC-AUC computed without external libraries
- 🤖 **Automated Testing Pipeline** — Headless verification suite for reproducible model validation
- 📈 **Trained on 253,680 CDC BRFSS Records** — Leverages one of the largest publicly available diabetes health indicator datasets
- 🔒 **Medical Disclaimer & Privacy-Conscious Design** — Responsible AI practices with clear usage guidelines
- 📖 **Comprehensive Documentation** — Architecture docs, methodology write-ups, and evaluation reports included

---

## 🏗️ Architecture

ProxiMed follows a clean three-module architecture, separating concerns between the machine learning engine, the user interface, and the testing infrastructure:

| Module | File | Description |
|--------|------|-------------|
| **ML Engine** | `src/model.py` | The heart of ProxiMed — a custom KNN classifier with min-max normalization, Euclidean distance calculation, majority voting, and a full confusion-matrix metrics suite (Accuracy, Precision, Recall, F1, ROC-AUC), all implemented from scratch |
| **Desktop GUI** | `src/main.py` | An interactive patient intake form built with tkinter and ttkbootstrap's superhero theme; features scrollable input fields for 18 health indicators and dynamic diagnostic result cards (🔴 risk / 🟢 healthy) |
| **Test Pipeline** | `src/robot.py` | An automated headless testing harness that runs the model against validation samples and reports performance metrics without manual interaction |

---

## 📊 Model Performance

ProxiMed's custom KNN classifier achieves competitive performance on the CDC BRFSS dataset:

| Metric | Value |
|--------|-------|
| **Accuracy** | ~85% |
| **Precision** | ~83% |
| **Recall** | ~78% |
| **F1 Score** | ~80% |
| **ROC-AUC** | ~0.90 |

> **Note:** Metrics are evaluated dynamically on random 50-sample validation subsets drawn at runtime. Results may vary slightly between runs due to stochastic sampling, reflecting real-world model behavior.

---

## 📁 Project Structure

```
ProxiMed/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   └── pull_request_template.md
├── assets/
│   ├── gui.png
│   ├── prediction.png
│   ├── workflow.png
│   └── demo.gif
├── data/
│   └── database.csv
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   └── evaluation.md
├── src/
│   ├── main.py
│   ├── model.py
│   └── robot.py
├── .gitignore
├── CHANGELOG.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── DISCLAIMER.md
├── LICENSE
├── README.md
├── requirements.txt
└── SECURITY.md
```

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- `pip` package manager
- A Kaggle account (to download the dataset)

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Armencho2012/ProxiMed.git
   cd ProxiMed
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset:**

   Download the [Diabetes Health Indicators Dataset](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) from Kaggle and place the CSV file in the `data/` directory:

   ```
   data/database.csv
   ```

---

## 🚀 Usage

### Launch the GUI

Start the interactive desktop application to input patient health indicators and receive real-time diabetes risk predictions:

```bash
python src/main.py
```

Fill in the 18 health indicator fields in the scrollable form, then click **Predict**. The result card will display dynamically — 🔴 **red** for elevated diabetes risk, 🟢 **green** for healthy classification.

### Run Automated Tests

Execute the headless test pipeline to validate model performance against the dataset:

```bash
python src/robot.py
```

---

## 📂 Dataset

ProxiMed is trained on the **CDC Behavioral Risk Factor Surveillance System (BRFSS)** dataset, one of the most comprehensive public health survey datasets available for diabetes prediction research.

| Detail | Description |
|--------|-------------|
| **Source** | [Diabetes Health Indicators Dataset — Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) |
| **Origin** | CDC BRFSS annual health survey |
| **Records** | 253,680 survey responses |
| **Features** | 22 health indicators (18 used for prediction) |
| **Task** | Binary classification (diabetes / no diabetes) |
| **Format** | CSV |

The BRFSS is the nation's premier system of health-related telephone surveys, collecting data on U.S. residents regarding their health-related risk behaviors, chronic health conditions, and use of preventive services. This dataset distills survey responses into structured, machine-learning-ready features ideal for predictive analytics and health screening applications.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| **Python** | Core programming language |
| **pandas** | CSV data loading and preprocessing |
| **tkinter** | Native GUI framework |
| **ttkbootstrap** | Modern UI theming and styling |
| **math** (stdlib) | Mathematical computations |
| **random** (stdlib) | Validation sampling |

> 💡 **No ML libraries required.** The entire machine learning pipeline — normalization, distance computation, classification, and evaluation — is implemented from scratch using only Python's standard library and pandas for data I/O.

---

## 🗺️ Roadmap

- ✅ Custom KNN algorithm from scratch
- ✅ Interactive desktop GUI
- ✅ Model evaluation metrics engine
- ✅ Automated testing pipeline
- ⬜ Web application (Flask/Django)
- ⬜ REST API endpoint
- ⬜ Docker containerization
- ⬜ Cloud deployment (AWS/GCP)
- ⬜ Explainable AI (SHAP values)
- ⬜ Mobile application

---

## 🤝 Contributing

Contributions are welcome and appreciated! Whether it's a bug fix, feature request, or documentation improvement — every contribution helps make ProxiMed better.

Please read the [Contributing Guide](CONTRIBUTING.md) before submitting a pull request. We follow a standard fork → branch → PR workflow.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

**ProxiMed is a research and educational tool only.** It is **not** a substitute for professional medical advice, diagnosis, or treatment. Predictions generated by this system should never be used as the sole basis for clinical decisions. Always consult a qualified healthcare professional for medical guidance.

For the full disclaimer, see [DISCLAIMER.md](DISCLAIMER.md).

---

## 📖 Citation

If you use ProxiMed in your research or coursework, please cite it:

```bibtex
@software{proximed2025,
  author    = {Armen},
  title     = {ProxiMed: Diabetes Risk Prediction using Machine Learning},
  year      = {2025},
  url       = {https://github.com/Armencho2012/ProxiMed},
  license   = {MIT}
}
```

For the full citation metadata, see [CITATION.cff](CITATION.cff).

---

## 🔐 Security

Found a vulnerability? Please report it responsibly. See our [Security Policy](SECURITY.md) for details on how to disclose security issues.

---

## ⭐ Show Your Support

If ProxiMed helped you learn about machine learning, healthcare AI, or building classification systems from scratch — give it a **⭐ star**! It helps others discover the project and motivates continued development.

```
⭐ github.com/Armencho2012/ProxiMed
```

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/Armencho2012">Armen</a>
</p>