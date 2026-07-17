# Contributing to ProxiMed

First off, thank you for taking the time to contribute! Contributions make the open-source community an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

---

## 1. Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please report unacceptable behavior to the project maintainers.

---

## 2. How Can I Contribute?

### Reporting Bugs
- Check the GitHub Issues tab to ensure the bug hasn't already been reported.
- Open a new issue using the **Bug Report** template.
- Provide a clear description, steps to reproduce, actual vs expected results, and environment details.

### Suggesting Enhancements
- Open a new issue using the **Feature Request** template.
- Explain the proposed feature, why it is useful, and the use case.
- Discuss implementation options with the community before writing code.

### Pull Requests
1. Fork the repository.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR-USERNAME/ProxiMed.git
   ```
3. Create a branch for your feature or bug fix:
   ```bash
   git checkout -b feature/amazing-feature
   ```
4. Set up your virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
5. Obtain the CDC BRFSS dataset from [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) and place `diabetes_binary_5050split_2015_clean.csv` (or the complete Clean dataset) into the `data/` folder renamed as `database.csv`.
6. Make your changes.
7. Run `python src/robot.py` to verify predictions and check model behavior.
8. Commit your changes using conventional messages:
   - `feat: add new metric`
   - `fix: resolve UI layout glitch`
   - `docs: update setup steps`
9. Push to your branch and open a Pull Request against the `main` branch.

---

## 3. Coding Style & Development Guidelines

- **PEP 8 Compliance**: Follow standard Python style guidelines.
- **De-coupled Architecture**: Ensure business logic in `src/model.py` remains completely separated from presentation in `src/main.py`.
- **No External ML Libraries**: To maintain the "built from scratch" focus of the project, do not introduce scikit-learn, PyTorch, TensorFlow, or other ML modeling dependencies.
- **Documentation**: Keep `docs/` and inline docstrings updated for any modified mathematical or UI modules.
