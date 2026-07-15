# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-07-15

### Added
- Created `description.md` providing in-depth mathematical documentation of the custom KNN algorithm.
- Added standard `LICENSE.md` using the MIT license framework.
- Added `DISCLAIMER.md` stating clear limits of medical liability and educational parameters.
- Added `.gitignore` to block cached Python compilation items, virtual environment files, and local CSV copies.
- Added `requirements.txt` containing dependencies (`pandas`, `ttkbootstrap`).
- Created standardized project management files (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`).

### Changed
- Refactored UI layout in `main.py` from a packed vertical orientation to a robust, responsive two-column grid.
- Isolated patient form inputs in a cleanly scrollable left-hand workspace frame.
- Consolidated metrics visualization, calculation warnings, and execution controls into a high-contrast right-hand diagnostic panel.
- Modified dynamic warning card styles to flash `danger` (red) for high diabetic risk predictions and `success` (green) for healthy results.

---

## [1.0.0] - 2026-07-10

### Added
- Implemented core mathematical backend in `model.py` with custom Min-Max scaling, Euclidean distance calculations, and diagnostic metrics (Accuracy, Precision, Recall, F1, and ROC-AUC) written from scratch without Scikit-Learn.
- Built interactive desktop application interface in `main.py` using `tkinter` and `ttkbootstrap` to collect 18 unique patient health survey variables.
- Created `robot.py` automated pipeline script to programmatically pass mock inputs and verify prediction vectors.
- Configured initial raw dataset link for Behavioral Risk Factor Surveillance System (BRFSS) data.
