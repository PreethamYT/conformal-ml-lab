# Project: Conformal Prediction for Classifiers (ML Lab)
Goal: add a split-conformal reliability layer to 5 classifiers and ship a Streamlit demo.
Stack: Python 3.11, pandas, numpy, scikit-learn, matplotlib, joblib, streamlit.

## Project Structure
- `data/`: Dataset storage (`diabetes.csv`).
- `src/`: Core logic modules
  - `preprocess.py`: Data loading, imputation, scaling, and stratified 60/20/20 splitting.
  - `models.py`: Classifier definitions and training.
  - `conformal.py`: Split-conformal calibration and prediction set construction.
  - `evaluate.py`: Evaluation metrics for models and conformal sets.
  - `plots.py`: Visualization routines.
- `app/`: Streamlit demo application (`streamlit_app.py`).
- `models/`: Directory for trained model artifacts.
- `outputs/`: Directory for generated plots and evaluation outputs.
- `tests/`: Unit test suite (`test_preprocess.py`).

## Rules
- Keep code simple and commented for students.
- Use `random_state=42`.
- Split 60/20/20 stratified (train/calibration/test).
- No GPU or heavy libraries.

## Conformal Method
- Score = 1 - p(true class) on calibration set.
- Threshold = ceil((n+1)(1-alpha))/n quantile.
- Prediction set = classes with prob >= 1 - threshold.

## Testing
- Run tests with: `pytest -q`
