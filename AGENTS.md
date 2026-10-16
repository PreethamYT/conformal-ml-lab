# Project: Conformal Prediction for Classifiers (ML Lab)
Goal: add a split-conformal reliability layer to 5 classifiers and ship a Streamlit demo.
Stack: Python 3.11, pandas, numpy, scikit-learn, matplotlib, joblib, streamlit.
Structure: data/, src/ (preprocess.py, models.py, conformal.py, evaluate.py, plots.py),
app/streamlit_app.py, models/, outputs/, tests/.
Rules: keep code simple and commented for students, use random_state=42,
split 60/20/20 stratified (train/calibration/test), no GPU or heavy libraries.
Conformal method: score = 1 - p(true class) on calibration set; threshold = ceil((n+1)(1-alpha))/n quantile;
prediction set = classes with prob >= 1 - threshold.
Run tests with: pytest -q
