"""
Conformal prediction module using split-conformal classification.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np


def get_threshold(model, X_cal, y_cal, alpha):
    """
    Compute the conformal threshold quantile on the calibration set.

    Parameters:
        model: Trained classifier with predict_proba method.
        X_cal (pd.DataFrame or np.ndarray): Calibration features.
        y_cal (pd.Series or np.ndarray): Calibration targets.
        alpha (float): Significance level (e.g. 0.1 for 90% coverage target).

    Returns:
        float: Conformal quantile threshold score.
    """
    probs = model.predict_proba(X_cal)
    y_cal_arr = np.asarray(y_cal, dtype=int)
    n = len(y_cal_arr)

    # Score = 1 - p(true class)
    true_class_probs = probs[np.arange(n), y_cal_arr]
    scores = 1.0 - true_class_probs

    # Conformal quantile rank: ceil((n + 1) * (1 - alpha))
    k = int(np.ceil((n + 1) * (1.0 - alpha)))

    if k >= n:
        threshold = float(np.max(scores))
    elif k <= 0:
        threshold = float(np.min(scores))
    else:
        sorted_scores = np.sort(scores)
        threshold = float(sorted_scores[k - 1])

    return threshold


def conformal_sets(model, X_cal, y_cal, X_test, alpha):
    """
    Construct conformal prediction sets for test instances.

    Parameters:
        model: Trained classifier.
        X_cal (pd.DataFrame or np.ndarray): Calibration features.
        y_cal (pd.Series or np.ndarray): Calibration targets.
        X_test (pd.DataFrame or np.ndarray): Test features.
        alpha (float): Significance level.

    Returns:
        np.ndarray: Boolean matrix of shape (n_test_samples, n_classes)
                    where entry [i, c] is True if prob(class c) >= 1 - threshold.
    """
    threshold = get_threshold(model, X_cal, y_cal, alpha)
    test_probs = model.predict_proba(X_test)

    # Prediction set = classes with prob >= 1 - threshold
    pred_sets = test_probs >= (1.0 - threshold)
    return pred_sets


if __name__ == "__main__":
    import joblib
    from src.preprocess import load_processed_data

    X_train, y_train, X_cal, y_cal, X_test, y_test = load_processed_data()
    alpha = 0.1

    print(f"Testing conformal prediction with alpha = {alpha}...")
    for model_file in Path("models").glob("*.joblib"):
        if model_file.name == "scaler.joblib":
            continue
        model = joblib.load(model_file)
        thresh = get_threshold(model, X_cal, y_cal, alpha)
        sets = conformal_sets(model, X_cal, y_cal, X_test, alpha)

        cov = np.mean(sets[np.arange(len(y_test)), np.asarray(y_test, dtype=int)])
        avg_set_size = np.mean(sets.sum(axis=1))

        print(f"Model: {model_file.stem:<20} Threshold: {thresh:.4f} Coverage: {cov:.4f} Avg Set Size: {avg_set_size:.2f}")
