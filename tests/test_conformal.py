"""
Tests for split-conformal prediction implementation.
"""

from pathlib import Path
import joblib
import numpy as np
import pytest

from src.conformal import conformal_sets, get_threshold
from src.preprocess import load_processed_data


@pytest.fixture
def data():
    return load_processed_data()


@pytest.fixture
def models():
    model_dir = Path("models")
    loaded_models = {}
    model_files = [
        "logistic_regression",
        "decision_tree",
        "gaussian_nb",
        "knn",
        "random_forest",
    ]
    for name in model_files:
        model_path = model_dir / f"{name}.joblib"
        assert model_path.exists(), f"Model artifact {name}.joblib missing"
        loaded_models[name] = joblib.load(model_path)
    return loaded_models


def test_conformal_threshold_and_coverage(data, models):
    """
    Test that for each model:
    1. get_threshold returns a valid numeric threshold.
    2. conformal_sets returns boolean matrix of shape (n_test, n_classes).
    3. Test set coverage is within 0.08 of (1 - alpha).
    4. Prediction set size is never larger than number of classes (n_classes).
    """
    X_train, y_train, X_cal, y_cal, X_test, y_test = data
    alpha = 0.1
    target_coverage = 1.0 - alpha
    n_classes = len(np.unique(y_train))

    for name, model in models.items():
        thresh = get_threshold(model, X_cal, y_cal, alpha)
        assert 0.0 <= thresh <= 1.0, f"Threshold out of bounds for {name}: {thresh}"

        sets = conformal_sets(model, X_cal, y_cal, X_test, alpha)

        # Check shape and type
        assert sets.shape == (len(X_test), n_classes)
        assert sets.dtype == bool

        # Check prediction set size constraint
        set_sizes = sets.sum(axis=1)
        assert (set_sizes <= n_classes).all(), f"Prediction set exceeds n_classes for {name}"
        assert (set_sizes >= 0).all()

        # Check coverage on test split
        y_test_arr = np.asarray(y_test, dtype=int)
        covered = sets[np.arange(len(y_test_arr)), y_test_arr]
        coverage = np.mean(covered)

        diff = abs(coverage - target_coverage)
        assert diff <= 0.08, f"Model {name} coverage {coverage:.4f} is not within 0.08 of target {target_coverage}"
