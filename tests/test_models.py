"""
Tests for model training and model artifacts.
"""

from pathlib import Path
import joblib
import pytest

from src.models import train_and_save_models
from src.preprocess import load_processed_data


def test_train_and_save_models():
    """
    Test that 5 model files are saved and produce valid predictions.
    """
    models, accuracies = train_and_save_models()

    model_names = [
        "logistic_regression",
        "decision_tree",
        "gaussian_nb",
        "knn",
        "random_forest",
    ]

    X_train, y_train, X_cal, y_cal, X_test, y_test = load_processed_data()

    for name in model_names:
        model_file = Path(f"models/{name}.joblib")
        assert model_file.exists()

        model = joblib.load(model_file)
        preds = model.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1})
        assert accuracies[name] > 0.5
