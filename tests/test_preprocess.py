"""
Tests for preprocessing and data splitting module.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from src.preprocess import load_data, load_processed_data, preprocess_and_save


def test_preprocess_and_split_sizes():
    """
    Test that preprocess_and_save produces split sizes corresponding to ~460/154/154
    and that data files and scaler are saved properly.
    """
    X_train, y_train, X_cal, y_cal, X_test, y_test, scaler = preprocess_and_save()

    total_samples = 768
    assert len(X_train) == 460
    assert len(X_cal) == 154
    assert len(X_test) == 154
    assert len(X_train) + len(X_cal) + len(X_test) == total_samples

    # Check file artifacts exist
    assert Path("models/scaler.joblib").exists()
    assert Path("data/processed/X_train.csv").exists()
    assert Path("data/processed/y_train.csv").exists()
    assert Path("data/processed/X_cal.csv").exists()
    assert Path("data/processed/y_cal.csv").exists()
    assert Path("data/processed/X_test.csv").exists()
    assert Path("data/processed/y_test.csv").exists()


def test_no_zeros_remain_in_imputed_columns():
    """
    Test that no zeros remain in Glucose, BloodPressure, SkinThickness, Insulin, BMI
    prior to standard scaling / in the imputed splits.
    """
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    X_train, y_train, X_cal, y_cal, X_test, y_test, scaler = preprocess_and_save()

    # Reconstruct unscaled features using inverse transform
    X_train_unscaled = pd.DataFrame(scaler.inverse_transform(X_train), columns=X_train.columns)
    X_cal_unscaled = pd.DataFrame(scaler.inverse_transform(X_cal), columns=X_cal.columns)
    X_test_unscaled = pd.DataFrame(scaler.inverse_transform(X_test), columns=X_test.columns)

    for df in [X_train_unscaled, X_cal_unscaled, X_test_unscaled]:
        for col in zero_cols:
            assert not (df[col] == 0).any(), f"Zero value found in column {col}"


def test_load_processed_data():
    """
    Test loading processed data from CSV files.
    """
    preprocess_and_save()
    X_train, y_train, X_cal, y_cal, X_test, y_test = load_processed_data()

    assert X_train.shape == (460, 8)
    assert X_cal.shape == (154, 8)
    assert X_test.shape == (154, 8)
    assert len(y_train) == 460
    assert len(y_cal) == 154
    assert len(y_test) == 154
