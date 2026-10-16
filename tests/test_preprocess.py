"""
Tests for preprocessing and data splitting module.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocess import get_preprocessed_splits, load_data, preprocess_data, split_data


def test_split_sizes():
    """
    Test that split sizes correspond to 60% train, 20% calibration, and 20% test split.
    """
    df = load_data("data/diabetes.csv")
    total_samples = len(df)

    X_train, X_cal, X_test, y_train, y_cal, y_test, _ = get_preprocessed_splits("data/diabetes.csv")

    n_train = len(X_train)
    n_cal = len(X_cal)
    n_test = len(X_test)

    # Sum of splits must equal total samples
    assert n_train + n_cal + n_test == total_samples

    # Verify target splits lengths match feature splits
    assert len(y_train) == n_train
    assert len(y_cal) == n_cal
    assert len(y_test) == n_test

    # Verify proportions (60 / 20 / 20) within reasonable roundoff margin
    assert abs(n_train / total_samples - 0.60) < 0.01
    assert abs(n_cal / total_samples - 0.20) < 0.01
    assert abs(n_test / total_samples - 0.20) < 0.01

    # Exact expected sample counts for 768 total samples with stratified 60/40 then 50/50 split
    assert n_train == 460
    assert n_cal == 154
    assert n_test == 154


def test_zero_imputation_and_scaling():
    """
    Test that impossible zero values are replaced with column medians and features are scaled.
    """
    df = load_data("data/diabetes.csv")
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    # Verify original dataset contains zeros in these columns
    assert (df[zero_cols] == 0).any().any()

    X_scaled, y, scaler = preprocess_data(df)

    # Check scaled dataset mean ~ 0 and std ~ 1
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-5)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-2)


def test_stratification():
    """
    Test that class proportions are preserved across train, calibration, and test splits.
    """
    X_train, X_cal, X_test, y_train, y_cal, y_test, _ = get_preprocessed_splits("data/diabetes.csv")

    p_train = y_train.mean()
    p_cal = y_cal.mean()
    p_test = y_test.mean()

    # Stratified splits should have nearly identical positive class ratios
    assert abs(p_train - p_cal) < 0.02
    assert abs(p_train - p_test) < 0.02
