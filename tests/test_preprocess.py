"""
Tests for preprocessing and data splitting module.
"""

from pathlib import Path
import pandas as pd

from src.preprocess import load_processed_data, preprocess_and_save


def test_preprocess_and_split_sizes():
    """
    Test that preprocess_and_save produces split sizes corresponding to
    ~460/154/154 and that data files and scaler are saved properly.
    """
    X_tr, y_tr, X_c, y_c, X_te, y_te, scaler = preprocess_and_save()

    total_samples = 768
    assert len(X_tr) == 460
    assert len(X_c) == 154
    assert len(X_te) == 154
    assert len(X_tr) + len(X_c) + len(X_te) == total_samples

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
    Test that no zeros remain in Glucose, BloodPressure, SkinThickness,
    Insulin, BMI prior to standard scaling / in the imputed splits.
    """
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    X_tr, y_tr, X_c, y_c, X_te, y_te, scaler = preprocess_and_save()

    # Reconstruct unscaled features using inverse transform
    X_tr_un = pd.DataFrame(
        scaler.inverse_transform(X_tr), columns=X_tr.columns
    )
    X_c_un = pd.DataFrame(
        scaler.inverse_transform(X_c), columns=X_c.columns
    )
    X_te_un = pd.DataFrame(
        scaler.inverse_transform(X_te), columns=X_te.columns
    )

    for df in [X_tr_un, X_c_un, X_te_un]:
        for col in zero_cols:
            assert not (df[col] == 0).any(), f"Zero in column {col}"


def test_load_processed_data():
    """
    Test loading processed data from CSV files.
    """
    preprocess_and_save()
    X_tr, y_tr, X_c, y_c, X_te, y_te = load_processed_data()

    assert X_tr.shape == (460, 8)
    assert X_c.shape == (154, 8)
    assert X_te.shape == (154, 8)
    assert len(y_tr) == 460
    assert len(y_c) == 154
    assert len(y_te) == 154
