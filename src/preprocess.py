"""
Preprocessing module for Diabetes dataset.
Handles data loading, stratified train/calibration/test splitting,
train-only median imputation for zero values, standard scaling, and saving splits.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath="data/diabetes.csv"):
    """
    Load the diabetes dataset from CSV.
    """
    return pd.read_csv(filepath)


def preprocess_and_save(filepath="data/diabetes.csv", output_dir="data/processed", models_dir="models", random_state=42):
    """
    Execute the full preprocessing pipeline:
    1. Load raw data and perform stratified 60/20/20 train/calibration/test split first.
    2. Treat 0s in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'] as missing.
    3. Compute feature medians from train split only and apply them to all splits.
    4. Fit StandardScaler on train split only and scale all splits.
    5. Save fitted scaler to models/scaler.joblib.
    6. Save preprocessed splits to data/processed/.

    Returns:
        tuple: (X_train, y_train, X_cal, y_cal, X_test, y_test, scaler)
    """
    df = load_data(filepath)

    X = df.drop(columns=["Outcome"]).copy()
    y = df["Outcome"].copy()

    # Step 1: Stratified 60/20/20 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=random_state
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )

    # Step 2 & 3: Median imputation from TRAIN only for zero-impossible columns
    zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    for col in zero_columns:
        if col in X.columns:
            X_train[col] = X_train[col].replace(0, np.nan)
            X_cal[col] = X_cal[col].replace(0, np.nan)
            X_test[col] = X_test[col].replace(0, np.nan)

    # Compute medians from train set
    train_medians = X_train[zero_columns].median()

    # Fill NaNs across all splits with train medians
    X_train = X_train.fillna(train_medians)
    X_cal = X_cal.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    # Step 4: Fit StandardScaler on TRAIN only
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns
    )
    X_cal_scaled = pd.DataFrame(
        scaler.transform(X_cal), columns=X_cal.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns
    )

    # Reset indices for target series
    y_train = y_train.reset_index(drop=True)
    y_cal = y_cal.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    # Step 5: Save scaler
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_path / "scaler.joblib")

    # Step 6: Save processed splits
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    X_train_scaled.to_csv(out_path / "X_train.csv", index=False)
    y_train.to_csv(out_path / "y_train.csv", index=False)
    X_cal_scaled.to_csv(out_path / "X_cal.csv", index=False)
    y_cal.to_csv(out_path / "y_cal.csv", index=False)
    X_test_scaled.to_csv(out_path / "X_test.csv", index=False)
    y_test.to_csv(out_path / "y_test.csv", index=False)

    return X_train_scaled, y_train, X_cal_scaled, y_cal, X_test_scaled, y_test, scaler


def load_processed_data(data_dir="data/processed"):
    """
    Load preprocessed data splits from CSV files.
    """
    path = Path(data_dir)
    X_train = pd.read_csv(path / "X_train.csv")
    y_train = pd.read_csv(path / "y_train.csv").squeeze("columns")
    X_cal = pd.read_csv(path / "X_cal.csv")
    y_cal = pd.read_csv(path / "y_cal.csv").squeeze("columns")
    X_test = pd.read_csv(path / "X_test.csv")
    y_test = pd.read_csv(path / "y_test.csv").squeeze("columns")
    return X_train, y_train, X_cal, y_cal, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_cal, y_cal, X_test, y_test, _ = preprocess_and_save()
    print(f"Processed splits saved successfully.")
    print(f"X_train shape: {X_train.shape}, X_cal shape: {X_cal.shape}, X_test shape: {X_test.shape}")
