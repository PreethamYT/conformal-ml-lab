"""
Preprocessing module for Diabetes dataset.
Handles data loading, missing value imputation, scaling, and stratified train/cal/test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath="data/diabetes.csv"):
    """
    Load the diabetes dataset from CSV.
    """
    return pd.read_csv(filepath)


def preprocess_data(df):
    """
    Preprocess the diabetes dataset:
    1. Replace impossible zero values in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
       with the median of non-zero values.
    2. Separate features (X) and target (Outcome).
    3. Scale features using StandardScaler.

    Parameters:
        df (pd.DataFrame): Raw diabetes dataset.

    Returns:
        tuple: (X_scaled, y, scaler) where X_scaled is a DataFrame of scaled features,
               y is the target series, and scaler is the fitted StandardScaler instance.
    """
    df = df.copy()
    zero_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    # Replace impossible 0 values with NaN, then fill with column median
    for col in zero_columns:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]

    # Scale feature columns
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled_df, y, scaler


def split_data(X, y, random_state=42):
    """
    Perform a stratified 60/20/20 split into train, calibration, and test sets.

    Parameters:
        X (pd.DataFrame or np.ndarray): Feature dataset.
        y (pd.Series or np.ndarray): Target dataset.
        random_state (int): Random seed for reproducibility (default 42).

    Returns:
        tuple: (X_train, X_cal, X_test, y_train, y_cal, y_test)
    """
    # First split: 60% train, 40% temporary (for calibration + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=random_state
    )

    # Second split: 50% of 40% = 20% calibration, 20% test
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )

    return X_train, X_cal, X_test, y_train, y_cal, y_test


def get_preprocessed_splits(filepath="data/diabetes.csv", random_state=42):
    """
    Convenience function to load, preprocess, and split the diabetes dataset.

    Returns:
        tuple: (X_train, X_cal, X_test, y_train, y_cal, y_test, scaler)
    """
    df = load_data(filepath)
    X, y, scaler = preprocess_data(df)
    X_train, X_cal, X_test, y_train, y_cal, y_test = split_data(X, y, random_state=random_state)
    return X_train, X_cal, X_test, y_train, y_cal, y_test, scaler


if __name__ == "__main__":
    X_train, X_cal, X_test, y_train, y_cal, y_test, scaler = get_preprocessed_splits()
    print(f"Train shape: {X_train.shape}, y_train counts:\n{y_train.value_counts(normalize=True)}")
    print(f"Calibration shape: {X_cal.shape}, y_cal counts:\n{y_cal.value_counts(normalize=True)}")
    print(f"Test shape: {X_test.shape}, y_test counts:\n{y_test.value_counts(normalize=True)}")
