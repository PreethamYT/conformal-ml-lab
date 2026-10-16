"""
Tests for evaluate module and results output.
"""

from pathlib import Path
from src.evaluate import evaluate_models


def test_evaluate_models():
    """
    Test that evaluate_models runs, generates outputs/results.csv,
    and checks output schema.
    """
    df = evaluate_models()

    csv_file = Path("outputs/results.csv")
    assert csv_file.exists()

    expected_cols = [
        "model",
        "alpha",
        "accuracy",
        "coverage",
        "avg_set_size",
        "uncertain_pct",
    ]
    assert list(df.columns) == expected_cols

    assert len(df) == 10  # 5 models * 2 alphas
    assert set(df["alpha"].unique()) == {0.1, 0.2}

    # Verify metrics are within reasonable bounds
    assert ((df["accuracy"] >= 0.0) & (df["accuracy"] <= 1.0)).all()
    assert ((df["coverage"] >= 0.0) & (df["coverage"] <= 1.0)).all()
    assert ((df["avg_set_size"] >= 0.0) & (df["avg_set_size"] <= 2.0)).all()
    assert ((df["uncertain_pct"] >= 0.0) & (df["uncertain_pct"] <= 1.0)).all()
