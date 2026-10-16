"""
Evaluation module for model accuracy and split-conformal metrics.
Generates outputs/results.csv containing accuracy, coverage, avg set size,
and uncertain_pct for alpha in [0.1, 0.2].
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.conformal import conformal_sets  # noqa: E402
from src.preprocess import load_processed_data  # noqa: E402


def evaluate_models(
    data_dir="data/processed",
    models_dir="models",
    output_dir="outputs",
    alphas=[0.1, 0.2],
):
    """
    Evaluate trained models and split-conformal prediction metrics.

    Returns:
        pd.DataFrame: Results table with columns:
                      [model, alpha, accuracy, coverage,
                       avg_set_size, uncertain_pct]
    """
    X_train, y_train, X_cal, y_cal, X_test, y_test = load_processed_data(
        data_dir
    )
    y_test_arr = np.asarray(y_test, dtype=int)

    model_names = [
        "logistic_regression",
        "decision_tree",
        "gaussian_nb",
        "knn",
        "random_forest",
    ]

    results = []
    models_path = Path(models_dir)

    for name in model_names:
        model_file = models_path / f"{name}.joblib"
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file {model_file} not found. Run src/models.py first."
            )

        model = joblib.load(model_file)
        preds = model.predict(X_test)
        accuracy = np.mean(preds == y_test_arr)

        for alpha in alphas:
            sets = conformal_sets(model, X_cal, y_cal, X_test, alpha)
            set_sizes = sets.sum(axis=1)

            # Coverage: true class in prediction set
            covered = sets[np.arange(len(y_test_arr)), y_test_arr]
            coverage = float(np.mean(covered))

            # Avg set size
            avg_set_size = float(np.mean(set_sizes))

            # Uncertain pct: set size != 1
            uncertain_pct = float(np.mean(set_sizes != 1))

            results.append({
                "model": name,
                "alpha": float(alpha),
                "accuracy": float(accuracy),
                "coverage": float(coverage),
                "avg_set_size": float(avg_set_size),
                "uncertain_pct": float(uncertain_pct)
            })

    df_results = pd.DataFrame(results)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_file = out_path / "results.csv"
    df_results.to_csv(csv_file, index=False)
    print(f"Evaluation results saved to {csv_file}")

    return df_results


if __name__ == "__main__":
    df = evaluate_models()
    print(df.to_string())
