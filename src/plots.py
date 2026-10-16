"""
Visualization module for conformal prediction results and analysis.
Generates bar charts for coverage and average set size, set-size histogram for Random Forest,
and a 2D PCA scatter plot highlighting uncertain prediction points.
"""

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.conformal import conformal_sets
from src.preprocess import load_processed_data


def generate_plots(results_csv="outputs/results.csv", output_dir="outputs"):
    """
    Generate and save all required PNG visualizations into output_dir.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results_file = Path(results_csv)
    if not results_file.exists():
        from src.evaluate import evaluate_models
        df_results = evaluate_models(output_dir=output_dir)
    else:
        df_results = pd.read_csv(results_file)

    # 1. Coverage Bar Chart
    plt.figure(figsize=(10, 6))
    models = df_results["model"].unique()
    alphas = df_results["alpha"].unique()
    x = np.arange(len(models))
    width = 0.35

    df_a1 = df_results[df_results["alpha"] == 0.1].set_index("model").reindex(models)
    df_a2 = df_results[df_results["alpha"] == 0.2].set_index("model").reindex(models)

    plt.bar(x - width/2, df_a1["coverage"], width, label="alpha = 0.1 (Target 90%)")
    plt.bar(x + width/2, df_a2["coverage"], width, label="alpha = 0.2 (Target 80%)")
    plt.axhline(0.90, color="blue", linestyle="--", alpha=0.7, label="Target 90%")
    plt.axhline(0.80, color="orange", linestyle="--", alpha=0.7, label="Target 80%")

    plt.xlabel("Model")
    plt.ylabel("Coverage")
    plt.title("Coverage by Model and Alpha")
    plt.xticks(x, models, rotation=15)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "coverage_bar_chart.png", dpi=300)
    plt.close()

    # 2. Avg Set Size Bar Chart
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, df_a1["avg_set_size"], width, label="alpha = 0.1")
    plt.bar(x + width/2, df_a2["avg_set_size"], width, label="alpha = 0.2")

    plt.xlabel("Model")
    plt.ylabel("Average Set Size")
    plt.title("Average Prediction Set Size by Model and Alpha")
    plt.xticks(x, models, rotation=15)
    plt.ylim(0, 2.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "avg_set_size_bar_chart.png", dpi=300)
    plt.close()

    # Load data and RF model for RF specific plots
    X_train, y_train, X_cal, y_cal, X_test, y_test = load_processed_data()
    rf_model_path = Path("models/random_forest.joblib")
    if not rf_model_path.exists():
        from src.models import train_and_save_models
        train_and_save_models()

    rf_model = joblib.load(rf_model_path)
    sets_rf_01 = conformal_sets(rf_model, X_cal, y_cal, X_test, alpha=0.1)
    set_sizes_rf = sets_rf_01.sum(axis=1)

    # 3. Set-Size Histogram for Random Forest at alpha = 0.1
    plt.figure(figsize=(8, 5))
    counts = pd.Series(set_sizes_rf).value_counts().sort_index()
    possible_sizes = [0, 1, 2]
    freqs = [counts.get(s, 0) for s in possible_sizes]

    plt.bar([str(s) for s in possible_sizes], freqs, color="skyblue", edgecolor="black")
    plt.xlabel("Prediction Set Size")
    plt.ylabel("Count")
    plt.title("Random Forest Prediction Set Size Distribution (alpha = 0.1)")
    plt.tight_layout()
    plt.savefig(out_path / "rf_set_size_histogram.png", dpi=300)
    plt.close()

    # 4. PCA 2D Scatter of Test Set with Uncertain Points Highlighted
    pca = PCA(n_components=2, random_state=42)
    X_test_pca = pca.fit_transform(X_test)

    uncertain_mask = (set_sizes_rf != 1)

    plt.figure(figsize=(9, 6))
    plt.scatter(
        X_test_pca[~uncertain_mask, 0],
        X_test_pca[~uncertain_mask, 1],
        c="lightgray",
        label="Certain (Set Size = 1)",
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5
    )
    plt.scatter(
        X_test_pca[uncertain_mask, 0],
        X_test_pca[uncertain_mask, 1],
        c="red",
        marker="X",
        s=70,
        label="Uncertain (Set Size != 1)",
        alpha=0.9
    )

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("PCA 2D Projection of Test Set with RF Uncertain Points (alpha = 0.1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path / "pca_uncertainty_scatter.png", dpi=300)
    plt.close()

    print("All plots generated successfully in outputs/ directory.")


if __name__ == "__main__":
    generate_plots()
