"""
Tests for visualization module.
"""

from pathlib import Path
import pytest

from src.plots import generate_plots


def test_generate_plots():
    """
    Test that generate_plots runs and creates the required PNG visualization files.
    """
    generate_plots()

    expected_files = [
        "coverage_bar_chart.png",
        "avg_set_size_bar_chart.png",
        "rf_set_size_histogram.png",
        "pca_uncertainty_scatter.png",
    ]

    out_dir = Path("outputs")
    for filename in expected_files:
        file_path = out_dir / filename
        assert file_path.exists(), f"Plot {filename} was not generated"
        assert file_path.stat().st_size > 0, f"Plot {filename} is empty"
