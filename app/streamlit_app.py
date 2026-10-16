"""
Streamlit demonstration app for Conformal Prediction for Diabetes.
"""

from pathlib import Path
import sys

# Ensure root directory is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.conformal import get_threshold  # noqa: E402
from src.preprocess import load_processed_data  # noqa: E402


@st.cache_resource
def load_scaler_and_models(models_dir="models"):
    """
    Cache loading of scaler and trained models.
    """
    path = Path(models_dir)
    scaler_path = path / "scaler.joblib"
    if not scaler_path.exists():
        return None, {}

    scaler = joblib.load(scaler_path)

    model_files = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "Gaussian Naive Bayes": "gaussian_nb.joblib",
        "K-Nearest Neighbors": "knn.joblib",
        "Random Forest": "random_forest.joblib",
    }

    models = {}
    for display_name, filename in model_files.items():
        m_path = path / filename
        if m_path.exists():
            models[display_name] = joblib.load(m_path)

    return scaler, models


@st.cache_data
def load_calibration_data(data_dir="data/processed"):
    """
    Cache loading of calibration dataset.
    """
    try:
        _, _, X_cal, y_cal, _, _ = load_processed_data(data_dir)
        return X_cal, y_cal
    except Exception:
        return None, None


def main():
    st.set_page_config(
        page_title="Conformal Prediction Demo", page_icon="🩺", layout="wide"
    )

    st.title("🩺 Split-Conformal Reliability Layer for Diabetes Prediction")
    st.caption("Machine Learning Lab — Conformal Prediction for Classifiers")

    # Medical Disclaimer Note
    st.warning(
        "⚠️ **Notice**: This demonstration is for educational purposes "
        "only and does **NOT** constitute medical advice. "
        "Always consult a qualified healthcare professional."
    )

    # Sidebar setup
    st.sidebar.header("1. Patient Feature Inputs")
    pregnancies = st.sidebar.number_input(
        "Pregnancies", min_value=0, max_value=20, value=1, step=1
    )
    glucose = st.sidebar.slider(
        "Glucose (mg/dL)",
        min_value=50.0,
        max_value=250.0,
        value=120.0,
        step=1.0,
    )
    blood_pressure = st.sidebar.slider(
        "Blood Pressure (mm Hg)",
        min_value=40.0,
        max_value=140.0,
        value=70.0,
        step=1.0,
    )
    skin_thickness = st.sidebar.slider(
        "Skin Thickness (mm)",
        min_value=10.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
    )
    insulin = st.sidebar.slider(
        "Insulin (mu U/ml)",
        min_value=10.0,
        max_value=900.0,
        value=80.0,
        step=1.0,
    )
    bmi = st.sidebar.slider(
        "BMI (kg/m²)", min_value=10.0, max_value=70.0, value=30.0, step=0.1
    )
    dpf = st.sidebar.slider(
        "Diabetes Pedigree Function",
        min_value=0.05,
        max_value=2.50,
        value=0.47,
        step=0.01,
    )
    age = st.sidebar.slider(
        "Age (years)", min_value=18, max_value=100, value=33, step=1
    )

    st.sidebar.header("2. Model & Conformal Settings")
    scaler, models = load_scaler_and_models()
    X_cal, y_cal = load_calibration_data()

    if not models or scaler is None:
        st.error(
            "Model or scaler artifacts missing. "
            "Please run `python src/preprocess.py` and `python src/models.py`."
        )
        return

    model_choice = st.sidebar.selectbox(
        "Select Classifier", list(models.keys())
    )
    alpha = st.sidebar.slider(
        "Significance Level (alpha)",
        min_value=0.05,
        max_value=0.30,
        value=0.10,
        step=0.01,
    )

    selected_model = models[model_choice]

    tab1, tab2 = st.tabs([
        "🔮 Single Patient Prediction",
        "📊 Model Performance & Conformal Analysis",
    ])

    with tab1:
        st.subheader("Patient Risk Assessment & Conformal Set")

        # Prepare input data
        raw_input = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age,
        }])

        scaled_input = pd.DataFrame(
            scaler.transform(raw_input), columns=raw_input.columns
        )

        # Compute conformal threshold and probabilities
        if X_cal is not None and y_cal is not None:
            thresh = get_threshold(selected_model, X_cal, y_cal, alpha)
            probs = selected_model.predict_proba(scaled_input)[0]

            prob_cutoff = 1.0 - thresh
            pred_set = probs >= prob_cutoff

            p_not_diabetic = probs[0]
            p_diabetic = probs[1]

            is_no_diabetes = pred_set[0]
            is_diabetes = pred_set[1]

            if is_diabetes and not is_no_diabetes:
                status_header = "🚨 Likely diabetic"
                status_color = "red"
                explanation = (
                    "The model reliably predicts diabetes with "
                    "high confidence at the chosen significance level."
                )
            elif is_no_diabetes and not is_diabetes:
                status_header = "✅ Likely not diabetic"
                status_color = "green"
                explanation = (
                    "The model reliably predicts no diabetes with "
                    "high confidence at the chosen significance level."
                )
            else:
                status_header = "❓ Uncertain, needs a doctor"
                status_color = "orange"
                explanation = (
                    "The model prediction is ambiguous at this confidence "
                    "level (prediction set contains 0 or 2 classes)."
                )

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(f"### Result: :{status_color}[{status_header}]")
                st.write(explanation)
                target_pct = (1 - alpha) * 100
                st.info(
                    f"**Target Coverage Level**: {target_pct:.1f}%\n\n"
                    f"**Conformal Cutoff Probability**: {prob_cutoff:.3f}"
                )

            with col2:
                st.markdown("### Class Probabilities")
                st.metric(
                    "Probability of No Diabetes", f"{p_not_diabetic:.1%}"
                )
                st.metric(
                    "Probability of Diabetes", f"{p_diabetic:.1%}"
                )

                included_classes = []
                if is_no_diabetes:
                    included_classes.append("Not Diabetic (0)")
                if is_diabetes:
                    included_classes.append("Diabetic (1)")

                st.write(f"**Prediction Set**: `{included_classes}`")
        else:
            st.error(
                "Calibration data missing for threshold calculation."
            )

    with tab2:
        st.subheader("Model Benchmark Results")

        results_path = Path("outputs/results.csv")
        if results_path.exists():
            df_results = pd.read_csv(results_path)
            st.dataframe(df_results, use_container_width=True)
        else:
            st.info(
                "Results CSV not found. "
                "Please run `python src/evaluate.py` to generate results."
            )

        st.subheader("Visualizations & Plots")

        col_a, col_b = st.columns(2)

        with col_a:
            p1 = Path("outputs/coverage_bar_chart.png")
            if p1.exists():
                st.image(
                    str(p1),
                    caption="Coverage",
                    use_container_width=True,
                )
            else:
                st.warning("coverage_bar_chart.png missing.")

            p3 = Path("outputs/rf_set_size_histogram.png")
            if p3.exists():
                st.image(
                    str(p3),
                    caption="RF Set Size Histogram",
                    use_container_width=True,
                )
            else:
                st.warning("rf_set_size_histogram.png missing.")

        with col_b:
            p2 = Path("outputs/avg_set_size_bar_chart.png")
            if p2.exists():
                st.image(
                    str(p2),
                    caption="Average Set Size",
                    use_container_width=True,
                )
            else:
                st.warning("avg_set_size_bar_chart.png missing.")

            p4 = Path("outputs/pca_uncertainty_scatter.png")
            if p4.exists():
                st.image(
                    str(p4),
                    caption="PCA Uncertainty Scatter",
                    use_container_width=True,
                )
            else:
                st.warning("pca_uncertainty_scatter.png missing.")


if __name__ == "__main__":
    main()
