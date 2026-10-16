"""
Model training module for Diabetes classification.
Trains 5 classifiers on preprocessed train data, evaluates test accuracy,
and saves trained models to models/*.joblib.
"""

from pathlib import Path
import sys

# Ensure root directory is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from src.preprocess import load_processed_data  # noqa: E402


def train_and_save_models(
    data_dir="data/processed", models_dir="models", random_state=42
):
    """
    Train 5 classifiers on training data, evaluate accuracy on test data,
    and save trained model instances to joblib files.

    Returns:
        dict: Trained model instances mapping name -> model
    """
    X_train, y_train, X_cal, y_cal, X_test, y_test = load_processed_data(
        data_dir
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=random_state
        ),
        "decision_tree": DecisionTreeClassifier(
            min_samples_leaf=10, random_state=random_state
        ),
        "gaussian_nb": GaussianNB(),
        "knn": KNeighborsClassifier(n_neighbors=15),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state
        ),
    }

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    accuracies = {}

    print("Training models and evaluating on test set...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        accuracies[name] = test_acc

        model_file = models_path / f"{name}.joblib"
        joblib.dump(model, model_file)
        print(f"[{name}] Test Accuracy: {test_acc:.4f} -> Saved {model_file}")

    return models, accuracies


if __name__ == "__main__":
    train_and_save_models()
