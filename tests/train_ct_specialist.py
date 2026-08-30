from pathlib import Path
import joblib
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_FEATURES = (
    PROJECT_ROOT
    / "training_data"
    / "ct_train_features.pkl"
)

VALIDATION_FEATURES = (
    PROJECT_ROOT
    / "training_data"
    / "ct_validation_features.pkl"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_OUTPUT = (
    MODEL_DIR
    / "ct_specialist.joblib"
)


# ============================================================
# LOAD FEATURE FILE
# ============================================================

def load_feature_file(path):

    print()
    print(f"Loading: {path.relative_to(PROJECT_ROOT)}")

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}"
        )

    data = joblib.load(path)

    print(
        "Loaded successfully using joblib."
    )

    print(
        f"Data type: {type(data)}"
    )

    if isinstance(data, dict):

        print(
            f"Keys: {list(data.keys())}"
        )

    required_keys = [
        "features",
        "labels",
        "query_ids",
        "image_paths",
    ]

    for key in required_keys:

        if key not in data:

            raise KeyError(
                f"Missing required key: {key}"
            )

    return data


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CT SPECIALIST TRAINING"
    )
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Only specialist training and validation "
        "features are used."
    )
    print(
        "Final test data is NOT loaded."
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if not TRAIN_FEATURES.exists():

        raise FileNotFoundError(
            f"Training features missing: "
            f"{TRAIN_FEATURES}"
        )

    if not VALIDATION_FEATURES.exists():

        raise FileNotFoundError(
            f"Validation features missing: "
            f"{VALIDATION_FEATURES}"
        )

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    train_data = load_feature_file(
        TRAIN_FEATURES
    )

    X_train = train_data["features"]
    y_train = train_data["labels"]

    print()
    print("Training data:")
    print(
        f"Feature shape: {X_train.shape}"
    )
    print(
        f"Labels: {len(y_train)}"
    )

    # --------------------------------------------------------
    # Load validation data
    # --------------------------------------------------------

    validation_data = load_feature_file(
        VALIDATION_FEATURES
    )

    X_validation = (
        validation_data["features"]
    )

    y_validation = (
        validation_data["labels"]
    )

    print()
    print("Validation data:")
    print(
        f"Feature shape: "
        f"{X_validation.shape}"
    )
    print(
        f"Labels: "
        f"{len(y_validation)}"
    )

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TRAINING LABEL DISTRIBUTION"
    )
    print("=" * 70)

    unique_labels, counts = np.unique(
        y_train,
        return_counts=True,
    )

    for label, count in zip(
        unique_labels,
        counts,
    ):

        print(
            f"{label:<40} {count}"
        )

    # --------------------------------------------------------
    # Train classifier
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TRAINING CLASSIFIER"
    )
    print("=" * 70)

    model = ExtraTreesClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "VALIDATION"
    )
    print("=" * 70)

    predictions = model.predict(
        X_validation
    )

    probabilities = model.predict_proba(
        X_validation
    )

    confidences = np.max(
        probabilities,
        axis=1,
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    print()
    print(
        f"Validation accuracy: "
        f"{accuracy:.4%}"
    )

    print()
    print(
        "Classification report:"
    )
    print()

    print(
        classification_report(
            y_validation,
            predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Individual validation results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "INDIVIDUAL VALIDATION RESULTS"
    )
    print("=" * 70)

    for index, (
        true_label,
        predicted_label,
        confidence,
    ) in enumerate(
        zip(
            y_validation,
            predictions,
            confidences,
        ),
        start=1,
    ):

        correct = (
            str(true_label).strip().lower()
            ==
            str(predicted_label).strip().lower()
        )

        status = (
            "CORRECT"
            if correct
            else "WRONG"
        )

        print()
        print(
            f"[{index}/{len(y_validation)}] "
            f"{status}"
        )

        print(
            f"True:       {true_label}"
        )

        print(
            f"Predicted:  "
            f"{predicted_label}"
        )

        print(
            f"Confidence: "
            f"{confidence:.4f}"
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "SAVING MODEL"
    )
    print("=" * 70)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_OUTPUT,
    )

    print()
    print(
        "Model saved to:"
    )
    print(
        MODEL_OUTPUT
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "SUCCESS"
    )
    print("=" * 70)

    print(
        "Final test data was NOT loaded."
    )


if __name__ == "__main__":
    main()