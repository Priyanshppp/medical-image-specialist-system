from pathlib import Path
import argparse

import joblib
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAINING_DATA_DIR = (
    PROJECT_ROOT / "training_data"
)

MODELS_DIR = (
    PROJECT_ROOT / "models"
)


# ============================================================
# LOAD FEATURE FILE
# ============================================================

def load_feature_file(path):

    print()
    print(f"Loading: {path.relative_to(PROJECT_ROOT)}")

    try:

        data = joblib.load(path)

        print(
            "Loaded successfully using joblib."
        )

        print(
            f"Data type: {type(data)}"
        )

        print(
            f"Keys: {list(data.keys())}"
        )

        return data

    except Exception as error:

        print()
        print(
            "ERROR: Could not load feature file."
        )

        print(
            f"Reason: {error}"
        )

        raise


# ============================================================
# VALIDATE FEATURE DATA
# ============================================================

def validate_feature_data(
    data,
    split_name,
):

    required_keys = [
        "features",
        "labels",
        "query_ids",
        "image_paths",
    ]

    for key in required_keys:

        if key not in data:

            raise KeyError(
                f"{split_name} data missing key: "
                f"{key}"
            )

    features = data["features"]
    labels = data["labels"]

    if len(features) != len(labels):

        raise ValueError(
            f"{split_name} features and labels "
            f"have different lengths."
        )

    if len(features) == 0:

        raise ValueError(
            f"{split_name} dataset is empty."
        )

    print()
    print(
        f"{split_name} data:"
    )

    print(
        f"Feature shape: {features.shape}"
    )

    print(
        f"Labels: {len(labels)}"
    )


# ============================================================
# PRINT LABEL DISTRIBUTION
# ============================================================

def print_label_distribution(labels):

    print()
    print("=" * 70)
    print("TRAINING LABEL DISTRIBUTION")
    print("=" * 70)
    print()

    unique_labels, counts = np.unique(
        labels,
        return_counts=True,
    )

    for label, count in zip(
        unique_labels,
        counts,
    ):

        print(
            f"{label}: {count}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train a modality specialist "
            "using extracted image features."
        )
    )

    parser.add_argument(
        "--modality",
        required=True,
        type=str,
        help=(
            "Modality name, for example: "
            "fundus, mri, oct"
        ),
    )

    args = parser.parse_args()

    modality = (
        args.modality
        .strip()
        .lower()
    )

    print()
    print("=" * 70)
    print(
        f"{modality.upper()} SPECIALIST TRAINING"
    )
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print(
        "Only specialist training and "
        "validation features are used."
    )
    print(
        "Final test data is NOT loaded."
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    train_features_path = (
        TRAINING_DATA_DIR
        / f"{modality}_train_features.pkl"
    )

    validation_features_path = (
        TRAINING_DATA_DIR
        / f"{modality}_validation_features.pkl"
    )

    model_path = (
        MODELS_DIR
        / f"{modality}_specialist.joblib"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not train_features_path.exists():

        raise FileNotFoundError(
            f"Training features not found:\n"
            f"{train_features_path}"
        )

    if not validation_features_path.exists():

        raise FileNotFoundError(
            f"Validation features not found:\n"
            f"{validation_features_path}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    train_data = load_feature_file(
        train_features_path
    )

    validation_data = load_feature_file(
        validation_features_path
    )

    validate_feature_data(
        train_data,
        "Training",
    )

    validate_feature_data(
        validation_data,
        "Validation",
    )

    X_train = train_data["features"]
    y_train = train_data["labels"]

    X_validation = validation_data["features"]
    y_validation = validation_data["labels"]

    # --------------------------------------------------------
    # Label distribution
    # --------------------------------------------------------

    print_label_distribution(
        y_train
    )

    # --------------------------------------------------------
    # Train classifier
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING CLASSIFIER")
    print("=" * 70)
    print()

    classifier = ExtraTreesClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    classifier.fit(
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
    print("VALIDATION")
    print("=" * 70)

    predictions = classifier.predict(
        X_validation
    )

    probabilities = classifier.predict_proba(
        X_validation
    )

    accuracy = accuracy_score(
        y_validation,
        predictions,
    )

    print()
    print(
        f"Validation accuracy: "
        f"{accuracy * 100:.4f}%"
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
    print("INDIVIDUAL VALIDATION RESULTS")
    print("=" * 70)

    query_ids = (
        validation_data["query_ids"]
    )

    for index in range(
        len(y_validation)
    ):

        true_label = (
            y_validation[index]
        )

        predicted_label = (
            predictions[index]
        )

        confidence = float(
            np.max(
                probabilities[index]
            )
        )

        correct = (
            true_label == predicted_label
        )

        status = (
            "CORRECT"
            if correct
            else "WRONG"
        )

        print()
        print(
            f"[{index + 1}/"
            f"{len(y_validation)}] "
            f"{status}"
        )

        print(
            f"ID:         "
            f"{query_ids[index]}"
        )

        print(
            f"True:       "
            f"{true_label}"
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
    # Save model package
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_package = {
        "model": classifier,
        "modality": modality,
        "feature_dimension": int(
            X_train.shape[1]
        ),
        "training_samples": int(
            len(X_train)
        ),
        "validation_samples": int(
            len(X_validation)
        ),
        "validation_accuracy": float(
            accuracy
        ),
        "classes": classifier.classes_,
    }

    joblib.dump(
        model_package,
        model_path,
    )

    print()
    print(
        f"Model saved to:\n"
        f"{model_path}"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print()
    print(
        f"Modality: {modality.upper()}"
    )

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Validation samples: "
        f"{len(X_validation)}"
    )

    print(
        f"Validation accuracy: "
        f"{accuracy * 100:.4f}%"
    )

    print()
    print(
        "Final test data was NOT loaded."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()