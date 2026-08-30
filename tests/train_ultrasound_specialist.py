from pathlib import Path
import pickle
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)
import joblib


# ============================================================
# PATHS
# ============================================================

TRAIN_FEATURES = Path(
    "training_data/ultrasound_train_features.pkl"
)

VALIDATION_FEATURES = Path(
    "training_data/ultrasound_validation_features.pkl"
)

MODEL_OUTPUT = Path(
    "models/ultrasound_specialist.joblib"
)


# ============================================================
# LOAD FEATURES
# ============================================================

def load_feature_file(path):

    print(f"\nLoading: {path}")

    import joblib

    data = joblib.load(path)

    print("Loaded successfully using joblib.")

    print(
        "Data type:",
        type(data),
    )

    if isinstance(data, dict):

        print(
            "Keys:",
            list(data.keys()),
        )

    return data


def extract_xy(data):

    # Support common dictionary structures
    possible_feature_keys = [
        "features",
        "X",
        "embeddings",
    ]

    possible_label_keys = [
        "labels",
        "y",
        "targets",
    ]

    X = None
    y = None

    for key in possible_feature_keys:
        if key in data:
            X = data[key]
            break

    for key in possible_label_keys:
        if key in data:
            y = data[key]
            break

    if X is None:
        raise KeyError(
            "Could not find feature array. "
            "Expected one of: "
            f"{possible_feature_keys}"
        )

    if y is None:
        raise KeyError(
            "Could not find labels. "
            "Expected one of: "
            f"{possible_label_keys}"
        )

    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ULTRASOUND SPECIALIST TRAINING")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )
    print(
        "Only specialist training and validation features "
        "are used."
    )
    print(
        "Final test data is NOT loaded."
    )

    # --------------------------------------------------------
    # LOAD TRAINING DATA
    # --------------------------------------------------------

    train_data = load_feature_file(
        TRAIN_FEATURES
    )

    X_train, y_train = extract_xy(
        train_data
    )

    print("\nTraining data:")
    print("Feature shape:", X_train.shape)
    print("Labels:", len(y_train))

    # --------------------------------------------------------
    # LOAD VALIDATION DATA
    # --------------------------------------------------------

    validation_data = load_feature_file(
        VALIDATION_FEATURES
    )

    X_val, y_val = extract_xy(
        validation_data
    )

    print("\nValidation data:")
    print("Feature shape:", X_val.shape)
    print("Labels:", len(y_val))

    # --------------------------------------------------------
    # LABEL DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING LABEL DISTRIBUTION")
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
            f"{str(label):30s} {count}"
        )

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING CLASSIFIER")
    print("=" * 70)

    model = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=42,
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
    # VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    predictions = model.predict(
        X_val
    )

    accuracy = accuracy_score(
        y_val,
        predictions,
    )

    print(
        f"\nValidation accuracy: "
        f"{accuracy:.4%}"
    )

    print("\nClassification report:\n")

    print(
        classification_report(
            y_val,
            predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # INDIVIDUAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INDIVIDUAL VALIDATION RESULTS")
    print("=" * 70)

    probabilities = model.predict_proba(
        X_val
    )

    confidences = np.max(
        probabilities,
        axis=1,
    )

    for i, (
        truth,
        prediction,
        confidence,
    ) in enumerate(
        zip(
            y_val,
            predictions,
            confidences,
        ),
        start=1,
    ):

        correct = (
            truth == prediction
        )

        status = (
            "CORRECT"
            if correct
            else "WRONG"
        )

        print(
            f"\n[{i}/{len(y_val)}] "
            f"{status}"
        )

        print(
            f"True:       {truth}"
        )

        print(
            f"Predicted:  {prediction}"
        )

        print(
            f"Confidence: "
            f"{confidence:.4f}"
        )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)

    MODEL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_OUTPUT,
    )

    print(
        f"\nModel saved to:\n"
        f"{MODEL_OUTPUT.resolve()}"
    )

    print("\n" + "=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print(
        "Final test data was NOT loaded."
    )


if __name__ == "__main__":
    main()