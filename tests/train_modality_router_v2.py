from pathlib import Path

import pandas as pd
import joblib

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


FEATURES = [

    "mean",
    "std",
    "dark",
    "bright",

    "colored",
    "red",
    "green",
    "blue",

    "horizontal_var",
    "vertical_var",

    "texture_v",
    "texture_h",

    "aspect_ratio",

    "entropy",

    "p05",
    "p25",
    "p50",
    "p75",
    "p95",

    "gradient_mean",
    "gradient_std",
    "gradient_max",

    "edge_density",

    "center_mean",
    "border_mean",
    "center_border_diff",

    "horizontal_symmetry",
    "vertical_symmetry",
]


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_val,
    y_val,
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_val
    )

    accuracy = accuracy_score(
        y_val,
        predictions,
    )

    print()
    print(
        f"Validation Accuracy: "
        f"{accuracy:.4%}"
    )

    print()
    print("Classification Report:")
    print()

    print(
        classification_report(
            y_val,
            predictions,
            digits=4,
        )
    )

    print("Confusion Matrix:")

    classes = sorted(
        y_train.unique()
    )

    matrix = confusion_matrix(
        y_val,
        predictions,
        labels=classes,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=classes,
        columns=classes,
    )

    print(
        matrix_df.to_string()
    )

    return accuracy, model


def main():

    print()
    print("=" * 70)
    print("MODALITY ROUTER V2 TRAINING")
    print("=" * 70)

    # -------------------------------------------------
    # Load ONLY train and validation
    # -------------------------------------------------

    train_df = pd.read_csv(
        "training_data/modality_train.csv"
    )

    validation_df = pd.read_csv(
        "training_data/modality_validation.csv"
    )

    print()
    print(
        f"Training samples: "
        f"{len(train_df)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_df)}"
    )

    # -------------------------------------------------
    # Features
    # -------------------------------------------------

    X_train = train_df[
        FEATURES
    ]

    y_train = train_df[
        "modality"
    ]

    X_val = validation_df[
        FEATURES
    ]

    y_val = validation_df[
        "modality"
    ]

    # -------------------------------------------------
    # Candidate models
    # -------------------------------------------------

    models = {

        "Random Forest": RandomForestClassifier(

            n_estimators=500,

            max_depth=None,

            min_samples_leaf=1,

            class_weight="balanced",

            random_state=42,

            n_jobs=-1,
        ),

        "Extra Trees": ExtraTreesClassifier(

            n_estimators=500,

            max_depth=None,

            min_samples_leaf=1,

            class_weight="balanced",

            random_state=42,

            n_jobs=-1,
        ),

        "Random Forest Shallow": RandomForestClassifier(

            n_estimators=500,

            max_depth=12,

            min_samples_leaf=2,

            class_weight="balanced",

            random_state=42,

            n_jobs=-1,
        ),
    }

    results = {}

    best_accuracy = -1
    best_model = None
    best_name = None

    # -------------------------------------------------
    # Train candidates
    # -------------------------------------------------

    for name, model in models.items():

        accuracy, trained_model = evaluate_model(

            name,

            model,

            X_train,

            y_train,

            X_val,

            y_val,
        )

        results[name] = accuracy

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_model = trained_model

            best_name = name

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    for name, accuracy in results.items():

        print(
            f"{name:<30} "
            f"{accuracy:.4%}"
        )

    print()
    print(
        f"BEST MODEL: {best_name}"
    )

    print(
        f"BEST VALIDATION ACCURACY: "
        f"{best_accuracy:.4%}"
    )

    # -------------------------------------------------
    # Save best model
    # -------------------------------------------------

    output_path = Path(
        "src/models/modality_router.pkl"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(

        {
            "model": best_model,
            "features": FEATURES,
            "validation_accuracy": best_accuracy,
            "model_name": best_name,
        },

        output_path,
    )

    print()
    print(
        f"Saved best model to: "
        f"{output_path}"
    )

    print()
    print(
        "Final test data was NOT used."
    )


if __name__ == "__main__":
    main()