from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main():

    print()
    print("=" * 70)
    print("ROUTER CONFIDENCE ANALYSIS")
    print("=" * 70)

    validation_path = Path(
        "training_data/modality_validation.csv"
    )

    model_path = Path(
        "src/models/modality_router.pkl"
    )

    # ------------------------------------------------------------
    # LOAD VALIDATION DATA
    # ------------------------------------------------------------

    df = pd.read_csv(
        validation_path
    )

    print()
    print(
        f"Validation samples: {len(df)}"
    )

    print()
    print("Validation columns:")

    print(
        df.columns.tolist()
    )

    # ------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------

    loaded = joblib.load(
        model_path
    )

    if isinstance(loaded, dict):

        model = loaded["model"]

        feature_names = loaded["features"]

    else:

        model = loaded

        if hasattr(
            model,
            "feature_names_in_",
        ):

            feature_names = list(
                model.feature_names_in_
            )

        else:

            feature_names = [

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
            ]

    print()
    print(
        f"Model: "
        f"{type(model).__name__}"
    )

    # ------------------------------------------------------------
    # PREPARE FEATURES
    # ------------------------------------------------------------

    X = df[
        feature_names
    ]

    y_true = df[
        "modality"
    ]

    # ------------------------------------------------------------
    # PREDICTIONS
    # ------------------------------------------------------------

    predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )

    confidences = np.max(
        probabilities,
        axis=1,
    )

    correct = (
        predictions == y_true.values
    )

    overall_accuracy = float(
        np.mean(correct)
    )

    # ------------------------------------------------------------
    # OVERALL RESULTS
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("BASELINE VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"Accuracy: "
        f"{overall_accuracy:.4%}"
    )

    print(
        f"Correct predictions: "
        f"{correct.sum()}"
    )

    print(
        f"Incorrect predictions: "
        f"{(~correct).sum()}"
    )

    # ------------------------------------------------------------
    # CONFIDENCE ANALYSIS
    # ------------------------------------------------------------

    correct_confidences = confidences[
        correct
    ]

    incorrect_confidences = confidences[
        ~correct
    ]

    print()
    print("=" * 70)
    print("CONFIDENCE ANALYSIS")
    print("=" * 70)

    print()

    print("Correct predictions:")

    print(
        f"Mean confidence: "
        f"{correct_confidences.mean():.4f}"
    )

    print(
        f"Minimum confidence: "
        f"{correct_confidences.min():.4f}"
    )

    print()

    print("Incorrect predictions:")

    print(
        f"Mean confidence: "
        f"{incorrect_confidences.mean():.4f}"
    )

    print(
        f"Maximum confidence: "
        f"{incorrect_confidences.max():.4f}"
    )

    # ------------------------------------------------------------
    # THRESHOLD ANALYSIS
    # ------------------------------------------------------------

    thresholds = [

        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

    results = []

    print()
    print("=" * 70)
    print("THRESHOLD ANALYSIS")
    print("=" * 70)

    for threshold in thresholds:

        accepted = (
            confidences >= threshold
        )

        rejected = (
            confidences < threshold
        )

        accepted_count = int(
            accepted.sum()
        )

        rejected_count = int(
            rejected.sum()
        )

        if accepted_count > 0:

            accepted_accuracy = float(
                np.mean(
                    correct[accepted]
                )
            )

        else:

            accepted_accuracy = 0.0

        # How many wrong predictions
        # does this threshold reject?
        incorrect_rejected = int(
            np.sum(
                (~correct) & rejected
            )
        )

        total_incorrect = int(
            (~correct).sum()
        )

        wrong_rejected_rate = (

            incorrect_rejected
            / total_incorrect

            if total_incorrect > 0

            else 0.0
        )

        results.append(
            {

                "threshold":
                    threshold,

                "accepted":
                    accepted_count,

                "rejected":
                    rejected_count,

                "accepted_accuracy":
                    accepted_accuracy,

                "wrong_predictions_rejected":
                    incorrect_rejected,

                "wrong_rejection_rate":
                    wrong_rejected_rate,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    print()

    print(
        results_df.to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # SAVE DETAILED ANALYSIS
    # ------------------------------------------------------------

    output = Path(
        "training_data/router_confidence_analysis.csv"
    )

    analysis_df = pd.DataFrame(
        {

            "query_id":
                df["query_id"],

            "true_modality":
                y_true,

            "predicted_modality":
                predictions,

            "confidence":
                confidences,

            "correct":
                correct,
        }
    )

    analysis_df.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)

    print(
        f"Detailed analysis: "
        f"{output}"
    )

    print()
    print(
        "IMPORTANT: Final test data "
        "was NOT used."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()