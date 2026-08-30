from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.domain_router import DomainRouter


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_FILE = Path(
    "training_data/modality_validation.csv"
)

MODEL_FILE = Path(
    "src/models/modality_router.pkl"
)


# Router domain -> original dataset modality
DOMAIN_TO_MODALITY = {
    "ct": "CT",
    "brain_mri": "MRI",
    "chest_xray": "X-ray",
    "dermoscopy": "Dermatology",
    "microscopy": "Microscopy",
    "oct": "OCT",
    "ultrasound": "Ultrasound",
    "general": "Fundus",
}


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("HYBRID ROUTER VALIDATION EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load validation data ONLY
    # --------------------------------------------------------

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VALIDATION_FILE}"
        )

    validation = pd.read_csv(
        VALIDATION_FILE
    )

    print()
    print(
        f"Validation samples: {len(validation)}"
    )

    print()
    print(
        "Final test data was NOT loaded."
    )

    # --------------------------------------------------------
    # Load saved classifier
    # --------------------------------------------------------

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    loaded = joblib.load(
        MODEL_FILE
    )

    if isinstance(loaded, dict):

        classifier = loaded["model"]

        feature_names = loaded.get(
            "features"
        )

    else:

        classifier = loaded

        if hasattr(
            classifier,
            "feature_names_in_",
        ):

            feature_names = list(
                classifier.feature_names_in_
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

    print(
        f"Model: "
        f"{classifier.__class__.__name__}"
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X = validation[
        feature_names
    ]

    y_true = validation[
        "modality"
    ].values

    # --------------------------------------------------------
    # RAW VISUAL CLASSIFIER
    # --------------------------------------------------------

    raw_predictions = classifier.predict(
        X
    )

    raw_probabilities = (
        classifier.predict_proba(X)
    )

    confidences = (
        raw_probabilities.max(axis=1)
    )

    raw_accuracy = accuracy_score(
        y_true,
        raw_predictions,
    )

    # --------------------------------------------------------
    # HYBRID ROUTER
    # --------------------------------------------------------

    router = DomainRouter()

    hybrid_predictions = []

    visual_accepted = 0
    visual_rejected = 0

    for index, row in validation.iterrows():

        predicted_modality = (
            raw_predictions[index]
        )

        confidence = (
            confidences[index]
        )

        threshold = getattr(
            router,
            "confidence_threshold",
            0.65,
        )

        # ----------------------------------------------------
        # Visual routing accepted
        # ----------------------------------------------------

        if confidence >= threshold:

            visual_accepted += 1

            hybrid_predictions.append(
                predicted_modality
            )

        # ----------------------------------------------------
        # Visual routing rejected
        # ----------------------------------------------------

        else:

            visual_rejected += 1

            # We do not have original images in this CSV.
            # Therefore evaluate fallback routing from question
            # and choices if available.

            question = row.get(
                "question",
                row.get(
                    "Question",
                    "",
                ),
            )

            choices = {}

            for column in row.index:

                column_lower = str(
                    column
                ).lower()

                if (
                    column_lower.startswith(
                        "option"
                    )
                    or column_lower
                    in [
                        "a",
                        "b",
                        "c",
                        "d",
                    ]
                ):

                    value = row[column]

                    if pd.notna(value):

                        choices[column] = value

            domain = router._text_route(
                question,
                choices,
            )

            modality = (
                DOMAIN_TO_MODALITY.get(
                    domain,
                    "Fundus",
                )
            )

            hybrid_predictions.append(
                modality
            )

    hybrid_predictions = np.array(
        hybrid_predictions
    )

    hybrid_accuracy = accuracy_score(
        y_true,
        hybrid_predictions,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("1. RAW VISUAL CLASSIFIER")
    print("=" * 70)

    print(
        f"Accuracy: "
        f"{raw_accuracy:.4%}"
    )

    print()

    print(
        classification_report(
            y_true,
            raw_predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("2. CONFIDENCE FILTER")
    print("=" * 70)

    print(
        f"Threshold: "
        f"{getattr(router, 'confidence_threshold', 0.65)}"
    )

    print(
        f"Visual predictions accepted: "
        f"{visual_accepted}"
    )

    print(
        f"Visual predictions rejected: "
        f"{visual_rejected}"
    )

    print(
        f"Acceptance rate: "
        f"{visual_accepted / len(validation):.4%}"
    )

    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("3. HYBRID ROUTER")
    print("=" * 70)

    print(
        f"Overall Accuracy: "
        f"{hybrid_accuracy:.4%}"
    )

    print()

    print(
        classification_report(
            y_true,
            hybrid_predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("4. HYBRID CONFUSION MATRIX")
    print("=" * 70)

    modalities = sorted(
        validation["modality"].unique()
    )

    matrix = confusion_matrix(
        y_true,
        hybrid_predictions,
        labels=modalities,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=modalities,
        columns=modalities,
    )

    print(
        matrix_df.to_string()
    )

    # ========================================================
    # PER-MODALITY ACCURACY
    # ========================================================

    print()
    print("=" * 70)
    print("5. PER-MODALITY ACCURACY")
    print("=" * 70)

    rows = []

    for modality in modalities:

        mask = (
            validation["modality"]
            == modality
        )

        total = int(mask.sum())

        correct_raw = int(
            np.sum(
                raw_predictions[mask]
                == modality
            )
        )

        correct_hybrid = int(
            np.sum(
                hybrid_predictions[mask]
                == modality
            )
        )

        rows.append(
            {
                "modality": modality,
                "samples": total,
                "raw_correct": correct_raw,
                "raw_accuracy": (
                    correct_raw / total
                ),
                "hybrid_correct": (
                    correct_hybrid
                ),
                "hybrid_accuracy": (
                    correct_hybrid / total
                ),
            }
        )

    results = pd.DataFrame(
        rows
    )

    print(
        results.to_string(
            index=False
        )
    )

    # ========================================================
    # SAVE DETAILED RESULTS
    # ========================================================

    output = Path(
        "training_data/"
        "hybrid_router_validation_results.csv"
    )

    detailed = validation.copy()

    detailed["raw_prediction"] = (
        raw_predictions
    )

    detailed["confidence"] = (
        confidences
    )

    threshold = getattr(
        router,
        "confidence_threshold",
        0.65,
    )

    detailed["visual_accepted"] = (
        confidences >= threshold
    )

    detailed["hybrid_prediction"] = (
        hybrid_predictions
    )

    detailed["raw_correct"] = (
        detailed["modality"]
        == detailed["raw_prediction"]
    )

    detailed["hybrid_correct"] = (
        detailed["modality"]
        == detailed["hybrid_prediction"]
    )

    detailed.to_csv(
        output,
        index=False,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)

    print(
        f"Raw visual accuracy:    "
        f"{raw_accuracy:.4%}"
    )

    print(
        f"Hybrid router accuracy: "
        f"{hybrid_accuracy:.4%}"
    )

    improvement = (
        hybrid_accuracy
        - raw_accuracy
    )

    print(
        f"Difference:             "
        f"{improvement:+.4%}"
    )

    print()

    print(
        f"Detailed results saved: "
        f"{output}"
    )

    print()

    print(
        "IMPORTANT: "
        "modality_final_test.csv "
        "was NOT loaded or used."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
    