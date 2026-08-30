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
# PATHS
# ============================================================

MANIFEST_PATH = Path(
    "training_data/validation_manifest.csv"
)

MODEL_PATH = Path(
    "src/models/modality_router.pkl"
)

FINAL_TEST_PATH = Path(
    "training_data/modality_final_test.csv"
)


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
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


# ============================================================
# TEXT ROUTING
# ============================================================

def strong_text_route(question, choices):
    """
    Conservative text router.

    Returns a modality only when textual evidence is strong.
    Otherwise returns None.

    This is intentionally conservative so weak text routing
    does not destroy a useful visual prediction.
    """

    question_text = str(question).lower()

    choices_text = " ".join(
        str(value).lower()
        for value in choices
        if pd.notna(value)
    )

    combined = (
        question_text
        + " "
        + choices_text
    )

    # --------------------------------------------------------
    # CT
    # --------------------------------------------------------

    ct_terms = [
        "ct scan",
        "computed tomography",
        "lung cancer",
        "pulmonary nodule",
    ]

    if any(
        term in combined
        for term in ct_terms
    ):
        return "CT"

    # --------------------------------------------------------
    # MRI
    # --------------------------------------------------------

    mri_terms = [
        "mri",
        "magnetic resonance",
        "alzheimer",
        "multiple sclerosis",
        "ischemic stroke",
        "brain tumor",
    ]

    if any(
        term in combined
        for term in mri_terms
    ):
        return "MRI"

    # --------------------------------------------------------
    # X-RAY
    # --------------------------------------------------------

    xray_terms = [
        "x-ray",
        "xray",
        "radiograph",
        "chest film",
        "fracture",
    ]

    if any(
        term in combined
        for term in xray_terms
    ):
        return "X-ray"

    # --------------------------------------------------------
    # DERMATOLOGY
    # --------------------------------------------------------

    derm_terms = [
        "skin lesion",
        "dermoscopic",
        "melanoma",
        "nevus",
        "skin cancer",
    ]

    if any(
        term in combined
        for term in derm_terms
    ):
        return "Dermatology"

    # --------------------------------------------------------
    # OCT
    # --------------------------------------------------------

    oct_terms = [
        "optical coherence tomography",
        "oct image",
        "retinal layer",
        "macular edema",
    ]

    if any(
        term in combined
        for term in oct_terms
    ):
        return "OCT"

    # --------------------------------------------------------
    # ULTRASOUND
    # --------------------------------------------------------

    ultrasound_terms = [
        "ultrasound",
        "sonography",
        "sonographic",
        "echogenic",
    ]

    if any(
        term in combined
        for term in ultrasound_terms
    ):
        return "Ultrasound"

    # --------------------------------------------------------
    # MICROSCOPY
    # --------------------------------------------------------

    microscopy_terms = [
        "microscopic",
        "microscopy",
        "cellular",
        "malaria",
        "parasite",
        "protein localization",
    ]

    if any(
        term in combined
        for term in microscopy_terms
    ):
        return "Microscopy"

    # --------------------------------------------------------
    # FUNDUS
    # --------------------------------------------------------

    fundus_terms = [
        "fundus",
        "optic disc",
        "retinal vessels",
        "eye axis",
    ]

    if any(
        term in combined
        for term in fundus_terms
    ):
        return "Fundus"

    return None


# ============================================================
# EVALUATION HELPER
# ============================================================

def print_results(
    title,
    y_true,
    y_pred,
):
    print()

    print("=" * 70)
    print(title)
    print("=" * 70)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    print(
        f"\nAccuracy: "
        f"{accuracy:.4%}"
    )

    print()

    print(
        classification_report(
            y_true,
            y_pred,
            digits=4,
        )
    )

    labels = sorted(
        set(y_true)
        | set(y_pred)
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    print(
        "Confusion Matrix:\n"
    )

    print(
        matrix_df
    )

    return accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print("=" * 70)
    print(
        "ROUTER STRATEGY EVALUATION"
    )
    print("=" * 70)

    print()

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Validation manifest not found: "
            f"{MANIFEST_PATH}"
        )

    print(
        "Loading validation manifest..."
    )

    validation_df = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"Validation samples: "
        f"{len(validation_df)}"
    )

    # Explicitly state that final test is untouched.

    print()

    print(
        "FINAL TEST SAFETY CHECK"
    )

    print(
        "-" * 70
    )

    print(
        "This script does NOT load:"
    )

    print(
        FINAL_TEST_PATH
    )

    print(
        "Final test remains untouched."
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print()

    print(
        "Loading modality classifier..."
    )

    loaded = joblib.load(
        MODEL_PATH
    )

    if isinstance(
        loaded,
        dict,
    ):

        model = loaded[
            "model"
        ]

        feature_names = loaded.get(
            "features",
            FEATURE_COLUMNS,
        )

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

            feature_names = (
                FEATURE_COLUMNS
            )

    print(
        f"Model: "
        f"{type(model).__name__}"
    )

    # --------------------------------------------------------
    # BUILD FEATURE MATRIX
    # --------------------------------------------------------

    print()

    print(
        "Preparing validation features..."
    )

    X = validation_df[
        feature_names
    ].copy()

    y_true = validation_df[
        "modality"
    ].tolist()

    # --------------------------------------------------------
    # GET VISUAL PREDICTIONS
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "GENERATING VISUAL PREDICTIONS"
    )

    print(
        "=" * 70
    )

    visual_predictions = model.predict(
        X
    )

    probabilities = model.predict_proba(
        X
    )

    confidences = np.max(
        probabilities,
        axis=1,
    )

    # --------------------------------------------------------
    # STRATEGY 1
    # PURE VISUAL
    # --------------------------------------------------------

    visual_accuracy = print_results(
        "STRATEGY 1: PURE VISUAL ROUTER",
        y_true,
        visual_predictions,
    )

    # --------------------------------------------------------
    # STRATEGY 2
    # CONFIDENCE FILTER
    #
    # IMPORTANT:
    # Low confidence does NOT automatically become General.
    #
    # We keep the visual prediction unless strong text evidence
    # explicitly suggests another modality.
    # --------------------------------------------------------

    thresholds = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
    ]

    strategy_results = []

    for threshold in thresholds:

        predictions = []

        text_overrides = 0

        low_confidence_count = 0

        for index, row in (
            validation_df.iterrows()
        ):

            visual_prediction = (
                visual_predictions[index]
            )

            confidence = (
                confidences[index]
            )

            final_prediction = (
                visual_prediction
            )

            # Only investigate text if confidence is low.

            if confidence < threshold:

                low_confidence_count += 1

                choices = [

                    row.get(
                        "Option_A",
                        "",
                    ),

                    row.get(
                        "Option_B",
                        "",
                    ),

                    row.get(
                        "Option_C",
                        "",
                    ),

                    row.get(
                        "Option_D",
                        "",
                    ),
                ]

                text_prediction = (
                    strong_text_route(
                        row.get(
                            "Question",
                            "",
                        ),
                        choices,
                    )
                )

                # Override ONLY if strong text evidence exists.

                if (
                    text_prediction
                    is not None
                ):

                    final_prediction = (
                        text_prediction
                    )

                    text_overrides += 1

            predictions.append(
                final_prediction
            )

        accuracy = accuracy_score(
            y_true,
            predictions,
        )

        strategy_results.append(
            {
                "strategy":
                    f"hybrid_{threshold}",
                "threshold":
                    threshold,
                "accuracy":
                    accuracy,
                "low_confidence":
                    low_confidence_count,
                "text_overrides":
                    text_overrides,
            }
        )

        print()

        print(
            "=" * 70
        )

        print(
            f"STRATEGY 2: "
            f"HYBRID THRESHOLD {threshold}"
        )

        print(
            "=" * 70
        )

        print(
            f"Accuracy: "
            f"{accuracy:.4%}"
        )

        print(
            f"Low-confidence samples: "
            f"{low_confidence_count}"
        )

        print(
            f"Strong text overrides: "
            f"{text_overrides}"
        )

    # --------------------------------------------------------
    # STRATEGY 3
    # ALWAYS ALLOW STRONG TEXT OVERRIDE
    #
    # This tests whether text can improve predictions even
    # when visual confidence is high.
    # --------------------------------------------------------

    always_hybrid_predictions = []

    always_text_overrides = 0

    for index, row in (
        validation_df.iterrows()
    ):

        visual_prediction = (
            visual_predictions[index]
        )

        choices = [

            row.get(
                "Option_A",
                "",
            ),

            row.get(
                "Option_B",
                "",
            ),

            row.get(
                "Option_C",
                "",
            ),

            row.get(
                "Option_D",
                "",
            ),
        ]

        text_prediction = (
            strong_text_route(
                row.get(
                    "Question",
                    "",
                ),
                choices,
            )
        )

        final_prediction = (
            visual_prediction
        )

        # Text overrides only when it provides
        # explicit modality evidence.

        if (
            text_prediction
            is not None
        ):

            final_prediction = (
                text_prediction
            )

            always_text_overrides += 1

        always_hybrid_predictions.append(
            final_prediction
        )

    always_hybrid_accuracy = (
        print_results(
            "STRATEGY 3: ALWAYS ALLOW STRONG TEXT OVERRIDE",
            y_true,
            always_hybrid_predictions,
        )
    )

    print(
        f"\nStrong text overrides: "
        f"{always_text_overrides}"
    )

    # --------------------------------------------------------
    # RESULTS SUMMARY
    # --------------------------------------------------------

    print()

    print("=" * 70)

    print(
        "STRATEGY COMPARISON"
    )

    print("=" * 70)

    summary_rows = [

        {
            "strategy":
                "pure_visual",
            "threshold":
                None,
            "accuracy":
                visual_accuracy,
            "low_confidence":
                0,
            "text_overrides":
                0,
        }

    ]

    summary_rows.extend(
        strategy_results
    )

    summary_rows.append(
        {
            "strategy":
                "always_strong_text",
            "threshold":
                None,
            "accuracy":
                always_hybrid_accuracy,
            "low_confidence":
                None,
            "text_overrides":
                always_text_overrides,
        }
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df = summary_df.sort_values(
        "accuracy",
        ascending=False,
    )

    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output_path = Path(
        "training_data/"
        "router_strategy_comparison.csv"
    )

    summary_df.to_csv(
        output_path,
        index=False,
    )

    print()

    print("=" * 70)

    print(
        "BEST STRATEGY"
    )

    print("=" * 70)

    best = summary_df.iloc[0]

    print()

    print(
        f"Strategy: "
        f"{best['strategy']}"
    )

    print(
        f"Validation accuracy: "
        f"{best['accuracy']:.4%}"
    )

    if pd.notna(
        best["threshold"]
    ):

        print(
            f"Threshold: "
            f"{best['threshold']}"
        )

    print()

    print(
        "Results saved to:"
    )

    print(
        output_path
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "modality_final_test.csv "
        "was NOT loaded or used."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
    