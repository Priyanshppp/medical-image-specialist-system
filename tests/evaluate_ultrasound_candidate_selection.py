from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_PATH = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound_validation_features.pkl"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound_validation_manifest.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ultrasound_specialist.joblib"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound_candidate_selection_results.csv"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text):
    """
    Normalize labels so that small formatting differences
    do not prevent matching.
    """

    if pd.isna(text):
        return ""

    text = str(text).strip().lower()

    text = " ".join(text.split())

    return text


# ============================================================
# LOAD DATA
# ============================================================

def main():

    print("=" * 70)
    print("ULTRASOUND CANDIDATE-CONSTRAINED EVALUATION")
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print("Only Ultrasound validation data is loaded.")
    print("Final test data is NOT loaded.")

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    for path in [
        FEATURES_PATH,
        MANIFEST_PATH,
        MODEL_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # --------------------------------------------------------
    # Load validation features
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING VALIDATION FEATURES")
    print("=" * 70)

    feature_data = joblib.load(
        FEATURES_PATH
    )

    X_val = np.asarray(
        feature_data["features"]
    )

    y_val = np.asarray(
        feature_data["labels"]
    )

    query_ids = np.asarray(
        feature_data["query_ids"]
    )

    print(
        f"Feature shape: {X_val.shape}"
    )

    print(
        f"Labels: {len(y_val)}"
    )

    # --------------------------------------------------------
    # Load validation manifest
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING VALIDATION MANIFEST")
    print("=" * 70)

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"Manifest samples: {len(manifest)}"
    )

    required_columns = [
        "query_id",
        "Option_A",
        "Option_B",
        "Option_C",
        "Option_D",
        "answer",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in manifest.columns
    ]

    if missing_columns:

        raise KeyError(
            "Missing manifest columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Align manifest with feature query IDs
    # --------------------------------------------------------

    manifest = manifest.set_index(
        "query_id"
    )

    aligned_rows = []

    for query_id in query_ids:

        if query_id not in manifest.index:

            raise KeyError(
                f"Query ID not found in manifest: "
                f"{query_id}"
            )

        row = manifest.loc[
            query_id
        ]

        aligned_rows.append(
            row
        )

    aligned_manifest = pd.DataFrame(
        aligned_rows
    ).reset_index()

    print(
        "Feature and manifest alignment successful."
    )

    # --------------------------------------------------------
    # Load specialist model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING ULTRASOUND SPECIALIST")
    print("=" * 70)

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Model loaded:",
        type(model).__name__
    )

    print()

    print(
        "Known training classes:"
    )

    for label in model.classes_:

        print(
            f" - {label}"
        )

    # --------------------------------------------------------
    # Get probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_val
    )

    classes = [
        normalize(label)
        for label in model.classes_
    ]

    # --------------------------------------------------------
    # Evaluation storage
    # --------------------------------------------------------

    raw_predictions = []

    constrained_predictions = []

    constrained_letters = []

    seen_answer_flags = []

    rows = []

    # --------------------------------------------------------
    # Evaluate each sample
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATING CANDIDATE-CONSTRAINED PREDICTIONS")
    print("=" * 70)

    for index in range(len(X_val)):

        query_id = query_ids[index]

        true_answer = normalize(
            y_val[index]
        )

        row = aligned_manifest.iloc[
            index
        ]

        # ----------------------------------------------------
        # Raw classifier prediction
        # ----------------------------------------------------

        sample_probs = probabilities[
            index
        ]

        raw_index = int(
            np.argmax(sample_probs)
        )

        raw_prediction = classes[
            raw_index
        ]

        raw_predictions.append(
            raw_prediction
        )

        # ----------------------------------------------------
        # Extract available choices
        # ----------------------------------------------------

        choices = {}

        option_columns = {
            "A": "Option_A",
            "B": "Option_B",
            "C": "Option_C",
            "D": "Option_D",
        }

        for letter, column in (
            option_columns.items()
        ):

            value = row[column]

            if pd.isna(value):
                continue

            value = str(value).strip()

            if not value:
                continue

            choices[letter] = value

        # ----------------------------------------------------
        # Score each available choice
        # ----------------------------------------------------

        choice_scores = {}

        for letter, choice_text in (
            choices.items()
        ):

            normalized_choice = normalize(
                choice_text
            )

            if normalized_choice in classes:

                class_index = classes.index(
                    normalized_choice
                )

                score = float(
                    sample_probs[class_index]
                )

            else:

                # Class was never seen during training.
                # The classifier cannot assign probability
                # to an unseen label.
                score = 0.0

            choice_scores[letter] = score

        # ----------------------------------------------------
        # Select highest scoring candidate
        # ----------------------------------------------------

        if not choice_scores:

            raise RuntimeError(
                f"No valid choices for "
                f"{query_id}"
            )

        best_letter = max(
            choice_scores,
            key=choice_scores.get,
        )

        best_prediction = normalize(
            choices[best_letter]
        )

        constrained_predictions.append(
            best_prediction
        )

        constrained_letters.append(
            best_letter
        )

        # ----------------------------------------------------
        # Check whether true answer was seen in training
        # ----------------------------------------------------

        answer_seen = (
            true_answer in classes
        )

        seen_answer_flags.append(
            answer_seen
        )

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        raw_correct = (
            raw_prediction == true_answer
        )

        constrained_correct = (
            best_prediction == true_answer
        )

        print()

        print(
            f"[{index + 1}/{len(X_val)}] "
            f"{query_id}"
        )

        print(
            f"True answer:          "
            f"{true_answer}"
        )

        print(
            f"Raw prediction:       "
            f"{raw_prediction}"
        )

        print(
            f"Constrained choice:   "
            f"{best_letter} -> "
            f"{best_prediction}"
        )

        print(
            f"Answer seen in train: "
            f"{answer_seen}"
        )

        print(
            f"Raw correct:          "
            f"{raw_correct}"
        )

        print(
            f"Constrained correct:  "
            f"{constrained_correct}"
        )

        print(
            "Choice scores:"
        )

        for letter, score in (
            choice_scores.items()
        ):

            print(
                f"  {letter}: "
                f"{choices[letter]} "
                f"-> {score:.4f}"
            )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result_row = {
            "query_id": query_id,
            "true_answer": true_answer,
            "raw_prediction": raw_prediction,
            "raw_correct": raw_correct,
            "constrained_letter": best_letter,
            "constrained_prediction": best_prediction,
            "constrained_correct": constrained_correct,
            "answer_seen_in_train": answer_seen,
        }

        for letter in ["A", "B", "C", "D"]:

            result_row[
                f"choice_{letter}"
            ] = choices.get(
                letter,
                ""
            )

            result_row[
                f"score_{letter}"
            ] = choice_scores.get(
                letter,
                np.nan,
            )

        rows.append(
            result_row
        )

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    raw_accuracy = accuracy_score(
        y_val,
        raw_predictions,
    )

    constrained_accuracy = accuracy_score(
        y_val,
        constrained_predictions,
    )

    print()

    print(
        f"Raw classifier accuracy:         "
        f"{raw_accuracy:.4%}"
    )

    print(
        f"Candidate-constrained accuracy:  "
        f"{constrained_accuracy:.4%}"
    )

    improvement = (
        constrained_accuracy
        - raw_accuracy
    )

    print(
        f"Difference:                      "
        f"{improvement:+.4%}"
    )

    # ========================================================
    # SEEN VS UNSEEN ANALYSIS
    # ========================================================

    results_df = pd.DataFrame(
        rows
    )

    seen_df = results_df[
        results_df[
            "answer_seen_in_train"
        ]
    ]

    unseen_df = results_df[
        ~results_df[
            "answer_seen_in_train"
        ]
    ]

    print()
    print("=" * 70)
    print("SEEN VS UNSEEN ANSWER ANALYSIS")
    print("=" * 70)

    print()

    print(
        f"Seen-answer samples: "
        f"{len(seen_df)}"
    )

    if len(seen_df) > 0:

        seen_raw = (
            seen_df["raw_correct"]
            .mean()
        )

        seen_constrained = (
            seen_df[
                "constrained_correct"
            ]
            .mean()
        )

        print(
            f"Raw accuracy: "
            f"{seen_raw:.4%}"
        )

        print(
            f"Constrained accuracy: "
            f"{seen_constrained:.4%}"
        )

    print()

    print(
        f"Unseen-answer samples: "
        f"{len(unseen_df)}"
    )

    if len(unseen_df) > 0:

        unseen_constrained = (
            unseen_df[
                "constrained_correct"
            ]
            .mean()
        )

        print(
            f"Constrained accuracy: "
            f"{unseen_constrained:.4%}"
        )

        print()

        print(
            "Unseen answer labels:"
        )

        for label in sorted(
            unseen_df[
                "true_answer"
            ].unique()
        ):

            print(
                f" - {label}"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()

    print(
        f"Results saved to:\n"
        f"{OUTPUT_PATH.resolve()}"
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"Raw classifier:        "
        f"{raw_accuracy:.4%}"
    )

    print(
        f"Candidate constrained: "
        f"{constrained_accuracy:.4%}"
    )

    if constrained_accuracy > raw_accuracy:

        print()
        print(
            "Candidate restriction improved "
            "the specialist."
        )

    elif constrained_accuracy == raw_accuracy:

        print()
        print(
            "Candidate restriction produced "
            "no improvement."
        )

    else:

        print()
        print(
            "Candidate restriction reduced "
            "performance."
        )

    print()
    print(
        "IMPORTANT: Final test data was NOT loaded."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()