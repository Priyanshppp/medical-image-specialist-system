from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALIDATION_FEATURES = (
    PROJECT_ROOT
    / "training_data"
    / "ct_validation_features.pkl"
)

VALIDATION_MANIFEST = (
    PROJECT_ROOT
    / "training_data"
    / "ct_validation_manifest.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "ct_specialist.joblib"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "training_data"
    / "ct_candidate_selection_results.csv"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalize labels for matching.

    Handles:
    - case differences
    - leading/trailing spaces
    - repeated spaces
    - trailing periods
    """

    if value is None:
        return ""

    text = str(value).strip().lower()

    text = " ".join(text.split())

    # Remove trailing punctuation differences such as:
    # "No" vs "No."
    text = text.rstrip(".")

    return text


# ============================================================
# LOAD DATA
# ============================================================

def load_feature_data(path):

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}"
        )

    data = joblib.load(path)

    required = [
        "features",
        "labels",
        "query_ids",
        "image_paths",
    ]

    for key in required:

        if key not in data:
            raise KeyError(
                f"Missing key '{key}' "
                f"in feature file"
            )

    return data


# ============================================================
# GET SCORE FOR ONE CHOICE
# ============================================================

def get_choice_score(
    choice_text,
    classes,
    probabilities,
):
    """
    Return classifier probability corresponding
    to a candidate answer choice.

    Matching is performed after normalization.
    """

    normalized_choice = normalize_text(
        choice_text
    )

    for index, class_name in enumerate(
        classes
    ):

        normalized_class = normalize_text(
            class_name
        )

        if normalized_choice == normalized_class:

            return float(
                probabilities[index]
            )

    # Candidate class was never seen in training.
    return 0.0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "CT CANDIDATE-CONSTRAINED EVALUATION"
    )
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Only CT validation data is loaded."
    )
    print(
        "Final test data is NOT loaded."
    )

    # --------------------------------------------------------
    # Load validation features
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "LOADING VALIDATION FEATURES"
    )
    print("=" * 70)

    validation_data = load_feature_data(
        VALIDATION_FEATURES
    )

    X_validation = (
        validation_data["features"]
    )

    y_validation = (
        validation_data["labels"]
    )

    query_ids = (
        validation_data["query_ids"]
    )

    print(
        f"Feature shape: "
        f"{X_validation.shape}"
    )

    print(
        f"Labels: "
        f"{len(y_validation)}"
    )

    # --------------------------------------------------------
    # Load validation manifest
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "LOADING VALIDATION MANIFEST"
    )
    print("=" * 70)

    if not VALIDATION_MANIFEST.exists():

        raise FileNotFoundError(
            f"Manifest not found: "
            f"{VALIDATION_MANIFEST}"
        )

    manifest = pd.read_csv(
        VALIDATION_MANIFEST
    )

    print(
        f"Manifest samples: "
        f"{len(manifest)}"
    )

    # --------------------------------------------------------
    # Align features with manifest
    # --------------------------------------------------------

    manifest = manifest.set_index(
        "query_id"
    )

    feature_query_ids = [
        str(x)
        for x in query_ids
    ]

    missing_ids = [
        query_id
        for query_id in feature_query_ids
        if query_id not in manifest.index
    ]

    if missing_ids:

        raise RuntimeError(
            "Feature IDs missing from "
            f"manifest: {missing_ids}"
        )

    manifest = manifest.loc[
        feature_query_ids
    ].reset_index()

    print(
        "Feature and manifest alignment "
        "successful."
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "LOADING CT SPECIALIST"
    )
    print("=" * 70)

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: "
            f"{MODEL_PATH}"
        )

    model = joblib.load(
        MODEL_PATH
    )

    print(
        f"Model loaded: "
        f"{type(model).__name__}"
    )

    classes = model.classes_

    print()
    print(
        "Known training classes:"
    )

    for class_name in classes:

        print(
            f" - {class_name}"
        )

    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            X_validation
        )
    )

    raw_predictions = (
        model.predict(
            X_validation
        )
    )

    # --------------------------------------------------------
    # Candidate constrained evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "EVALUATING CANDIDATE-CONSTRAINED "
        "PREDICTIONS"
    )
    print("=" * 70)

    results = []

    raw_correct_count = 0
    constrained_correct_count = 0

    seen_answer_count = 0
    unseen_answer_count = 0

    seen_correct_count = 0
    unseen_correct_count = 0

    for index in range(
        len(X_validation)
    ):

        row = manifest.iloc[index]

        query_id = str(
            row["query_id"]
        )

        true_answer = str(
            row["answer"]
        )

        choices = {
            "A": row.get(
                "Option_A",
                "",
            ),
            "B": row.get(
                "Option_B",
                "",
            ),
            "C": row.get(
                "Option_C",
                "",
            ),
            "D": row.get(
                "Option_D",
                "",
            ),
        }

        # Remove NaN choices.
        cleaned_choices = {}

        for letter, choice in choices.items():

            if pd.isna(choice):
                continue

            choice_text = str(
                choice
            ).strip()

            if choice_text:
                cleaned_choices[
                    letter
                ] = choice_text

        # ----------------------------------------------------
        # Raw prediction
        # ----------------------------------------------------

        raw_prediction = str(
            raw_predictions[index]
        )

        raw_correct = (
            normalize_text(
                raw_prediction
            )
            ==
            normalize_text(
                true_answer
            )
        )

        if raw_correct:
            raw_correct_count += 1

        # ----------------------------------------------------
        # Score every candidate
        # ----------------------------------------------------

        candidate_scores = {}

        for letter, choice_text in (
            cleaned_choices.items()
        ):

            score = get_choice_score(
                choice_text,
                classes,
                probabilities[index],
            )

            candidate_scores[
                letter
            ] = score

        # ----------------------------------------------------
        # Select highest scoring choice
        # ----------------------------------------------------

        selected_letter = max(
            candidate_scores,
            key=candidate_scores.get,
        )

        selected_answer = (
            cleaned_choices[
                selected_letter
            ]
        )

        constrained_correct = (
            normalize_text(
                selected_answer
            )
            ==
            normalize_text(
                true_answer
            )
        )

        if constrained_correct:
            constrained_correct_count += 1

        # ----------------------------------------------------
        # Check if answer class was seen
        # ----------------------------------------------------

        normalized_classes = {
            normalize_text(x)
            for x in classes
        }

        answer_seen_in_train = (
            normalize_text(
                true_answer
            )
            in normalized_classes
        )

        if answer_seen_in_train:

            seen_answer_count += 1

            if constrained_correct:
                seen_correct_count += 1

        else:

            unseen_answer_count += 1

            if constrained_correct:
                unseen_correct_count += 1

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print()
        print(
            f"[{index + 1}/"
            f"{len(X_validation)}] "
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
            f"{selected_letter} -> "
            f"{selected_answer}"
        )

        print(
            f"Answer seen in train: "
            f"{answer_seen_in_train}"
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
            candidate_scores.items()
        ):

            print(
                f"  {letter}: "
                f"{cleaned_choices[letter]} "
                f"-> {score:.4f}"
            )

        # ----------------------------------------------------
        # Save row
        # ----------------------------------------------------

        result_row = {
            "query_id": query_id,
            "true_answer": true_answer,
            "raw_prediction": raw_prediction,
            "selected_letter": selected_letter,
            "selected_answer": selected_answer,
            "answer_seen_in_train": (
                answer_seen_in_train
            ),
            "raw_correct": raw_correct,
            "constrained_correct": (
                constrained_correct
            ),
        }

        for letter, choice_text in (
            cleaned_choices.items()
        ):

            result_row[
                f"option_{letter}"
            ] = choice_text

            result_row[
                f"score_{letter}"
            ] = candidate_scores[
                letter
            ]

        results.append(
            result_row
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    total_samples = len(
        X_validation
    )

    raw_accuracy = (
        raw_correct_count
        / total_samples
    )

    constrained_accuracy = (
        constrained_correct_count
        / total_samples
    )

    improvement = (
        constrained_accuracy
        - raw_accuracy
    )

    print()
    print("=" * 70)
    print(
        "RESULTS"
    )
    print("=" * 70)

    print()
    print(
        f"Raw classifier accuracy:         "
        f"{raw_accuracy:.4%}"
    )

    print(
        f"Candidate-constrained accuracy:  "
        f"{constrained_accuracy:.4%}"
    )

    print(
        f"Difference:                      "
        f"{improvement:+.4%}"
    )

    # ========================================================
    # SEEN VS UNSEEN
    # ========================================================

    print()
    print("=" * 70)
    print(
        "SEEN VS UNSEEN ANSWER ANALYSIS"
    )
    print("=" * 70)

    print()
    print(
        f"Seen-answer samples: "
        f"{seen_answer_count}"
    )

    if seen_answer_count > 0:

        seen_raw_correct = sum(
            1
            for result in results
            if result[
                "answer_seen_in_train"
            ]
            and result[
                "raw_correct"
            ]
        )

        seen_raw_accuracy = (
            seen_raw_correct
            / seen_answer_count
        )

        seen_constrained_accuracy = (
            seen_correct_count
            / seen_answer_count
        )

        print(
            f"Raw accuracy: "
            f"{seen_raw_accuracy:.4%}"
        )

        print(
            f"Constrained accuracy: "
            f"{seen_constrained_accuracy:.4%}"
        )

    print()
    print(
        f"Unseen-answer samples: "
        f"{unseen_answer_count}"
    )

    if unseen_answer_count > 0:

        unseen_constrained_accuracy = (
            unseen_correct_count
            / unseen_answer_count
        )

        print(
            f"Constrained accuracy: "
            f"{unseen_constrained_accuracy:.4%}"
        )

        unseen_labels = sorted(
            {
                result["true_answer"]
                for result in results
                if not result[
                    "answer_seen_in_train"
                ]
            }
        )

        print()
        print(
            "Unseen answer labels:"
        )

        for label in unseen_labels:

            print(
                f" - {label}"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print(
        "SAVING RESULTS"
    )
    print("=" * 70)

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_PATH
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "FINAL SUMMARY"
    )
    print("=" * 70)

    print()
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
            "no change."
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