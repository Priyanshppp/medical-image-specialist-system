from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAINING_DATA = PROJECT_ROOT / "training_data"
MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).strip().lower()

    # Remove trailing punctuation differences.
    text = text.rstrip(".")
    text = text.rstrip(",")

    # Normalize repeated spaces.
    text = " ".join(text.split())

    return text


# ============================================================
# LOAD DATA
# ============================================================

def load_features(path):

    print(f"Loading: {path.relative_to(PROJECT_ROOT)}")

    data = joblib.load(path)

    print("Loaded successfully using joblib.")

    return data


# ============================================================
# GET CANDIDATE COLUMNS
# ============================================================

def find_candidate_columns(df):

    possible_patterns = [
        ["Option_A", "Option_B", "Option_C", "Option_D"],
        ["option_a", "option_b", "option_c", "option_d"],
        ["A", "B", "C", "D"],
        ["choice_a", "choice_b", "choice_c", "choice_d"],
    ]

    for pattern in possible_patterns:

        if all(column in df.columns for column in pattern):
            return pattern

    raise ValueError(
        "Could not identify candidate option columns.\n"
        f"Available columns: {list(df.columns)}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modality",
        required=True,
        type=str,
        help="Modality name, e.g. dermatology",
    )

    args = parser.parse_args()

    modality = args.modality.lower()

    print()
    print("=" * 70)
    print(
        f"{modality.upper()} CANDIDATE-CONSTRAINED EVALUATION"
    )
    print("=" * 70)

    print()
    print(
        "IMPORTANT:\n"
        "Only specialist validation data is loaded.\n"
        "Final test data is NOT loaded."
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    validation_features_path = (
        TRAINING_DATA
        / f"{modality}_validation_features.pkl"
    )

    validation_manifest_path = (
        TRAINING_DATA
        / f"{modality}_validation_manifest.csv"
    )

    model_path = (
        MODELS_DIR
        / f"{modality}_specialist.joblib"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    for path in [
        validation_features_path,
        validation_manifest_path,
        model_path,
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

    validation_data = load_features(
        validation_features_path
    )

    X_validation = validation_data["features"]
    y_validation = validation_data["labels"]
    query_ids = validation_data["query_ids"]

    print(
        f"Feature shape: {X_validation.shape}"
    )

    print(
        f"Labels: {len(y_validation)}"
    )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING VALIDATION MANIFEST")
    print("=" * 70)

    manifest = pd.read_csv(
        validation_manifest_path
    )

    print(
        f"Manifest samples: {len(manifest)}"
    )

    if len(manifest) != len(X_validation):

        raise RuntimeError(
            "Feature and manifest lengths do not match."
        )

    # Align by query ID if possible.
    if "query_id" in manifest.columns:

        manifest = (
            manifest
            .set_index("query_id")
            .loc[query_ids]
            .reset_index()
        )

    print(
        "Feature and manifest alignment successful."
    )

    # --------------------------------------------------------
    # Find answer column
    # --------------------------------------------------------

    if "answer" not in manifest.columns:

        raise ValueError(
            "Manifest does not contain an 'answer' column."
        )

    # --------------------------------------------------------
    # Candidate columns
    # --------------------------------------------------------

    candidate_columns = find_candidate_columns(
        manifest
    )

    print(
        f"Candidate columns: {candidate_columns}"
    )

    # --------------------------------------------------------
    # Load specialist
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"LOADING {modality.upper()} SPECIALIST"
    )
    print("=" * 70)

    loaded_model = joblib.load(
    model_path
)

print(
    f"Loaded object type: "
    f"{type(loaded_model).__name__}"
)

# Some specialist files store metadata together
# with the trained classifier.
if isinstance(loaded_model, dict):

    print(
        f"Model dictionary keys: "
        f"{list(loaded_model.keys())}"
    )

    if "model" in loaded_model:

        model = loaded_model["model"]

    elif "classifier" in loaded_model:

        model = loaded_model["classifier"]

    else:

        raise KeyError(
            "Model file is a dictionary, but neither "
            "'model' nor 'classifier' key was found."
        )

else:

    model = loaded_model


print(
    f"Actual classifier: "
    f"{type(model).__name__}"
)

classes = model.classes_

    print()
    print("Known training classes:")

    for label in classes:

        print(f" - {label}")

    # --------------------------------------------------------
    # Raw predictions
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_validation
    )

    raw_predictions = model.predict(
        X_validation
    )

    # Normalized class lookup.
    normalized_class_map = {}

    for index, class_name in enumerate(classes):

        normalized = normalize_text(
            class_name
        )

        normalized_class_map[normalized] = (
            index,
            class_name,
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "EVALUATING CANDIDATE-CONSTRAINED PREDICTIONS"
    )
    print("=" * 70)

    results = []

    raw_correct_count = 0
    constrained_correct_count = 0

    seen_samples = []
    unseen_samples = []

    for index in range(len(manifest)):

        row = manifest.iloc[index]

        true_answer = str(
            row["answer"]
        )

        raw_prediction = str(
            raw_predictions[index]
        )

        candidates = []

        for column in candidate_columns:

            candidates.append(
                str(row[column])
            )

        choice_scores = []

        for candidate in candidates:

            normalized_candidate = normalize_text(
                candidate
            )

            score = 0.0
            matched_label = None

            # Exact normalized match.
            if normalized_candidate in normalized_class_map:

                class_index, matched_label = (
                    normalized_class_map[
                        normalized_candidate
                    ]
                )

                score = float(
                    probabilities[
                        index,
                        class_index,
                    ]
                )

            choice_scores.append(
                score
            )

        best_choice_index = int(
            np.argmax(choice_scores)
        )

        constrained_answer = (
            candidates[best_choice_index]
        )

        constrained_score = (
            choice_scores[best_choice_index]
        )

        raw_correct = (
            normalize_text(raw_prediction)
            ==
            normalize_text(true_answer)
        )

        constrained_correct = (
            normalize_text(constrained_answer)
            ==
            normalize_text(true_answer)
        )

        answer_seen = (
            normalize_text(true_answer)
            in normalized_class_map
        )

        if raw_correct:
            raw_correct_count += 1

        if constrained_correct:
            constrained_correct_count += 1

        result = {
            "query_id": query_ids[index],
            "true_answer": true_answer,
            "raw_prediction": raw_prediction,
            "constrained_answer": constrained_answer,
            "constrained_score": constrained_score,
            "raw_correct": raw_correct,
            "constrained_correct": constrained_correct,
            "answer_seen_in_train": answer_seen,
        }

        results.append(result)

        if answer_seen:
            seen_samples.append(result)
        else:
            unseen_samples.append(result)

        letters = ["A", "B", "C", "D"]

        print()
        print(
            f"[{index + 1}/{len(manifest)}] "
            f"{query_ids[index]}"
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
            f"{letters[best_choice_index]} "
            f"-> {constrained_answer}"
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

        print("Choice scores:")

        for choice_index, candidate in enumerate(candidates):

            print(
                f"  {letters[choice_index]}: "
                f"{candidate} "
                f"-> {choice_scores[choice_index]:.4f}"
            )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    total = len(results)

    raw_accuracy = (
        raw_correct_count / total
    )

    constrained_accuracy = (
        constrained_correct_count / total
    )

    print()
    print("=" * 70)
    print("RESULTS")
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
        f"{(constrained_accuracy - raw_accuracy):+.4%}"
    )

    # --------------------------------------------------------
    # Seen vs unseen analysis
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SEEN VS UNSEEN ANSWER ANALYSIS")
    print("=" * 70)

    if seen_samples:

        seen_raw_accuracy = np.mean(
            [
                sample["raw_correct"]
                for sample in seen_samples
            ]
        )

        seen_constrained_accuracy = np.mean(
            [
                sample["constrained_correct"]
                for sample in seen_samples
            ]
        )

        print()
        print(
            f"Seen-answer samples: "
            f"{len(seen_samples)}"
        )

        print(
            f"Raw accuracy: "
            f"{seen_raw_accuracy:.4%}"
        )

        print(
            f"Constrained accuracy: "
            f"{seen_constrained_accuracy:.4%}"
        )

    else:

        print()
        print("Seen-answer samples: 0")

    if unseen_samples:

        unseen_constrained_accuracy = np.mean(
            [
                sample["constrained_correct"]
                for sample in unseen_samples
            ]
        )

        print()
        print(
            f"Unseen-answer samples: "
            f"{len(unseen_samples)}"
        )

        print(
            f"Constrained accuracy: "
            f"{unseen_constrained_accuracy:.4%}"
        )

        print()
        print("Unseen answer labels:")

        unseen_labels = sorted(
            set(
                sample["true_answer"]
                for sample in unseen_samples
            )
        )

        for label in unseen_labels:

            print(f" - {label}")

    else:

        print()
        print("Unseen-answer samples: 0")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_path = (
        TRAINING_DATA
        / f"{modality}_candidate_selection_results.csv"
    )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    print()
    print(
        f"Results saved to:\n"
        f"{output_path}"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print()
    print(
        f"Modality:              "
        f"{modality.upper()}"
    )

    print(
        f"Raw classifier:        "
        f"{raw_accuracy:.4%}"
    )

    print(
        f"Candidate constrained: "
        f"{constrained_accuracy:.4%}"
    )

    print()

    if constrained_accuracy > raw_accuracy:

        print(
            "Candidate restriction improved "
            "the specialist."
        )

    else:

        print(
            "Candidate restriction did not "
            "improve this specialist."
        )

    print()
    print(
        "IMPORTANT: Final test data was NOT loaded."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
