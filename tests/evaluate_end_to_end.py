from pathlib import Path
import traceback

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.model import YourModel


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALIDATION_MANIFEST = (
    PROJECT_ROOT
    / "training_data"
    / "validation_manifest.csv"
)

FINAL_TEST_PATH = (
    PROJECT_ROOT
    / "training_data"
    / "modality_final_test.csv"
)


# ==========================================================
# IMAGE PATH RESOLUTION
# ==========================================================

def resolve_image_path(image_value):
    """
    Resolve image paths stored in the validation manifest.

    Manifest example:
        Images/007062.png

    Actual location:
        test_data/Images/007062.png
    """

    image_value = str(image_value).strip()

    original_path = Path(image_value)

    # ------------------------------------------------------
    # Absolute path
    # ------------------------------------------------------

    if (
        original_path.is_absolute()
        and original_path.exists()
    ):
        return original_path

    # ------------------------------------------------------
    # Project-relative path
    # ------------------------------------------------------

    candidate = (
        PROJECT_ROOT
        / original_path
    )

    if candidate.exists():
        return candidate

    # ------------------------------------------------------
    # Expected test-data location
    # ------------------------------------------------------

    candidate = (
        PROJECT_ROOT
        / "test_data"
        / original_path
    )

    if candidate.exists():
        return candidate

    # ------------------------------------------------------
    # Directly inside test_data/Images
    # ------------------------------------------------------

    candidate = (
        PROJECT_ROOT
        / "test_data"
        / "Images"
        / original_path.name
    )

    if candidate.exists():
        return candidate

    return None


# ==========================================================
# ANSWER NORMALIZATION
# ==========================================================

def normalize_text(value):
    """
    Normalize text for robust comparison.
    """

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
    )


def answer_to_option_letter(
    answer,
    choices,
):
    """
    Convert an answer into A/B/C/D.

    Handles:
        A
        B
        C
        D

    Also handles answer text:
        "fibroid" -> "D"
    """

    answer_normalized = normalize_text(
        answer
    )

    # ------------------------------------------------------
    # Already an option letter
    # ------------------------------------------------------

    valid_letters = {
        "a",
        "b",
        "c",
        "d",
    }

    if answer_normalized in valid_letters:
        return answer_normalized.upper()

    # ------------------------------------------------------
    # Match against choice text
    # ------------------------------------------------------

    for letter, choice_text in choices.items():

        choice_normalized = normalize_text(
            choice_text
        )

        if (
            answer_normalized
            == choice_normalized
        ):
            return letter.upper()

    # ------------------------------------------------------
    # No match
    # ------------------------------------------------------

    return None


# ==========================================================
# MAIN EVALUATION
# ==========================================================

def main():

    print("=" * 70)
    print("END-TO-END VALIDATION EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------
    # FINAL TEST SAFETY CHECK
    # ------------------------------------------------------

    print("\nFINAL TEST SAFETY CHECK")
    print("-" * 70)

    print(
        "This evaluation uses ONLY:"
    )

    print(
        VALIDATION_MANIFEST
    )

    print(
        "\nThe following file will NOT be loaded:"
    )

    print(
        FINAL_TEST_PATH
    )

    print(
        "\nFinal test set remains untouched."
    )

    # ------------------------------------------------------
    # LOAD VALIDATION DATA
    # ------------------------------------------------------

    if not VALIDATION_MANIFEST.exists():

        raise FileNotFoundError(
            f"Validation manifest not found:\n"
            f"{VALIDATION_MANIFEST}"
        )

    df = pd.read_csv(
        VALIDATION_MANIFEST
    )

    print("\n" + "=" * 70)
    print("VALIDATION DATA")
    print("=" * 70)

    print(
        f"\nValidation samples: {len(df)}"
    )

    # ------------------------------------------------------
    # CHECK REQUIRED COLUMNS
    # ------------------------------------------------------

    required_columns = [
        "query_id",
        "image",
        "Question",
        "Option_A",
        "Option_B",
        "Option_C",
        "Option_D",
        "answer",
        "modality",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns:\n"
            f"{missing_columns}"
        )

    # ------------------------------------------------------
    # INITIALIZE MODEL ONCE
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)

    model = YourModel()

    # ------------------------------------------------------
    # STORAGE
    # ------------------------------------------------------

    results = []

    correct_count = 0
    error_count = 0

    total_samples = len(df)

    # ------------------------------------------------------
    # EVALUATE EACH SAMPLE
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("RUNNING END-TO-END EVALUATION")
    print("=" * 70)

    for index, row in df.iterrows():

        sample_number = index + 1

        query_id = row["query_id"]

        expected_modality = row["modality"]

        print(
            f"\n[{sample_number}/{total_samples}] "
            f"{query_id} "
            f"| Expected modality: "
            f"{expected_modality}"
        )

        # --------------------------------------------------
        # BUILD CHOICES
        # --------------------------------------------------

        choices = {
            "A": row["Option_A"],
            "B": row["Option_B"],
            "C": row["Option_C"],
            "D": row["Option_D"],
        }

        # --------------------------------------------------
        # GET EXPECTED ANSWER LETTER
        # --------------------------------------------------

        ground_truth_raw = row["answer"]

        expected_letter = (
            answer_to_option_letter(
                ground_truth_raw,
                choices,
            )
        )

        if expected_letter is None:

            print(
                "WARNING: Could not map "
                "ground truth to A/B/C/D"
            )

            error_count += 1

            results.append(
                {
                    "query_id": query_id,
                    "modality": expected_modality,
                    "image": row["image"],
                    "question": row["Question"],
                    "ground_truth_raw": ground_truth_raw,
                    "expected_option": None,
                    "prediction_raw": None,
                    "predicted_option": None,
                    "correct": False,
                    "status": "ground_truth_mapping_error",
                    "error": "",
                }
            )

            continue

        # --------------------------------------------------
        # RESOLVE IMAGE
        # --------------------------------------------------

        image_path = resolve_image_path(
            row["image"]
        )

        if image_path is None:

            print(
                "ERROR: Image not found"
            )

            error_count += 1

            results.append(
                {
                    "query_id": query_id,
                    "modality": expected_modality,
                    "image": row["image"],
                    "question": row["Question"],
                    "ground_truth_raw": ground_truth_raw,
                    "expected_option": expected_letter,
                    "prediction_raw": None,
                    "predicted_option": None,
                    "correct": False,
                    "status": "image_not_found",
                    "error": "",
                }
            )

            continue

        # --------------------------------------------------
        # LOAD IMAGE
        # --------------------------------------------------

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as error:

            print(
                f"ERROR loading image: {error}"
            )

            error_count += 1

            results.append(
                {
                    "query_id": query_id,
                    "modality": expected_modality,
                    "image": str(image_path),
                    "question": row["Question"],
                    "ground_truth_raw": ground_truth_raw,
                    "expected_option": expected_letter,
                    "prediction_raw": None,
                    "predicted_option": None,
                    "correct": False,
                    "status": "image_load_error",
                    "error": str(error),
                }
            )

            continue

        # --------------------------------------------------
        # RUN MODEL
        # --------------------------------------------------

        try:

            prediction = model.answer(
                images=[image],
                question=row["Question"],
                choices=choices,
            )

            predicted_letter = (
                answer_to_option_letter(
                    prediction,
                    choices,
                )
            )

            is_correct = (
                predicted_letter
                == expected_letter
            )

            if is_correct:
                correct_count += 1

            status = (
                "correct"
                if is_correct
                else "incorrect"
            )

            print(
                f"Expected: {expected_letter} "
                f"| Predicted: {predicted_letter} "
                f"| {status.upper()}"
            )

            results.append(
                {
                    "query_id": query_id,
                    "modality": expected_modality,
                    "image": str(image_path),
                    "question": row["Question"],
                    "ground_truth_raw": ground_truth_raw,
                    "expected_option": expected_letter,
                    "prediction_raw": prediction,
                    "predicted_option": predicted_letter,
                    "correct": is_correct,
                    "status": status,
                    "error": "",
                }
            )

        except Exception as error:

            print(
                f"MODEL ERROR: {error}"
            )

            traceback.print_exc()

            error_count += 1

            results.append(
                {
                    "query_id": query_id,
                    "modality": expected_modality,
                    "image": str(image_path),
                    "question": row["Question"],
                    "ground_truth_raw": ground_truth_raw,
                    "expected_option": expected_letter,
                    "prediction_raw": None,
                    "predicted_option": None,
                    "correct": False,
                    "status": "model_error",
                    "error": str(error),
                }
            )

    # ======================================================
    # CREATE RESULTS DATAFRAME
    # ======================================================

    results_df = pd.DataFrame(
        results
    )

    # ======================================================
    # OVERALL RESULTS
    # ======================================================

    valid_results = results_df[
        results_df["status"].isin(
            [
                "correct",
                "incorrect",
            ]
        )
    ].copy()

    print("\n" + "=" * 70)
    print("END-TO-END RESULTS")
    print("=" * 70)

    print(
        f"\nTotal validation samples: "
        f"{total_samples}"
    )

    print(
        f"Successfully evaluated: "
        f"{len(valid_results)}"
    )

    print(
        f"Evaluation errors: "
        f"{error_count}"
    )

    if len(valid_results) > 0:

        overall_accuracy = (
            valid_results["correct"]
            .mean()
        )

        print(
            f"\nOverall Accuracy: "
            f"{overall_accuracy:.4%}"
        )

        print(
            f"Correct predictions: "
            f"{int(valid_results['correct'].sum())}"
        )

        print(
            f"Incorrect predictions: "
            f"{int((~valid_results['correct']).sum())}"
        )

    else:

        overall_accuracy = 0.0

        print(
            "\nNo valid predictions available."
        )

    # ======================================================
    # PER-MODALITY RESULTS
    # ======================================================

    print("\n" + "=" * 70)
    print("PER-MODALITY ACCURACY")
    print("=" * 70)

    if len(valid_results) > 0:

        modality_results = (
            valid_results
            .groupby("modality")
            .agg(
                samples=(
                    "correct",
                    "count",
                ),
                correct=(
                    "correct",
                    "sum",
                ),
                accuracy=(
                    "correct",
                    "mean",
                ),
            )
            .sort_values(
                "accuracy"
            )
        )

        modality_results["accuracy"] = (
            modality_results["accuracy"]
            * 100
        )

        print()

        print(
            modality_results.to_string(
                float_format="%.2f"
            )
        )

    else:

        modality_results = pd.DataFrame()

    # ======================================================
    # CLASSIFICATION REPORT
    # ======================================================

    if len(valid_results) > 0:

        print("\n" + "=" * 70)
        print("OPTION-LEVEL CLASSIFICATION REPORT")
        print("=" * 70)

        print()

        print(
            classification_report(
                valid_results[
                    "expected_option"
                ],
                valid_results[
                    "predicted_option"
                ],
                labels=[
                    "A",
                    "B",
                    "C",
                    "D",
                ],
                zero_division=0,
            )
        )

        print("\n" + "=" * 70)
        print("OPTION CONFUSION MATRIX")
        print("=" * 70)

        matrix = confusion_matrix(
            valid_results[
                "expected_option"
            ],
            valid_results[
                "predicted_option"
            ],
            labels=[
                "A",
                "B",
                "C",
                "D",
            ],
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=[
                "Actual_A",
                "Actual_B",
                "Actual_C",
                "Actual_D",
            ],
            columns=[
                "Pred_A",
                "Pred_B",
                "Pred_C",
                "Pred_D",
            ],
        )

        print()
        print(matrix_df)

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    output_path = (
        PROJECT_ROOT
        / "training_data"
        / "end_to_end_validation_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        f"\nDetailed results:\n"
        f"{output_path}"
    )

    # ======================================================
    # SAVE MODALITY SUMMARY
    # ======================================================

    if not modality_results.empty:

        modality_output_path = (
            PROJECT_ROOT
            / "training_data"
            / "end_to_end_modality_results.csv"
        )

        modality_results.to_csv(
            modality_output_path
        )

        print(
            f"\nPer-modality summary:\n"
            f"{modality_output_path}"
        )

    # ======================================================
    # FINAL SAFETY STATEMENT
    # ======================================================

    print("\n" + "=" * 70)
    print("FINAL TEST SAFETY")
    print("=" * 70)

    print(
        "\nmodality_final_test.csv "
        "was NOT loaded."
    )

    print(
        "No final test samples were used."
    )

    print(
        "Final test data remains untouched."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()