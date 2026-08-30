from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report

from src.model import YourModel


# ============================================================
# PATHS
# ============================================================

MANIFEST_PATH = Path(
    "training_data/validation_manifest.csv"
)

FINAL_TEST_PATH = Path(
    "training_data/modality_final_test.csv"
)


# ============================================================
# HELPERS
# ============================================================

def build_choices(row):

    choices = {}

    option_columns = [
        "Option_A",
        "Option_B",
        "Option_C",
        "Option_D",
    ]

    letters = [
        "A",
        "B",
        "C",
        "D",
    ]

    for letter, column in zip(
        letters,
        option_columns,
    ):

        value = row.get(column)

        if pd.notna(value):
            choices[letter] = str(value)

    return choices


def normalize_answer(answer):

    if answer is None:
        return None

    answer = str(answer).strip()

    # Handle formats such as:
    # A
    # a
    # Option_A
    # option a

    lower = answer.lower()

    mapping = {
        "option_a": "A",
        "option_b": "B",
        "option_c": "C",
        "option_d": "D",
        "option a": "A",
        "option b": "B",
        "option c": "C",
        "option d": "D",
        "a": "A",
        "b": "B",
        "c": "C",
        "d": "D",
    }

    if lower in mapping:
        return mapping[lower]

    return answer


def normalize_prediction(prediction, choices):

    if prediction is None:
        return None

    prediction = str(prediction).strip()

    # Direct option letter
    normalized = normalize_answer(prediction)

    if normalized in choices:
        return normalized

    # If model returns answer text instead of letter,
    # match against choice text.

    prediction_lower = prediction.lower()

    for key, value in choices.items():

        if prediction_lower == str(value).strip().lower():
            return key

    return prediction


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("FULL PIPELINE VALIDATION EVALUATION")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    print("FINAL TEST SAFETY CHECK")
    print("-" * 70)

    if not MANIFEST_PATH.exists():

        raise FileNotFoundError(
            f"Validation manifest not found: "
            f"{MANIFEST_PATH}"
        )

    print(
        "Using validation manifest only:"
    )

    print(
        MANIFEST_PATH
    )

    print()
    print(
        "Final test file will NOT be loaded:"
    )

    print(
        FINAL_TEST_PATH
    )

    print()

    # --------------------------------------------------------
    # LOAD VALIDATION DATA
    # --------------------------------------------------------

    df = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        f"Validation samples: {len(df)}"
    )

    print()

    # --------------------------------------------------------
    # INITIALIZE COMPLETE MODEL
    # --------------------------------------------------------

    print("Initializing full model pipeline...")

    model = YourModel()

    print(
        "Full model initialized."
    )

    print()

    # --------------------------------------------------------
    # RESULTS STORAGE
    # --------------------------------------------------------

    results = []

    correct_answers = []
    predicted_answers = []

    correct_modalities = []
    predicted_modalities = []

    # --------------------------------------------------------
    # EVALUATION LOOP
    # --------------------------------------------------------

    print("=" * 70)
    print("RUNNING FULL PIPELINE")
    print("=" * 70)
    print()

    for index, row in df.iterrows():

        query_id = row["query_id"]

        image_path = Path(
            row["image"]
        )

        question = str(
            row["Question"]
        )

        true_answer = normalize_answer(
            row["answer"]
        )

        true_modality = str(
            row["modality"]
        )

        choices = build_choices(
            row
        )

        # ----------------------------------------------------
        # LOAD IMAGE
        # ----------------------------------------------------

        if not image_path.exists():

            print(
                f"[{index + 1}/{len(df)}] "
                f"IMAGE NOT FOUND"
            )

            print(
                f"Query ID: {query_id}"
            )

            print(
                f"Path: {image_path}"
            )

            continue

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as error:

            print(
                f"Could not load image: "
                f"{image_path}"
            )

            print(
                f"Error: {error}"
            )

            continue

        # ----------------------------------------------------
        # GET ROUTING PREDICTION
        # ----------------------------------------------------

        try:

            selected_domain = model.router.route(
                question=question,
                choices=choices,
                images=[image],
            )

        except Exception as error:

            print(
                f"Routing error for "
                f"{query_id}: {error}"
            )

            selected_domain = "ERROR"

        # ----------------------------------------------------
        # GET FINAL ANSWER
        # ----------------------------------------------------

        try:

            prediction = model.answer(
                images=[image],
                question=question,
                choices=choices,
            )

        except Exception as error:

            print(
                f"Model error for "
                f"{query_id}: {error}"
            )

            prediction = None

        predicted_answer = normalize_prediction(
            prediction,
            choices,
        )

        # ----------------------------------------------------
        # EVALUATE
        # ----------------------------------------------------

        answer_correct = (
            predicted_answer == true_answer
        )

        # Router domain names differ from dataset labels.
        modality_map = {

            "CT": "ct",

            "MRI": "brain_mri",

            "X-ray": "chest_xray",

            "Fundus": "general",

            "Dermatology": "dermoscopy",

            "Microscopy": "microscopy",

            "OCT": "oct",

            "Ultrasound": "ultrasound",
        }

        expected_domain = modality_map.get(
            true_modality,
            "general",
        )

        routing_correct = (
            selected_domain == expected_domain
        )

        correct_answers.append(
            true_answer
        )

        predicted_answers.append(
            predicted_answer
        )

        correct_modalities.append(
            expected_domain
        )

        predicted_modalities.append(
            selected_domain
        )

        # ----------------------------------------------------
        # SAVE RESULT
        # ----------------------------------------------------

        results.append({

            "query_id": query_id,

            "true_modality": true_modality,

            "expected_domain": expected_domain,

            "selected_domain": selected_domain,

            "routing_correct": routing_correct,

            "true_answer": true_answer,

            "predicted_answer": predicted_answer,

            "answer_correct": answer_correct,

            "question": question,

        })

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        print(
            f"[{index + 1}/{len(df)}] "
            f"{query_id}"
        )

        print(
            f"Modality: "
            f"{true_modality}"
        )

        print(
            f"Route: "
            f"{selected_domain}"
        )

        print(
            f"Answer: "
            f"{predicted_answer}"
        )

        print(
            f"Expected: "
            f"{true_answer}"
        )

        print(
            f"Correct: "
            f"{answer_correct}"
        )

        print("-" * 50)

    # ========================================================
    # RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    if len(results_df) == 0:

        print(
            "No samples were successfully evaluated."
        )

        return

    # ========================================================
    # OVERALL ACCURACY
    # ========================================================

    answer_accuracy = (
        results_df["answer_correct"].mean()
    )

    routing_accuracy = (
        results_df["routing_correct"].mean()
    )

    print()
    print("=" * 70)
    print("FINAL VALIDATION RESULTS")
    print("=" * 70)

    print()

    print(
        f"Samples evaluated: "
        f"{len(results_df)}"
    )

    print(
        f"Routing accuracy: "
        f"{routing_accuracy:.4%}"
    )

    print(
        f"Answer accuracy: "
        f"{answer_accuracy:.4%}"
    )

    # ========================================================
    # PER-MODALITY RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("PER-MODALITY ANSWER ACCURACY")
    print("=" * 70)
    print()

    modality_summary = (
        results_df
        .groupby("true_modality")
        .agg(
            samples=(
                "answer_correct",
                "count",
            ),
            correct=(
                "answer_correct",
                "sum",
            ),
            accuracy=(
                "answer_correct",
                "mean",
            ),
            routing_accuracy=(
                "routing_correct",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        modality_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # ROUTING IMPACT
    # ========================================================

    print()
    print("=" * 70)
    print("ROUTING IMPACT ON ANSWER ACCURACY")
    print("=" * 70)
    print()

    correctly_routed = results_df[
        results_df["routing_correct"]
    ]

    incorrectly_routed = results_df[
        ~results_df["routing_correct"]
    ]

    if len(correctly_routed) > 0:

        print(
            "When routing was correct:"
        )

        print(
            f"Samples: "
            f"{len(correctly_routed)}"
        )

        print(
            f"Answer accuracy: "
            f"{correctly_routed['answer_correct'].mean():.4%}"
        )

        print()

    if len(incorrectly_routed) > 0:

        print(
            "When routing was incorrect:"
        )

        print(
            f"Samples: "
            f"{len(incorrectly_routed)}"
        )

        print(
            f"Answer accuracy: "
            f"{incorrectly_routed['answer_correct'].mean():.4%}"
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_path = Path(
        "training_data/"
        "full_pipeline_validation_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("SAVED")
    print("=" * 70)

    print()

    print(
        f"Detailed results saved to:"
    )

    print(
        output_path
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "modality_final_test.csv was NOT loaded."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()