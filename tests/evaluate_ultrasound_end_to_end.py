from pathlib import Path
import sys

import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT)
)

from src.models.ultrasound import (
    UltrasoundModel
)


MANIFEST_PATH = (
    PROJECT_ROOT
    / "training_data"
    / "ultrasound_validation_manifest.csv"
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "test_data"
)


def normalize(text):

    return str(text).strip().lower()


def main():

    print()
    print("=" * 70)
    print(
        "ULTRASOUND END-TO-END VALIDATION"
    )
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Only Ultrasound validation data "
        "is being used."
    )
    print(
        "Final test data is NOT loaded."
    )

    # --------------------------------------------------
    # Load validation manifest
    # --------------------------------------------------

    df = pd.read_csv(
        MANIFEST_PATH
    )

    print()
    print(
        f"Validation samples: "
        f"{len(df)}"
    )

    # --------------------------------------------------
    # Initialize model
    # --------------------------------------------------

    print()
    print(
        "Initializing UltrasoundModel..."
    )

    model = UltrasoundModel()

    correct = 0

    results = []

    # --------------------------------------------------
    # Evaluate
    # --------------------------------------------------

    for index, row in df.iterrows():

        print()

        print("=" * 70)

        print(
            f"[{index + 1}/{len(df)}]"
        )

        print(
            f"Query ID: "
            f"{row['query_id']}"
        )

        image_path = (
            IMAGE_ROOT
            / Path(
                row["image"]
            )
        )

        if not image_path.exists():

            print(
                f"Missing image: "
                f"{image_path}"
            )

            continue

        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        choices = {
            "A": row["Option_A"],
            "B": row["Option_B"],
            "C": row["Option_C"],
            "D": row["Option_D"],
        }

        # Remove NaN choices
        choices = {
            key: value
            for key, value in choices.items()
            if pd.notna(value)
        }

        prediction = model.answer(
            images=[image],
            question=row["Question"],
            choices=choices,
        )

        true_answer = normalize(
            row["answer"]
        )

        predicted_text = normalize(
            choices.get(
                prediction,
                "",
            )
        )

        is_correct = (
            predicted_text
            == true_answer
        )

        if is_correct:

            correct += 1

        print()

        print(
            f"Ground truth: "
            f"{row['answer']}"
        )

        print(
            f"Prediction letter: "
            f"{prediction}"
        )

        print(
            f"Prediction text: "
            f"{choices.get(prediction)}"
        )

        print(
            f"Correct: "
            f"{is_correct}"
        )

        results.append(
            {
                "query_id": row[
                    "query_id"
                ],
                "ground_truth": row[
                    "answer"
                ],
                "prediction_letter": prediction,
                "prediction_text": choices.get(
                    prediction
                ),
                "correct": is_correct,
            }
        )

    # --------------------------------------------------
    # Final results
    # --------------------------------------------------

    total = len(
        results
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    print()

    print("=" * 70)
    print(
        "FINAL RESULTS"
    )
    print("=" * 70)

    print()

    print(
        f"Correct: "
        f"{correct}/{total}"
    )

    print(
        f"Accuracy: "
        f"{accuracy:.4%}"
    )

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------

    output_path = (
        PROJECT_ROOT
        / "training_data"
        / "ultrasound_end_to_end_results.csv"
    )

    pd.DataFrame(
        results
    ).to_csv(
        output_path,
        index=False,
    )

    print()

    print(
        f"Results saved to:\n"
        f"{output_path}"
    )

    print()

    print(
        "Final test data was NOT loaded."
    )


if __name__ == "__main__":
    main()