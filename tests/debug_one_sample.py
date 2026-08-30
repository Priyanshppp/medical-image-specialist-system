from pathlib import Path

import pandas as pd
from PIL import Image

from src.model import YourModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_ROOTS = [
    PROJECT_ROOT / "test_data",
    PROJECT_ROOT / "test_data" / "Images",
    PROJECT_ROOT / "Images",
    PROJECT_ROOT / "data",
]


def resolve_image_path(image_value):
    """
    Resolve an image path stored in the manifest.

    Example manifest value:
        Images/007062.png

    Actual location:
        test_data/Images/007062.png
    """

    image_value = str(image_value).strip()

    original_path = Path(image_value)

    # --------------------------------------------------
    # 1. Already absolute
    # --------------------------------------------------

    if original_path.is_absolute() and original_path.exists():
        return original_path

    # --------------------------------------------------
    # 2. Relative to project root
    # --------------------------------------------------

    candidate = PROJECT_ROOT / original_path

    if candidate.exists():
        return candidate

    # --------------------------------------------------
    # 3. Relative to known image roots
    # --------------------------------------------------

    for root in IMAGE_ROOTS:

        # Try complete relative path
        candidate = root / original_path

        if candidate.exists():
            return candidate

        # Try only filename
        candidate = root / original_path.name

        if candidate.exists():
            return candidate

    # --------------------------------------------------
    # 4. Recursive fallback
    # --------------------------------------------------

    matches = list(
        PROJECT_ROOT.rglob(
            original_path.name
        )
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:

        print(
            "\nWARNING: Multiple matching images found:"
        )

        for match in matches:
            print(match)

        return matches[0]

    return None


def main():

    print("=" * 70)
    print("DEBUGGING ONE END-TO-END SAMPLE")
    print("=" * 70)

    manifest_path = (
        PROJECT_ROOT
        / "training_data"
        / "validation_manifest.csv"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Validation manifest not found: "
            f"{manifest_path}"
        )

    df = pd.read_csv(
        manifest_path
    )

    print(f"\nValidation samples: {len(df)}")

    row = df.iloc[0]

    print("\nQuery ID:")
    print(row["query_id"])

    print("\nExpected modality:")
    print(row["modality"])

    print("\nManifest image path:")
    print(row["image"])

    print("\nQuestion:")
    print(row["Question"])

    choices = {
        "A": row["Option_A"],
        "B": row["Option_B"],
        "C": row["Option_C"],
        "D": row["Option_D"],
    }

    print("\nChoices:")

    for key, value in choices.items():
        print(f"{key}: {value}")

    print("\nGround truth answer:")
    print(row["answer"])

    # ==================================================
    # RESOLVE IMAGE
    # ==================================================

    image_path = resolve_image_path(
        row["image"]
    )

    print("\nResolved image path:")
    print(image_path)

    if image_path is None:

        print(
            "\nERROR: Could not resolve image path."
        )

        return

    # ==================================================
    # LOAD IMAGE
    # ==================================================

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        print(
            f"\nImage loaded successfully."
        )

        print(
            f"Image size: {image.size}"
        )

    except Exception as error:

        print(
            f"\nERROR loading image: {error}"
        )

        return

    # ==================================================
    # INITIALIZE MODEL
    # ==================================================

    print("\n" + "=" * 70)
    print("INITIALIZING MODEL")
    print("=" * 70)

    model = YourModel()

    # ==================================================
    # RUN END-TO-END PREDICTION
    # ==================================================

    print("\n" + "=" * 70)
    print("RUNNING PREDICTION")
    print("=" * 70)

    try:

        prediction = model.answer(
            images=[image],
            question=row["Question"],
            choices=choices,
        )

    except Exception as error:

        print(
            f"\nMODEL ERROR: {error}"
        )

        import traceback

        traceback.print_exc()

        return

    # ==================================================
    # RESULTS
    # ==================================================

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(
        f"\nPrediction:   {prediction}"
    )

    print(
        f"Ground truth: {row['answer']}"
    )

    print(
        f"\nPrediction type: "
        f"{type(prediction).__name__}"
    )

    print(
        f"Ground truth type: "
        f"{type(row['answer']).__name__}"
    )

    # Normalize for preliminary comparison
    prediction_normalized = (
        str(prediction)
        .strip()
        .lower()
    )

    answer_normalized = (
        str(row["answer"])
        .strip()
        .lower()
    )

    print("\nNormalized prediction:")
    print(prediction_normalized)

    print("\nNormalized ground truth:")
    print(answer_normalized)

    print("\nCorrect:")

    print(
        prediction_normalized
        == answer_normalized
    )


if __name__ == "__main__":
    main()