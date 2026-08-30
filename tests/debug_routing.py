from pathlib import Path
import pandas as pd

from src.preprocess import load_representative_views
from src.domain_router import DomainRouter


def main():

    metadata = pd.read_csv(
        "test_data/test_metadata.csv"
    )

    answers = pd.read_csv(
        "test_data/answer_key.csv"
    )

    df = metadata.merge(
        answers[
            [
                "query_id",
                "modality",
            ]
        ],
        on="query_id",
    )

    router = DomainRouter()

    # Test one example from each modality
    modalities = [
        "CT",
        "MRI",
        "X-ray",
        "Fundus",
        "Dermatology",
        "Microscopy",
        "OCT",
        "Ultrasound",
    ]

    print()
    print("=" * 70)
    print("VISUAL ROUTING TEST")
    print("=" * 70)

    for modality in modalities:

        row = df[
            df["modality"] == modality
        ].iloc[0]

        image_path = (
            Path("test_data")
            / row["image"]
        )

        choices = {}

        for letter in ["A", "B", "C", "D"]:

            column = f"Option_{letter}"

            if (
                column in row
                and pd.notna(row[column])
            ):
                choices[letter] = row[column]

        images = load_representative_views(
            image_path
        )

        predicted = router.route(
            row["Question"],
            choices,
            images=images,
        )

        print()
        print(f"ID:        {row['query_id']}")
        print(f"Expected:  {modality}")
        print(f"Predicted: {predicted}")
        print(f"Question:  {row['Question']}")


if __name__ == "__main__":
    main()
