from pathlib import Path

import pandas as pd

from src.preprocess import load_representative_views
from src.domain_router import DomainRouter


def main():

    print()
    print("=" * 70)
    print("LEARNED MODALITY ROUTER TEST")
    print("=" * 70)

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

    results = []

    for modality in sorted(
        df["modality"].unique()
    ):

        print()
        print(
            f"Testing {modality}"
        )

        subset = df[
            df["modality"] == modality
        ].head(100)

        correct = 0
        total = 0

        for _, row in subset.iterrows():

            image_path = (
                Path("test_data")
                / row["image"]
            )

            try:

                images = (
                    load_representative_views(
                        image_path
                    )
                )

                prediction = router.route(
                    row["Question"],
                    {},
                    images=images,
                )

                mapping = {
                    "CT": "ct",
                    "MRI": "brain_mri",
                    "X-ray": "chest_xray",
                    "Dermatology": "dermoscopy",
                    "Microscopy": "microscopy",
                    "OCT": "oct",
                    "Ultrasound": "ultrasound",
                    "Fundus": "general",
                }

                expected = mapping[
                    row["modality"]
                ]

                if prediction == expected:
                    correct += 1

                total += 1

            except Exception as error:

                print(
                    f"Error: {error}"
                )

        accuracy = (
            correct / total
            if total else 0
        )

        print(
            f"{modality}: "
            f"{correct}/{total} "
            f"= {accuracy:.2%}"
        )

        results.append(
            {
                "modality": modality,
                "correct": correct,
                "total": total,
                "accuracy": accuracy,
            }
        )

    results_df = pd.DataFrame(
        results
    )

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
