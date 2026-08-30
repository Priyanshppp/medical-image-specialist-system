from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)

from src.domain_router import DomainRouter
from src.preprocess import load_representative_views


MODALITY_TO_DOMAIN = {
    "CT": "ct",
    "MRI": "brain_mri",
    "X-ray": "general",
    "Fundus": "general",
    "Dermatology": "dermoscopy",
    "Microscopy": "microscopy",
    "OCT": "oct",
    "Ultrasound": "ultrasound",
}


def main():

    print()
    print("=" * 80)
    print("MODALITY ROUTER EVALUATION")
    print("=" * 80)

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

    y_true = []
    y_pred = []

    # Diagnostic sample only.
    # Do NOT use this for training.
    for modality in sorted(df["modality"].unique()):

        subset = df[
            df["modality"] == modality
        ].head(100)

        print()
        print(f"Testing {modality}")

        for _, row in subset.iterrows():

            image_path = (
                Path("test_data")
                / row["image"]
            )

            try:

                images = load_representative_views(
                    image_path
                )

                if not images:
                    continue

                predicted_domain = router.route(
                    question="",
                    choices={},
                    images=images,
                )

                expected_domain = (
                    MODALITY_TO_DOMAIN[
                        row["modality"]
                    ]
                )

                y_true.append(
                    expected_domain
                )

                y_pred.append(
                    predicted_domain
                )

            except Exception as error:

                print(
                    f"Error {row['query_id']}: "
                    f"{error}"
                )

    print()
    print("=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    labels = sorted(
        set(y_true) | set(y_pred)
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

    print()
    print("=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)

    print(
        matrix_df.to_string()
    )

    accuracy = sum(
        true == pred
        for true, pred
        in zip(y_true, y_pred)
    ) / len(y_true)

    print()
    print(
        f"Overall routing accuracy: "
        f"{accuracy:.2%}"
    )


if __name__ == "__main__":
    main()
    