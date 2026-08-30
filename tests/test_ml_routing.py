from pathlib import Path

import pandas as pd
from PIL import Image

from src.models.visual_router import VisualRouter


def main():

    print()
    print("=" * 70)
    print("ML VISUAL ROUTER TEST")
    print("=" * 70)

    metadata = pd.read_csv(
        "test_data/test_metadata.csv"
    )

    answers = pd.read_csv(
        "test_data/answer_key.csv"
    )

    df = metadata.merge(
        answers[
            ["query_id", "modality"]
        ],
        on="query_id",
    )

    router = VisualRouter()

    correct = 0
    total = 0

    results = []

    for _, row in df.iterrows():

        image_path = (
            Path("test_data")
            / row["image"]
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            predicted, confidence = (
                router.predict([image])
            )

            expected = row["modality"]

            is_correct = (
                predicted == expected
            )

            correct += int(is_correct)
            total += 1

            results.append({
                "query_id": row["query_id"],
                "expected": expected,
                "predicted": predicted,
                "confidence": confidence,
                "correct": is_correct,
            })

        except Exception as error:

            print(
                f"ERROR {row['query_id']}: "
                f"{error}"
            )

        if total % 500 == 0:

            accuracy = correct / total

            print(
                f"Processed {total}/8000 "
                f"| Accuracy: {accuracy:.4%}"
            )

    accuracy = correct / max(total, 1)

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(f"Total:    {total}")
    print(f"Correct:  {correct}")
    print(
        f"Accuracy: {accuracy:.4%}"
    )

    results_df = pd.DataFrame(
        results
    )

    print()
    print("ACCURACY BY MODALITY")

    print(
        results_df
        .groupby("expected")["correct"]
        .agg(["count", "mean"])
    )

    results_df.to_csv(
        "test_data/routing_results.csv",
        index=False,
    )

    print()
    print(
        "Saved: "
        "test_data/routing_results.csv"
    )


if __name__ == "__main__":
    main()
