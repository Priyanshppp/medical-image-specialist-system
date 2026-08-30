from pathlib import Path
import time

import pandas as pd

from src.model import YourModel
from src.predictor import Predictor


def extract_choices(row):
    choices = {}

    mapping = {
        "A": "Option_A",
        "B": "Option_B",
        "C": "Option_C",
        "D": "Option_D",
    }

    for letter, column in mapping.items():
        value = row.get(column)

        if pd.notna(value):
            choices[letter] = str(value)

    return choices


def main():

    print()
    print("=" * 70)
    print("OMNI TEST DATASET EVALUATION")
    print("=" * 70)

    dataset_root = Path("test_data")

    metadata_path = dataset_root / "test_metadata.csv"
    answer_key_path = dataset_root / "answer_key.csv"

    print("\nLoading metadata...")

    df = pd.read_csv(metadata_path).head(10)
    answers_df = pd.read_csv(answer_key_path)

    print(f"Test queries: {len(df)}")
    print(f"Answer keys:  {len(answers_df)}")

    print("\nInitializing model...")

    model = YourModel()
    predictor = Predictor(model)

    print("Model initialized.")

    results = []

    total_start = time.time()

    for index, row in df.iterrows():

        query_id = row["query_id"]

        image_path = dataset_root / row["image"]

        question = row["Question"]

        choices = extract_choices(row)

        if not image_path.exists():

            print(
                f"\n[{index + 1}/{len(df)}] "
                f"{query_id} - IMAGE NOT FOUND"
            )

            results.append({
                "query_id": query_id,
                "prediction": None,
                "inference_time": None,
                "status": "image_not_found",
            })

            continue

        try:

            answer, inference_time = predictor.predict(
                image_path=image_path,
                query=question,
                choices=choices,
            )

            results.append({
                "query_id": query_id,
                "prediction": answer,
                "inference_time": inference_time,
                "status": "success",
            })

        except Exception as error:

            print(
                f"\nERROR on {query_id}: {error}"
            )

            results.append({
                "query_id": query_id,
                "prediction": None,
                "inference_time": None,
                "status": f"error: {error}",
            })

        # Progress update every 100 images
        if (index + 1) % 100 == 0:

            elapsed = time.time() - total_start

            print(
                f"Processed {index + 1}/{len(df)} "
                f"({elapsed:.1f}s)"
            )

    total_time = time.time() - total_start

    results_df = pd.DataFrame(results)

    # Merge predictions with answer key
    evaluation = results_df.merge(
        answers_df[
            [
                "query_id",
                "answer",
                "modality",
                "source_dataset",
            ]
        ],
        on="query_id",
        how="left",
    )

    # Normalize answers
    evaluation["prediction"] = (
        evaluation["prediction"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    evaluation["answer"] = (
        evaluation["answer"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    evaluation["correct"] = (
        evaluation["prediction"]
        == evaluation["answer"]
    )

    # Save detailed results
    output_path = dataset_root / "evaluation_results.csv"

    evaluation.to_csv(
        output_path,
        index=False,
    )

    successful = (
        evaluation["status"] == "success"
    ).sum()

    correct = evaluation["correct"].sum()

    accuracy = (
        correct / successful
        if successful > 0
        else 0
    )

    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    print(f"Total queries:      {len(df)}")
    print(f"Successful:         {successful}")
    print(f"Correct:            {correct}")
    print(f"Accuracy:           {accuracy:.4%}")
    print(f"Total time:         {total_time:.2f}s")

    if successful > 0:

        avg_time = (
            evaluation.loc[
                evaluation["status"] == "success",
                "inference_time"
            ]
            .mean()
        )

        print(
            f"Average inference:  "
            f"{avg_time:.4f}s/image"
        )

    print()
    print(f"Results saved to:")
    print(output_path)

    # Accuracy by modality
    print()
    print("=" * 70)
    print("ACCURACY BY MODALITY")
    print("=" * 70)

    modality_results = (
        evaluation[
            evaluation["status"] == "success"
        ]
        .groupby("modality")
        ["correct"]
        .agg(["count", "sum", "mean"])
        .sort_values("count", ascending=False)
    )

    print(modality_results)


if __name__ == "__main__":
    main()
