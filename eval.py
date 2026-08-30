import argparse
import time
from pathlib import Path

import pandas as pd

from src.metadata import (
    load_queries,
    extract_choices,
)
from src.model import YourModel
from src.predictor import Predictor


def resolve_image_path(input_dir, image_reference):
    image_reference = str(image_reference)

    path = input_dir / image_reference

    if path.exists():
        return path

    raise FileNotFoundError(
        f"Image path not found: {path}"
    )


def main(input_dir):

    # ==============================
    # LOAD MODEL ONCE
    # ==============================

    model_start = time.perf_counter()

    model = YourModel()

    predictor = Predictor(model)

    model_load_time = (
        time.perf_counter() - model_start
    )

    print(
        f"Model initialized in "
        f"{model_load_time:.2f} seconds"
    )

    # ==============================
    # LOAD QUERIES
    # ==============================

    df, input_dir = load_queries(input_dir)

    results = []

    # ==============================
    # INFERENCE
    # ==============================

    for _, row in df.iterrows():

        query_id = row["query_id"]

        image_path = resolve_image_path(
            input_dir,
            row["image"],
        )

        choices = extract_choices(row)

        answer, inference_time = predictor.predict(
            image_path=image_path,
            query=row["question"],
            choices=choices,
        )

        results.append(
            {
                "query_id": query_id,
                "answer": answer,
                "inference_time": inference_time,
            }
        )

        print(
            f"{query_id} -> {answer} "
            f"({inference_time:.3f}s)"
        )

    # ==============================
    # VALIDATE OUTPUT
    # ==============================

    result_df = pd.DataFrame(results)

    if len(result_df) != len(df):
        raise RuntimeError(
            "Number of predictions does not "
            "match number of queries"
        )

    if result_df["query_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate query IDs found"
        )

    valid = {"A", "B", "C", "D"}

    if not result_df["answer"].isin(valid).all():
        raise RuntimeError(
            "Invalid answer detected"
        )

    # ==============================
    # WRITE REQUIRED OUTPUT
    # ==============================

    result_df.to_csv(
        "predictions.csv",
        index=False,
    )

    print(
        "\nSaved predictions.csv"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
    )

    args = parser.parse_args()

    main(
        Path(args.input_dir)
    )
