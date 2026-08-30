from pathlib import Path

from src.metadata import load_queries, extract_choices
from src.model import YourModel
from src.predictor import Predictor


def main():
    print()
    print("FULL MODEL TEST")
    print("=" * 70)

    # Load metadata
    df, root = load_queries("dev")

    print(f"Loaded {len(df)} queries")

    # Load model once
    print()
    print("Loading model...")

    model = YourModel()
    predictor = Predictor(model)

    print("Model loaded.")

    print()
    print("=" * 70)

    # Test every query
    for _, row in df.iterrows():

        query_id = row["query_id"]

        image_path = (
            root / row["image"]
        )

        choices = extract_choices(row)

        print()
        print(f"QUERY {query_id}")
        print("-" * 70)
        print(f"Image: {image_path}")
        print(f"Question: {row['question']}")
        print("Choices:")

        for letter, choice in choices.items():
            print(f"  {letter}. {choice}")

        answer, inference_time = predictor.predict(
            image_path=image_path,
            query=row["question"],
            choices=choices,
        )

        print()
        print(f"ANSWER: {answer}")
        print(f"TIME: {inference_time:.3f}s")

    print()
    print("=" * 70)
    print("FULL MODEL TEST COMPLETE")


if __name__ == "__main__":
    main()
