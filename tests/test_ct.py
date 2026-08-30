from src.preprocess import load_representative_views
from src.models.ct import CTModel


def main():

    model = CTModel()

    test_cases = [
        (
            "dev/images/0004",
            "Which anatomical region is primarily depicted in the provided CT scan?",
            {
                "A": "Abdomen",
                "B": "Chest",
                "C": "Pelvis",
                "D": "Brain",
            },
        ),
        (
            "dev/images/0005",
            "Does the patient have primary lung cancer? (Malignant)",
            {
                "A": "Yes",
                "B": "No",
            },
        ),
    ]

    for image_path, question, choices in test_cases:

        print()
        print("=" * 70)
        print(f"IMAGE: {image_path}")
        print(f"QUESTION: {question}")

        images = load_representative_views(
            image_path
        )

        print()
        print(f"Number of views: {len(images)}")

        for i, image in enumerate(images):
            print(
                f"View {i}: "
                f"size={image.size}, "
                f"mode={image.mode}"
            )

        answer = model.answer(
            images=images,
            question=question,
            choices=choices,
        )

        print()
        print(f"MODEL ANSWER: {answer}")


if __name__ == "__main__":
    main()
