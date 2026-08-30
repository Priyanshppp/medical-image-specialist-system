from pathlib import Path

from src.models.brain_mri import BrainMRIModel
from src.preprocess import load_representative_views


def main():

    print()
    print("=" * 70)
    print("BRAIN MRI MODEL TEST")
    print("=" * 70)

    model = BrainMRIModel()

    tests = [
        {
            "image": "dev/images/0006.nii",
            "question": (
                "Which diagnosis is most consistent "
                "with the provided image?"
            ),
            "choices": {
                "A": "Alzheimer's disease",
                "B": "Multiple sclerosis",
                "C": "Acute ischemic stroke",
                "D": "Primary brain tumor",
            },
        },
        {
            "image": "dev/images/0007.nii",
            "question": (
                "Which anatomical region is primarily "
                "depicted in the provided MRI volume?"
            ),
            "choices": {
                "A": "Lungs",
                "B": "Chest",
                "C": "Brain",
                "D": "Stomach",
            },
        },
    ]

    for test in tests:

        image_path = Path(test["image"])

        print()
        print("=" * 70)
        print("IMAGE:", image_path)
        print("QUESTION:", test["question"])

        print()
        print("Loading representative views...")

        images = load_representative_views(
            image_path
        )

        print(
            "Number of views:",
            len(images)
        )

        for i, image in enumerate(images):
            print(
                f"View {i}: "
                f"size={image.size}, "
                f"mode={image.mode}"
            )

        answer = model.answer(
            images=images,
            question=test["question"],
            choices=test["choices"],
        )

        print()
        print("MODEL ANSWER:", answer)

    print()
    print("=" * 70)
    print("BRAIN MRI TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
