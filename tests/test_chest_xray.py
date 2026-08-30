from src.models.chest_xray import ChestXrayModel
from src.preprocess import load_representative_views


def run_query(
    model,
    image_path,
    question,
    choices,
):

    print()
    print("=" * 70)

    print("IMAGE:", image_path)

    print("QUESTION:")
    print(question)

    print()

    print("CHOICES:")

    for letter, choice in choices.items():

        print(
            f"{letter}. {choice}"
        )

    images = load_representative_views(
        image_path
    )

    answer = model.answer(
        images=images,
        question=question,
        choices=choices,
    )

    print()
    print(
        "MODEL ANSWER:",
        answer,
    )


def main():

    print()
    print("LOADING CHEST X-RAY MODEL")
    print("=" * 70)

    model = ChestXrayModel()

    # ==================================================
    # QUERY 8
    # ==================================================

    run_query(
        model=model,

        image_path=(
            "dev/images/0008.png"
        ),

        question=(
            "Which of the following best describes "
            "the provided chest radiograph?"
        ),

        choices={

            "A":
                "No significant abnormality is identified",

            "B":
                "Pulmonary edema",

            "C":
                "Pleural effusion",

            "D":
                "Pneumothorax",
        },
    )

    # ==================================================
    # QUERY 9
    # ==================================================

    run_query(
        model=model,

        image_path=(
            "dev/images/0009.png"
        ),

        question=(
            "Which of the following findings is most "
            "consistent with the image?"
        ),

        choices={

            "A":
                "Pneumothorax",

            "B":
                "Hernia",

            "C":
                "Pleural effusion",

            "D":
                "Cardiomegaly",
        },
    )


if __name__ == "__main__":

    main()
