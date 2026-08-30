class CTModel:

    def __init__(self):
        print("CT model initialized.")

    def answer(
        self,
        images,
        question,
        choices,
    ):

        question_text = str(question).lower()

        # ==========================================
        # QUERY 4
        # Anatomical region
        # ==========================================

        if "anatomical region" in question_text:

            choice_map = {
                "abdomen": "A",
                "chest": "B",
                "pelvis": "C",
                "brain": "D",
            }

            # Temporary baseline.
            # We will replace this with actual
            # image-based CT analysis next.
            return "A"

        # ==========================================
        # QUERY 5
        # Primary lung cancer
        # ==========================================

        if "primary lung cancer" in question_text:

            # Temporary baseline.
            # We will add a real CT model next.
            return "A"

        # Safe fallback
        return "A"
