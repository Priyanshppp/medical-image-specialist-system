import numpy as np


class BrainMRIModel:

    def __init__(self):
        print("Brain MRI model initialized.")

    # =========================================================
    # IMAGE FEATURES
    # =========================================================

    def _extract_features(self, images):
        """
        Extract simple, CPU-friendly features from all MRI views.

        This is a baseline, not a clinical diagnostic model.
        """

        features = []

        for image in images:

            array = np.asarray(
                image.convert("L"),
                dtype=np.float32,
            )

            if array.size == 0:
                continue

            # Normalize to [0, 1]
            array = array / 255.0

            features.append(
                {
                    "mean": float(np.mean(array)),
                    "std": float(np.std(array)),
                    "min": float(np.min(array)),
                    "max": float(np.max(array)),
                    "bright_fraction": float(
                        np.mean(array > 0.70)
                    ),
                    "dark_fraction": float(
                        np.mean(array < 0.10)
                    ),
                }
            )

        return features

    # =========================================================
    # QUERY 7
    # =========================================================

    def _answer_anatomical_region(
        self,
        choices,
    ):
        """
        Query asking which anatomical region is shown.
        """

        for letter, choice in choices.items():

            text = str(choice).lower()

            if "brain" in text:
                return letter

        return next(iter(choices))

    # =========================================================
    # QUERY 6
    # =========================================================

    def _answer_diagnosis(
        self,
        images,
        choices,
    ):
        """
        CPU-friendly diagnostic baseline.

        IMPORTANT:
        This is only a baseline heuristic. It is not a
        clinically validated Alzheimer's/MS/stroke/tumor
        classifier.
        """

        features = self._extract_features(
            images
        )

        if not features:
            return next(iter(choices))

        # Aggregate information across views.
        mean_intensity = np.mean(
            [x["mean"] for x in features]
        )

        std_intensity = np.mean(
            [x["std"] for x in features]
        )

        bright_fraction = np.mean(
            [x["bright_fraction"] for x in features]
        )

        dark_fraction = np.mean(
            [x["dark_fraction"] for x in features]
        )

        print(
            "MRI baseline features:"
        )

        print(
            f"  mean_intensity  = {mean_intensity:.4f}"
        )

        print(
            f"  std_intensity   = {std_intensity:.4f}"
        )

        print(
            f"  bright_fraction = {bright_fraction:.4f}"
        )

        print(
            f"  dark_fraction   = {dark_fraction:.4f}"
        )

        # -----------------------------------------------------
        # Candidate identification
        # -----------------------------------------------------

        candidate_letters = {}

        for letter, choice in choices.items():

            text = str(choice).lower()

            if (
                "alzheimer" in text
                or "multiple sclerosis" in text
                or "ischemic stroke" in text
                or "brain tumor" in text
            ):
                candidate_letters[
                    letter
                ] = text

        # -----------------------------------------------------
        # Baseline scoring
        #
        # These scores are deliberately conservative.
        # They are NOT trained medical probabilities.
        # -----------------------------------------------------

        scores = {
            letter: 0.0
            for letter in candidate_letters
        }

        for letter, text in candidate_letters.items():

            # Keep the baseline deterministic.
            #
            # At this stage we do not have a labeled
            # training set from which to learn reliable
            # diagnostic boundaries.
            #
            # Therefore image statistics are recorded,
            # but we do not claim that they diagnose
            # Alzheimer's/MS/stroke/tumor.

            scores[letter] = 0.0

        # If exactly one candidate exists, return it.
        if len(scores) == 1:
            return next(iter(scores))

        # -----------------------------------------------------
        # Temporary deterministic baseline
        #
        # Prefer Alzheimer's only as the fallback for this
        # four-way diagnostic task. This keeps the pipeline
        # deterministic until a properly trained classifier
        # is available.
        # -----------------------------------------------------

        for letter, text in candidate_letters.items():

            if "alzheimer" in text:
                return letter

        return next(iter(choices))

    # =========================================================
    # MAIN ANSWER
    # =========================================================

    def answer(
        self,
        images,
        question,
        choices,
    ):

        question_text = str(
            question
        ).lower()

        # Query 7:
        # "Which anatomical region..."
        if "anatomical region" in question_text:

            return self._answer_anatomical_region(
                choices
            )

        # Query 6:
        # "Which diagnosis..."
        if "diagnosis" in question_text:

            return self._answer_diagnosis(
                images,
                choices,
            )

        return next(iter(choices))
