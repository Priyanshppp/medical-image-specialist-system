from pathlib import Path

import joblib
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


class UltrasoundModel:

    def __init__(self):

        print("Ultrasound model initialized.")

        project_root = Path(__file__).resolve().parent.parent.parent

        model_path = (
            project_root
            / "models"
            / "ultrasound_specialist.joblib"
        )

        if not model_path.exists():

            raise FileNotFoundError(
                f"Ultrasound specialist not found: {model_path}"
            )

        self.classifier = joblib.load(
            model_path
        )

        print(
            "Ultrasound specialist loaded."
        )

        print(
            "Known classes:"
        )

        for label in self.classifier.classes_:

            print(
                f" - {label}"
            )

        # ---------------------------------------------
        # Load ResNet18 feature extractor
        # ---------------------------------------------

        weights = models.ResNet18_Weights.DEFAULT

        base_model = models.resnet18(
            weights=weights
        )

        self.feature_extractor = nn.Sequential(
            *list(base_model.children())[:-1]
        )

        self.feature_extractor.eval()

        # ---------------------------------------------
        # Image preprocessing
        # ---------------------------------------------

        self.transform = transforms.Compose(
            [
                transforms.Resize(256),

                transforms.CenterCrop(224),

                transforms.ToTensor(),

                transforms.Normalize(
                    mean=[
                        0.485,
                        0.456,
                        0.406,
                    ],
                    std=[
                        0.229,
                        0.224,
                        0.225,
                    ],
                ),
            ]
        )

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================

    def _extract_feature(
        self,
        image,
    ):

        if not isinstance(
            image,
            Image.Image,
        ):

            image = Image.fromarray(
                np.asarray(image)
            )

        image = image.convert(
            "RGB"
        )

        tensor = self.transform(
            image
        )

        tensor = tensor.unsqueeze(
            0
        )

        with torch.no_grad():

            feature = self.feature_extractor(
                tensor
            )

        feature = (
            feature
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return feature

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def _normalize_text(
        self,
        text,
    ):

        text = str(text).lower().strip()

        text = (
            text
            .replace("-", " ")
            .replace("_", " ")
        )

        text = " ".join(
            text.split()
        )

        return text

    # =====================================================
    # EXACT CLASS MATCH
    # =====================================================

    def _get_exact_class(
        self,
        choice,
    ):

        choice_normalized = (
            self._normalize_text(choice)
        )

        for class_name in self.classifier.classes_:

            class_normalized = (
                self._normalize_text(class_name)
            )

            if choice_normalized == class_normalized:

                return class_name

        return None

    # =====================================================
    # SAFE ALIAS MATCHING
    # =====================================================

    def _get_alias_class(
        self,
        choice,
    ):

        choice_normalized = (
            self._normalize_text(choice)
        )

        # -------------------------------------------------
        # IMPORTANT:
        #
        # Only use aliases where they are genuinely
        # equivalent dataset labels.
        #
        # Do NOT collapse anatomical substructures into
        # organs.
        #
        # Example:
        #
        # "renal cortex" != "kidney"
        # "thyroid gland" == "thyroid"
        # -------------------------------------------------

        aliases = {

            "thyroid gland": "thyroid",

            "urinary bladder": "bladder",

            "common bile duct": "common bile duct",

        }

        if choice_normalized in aliases:

            target = aliases[
                choice_normalized
            ]

            for class_name in self.classifier.classes_:

                if (
                    self._normalize_text(
                        class_name
                    )
                    ==
                    target
                ):

                    return class_name

        return None

    # =====================================================
    # MAP CHOICE TO MODEL CLASS
    # =====================================================

    def _map_choice_to_class(
        self,
        choice,
    ):

        # Priority 1:
        # Exact match

        exact_match = (
            self._get_exact_class(
                choice
            )
        )

        if exact_match is not None:

            return (
                exact_match,
                "exact",
            )

        # Priority 2:
        # Explicit safe aliases

        alias_match = (
            self._get_alias_class(
                choice
            )
        )

        if alias_match is not None:

            return (
                alias_match,
                "alias",
            )

        # No mapping

        return (
            None,
            "unseen",
        )

    # =====================================================
    # MAIN ANSWER FUNCTION
    # =====================================================

    def answer(
        self,
        images,
        question,
        choices,
    ):

        # -------------------------------------------------
        # Safety
        # -------------------------------------------------

        if not images:

            print(
                "No image provided."
            )

            return next(
                iter(choices)
            )

        # -------------------------------------------------
        # Extract image feature
        # -------------------------------------------------

        feature = self._extract_feature(
            images[0]
        )

        feature = feature.reshape(
            1,
            -1,
        )

        # -------------------------------------------------
        # Get classifier probabilities
        # -------------------------------------------------

        probabilities = (
            self.classifier.predict_proba(
                feature
            )[0]
        )

        class_scores = {

            class_name: float(probability)

            for (
                class_name,
                probability,
            )

            in zip(
                self.classifier.classes_,
                probabilities,
            )
        }

        # -------------------------------------------------
        # Print raw prediction
        # -------------------------------------------------

        raw_index = int(
            np.argmax(
                probabilities
            )
        )

        raw_prediction = (
            self.classifier.classes_[
                raw_index
            ]
        )

        raw_confidence = float(
            probabilities[
                raw_index
            ]
        )

        print()

        print(
            "Ultrasound raw prediction:"
        )

        print(
            f"  Class: {raw_prediction}"
        )

        print(
            f"  Confidence: "
            f"{raw_confidence:.4f}"
        )

        # -------------------------------------------------
        # Candidate constrained selection
        # -------------------------------------------------

        candidate_results = []

        for letter, choice in choices.items():

            mapped_class, match_type = (
                self._map_choice_to_class(
                    choice
                )
            )

            if mapped_class is not None:

                score = class_scores.get(
                    mapped_class,
                    0.0,
                )

            else:

                score = 0.0

            candidate_results.append(
                {
                    "letter": letter,
                    "choice": choice,
                    "mapped_class": mapped_class,
                    "match_type": match_type,
                    "score": score,
                }
            )

        # -------------------------------------------------
        # Display candidate scores
        # -------------------------------------------------

        print()

        print(
            "Ultrasound candidate scores:"
        )

        for candidate in candidate_results:

            print(
                f"  {candidate['letter']}: "
                f"{candidate['choice']}"
            )

            print(
                f"      mapped class: "
                f"{candidate['mapped_class']}"
            )

            print(
                f"      match type: "
                f"{candidate['match_type']}"
            )

            print(
                f"      score: "
                f"{candidate['score']:.4f}"
            )

        # -------------------------------------------------
        # Select highest scoring VALID candidate
        # -------------------------------------------------

        best_candidate = max(
            candidate_results,
            key=lambda x: x["score"],
        )

        # -------------------------------------------------
        # If no candidate maps to a known class
        # -------------------------------------------------

        if best_candidate["score"] <= 0:

            print()

            print(
                "No answer choice maps to a known "
                "Ultrasound training class."
            )

            print(
                "Using deterministic fallback."
            )

            return next(
                iter(choices)
            )

        # -------------------------------------------------
        # Final decision
        # -------------------------------------------------

        print()

        print(
            "Ultrasound final selection:"
        )

        print(
            f"  Choice: "
            f"{best_candidate['letter']}"
        )

        print(
            f"  Answer: "
            f"{best_candidate['choice']}"
        )

        print(
            f"  Model class: "
            f"{best_candidate['mapped_class']}"
        )

        print(
            f"  Match type: "
            f"{best_candidate['match_type']}"
        )

        print(
            f"  Score: "
            f"{best_candidate['score']:.4f}"
        )

        return best_candidate[
            "letter"
        ]