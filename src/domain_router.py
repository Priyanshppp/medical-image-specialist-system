from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class DomainRouter:
    """
    Routes a medical image/question to the appropriate specialized model.

    Production strategy:
    - Use the trained visual modality classifier when an image is available.
    - Do NOT use confidence threshold rejection.
    - Do NOT override visual predictions with text heuristics.

    This strategy was selected because pure visual routing achieved the
    best validation accuracy.
    """

    def __init__(self):

        model_path = Path(
            "src/models/modality_router.pkl"
        )

        self.model = None
        self.feature_names = None

        if model_path.exists():

            loaded = joblib.load(
                model_path
            )

            # Model saved as dictionary
            if isinstance(loaded, dict):

                self.model = loaded.get(
                    "model"
                )

                self.feature_names = loaded.get(
                    "features"
                )

            # Fallback for raw sklearn model
            else:

                self.model = loaded

                if hasattr(
                    self.model,
                    "feature_names_in_"
                ):
                    self.feature_names = list(
                        self.model.feature_names_in_
                    )

            print(
                "Visual modality router loaded."
            )

            if self.model is not None:

                print(
                    f"Router model: "
                    f"{type(self.model).__name__}"
                )

        else:

            print(
                "Visual modality router not found."
            )

        # Default feature order.
        # Must match training feature order.
        if self.feature_names is None:

            self.feature_names = [

                "mean",
                "std",

                "dark",
                "bright",

                "colored",

                "red",
                "green",
                "blue",

                "horizontal_var",
                "vertical_var",

                "texture_v",
                "texture_h",
            ]

    def route(
        self,
        question,
        choices=None,
        images=None,
    ):

        # =====================================
        # PRIMARY STRATEGY:
        # PURE VISUAL ROUTING
        # =====================================

        if (
            self.model is not None
            and images
            and len(images) > 0
        ):

            try:

                image = images[0]

                features = (
                    self._extract_features(
                        image
                    )
                )

                # Create DataFrame with the exact
                # feature names expected by sklearn.
                feature_vector = pd.DataFrame(
                    [
                        {
                            name: features[name]
                            for name in self.feature_names
                        }
                    ],
                    columns=self.feature_names,
                )

                # =================================
                # PREDICTION
                # =================================

                prediction = (
                    self.model.predict(
                        feature_vector
                    )[0]
                )

                # =================================
                # CONFIDENCE
                # =================================

                confidence = None

                if hasattr(
                    self.model,
                    "predict_proba"
                ):

                    probabilities = (
                        self.model.predict_proba(
                            feature_vector
                        )[0]
                    )

                    confidence = float(
                        np.max(
                            probabilities
                        )
                    )

                # =================================
                # LOGGING
                # =================================

                if confidence is not None:

                    print(
                        f"Visual modality prediction: "
                        f"{prediction} "
                        f"(confidence: "
                        f"{confidence:.3f})"
                    )

                else:

                    print(
                        f"Visual modality prediction: "
                        f"{prediction}"
                    )

                print(
                    "Visual routing selected."
                )

                # =================================
                # MAP DATASET LABEL
                # TO INTERNAL DOMAIN
                # =================================

                modality_map = {

                    "CT":
                        "ct",

                    "MRI":
                        "brain_mri",

                    "X-ray":
                        "chest_xray",

                    "Fundus":
                        "general",

                    "Dermatology":
                        "dermoscopy",

                    "Microscopy":
                        "microscopy",

                    "OCT":
                        "oct",

                    "Ultrasound":
                        "ultrasound",
                }

                if prediction in modality_map:

                    return modality_map[
                        prediction
                    ]

                print(
                    "Unknown visual modality. "
                    "Using text fallback."
                )

            except Exception as error:

                print(
                    f"Visual routing error: "
                    f"{error}"
                )

        # =====================================
        # FALLBACK ONLY
        #
        # Text routing is used only when:
        #
        # 1. No visual model exists
        # 2. No image is provided
        # 3. Visual extraction fails
        # 4. Prediction is unknown
        # =====================================

        domain = self._text_route(
            question,
            choices,
        )

        print(
            f"Text routing selected: "
            f"{domain}"
        )

        return domain

    def _extract_features(
        self,
        image,
    ):

        # =====================================
        # RGB CONVERSION
        # =====================================

        image = image.convert(
            "RGB"
        )

        # =====================================
        # NORMALIZE IMAGE
        # =====================================

        array = (
            np.asarray(
                image,
                dtype=np.float32,
            )
            / 255.0
        )

        # =====================================
        # GRAYSCALE
        # =====================================

        gray = (

            0.299 * array[:, :, 0]

            + 0.587 * array[:, :, 1]

            + 0.114 * array[:, :, 2]
        )

        # =====================================
        # COLOR SATURATION
        # =====================================

        saturation = (

            np.max(
                array,
                axis=2,
            )

            -

            np.min(
                array,
                axis=2,
            )
        )

        # =====================================
        # SPATIAL PROFILES
        # =====================================

        horizontal_profile = (
            gray.mean(
                axis=1
            )
        )

        vertical_profile = (
            gray.mean(
                axis=0
            )
        )

        # =====================================
        # TEXTURE FEATURES
        # =====================================

        texture_vertical = float(

            np.mean(

                np.abs(

                    np.diff(
                        gray,
                        axis=0,
                    )

                )

            )

        )

        texture_horizontal = float(

            np.mean(

                np.abs(

                    np.diff(
                        gray,
                        axis=1,
                    )

                )

            )

        )

        # =====================================
        # RETURN FEATURES
        # =====================================

        return {

            "mean":

                float(
                    gray.mean()
                ),

            "std":

                float(
                    gray.std()
                ),

            "dark":

                float(

                    np.mean(
                        gray < 0.15
                    )

                ),

            "bright":

                float(

                    np.mean(
                        gray > 0.85
                    )

                ),

            "colored":

                float(

                    np.mean(
                        saturation > 0.15
                    )

                ),

            "red":

                float(
                    array[:, :, 0].mean()
                ),

            "green":

                float(
                    array[:, :, 1].mean()
                ),

            "blue":

                float(
                    array[:, :, 2].mean()
                ),

            "horizontal_var":

                float(
                    horizontal_profile.std()
                ),

            "vertical_var":

                float(
                    vertical_profile.std()
                ),

            "texture_v":

                texture_vertical,

            "texture_h":

                texture_horizontal,
        }

    def _text_route(
        self,
        question,
        choices=None,
    ):

        # =====================================
        # NORMALIZE QUESTION
        # =====================================

        question_text = str(
            question
        ).lower()

        # =====================================
        # NORMALIZE CHOICES
        # =====================================

        choice_values = []

        if choices is not None:

            if isinstance(
                choices,
                dict,
            ):

                choice_values = [

                    str(value).lower()

                    for value
                    in choices.values()

                    if value is not None
                ]

            elif isinstance(
                choices,
                (
                    list,
                    tuple,
                ),
            ):

                choice_values = [

                    str(value).lower()

                    for value
                    in choices

                    if value is not None
                ]

        choice_text = " ".join(
            choice_values
        )

        # Combine for fallback matching
        combined_text = (
            question_text
            + " "
            + choice_text
        )

        # =====================================
        # CT
        # =====================================

        ct_keywords = [

            "ct scan",

            "computed tomography",

            "primary lung cancer",

            "lung mass",

            "pulmonary nodule",
        ]

        if any(
            keyword in combined_text
            for keyword in ct_keywords
        ):

            return "ct"

        # =====================================
        # MRI
        # =====================================

        mri_keywords = [

            "mri volume",

            "magnetic resonance",

            "brain mri",

            "alzheimer",

            "multiple sclerosis",

            "acute ischemic stroke",

            "primary brain tumor",
        ]

        if any(
            keyword in combined_text
            for keyword in mri_keywords
        ):

            return "brain_mri"

        # =====================================
        # CHEST X-RAY
        # =====================================

        xray_keywords = [

            "chest radiograph",

            "chest x-ray",

            "chest xray",

            "medical device",

            "pleural effusion",

            "pneumothorax",
        ]

        if any(
            keyword in combined_text
            for keyword in xray_keywords
        ):

            return "chest_xray"

        # =====================================
        # DERMOSCOPY
        # =====================================

        dermoscopy_keywords = [

            "dermoscopic",

            "dermoscopy",

            "skin lesion",

            "melanoma",

            "vascular pattern",

            "pigmented lesion",
        ]

        if any(
            keyword in combined_text
            for keyword in dermoscopy_keywords
        ):

            return "dermoscopy"

        # =====================================
        # OCT
        # =====================================

        oct_keywords = [

            "oct image",

            "optical coherence tomography",

            "retinal condition",

            "retinal abnormalities",

            "macular",
        ]

        if any(
            keyword in combined_text
            for keyword in oct_keywords
        ):

            return "oct"

        # =====================================
        # ULTRASOUND
        # =====================================

        ultrasound_keywords = [

            "ultrasound",

            "sonography",

            "sonogram",

            "breast lesion",
        ]

        if any(
            keyword in combined_text
            for keyword in ultrasound_keywords
        ):

            return "ultrasound"

        # =====================================
        # MICROSCOPY
        # =====================================

        microscopy_keywords = [

            "cellular structure",

            "protein localization",

            "malaria infection",

            "microscope",

            "microscopy",

            "cell culture",
        ]

        if any(
            keyword in combined_text
            for keyword in microscopy_keywords
        ):

            return "microscopy"

        # =====================================
        # DEFAULT
        # =====================================

        return "general"