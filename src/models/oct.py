import numpy as np
from PIL import Image


class OCTModel:

    def __init__(self):
        print("OCT model initialized.")

    def _extract_features(self, images):

        features = []

        for image in images:

            if not isinstance(image, Image.Image):
                image = Image.fromarray(
                    np.asarray(image)
                )

            gray = np.asarray(
                image.convert("L"),
                dtype=np.float32,
            )

            if gray.size == 0:
                continue

            gray = gray / 255.0

            features.append(
                {
                    "mean": float(np.mean(gray)),
                    "std": float(np.std(gray)),
                    "dark_fraction": float(
                        np.mean(gray < 0.20)
                    ),
                    "bright_fraction": float(
                        np.mean(gray > 0.80)
                    ),
                }
            )

        return features

    def answer(
        self,
        images,
        question,
        choices,
    ):

        question_text = str(
            question
        ).lower()

        features = self._extract_features(
            images
        )

        if features:

            mean_intensity = np.mean(
                [x["mean"] for x in features]
            )

            std_intensity = np.mean(
                [x["std"] for x in features]
            )

            dark_fraction = np.mean(
                [x["dark_fraction"] for x in features]
            )

            bright_fraction = np.mean(
                [x["bright_fraction"] for x in features]
            )

            print("OCT baseline features:")

            print(
                f"  mean_intensity  = "
                f"{mean_intensity:.4f}"
            )

            print(
                f"  std_intensity   = "
                f"{std_intensity:.4f}"
            )

            print(
                f"  dark_fraction   = "
                f"{dark_fraction:.4f}"
            )

            print(
                f"  bright_fraction = "
                f"{bright_fraction:.4f}"
            )

        # =========================================
        # CURRENT BASELINE
        # =========================================

        # The image is being processed correctly,
        # but no trained OCT classifier is currently
        # available in the project.

        return next(iter(choices))
