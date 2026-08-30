import numpy as np
from PIL import Image


class DermoscopyModel:

    def __init__(self):
        print("Dermoscopy model initialized.")

    def _extract_features(self, images):

        features = []

        for image in images:

            if not isinstance(image, Image.Image):
                image = Image.fromarray(
                    np.asarray(image)
                )

            rgb = np.asarray(
                image.convert("RGB"),
                dtype=np.float32,
            )

            if rgb.size == 0:
                continue

            rgb = rgb / 255.0

            gray = np.mean(rgb, axis=2)

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
                    "red_mean": float(
                        np.mean(rgb[:, :, 0])
                    ),
                    "green_mean": float(
                        np.mean(rgb[:, :, 1])
                    ),
                    "blue_mean": float(
                        np.mean(rgb[:, :, 2])
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

            red_mean = np.mean(
                [x["red_mean"] for x in features]
            )

            green_mean = np.mean(
                [x["green_mean"] for x in features]
            )

            blue_mean = np.mean(
                [x["blue_mean"] for x in features]
            )

            print(
                "Dermoscopy baseline features:"
            )

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

            print(
                f"  red_mean        = "
                f"{red_mean:.4f}"
            )

            print(
                f"  green_mean      = "
                f"{green_mean:.4f}"
            )

            print(
                f"  blue_mean       = "
                f"{blue_mean:.4f}"
            )

        # =========================================
        # QUERY 13
        # VASCULAR PATTERN
        # =========================================

        if "vascular pattern" in question_text:

            # Baseline only.
            # Actual vascular-pattern recognition
            # requires a trained dermoscopy model.

            return next(iter(choices))

        # =========================================
        # QUERY 12
        # SKIN LESION DIAGNOSIS
        # =========================================

        if "diagnosis" in question_text:
            return next(iter(choices))

        # =========================================
        # SAFE FALLBACK
        # =========================================

        return next(iter(choices))
