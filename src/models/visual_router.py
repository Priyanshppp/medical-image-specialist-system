from pathlib import Path

import joblib
import numpy as np


class VisualRouter:

    def __init__(self):

        model_path = (
            Path(__file__).parent
            / "modality_router.pkl"
        )

        data = joblib.load(model_path)

        self.model = data["model"]
        self.feature_names = data["features"]

        print(
            "Visual modality router loaded."
        )

    def extract_features(self, images):

        if not images:
            return None

        image = images[0].convert("RGB")

        array = (
            np.asarray(image)
            .astype(np.float32)
            / 255.0
        )

        gray = (
            0.299 * array[:, :, 0]
            + 0.587 * array[:, :, 1]
            + 0.114 * array[:, :, 2]
        )

        horizontal_diff = np.abs(
            np.diff(gray, axis=1)
        )

        vertical_diff = np.abs(
            np.diff(gray, axis=0)
        )

        saturation = (
            np.max(array, axis=2)
            - np.min(array, axis=2)
        )

        features = {
            "mean": float(gray.mean()),
            "std": float(gray.std()),

            "dark": float(
                np.mean(gray < 0.15)
            ),

            "bright": float(
                np.mean(gray > 0.85)
            ),

            "colored": float(
                np.mean(saturation > 0.10)
            ),

            "red": float(
                array[:, :, 0].mean()
            ),

            "green": float(
                array[:, :, 1].mean()
            ),

            "blue": float(
                array[:, :, 2].mean()
            ),

            "horizontal_var": float(
                np.var(horizontal_diff)
            ),

            "vertical_var": float(
                np.var(vertical_diff)
            ),

            "texture_v": float(
                vertical_diff.mean()
            ),

            "texture_h": float(
                horizontal_diff.mean()
            ),
        }

        return features

    def predict(self, images):

        features = self.extract_features(images)

        if features is None:
            return "general", 0.0

        vector = np.array([
            features[name]
            for name in self.feature_names
        ]).reshape(1, -1)

        prediction = self.model.predict(
            vector
        )[0]

        probabilities = (
            self.model.predict_proba(vector)[0]
        )

        confidence = float(
            probabilities.max()
        )

        return prediction, confidence
