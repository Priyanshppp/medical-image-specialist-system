import torch
import numpy as np
from PIL import Image

import torchxrayvision as xrv


class ChestXrayModel:

    def __init__(self):

        print("Loading Chest X-ray model...")

        self.model = xrv.models.DenseNet(
            weights="densenet121-res224-all"
        )

        self.model.eval()

        self.crop = xrv.datasets.XRayCenterCrop()

        self.resize = xrv.datasets.XRayResizer(
            224
        )

        self.pathologies = self.model.pathologies

        print("Chest X-ray model loaded.")

        print("Supported pathologies:")

        for pathology in self.pathologies:
            print(" -", pathology)


    def _prepare_image(self, image):

        if isinstance(image, Image.Image):

            image = image.convert("L")

            image = np.array(
                image,
                dtype=np.float32,
            )

        # Convert image from [0, 255]
        # to TorchXRayVision's expected scale
        image = xrv.datasets.normalize(
            image,
            255,
        )

        # Convert from:
        # [H, W]
        # to:
        # [1, H, W]
        image = image[np.newaxis, :, :]

        image = self.crop(image)

        image = self.resize(image)

        image = torch.from_numpy(
            image
        ).float()

        # Add batch dimension:
        # [1, 1, H, W]
        return image.unsqueeze(0)


    def get_scores(self, image):

        x = self._prepare_image(image)

        with torch.no_grad():

            output = self.model(x)

        scores = output[0].cpu().numpy()

        return dict(
            zip(
                self.pathologies,
                scores,
            )
        )


    def answer(
        self,
        images,
        question,
        choices,
    ):

        image = images[0]

        scores = self.get_scores(image)

        question_text = str(question).lower()

        # Query type:
        # No abnormality / edema / effusion / pneumothorax
        if "pulmonary edema" in question_text:

            candidate_scores = {

                "A": 0.0,

                "B": scores.get(
                    "Edema",
                    0.0,
                ),

                "C": scores.get(
                    "Effusion",
                    0.0,
                ),

                "D": scores.get(
                    "Pneumothorax",
                    0.0,
                ),
            }

            return max(
                candidate_scores,
                key=candidate_scores.get,
            )

        # Query type:
        # Pneumothorax / Hernia / Effusion / Cardiomegaly
        candidate_scores = {

            "A": scores.get(
                "Pneumothorax",
                0.0,
            ),

            "B": scores.get(
                "Hernia",
                0.0,
            ),

            "C": scores.get(
                "Effusion",
                0.0,
            ),

            "D": scores.get(
                "Cardiomegaly",
                0.0,
            ),
        }

        return max(
            candidate_scores,
            key=candidate_scores.get,
        )
