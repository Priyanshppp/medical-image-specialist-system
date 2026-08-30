import time

from src.config import VALID_ANSWERS
from src.preprocess import load_representative_views
from src.metadata import extract_choices


class Predictor:

    def __init__(self, model):
        # Model must already be loaded here.
        self.model = model

    def predict(self, image_path, query, choices):
        start = time.perf_counter()

        images = load_representative_views(
            image_path
        )

        answer = self.model.answer(
            images=images,
            question=query,
            choices=choices,
        )

        answer = str(answer).strip().upper()

        if answer not in VALID_ANSWERS:
            raise ValueError(
                f"Invalid model answer: {answer}"
            )

        inference_time = (
            time.perf_counter() - start
        )

        return answer, inference_time
