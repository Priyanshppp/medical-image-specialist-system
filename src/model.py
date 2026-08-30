from src.domain_router import DomainRouter

from src.models.chest_xray import ChestXrayModel
from src.models.ct import CTModel
from src.models.brain_mri import BrainMRIModel
from src.models.ultrasound import UltrasoundModel
from src.models.dermoscopy import DermoscopyModel
from src.models.oct import OCTModel
from src.models.microscopy import MicroscopyModel
from src.models.general import GeneralModel


class YourModel:

    def __init__(self):

        print("Initializing models...")

        self.router = DomainRouter()

        # =====================================
        # Specialized models
        # =====================================

        self.chest_xray_model = ChestXrayModel()
        self.ct_model = CTModel()
        self.brain_mri_model = BrainMRIModel()
        self.ultrasound_model = UltrasoundModel()
        self.dermoscopy_model = DermoscopyModel()
        self.oct_model = OCTModel()
        self.microscopy_model = MicroscopyModel()
        self.general_model = GeneralModel()

        print("Models initialized.")

    def answer(
        self,
        images,
        question,
        choices,
    ):

        domain = self.router.route(
            question,
            choices,
            images=images,

        )

        print(
            f"Domain selected: {domain}"
        )

        # =====================================
        # CHEST X-RAY
        # =====================================

        if domain == "chest_xray":

            return self.chest_xray_model.answer(
                images=images,
                question=question,
                choices=choices,
            )

        # =====================================
        # CT
        # =====================================

        if domain == "ct":

            return self.ct_model.answer(
                images=images,
                question=question,
                choices=choices,
            )

        # =====================================
        # BRAIN MRI
        # =====================================

        if domain == "brain_mri":

            return self.brain_mri_model.answer(
                images=images,
                question=question,
                choices=choices,
            )

        # =====================================
        # ULTRASOUND
        # =====================================

        if domain == "ultrasound":

            return self.ultrasound_model.answer(
                images=images,
                question=question,
                choices=choices,
            )

        # =====================================
        # DERMOSCOPY
        # =====================================

        if domain == "dermoscopy":

            return self.dermoscopy_model.answer(
                images=images,
                question=question,
                choices=choices,
            )

        # =====================================
        # OCT
        # =====================================

        if domain == "oct":

            return self.oct_model.answer(
                images=images,
                question=question,
                choices=choices,
            )

        # =====================================
        # MICROSCOPY
        # =====================================

        if domain == "microscopy":

            return self.microscopy_model.answer(
                images=images,
                question=question,
                choices=choices,
            )

        # =====================================
        # GENERAL IMAGE MODEL
        # =====================================

        if domain == "general":

            return self.general_model.answer(
                images=images,
                question=question,
                choices=choices,
            )

        # =====================================
        # FINAL FALLBACK
        # =====================================

        return next(iter(choices))
