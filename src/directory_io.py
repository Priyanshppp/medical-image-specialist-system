from pathlib import Path

from src.config import IMAGE_EXTENSIONS
from src.image_io import load_image
from src.dicom_io import load_dicom_representative_views


def load_directory_views(directory):
    directory = Path(directory)

    dicom_files = list(directory.glob("*.dcm"))

    # Case 1: DICOM series
    if dicom_files:
        return load_dicom_representative_views(
            directory,
            num_slices=3,
        )

    # Case 2: ordinary images
    image_files = sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    if not image_files:
        raise ValueError(
            f"No supported files found in {directory}"
        )

    # For 0016 and 0017, load all four images.
    if len(image_files) <= 4:
        return [
            load_image(path)
            for path in image_files
        ]

    # General fallback: representative sampling.
    import numpy as np

    indices = np.linspace(
        0,
        len(image_files) - 1,
        3,
    ).astype(int)

    return [
        load_image(image_files[index])
        for index in indices
    ]
