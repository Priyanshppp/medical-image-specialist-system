from src.input_router import detect_input_type
from src.image_io import load_image
from src.volume_io import load_volume_slices
from src.directory_io import load_directory_views


def load_representative_views(path):
    """
    Load representative views from an image, volume,
    or directory-based medical imaging input.
    """

    input_type = detect_input_type(path)

    # =========================================
    # Standard image
    # =========================================

    if input_type == "image":
        return [
            load_image(path)
        ]

    # =========================================
    # NIfTI / volume
    # =========================================

    if input_type == "volume":
        return load_volume_slices(
            path,
            num_slices=5,
        )

    # =========================================
    # DICOM directory / image directory
    # =========================================

    if input_type == "directory":
        return load_directory_views(path)

    # =========================================
    # Standalone DICOM
    # =========================================

    if input_type == "dicom":
        raise ValueError(
            "Standalone DICOM files are not "
            "currently expected in this dataset."
        )

    # =========================================
    # Unknown input
    # =========================================

    raise ValueError(
        f"Unhandled input type: {input_type}"
    )
