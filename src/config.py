from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

VOLUME_EXTENSIONS = {
    ".nii",
    ".nii.gz",
}

DICOM_EXTENSIONS = {
    ".dcm",
    ".dicom",
}

NUM_REPRESENTATIVE_SLICES = 3

VALID_ANSWERS = {"A", "B", "C", "D"}
