from pathlib import Path

from src.config import IMAGE_EXTENSIONS


def detect_input_type(path: str | Path) -> str:
    path = Path(path)

    if path.is_dir():
        return "directory"

    name = path.name.lower()

    if name.endswith(".nii") or name.endswith(".nii.gz"):
        return "volume"

    if path.suffix.lower() in IMAGE_EXTENSIONS:
        return "image"

    if path.suffix.lower() in {".dcm", ".dicom"}:
        return "dicom"

    raise ValueError(
        f"Unsupported input type: {path}"
    )
