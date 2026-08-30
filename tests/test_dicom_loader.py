from pathlib import Path

from src.dicom_io import (
    read_dicom_series,
    load_dicom_representative_views,
)


for directory in [
    Path("dev/images/0004"),
    Path("dev/images/0005"),
]:

    print("\n" + "=" * 70)
    print(f"TESTING: {directory}")
    print("=" * 70)

    slices = read_dicom_series(
        directory
    )

    print(
        f"Total slices: {len(slices)}"
    )

    print(
        "First position:",
        slices[0]["position"],
    )

    print(
        "Last position:",
        slices[-1]["position"],
    )

    print(
        "First instance:",
        slices[0]["instance"],
    )

    print(
        "Last instance:",
        slices[-1]["instance"],
    )

    views = load_dicom_representative_views(
        directory,
        num_slices=3,
    )

    print(
        f"Representative views: {len(views)}"
    )

    for i, image in enumerate(views):
        print(
            f"View {i}: size={image.size}, "
            f"mode={image.mode}"
        )
