from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


def get_slice_position(ds):
    """
    Return a geometric position for ordering DICOM slices.

    Uses ImageOrientationPatient and ImagePositionPatient
    when available.
    """

    orientation = getattr(
        ds,
        "ImageOrientationPatient",
        None,
    )

    position = getattr(
        ds,
        "ImagePositionPatient",
        None,
    )

    if orientation is not None and position is not None:

        orientation = np.asarray(
            orientation,
            dtype=np.float64,
        )

        position = np.asarray(
            position,
            dtype=np.float64,
        )

        row_direction = orientation[:3]
        column_direction = orientation[3:]

        normal = np.cross(
            row_direction,
            column_direction,
        )

        return float(
            np.dot(
                position,
                normal,
            )
        )

    return None


def normalize_ct_pixels(ds):
    """
    Convert DICOM pixel data into an 8-bit image.
    """

    pixels = ds.pixel_array.astype(np.float32)

    slope = float(
        getattr(
            ds,
            "RescaleSlope",
            1.0,
        )
    )

    intercept = float(
        getattr(
            ds,
            "RescaleIntercept",
            0.0,
        )
    )

    pixels = pixels * slope + intercept

    low = np.percentile(
        pixels,
        1,
    )

    high = np.percentile(
        pixels,
        99,
    )

    if high <= low:
        return np.zeros(
            pixels.shape,
            dtype=np.uint8,
        )

    pixels = np.clip(
        pixels,
        low,
        high,
    )

    pixels = (
        (pixels - low)
        / (high - low)
        * 255
    )

    return pixels.astype(np.uint8)


def read_dicom_series(directory):
    """
    Read and geometrically sort all DICOM files
    in a directory.
    """

    directory = Path(directory)

    dicom_files = list(
        directory.glob("*.dcm")
    )

    if not dicom_files:
        raise ValueError(
            f"No DICOM files found in {directory}"
        )

    slices = []

    for file_path in dicom_files:

        ds = pydicom.dcmread(
            file_path,
            force=True,
        )

        position = get_slice_position(ds)

        instance = getattr(
            ds,
            "InstanceNumber",
            None,
        )

        slices.append(
            {
                "dataset": ds,
                "position": position,
                "instance": instance,
            }
        )

    # Prefer geometric ordering.
    if all(
        item["position"] is not None
        for item in slices
    ):

        slices.sort(
            key=lambda x: x["position"]
        )

    # Fallback to InstanceNumber.
    elif all(
        item["instance"] is not None
        for item in slices
    ):

        slices.sort(
            key=lambda x: int(x["instance"])
        )

    else:

        raise ValueError(
            "Cannot reliably determine "
            "DICOM slice order"
        )

    return slices


def load_dicom_representative_views(
    directory,
    num_slices=3,
):
    """
    Select representative slices from
    a DICOM series.
    """

    slices = read_dicom_series(directory)

    total = len(slices)

    indices = np.linspace(
        int(total * 0.25),
        int(total * 0.75),
        num_slices,
    ).astype(int)

    images = []

    for index in indices:

        index = max(
            0,
            min(
                index,
                total - 1,
            ),
        )

        ds = slices[index]["dataset"]

        pixels = normalize_ct_pixels(ds)

        image = Image.fromarray(
            pixels
        ).convert("RGB")

        images.append(image)

    return images
