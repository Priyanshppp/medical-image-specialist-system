import numpy as np
import nibabel as nib
from PIL import Image


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    slice_2d = np.asarray(
        slice_2d,
        dtype=np.float32,
    )

    low = np.percentile(slice_2d, 1)
    high = np.percentile(slice_2d, 99)

    if high <= low:
        return np.zeros_like(
            slice_2d,
            dtype=np.uint8,
        )

    slice_2d = np.clip(
        slice_2d,
        low,
        high,
    )

    slice_2d = (
        slice_2d - low
    ) / (
        high - low
    )

    return (
        slice_2d * 255
    ).astype(np.uint8)


def load_volume_slices(
    path,
    num_slices=5,
):
    """
    Load representative slices from a NIfTI volume.

    The volume is first reoriented to canonical RAS
    orientation. Only requested slices are loaded.
    """

    nii = nib.load(str(path))

    # Convert orientation to canonical RAS.
    nii = nib.as_closest_canonical(nii)

    shape = nii.shape

    if len(shape) == 4:
        depth = shape[2]
    elif len(shape) == 3:
        depth = shape[2]
    else:
        raise ValueError(
            f"Expected 3D or 4D NIfTI volume, "
            f"got shape {shape}"
        )

    if depth <= 0:
        raise ValueError(
            f"Invalid volume depth: {depth}"
        )

    positions = np.linspace(
        int(depth * 0.10),
        int(depth * 0.90),
        num=num_slices,
        dtype=int,
    )

    positions = np.unique(positions)

    images = []

    for index in positions:

        if len(shape) == 4:
            slice_2d = np.asanyarray(
                nii.dataobj[:, :, index, 0]
            )
        else:
            slice_2d = np.asanyarray(
                nii.dataobj[:, :, index]
            )

        slice_2d = normalize_slice(
            slice_2d
        )

        image = (
            Image
            .fromarray(slice_2d)
            .convert("RGB")
        )

        images.append(image)

    return images
