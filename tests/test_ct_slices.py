from pathlib import Path
import pydicom
import numpy as np


def inspect_ct(directory):

    directory = Path(directory)

    files = sorted(
        directory.glob("*.dcm")
    )

    slices = []

    for file in files:

        ds = pydicom.dcmread(
            file,
            stop_before_pixels=True,
        )

        position = float(
            ds.ImagePositionPatient[2]
        )

        instance = int(
            ds.InstanceNumber
        )

        slices.append(
            {
                "file": file.name,
                "position": position,
                "instance": instance,
            }
        )

    slices.sort(
        key=lambda x: x["position"]
    )

    print()
    print("=" * 70)
    print(f"DIRECTORY: {directory}")
    print(f"TOTAL SLICES: {len(slices)}")
    print("=" * 70)

    indices = [
        0,
        len(slices) // 4,
        len(slices) // 2,
        3 * len(slices) // 4,
        len(slices) - 1,
    ]

    for index in indices:

        item = slices[index]

        print(
            f"Index {index:3d} | "
            f"Instance {item['instance']:3d} | "
            f"Position {item['position']:.2f}"
        )


def main():

    inspect_ct("dev/images/0004")

    inspect_ct("dev/images/0005")


if __name__ == "__main__":
    main()
    
