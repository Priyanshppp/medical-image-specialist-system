from pathlib import Path
from collections import Counter

import pydicom


DICOM_DIRS = [
    Path("dev/images/0004"),
    Path("dev/images/0005"),
]


for dicom_dir in DICOM_DIRS:

    print("\n" + "=" * 80)
    print(f"DIRECTORY: {dicom_dir}")
    print("=" * 80)

    files = sorted(dicom_dir.glob("*.dcm"))

    print(f"Number of DICOM files: {len(files)}\n")

    rows = []

    for file_path in files:
        try:
            ds = pydicom.dcmread(
                file_path,
                stop_before_pixels=True,
                force=True,
            )

            rows.append({
                "file": file_path.name,
                "series_uid": str(
                    getattr(ds, "SeriesInstanceUID", "MISSING")
                ),
                "instance": getattr(
                    ds,
                    "InstanceNumber",
                    None,
                ),
                "position": getattr(
                    ds,
                    "ImagePositionPatient",
                    None,
                ),
                "orientation": getattr(
                    ds,
                    "ImageOrientationPatient",
                    None,
                ),
                "modality": getattr(
                    ds,
                    "Modality",
                    None,
                ),
                "description": getattr(
                    ds,
                    "SeriesDescription",
                    None,
                ),
            })

        except Exception as error:
            print(
                f"FAILED: {file_path.name} -> {error}"
            )

    print(f"Successfully read: {len(rows)}")

    print("\nSeriesInstanceUID counts:")

    counts = Counter(
        row["series_uid"]
        for row in rows
    )

    for uid, count in counts.items():
        print(f"  {uid}: {count} files")

    print("\nFirst 5 records:")

    for row in rows[:5]:
        print()
        for key, value in row.items():
            print(f"  {key}: {value}")
