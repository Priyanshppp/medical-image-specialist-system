from pathlib import Path

import pandas as pd


FILES = [
    "dev/dev_metadata.csv",
    "test_data/test_metadata.csv",
    "test_data/answer_key.csv",
]


def read_csv_safe(path):

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1",
    ]

    for encoding in encodings:

        try:

            df = pd.read_csv(
                path,
                encoding=encoding,
            )

            print(
                f"Successfully loaded with encoding: "
                f"{encoding}"
            )

            return df

        except UnicodeDecodeError:
            continue

    raise RuntimeError(
        f"Could not decode file: {path}"
    )


def main():

    print("=" * 70)
    print("INSPECTING DATA FILES")
    print("=" * 70)

    for file_path in FILES:

        path = Path(file_path)

        print()
        print("=" * 70)
        print(f"FILE: {path}")
        print("=" * 70)

        df = read_csv_safe(path)

        print()
        print("Shape:")
        print(df.shape)

        print()
        print("Columns:")
        print(df.columns.tolist())

        print()
        print("First 3 rows:")
        print(
            df.head(3).to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()