from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "query_id",
    "image",
    "question",
    "choice_A",
    "choice_B",
    "choice_C",
    "choice_D",
]


def find_metadata_file(input_dir):
    input_dir = Path(input_dir)

    csv_files = []

    for csv_path in input_dir.glob("*.csv"):
        # Ignore empty CSV files
        if csv_path.stat().st_size > 0:
            csv_files.append(csv_path)

    if not csv_files:
        raise FileNotFoundError(
            f"No non-empty CSV file found in: {input_dir}"
        )

    print("Found CSV files:")

    for csv_file in csv_files:
        print(
            f"  {csv_file} "
            f"({csv_file.stat().st_size} bytes)"
        )

    # Prefer files containing "metadata"
    metadata_files = [
        f for f in csv_files
        if "metadata" in f.name.lower()
    ]

    if len(metadata_files) == 1:
        return metadata_files[0]

    if len(csv_files) == 1:
        return csv_files[0]

    raise RuntimeError(
        "Multiple possible metadata CSV files found:\n"
        + "\n".join(str(f) for f in csv_files)
    )


def load_queries(input_dir):
    input_dir = Path(input_dir)

    metadata_path = find_metadata_file(input_dir)

    df = pd.read_csv(
    metadata_path,
    encoding="cp1252"
)

    missing = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df, input_dir 
import pandas as pd


def extract_choices(row):
    choices = {}

    for letter in ["A", "B", "C", "D"]:
        value = row[f"choice_{letter}"]

        if pd.notna(value):
            value = str(value).strip()

            if value:
                choices[letter] = value

    return choices
