from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():

    input_file = Path(
        "test_data/modality_features.csv"
    )

    output_dir = Path(
        "training_data"
    )

    output_dir.mkdir(
        exist_ok=True
    )

    df = pd.read_csv(
        input_file
    )

    print("=" * 70)
    print("CREATING MODALITY DATA SPLITS")
    print("=" * 70)

    print()
    print("Total samples:")
    print(len(df))

    # -------------------------------------------------
    # First split:
    # 80% development
    # 20% completely untouched final test
    # -------------------------------------------------

    development_df, final_test_df = train_test_split(

        df,

        test_size=0.20,

        stratify=df["modality"],

        random_state=42,
    )

    # -------------------------------------------------
    # Second split:
    # Development data -> train + validation
    #
    # 640 samples:
    # 75% train = 480
    # 25% validation = 160
    # -------------------------------------------------

    train_df, validation_df = train_test_split(

        development_df,

        test_size=0.25,

        stratify=development_df["modality"],

        random_state=42,
    )

    # -------------------------------------------------
    # Save splits
    # -------------------------------------------------

    train_path = (
        output_dir / "modality_train.csv"
    )

    validation_path = (
        output_dir / "modality_validation.csv"
    )

    final_test_path = (
        output_dir / "modality_final_test.csv"
    )

    train_df.to_csv(
        train_path,
        index=False,
    )

    validation_df.to_csv(
        validation_path,
        index=False,
    )

    final_test_df.to_csv(
        final_test_path,
        index=False,
    )

    # -------------------------------------------------
    # Report
    # -------------------------------------------------

    print()
    print("SPLIT SUMMARY")
    print("-" * 70)

    print(
        f"Training samples:   {len(train_df)}"
    )

    print(
        f"Validation samples: {len(validation_df)}"
    )

    print(
        f"Final test samples: {len(final_test_df)}"
    )

    print()
    print("TRAIN DISTRIBUTION")
    print(
        train_df["modality"]
        .value_counts()
        .sort_index()
    )

    print()
    print("VALIDATION DISTRIBUTION")
    print(
        validation_df["modality"]
        .value_counts()
        .sort_index()
    )

    print()
    print("FINAL TEST DISTRIBUTION")
    print(
        final_test_df["modality"]
        .value_counts()
        .sort_index()
    )

    print()
    print("=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print(
        "The final test file must NOT be used "
        "for model tuning."
    )

    print(
        f"Final test saved at: {final_test_path}"
    )


if __name__ == "__main__":
    main()