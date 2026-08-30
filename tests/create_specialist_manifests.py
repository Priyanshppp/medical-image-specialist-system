from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

TRAIN_SPLIT_PATH = Path(
    "training_data/modality_train.csv"
)

VALIDATION_SPLIT_PATH = Path(
    "training_data/modality_validation.csv"
)

METADATA_PATH = Path(
    "test_data/test_metadata.csv"
)

ANSWER_KEY_PATH = Path(
    "test_data/answer_key.csv"
)

OUTPUT_DIR = Path(
    "training_data"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("CREATING SPECIALIST TRAINING MANIFESTS")
    print("=" * 70)

    print()
    print("Loading modality training split...")

    train_split = pd.read_csv(
        TRAIN_SPLIT_PATH
    )

    print(
        f"Training split samples: "
        f"{len(train_split)}"
    )

    print("Loading modality validation split...")

    validation_split = pd.read_csv(
        VALIDATION_SPLIT_PATH
    )

    print(
        f"Validation split samples: "
        f"{len(validation_split)}"
    )

    print("Loading metadata...")

    metadata = pd.read_csv(
        METADATA_PATH
    )

    print(
        f"Metadata samples: "
        f"{len(metadata)}"
    )

    print("Loading answer key...")

    answers = pd.read_csv(
        ANSWER_KEY_PATH
    )

    print(
        f"Answer key samples: "
        f"{len(answers)}"
    )

    return (
        train_split,
        validation_split,
        metadata,
        answers,
    )


# ============================================================
# NORMALIZE QUERY IDS
# ============================================================

def normalize_ids(df):

    df = df.copy()

    df["query_id"] = (
        df["query_id"]
        .astype(str)
        .str.strip()
    )

    return df


# ============================================================
# BUILD FULL MASTER TABLE
# ============================================================

def build_master_table(
    metadata,
    answers,
):

    metadata = normalize_ids(
        metadata
    )

    answers = normalize_ids(
        answers
    )

    print()
    print("Merging metadata with answers...")

    master = metadata.merge(
        answers,
        on="query_id",
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Master table samples: "
        f"{len(master)}"
    )

    expected_columns = [
        "query_id",
        "image",
        "Question",
        "Option_A",
        "Option_B",
        "Option_C",
        "Option_D",
        "answer",
        "modality",
    ]

    missing = [
        column
        for column in expected_columns
        if column not in master.columns
    ]

    if missing:

        raise ValueError(
            f"Missing columns in master table: "
            f"{missing}"
        )

    return master


# ============================================================
# CREATE MANIFEST
# ============================================================

def create_manifest(
    split_df,
    master_df,
    split_name,
):

    split_df = normalize_ids(
        split_df
    )

    master_df = normalize_ids(
        master_df
    )

    split_ids = set(
        split_df["query_id"]
    )

    manifest = master_df[
        master_df["query_id"].isin(
            split_ids
        )
    ].copy()

    # Preserve the exact modality assignment
    # from the split file.
    modality_lookup = (
        split_df[
            [
                "query_id",
                "modality",
            ]
        ]
        .drop_duplicates(
            subset=["query_id"]
        )
        .rename(
            columns={
                "modality": "split_modality"
            }
        )
    )

    manifest = (
        manifest
        .drop(
            columns=["modality"],
            errors="ignore",
        )
        .merge(
            modality_lookup,
            on="query_id",
            how="left",
            validate="one_to_one",
        )
        .rename(
            columns={
                "split_modality": "modality"
            }
        )
    )

    # Safety check
    missing_ids = (
        split_ids
        - set(manifest["query_id"])
    )

    if missing_ids:

        raise ValueError(
            f"{split_name}: "
            f"{len(missing_ids)} split IDs "
            f"could not be found in metadata."
        )

    if len(manifest) != len(split_df):

        raise ValueError(
            f"{split_name}: sample count mismatch. "
            f"Split={len(split_df)}, "
            f"Manifest={len(manifest)}"
        )

    print()
    print(
        f"{split_name.upper()} MANIFEST CREATED"
    )

    print(
        f"Samples: {len(manifest)}"
    )

    print()
    print("Modality distribution:")

    print(
        manifest["modality"]
        .value_counts()
        .sort_index()
    )

    return manifest


# ============================================================
# CREATE PER-MODALITY FILES
# ============================================================

def save_modality_manifests(
    manifest,
    split_name,
):

    modalities = sorted(
        manifest["modality"]
        .unique()
    )

    print()
    print(
        f"Saving {split_name} "
        f"per-modality manifests..."
    )

    for modality in modalities:

        subset = manifest[
            manifest["modality"] == modality
        ].copy()

        safe_name = (
            modality
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        output_path = (
            OUTPUT_DIR
            / f"{safe_name}_{split_name}_manifest.csv"
        )

        subset.to_csv(
            output_path,
            index=False,
        )

        print(
            f"{modality:15} "
            f"{len(subset):3} samples -> "
            f"{output_path}"
        )


# ============================================================
# SAFETY CHECK
# ============================================================

def check_no_overlap(
    train_manifest,
    validation_manifest,
):

    train_ids = set(
        train_manifest["query_id"]
    )

    validation_ids = set(
        validation_manifest["query_id"]
    )

    overlap = (
        train_ids
        & validation_ids
    )

    print()
    print("=" * 70)
    print("SPLIT SAFETY CHECK")
    print("=" * 70)

    print(
        f"Training IDs:   "
        f"{len(train_ids)}"
    )

    print(
        f"Validation IDs: "
        f"{len(validation_ids)}"
    )

    print(
        f"Overlap:        "
        f"{len(overlap)}"
    )

    if overlap:

        raise RuntimeError(
            "ERROR: Training and validation "
            "sets overlap."
        )

    print()
    print(
        "PASSED: Training and validation "
        "sets have zero overlap."
    )

    print()
    print(
        "IMPORTANT: "
        "modality_final_test.csv was NOT loaded."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    (
        train_split,
        validation_split,
        metadata,
        answers,
    ) = load_data()

    master = build_master_table(
        metadata,
        answers,
    )

    train_manifest = create_manifest(
        train_split,
        master,
        "train",
    )

    validation_manifest = create_manifest(
        validation_split,
        master,
        "validation",
    )

    check_no_overlap(
        train_manifest,
        validation_manifest,
    )

    # --------------------------------------------------------
    # Save complete manifests
    # --------------------------------------------------------

    train_output = (
        OUTPUT_DIR
        / "specialist_train_manifest.csv"
    )

    validation_output = (
        OUTPUT_DIR
        / "specialist_validation_manifest.csv"
    )

    train_manifest.to_csv(
        train_output,
        index=False,
    )

    validation_manifest.to_csv(
        validation_output,
        index=False,
    )

    print()
    print("=" * 70)
    print("SAVING COMPLETE MANIFESTS")
    print("=" * 70)

    print(
        f"Training manifest:\n"
        f"{train_output}"
    )

    print(
        f"Validation manifest:\n"
        f"{validation_output}"
    )

    # --------------------------------------------------------
    # Save individual modality manifests
    # --------------------------------------------------------

    save_modality_manifests(
        train_manifest,
        "train",
    )

    save_modality_manifests(
        validation_manifest,
        "validation",
    )

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print(
        "Specialist manifests created successfully."
    )

    print()
    print(
        "Final test data remains untouched."
    )

    print(
        "modality_final_test.csv was never loaded."
    )


if __name__ == "__main__":
    main()