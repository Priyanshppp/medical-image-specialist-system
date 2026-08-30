from pathlib import Path

import pandas as pd


def main():

    print()
    print("=" * 70)
    print("CREATING VALIDATION MANIFEST")
    print("=" * 70)

    # ============================================================
    # FILE PATHS
    # ============================================================

    validation_file = Path(
        "training_data/modality_validation.csv"
    )

    final_test_file = Path(
        "training_data/modality_final_test.csv"
    )

    metadata_file = Path(
        "test_data/test_metadata.csv"
    )

    answers_file = Path(
        "test_data/answer_key.csv"
    )

    output_file = Path(
        "training_data/validation_manifest.csv"
    )

    # ============================================================
    # LOAD VALIDATION IDS
    # ============================================================

    validation = pd.read_csv(
        validation_file
    )

    print()
    print(
        f"Validation samples: {len(validation)}"
    )

    validation_ids = set(
        validation["query_id"]
    )

    # ============================================================
    # LOAD FINAL TEST IDS
    #
    # ONLY FOR SAFETY CHECKING.
    # DO NOT USE ITS FEATURES OR LABELS.
    # ============================================================

    final_test = pd.read_csv(
        final_test_file
    )

    final_test_ids = set(
        final_test["query_id"]
    )

    print(
        f"Final test samples: {len(final_test_ids)}"
    )

    # ============================================================
    # CRITICAL SAFETY CHECK
    # ============================================================

    overlap = (
        validation_ids
        & final_test_ids
    )

    if overlap:

        raise RuntimeError(
            "DATA LEAKAGE DETECTED!\n"
            f"Validation and final test overlap: "
            f"{len(overlap)} samples"
        )

    print()
    print(
        "Safety check passed."
    )

    print(
        "Validation and final test sets "
        "have zero overlap."
    )

    # ============================================================
    # LOAD ORIGINAL METADATA
    # ============================================================

    metadata = pd.read_csv(
        metadata_file
    )

    answers = pd.read_csv(
        answers_file
    )

    print()
    print(
        "Metadata columns:"
    )

    print(
        metadata.columns.tolist()
    )

    print()
    print(
        "Answer key columns:"
    )

    print(
        answers.columns.tolist()
    )

    # ============================================================
    # FILTER STRICTLY TO VALIDATION IDS
    # ============================================================

    validation_metadata = metadata[
        metadata["query_id"].isin(
            validation_ids
        )
    ].copy()

    validation_answers = answers[
        answers["query_id"].isin(
            validation_ids
        )
    ].copy()

    # ============================================================
    # SECOND SAFETY CHECK
    # ============================================================

    metadata_ids = set(
        validation_metadata["query_id"]
    )

    answer_ids = set(
        validation_answers["query_id"]
    )

    metadata_overlap = (
        metadata_ids
        & final_test_ids
    )

    answer_overlap = (
        answer_ids
        & final_test_ids
    )

    if metadata_overlap:

        raise RuntimeError(
            "FINAL TEST LEAKAGE INTO METADATA!"
        )

    if answer_overlap:

        raise RuntimeError(
            "FINAL TEST LEAKAGE INTO ANSWERS!"
        )

    # ============================================================
    # MERGE VALIDATION DATA
    # ============================================================

    manifest = validation.merge(
        validation_metadata,
        on="query_id",
        how="left",
        suffixes=(
            "",
            "_metadata",
        ),
    )

    manifest = manifest.merge(
        validation_answers,
        on="query_id",
        how="left",
        suffixes=(
            "",
            "_answer",
        ),
    )

    # ============================================================
    # FINAL SAFETY CHECK
    # ============================================================

    manifest_ids = set(
        manifest["query_id"]
    )

    final_overlap = (
        manifest_ids
        & final_test_ids
    )

    if final_overlap:

        raise RuntimeError(
            "FINAL TEST DATA LEAKED "
            "INTO VALIDATION MANIFEST!"
        )

    # ============================================================
    # VALIDATION CHECKS
    # ============================================================

    print()
    print("=" * 70)
    print("MANIFEST CHECKS")
    print("=" * 70)

    print(
        f"Manifest samples: {len(manifest)}"
    )

    print()
    print(
        "Manifest columns:"
    )

    print(
        manifest.columns.tolist()
    )

    missing_metadata = (
        manifest.isna()
        .all(axis=1)
        .sum()
    )

    print()
    print(
        f"Rows completely empty: "
        f"{missing_metadata}"
    )

    # ============================================================
    # SAVE
    # ============================================================

    manifest.to_csv(
        output_file,
        index=False,
    )

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print(
        f"Validation manifest saved to:\n"
        f"{output_file}"
    )

    print()
    print(
        "This manifest contains ONLY "
        "validation IDs."
    )

    print(
        "Final test IDs were explicitly "
        "excluded."
    )

    print("=" * 70)


if __name__ == "__main__":
    main()