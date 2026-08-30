from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# DIRECTORIES TO CREATE
# ============================================================

DIRECTORIES = [
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "training_data",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "results",
]


# ============================================================
# EXPECTED DATA FILES
# ============================================================

EXPECTED_FILES = [

    # Original data
    PROJECT_ROOT / "dev" / "dev_metadata.csv",
    PROJECT_ROOT / "test_data" / "test_metadata.csv",
    PROJECT_ROOT / "test_data" / "answer_key.csv",

    # Main manifests
    PROJECT_ROOT / "training_data" / "specialist_train_manifest.csv",
    PROJECT_ROOT / "training_data" / "specialist_validation_manifest.csv",

]


# ============================================================
# MODALITIES
# ============================================================

MODALITIES = [
    "ct",
    "dermatology",
    "fundus",
    "mri",
    "microscopy",
    "oct",
    "ultrasound",
    "x_ray",
]


# ============================================================
# CREATE DIRECTORIES
# ============================================================

def create_directories():

    print()
    print("=" * 70)
    print("CREATING PROJECT STRUCTURE")
    print("=" * 70)

    print()
    print("Creating directories...")

    for directory in DIRECTORIES:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            f"[OK] {directory.relative_to(PROJECT_ROOT)}"
        )


# ============================================================
# CHECK REQUIRED DATA FILES
# ============================================================

def check_required_files():

    print()
    print("=" * 70)
    print("CHECKING REQUIRED DATA FILES")
    print("=" * 70)

    all_present = True

    for file_path in EXPECTED_FILES:

        exists = file_path.exists()

        status = "FOUND" if exists else "MISSING"

        print(
            f"[{status}] "
            f"{file_path.relative_to(PROJECT_ROOT)}"
        )

        if not exists:
            all_present = False

    return all_present


# ============================================================
# CHECK MODALITY MANIFESTS
# ============================================================

def check_modality_manifests():

    print()
    print("=" * 70)
    print("CHECKING MODALITY MANIFESTS")
    print("=" * 70)

    manifest_count = 0

    for modality in MODALITIES:

        train_manifest = (
            PROJECT_ROOT
            / "training_data"
            / f"{modality}_train_manifest.csv"
        )

        validation_manifest = (
            PROJECT_ROOT
            / "training_data"
            / f"{modality}_validation_manifest.csv"
        )

        train_status = (
            "FOUND"
            if train_manifest.exists()
            else "MISSING"
        )

        validation_status = (
            "FOUND"
            if validation_manifest.exists()
            else "MISSING"
        )

        print()
        print(f"{modality.upper()}")

        print(
            f"  Train:      "
            f"{train_status}"
        )

        print(
            f"  Validation: "
            f"{validation_status}"
        )

        if train_manifest.exists():
            manifest_count += 1

        if validation_manifest.exists():
            manifest_count += 1

    return manifest_count


# ============================================================
# CHECK EXISTING MODELS
# ============================================================

def check_existing_models():

    print()
    print("=" * 70)
    print("CHECKING EXISTING SPECIALIST MODELS")
    print("=" * 70)

    for modality in MODALITIES:

        model_path = (
            PROJECT_ROOT
            / "models"
            / f"{modality}_specialist.joblib"
        )

        if model_path.exists():

            print(
                f"[EXISTS] "
                f"{model_path.relative_to(PROJECT_ROOT)}"
            )

        else:

            print(
                f"[NOT CREATED] "
                f"{model_path.relative_to(PROJECT_ROOT)}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    create_directories()

    required_files_ok = (
        check_required_files()
    )

    manifest_count = (
        check_modality_manifests()
    )

    check_existing_models()

    print()
    print("=" * 70)
    print("PROJECT STRUCTURE SUMMARY")
    print("=" * 70)

    print()

    print(
        f"Required files status: "
        f"{'READY' if required_files_ok else 'INCOMPLETE'}"
    )

    print(
        f"Modality manifests found: "
        f"{manifest_count}/16"
    )

    print()

    if required_files_ok and manifest_count == 16:

        print(
            "PROJECT STRUCTURE IS READY."
        )

        print(
            "You can proceed with specialist "
            "feature extraction and training."
        )

    else:

        print(
            "PROJECT STRUCTURE IS NOT YET COMPLETE."
        )

        print(
            "Create the missing manifests before "
            "continuing."
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()