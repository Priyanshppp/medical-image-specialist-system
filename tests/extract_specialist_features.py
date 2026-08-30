from pathlib import Path
import argparse

import joblib
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAINING_DATA_DIR = (
    PROJECT_ROOT / "training_data"
)

IMAGE_ROOT = (
    PROJECT_ROOT / "test_data"
)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# ============================================================
# LOAD FEATURE EXTRACTOR
# ============================================================

def load_feature_extractor():

    print("Loading pretrained ResNet18...")

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

    # Remove classification layer.
    # ResNet18 produces a 512-dimensional feature vector.
    feature_extractor = nn.Sequential(
        *list(model.children())[:-1]
    )

    feature_extractor.eval()

    print("Feature extractor ready.")

    return feature_extractor


# ============================================================
# EXTRACT SINGLE IMAGE FEATURE
# ============================================================

def extract_image_feature(
    image_path,
    feature_extractor,
):

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

        tensor = IMAGE_TRANSFORM(
            image
        )

        tensor = tensor.unsqueeze(0)

        with torch.no_grad():

            feature = feature_extractor(
                tensor
            )

        feature = (
            feature
            .squeeze()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        return feature

    except Exception as error:

        print(
            f"\nERROR processing image:"
        )

        print(image_path)

        print(
            f"Reason: {error}"
        )

        return None


# ============================================================
# PROCESS MANIFEST
# ============================================================

def process_manifest(
    manifest_path,
    split_name,
    feature_extractor,
):

    print()
    print("=" * 70)
    print(
        f"PROCESSING {split_name.upper()} SPLIT"
    )
    print("=" * 70)

    df = pd.read_csv(
        manifest_path
    )

    print(
        f"Samples: {len(df)}"
    )

    features = []
    labels = []
    query_ids = []
    image_paths = []

    for index, row in df.iterrows():

        relative_path = Path(
            row["image"]
        )

        image_path = (
            IMAGE_ROOT
            / relative_path
        )

        if not image_path.exists():

            print(
                f"MISSING IMAGE: {image_path}"
            )

            continue

        feature = extract_image_feature(
            image_path,
            feature_extractor,
        )

        if feature is None:
            continue

        features.append(feature)

        labels.append(
            str(row["answer"])
        )

        query_ids.append(
            str(row["query_id"])
        )

        image_paths.append(
            str(row["image"])
        )

        print(
            f"[{index + 1}/{len(df)}] "
            f"{row['query_id']} "
            f"-> {row['answer']}"
        )

    if not features:

        raise RuntimeError(
            f"No features extracted for "
            f"{split_name} split."
        )

    features = np.vstack(
        features
    )

    print()
    print(
        f"{split_name} feature shape: "
        f"{features.shape}"
    )

    print(
        f"{split_name} labels: "
        f"{len(labels)}"
    )

    return {
        "features": features,
        "labels": np.array(
            labels,
            dtype=str,
        ),
        "query_ids": np.array(
            query_ids,
            dtype=str,
        ),
        "image_paths": np.array(
            image_paths,
            dtype=str,
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract ResNet18 features for "
            "a modality specialist."
        )
    )

    parser.add_argument(
        "--modality",
        required=True,
        type=str,
        help=(
            "Modality name, for example: "
            "dermatology, fundus, mri"
        ),
    )

    args = parser.parse_args()

    modality = (
        args.modality
        .strip()
        .lower()
    )

    print()
    print("=" * 70)
    print(
        f"{modality.upper()} FEATURE EXTRACTION"
    )
    print("=" * 70)

    print()
    print("IMPORTANT:")
    print(
        "Only training and validation manifests "
        "will be loaded."
    )
    print(
        "Final test data will NOT be loaded."
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    train_manifest = (
        TRAINING_DATA_DIR
        / f"{modality}_train_manifest.csv"
    )

    validation_manifest = (
        TRAINING_DATA_DIR
        / f"{modality}_validation_manifest.csv"
    )

    train_output = (
        TRAINING_DATA_DIR
        / f"{modality}_train_features.pkl"
    )

    validation_output = (
        TRAINING_DATA_DIR
        / f"{modality}_validation_features.pkl"
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if not train_manifest.exists():

        raise FileNotFoundError(
            f"Training manifest not found:\n"
            f"{train_manifest}"
        )

    if not validation_manifest.exists():

        raise FileNotFoundError(
            f"Validation manifest not found:\n"
            f"{validation_manifest}"
        )

    # --------------------------------------------------------
    # Load extractor
    # --------------------------------------------------------

    feature_extractor = (
        load_feature_extractor()
    )

    # --------------------------------------------------------
    # Training features
    # --------------------------------------------------------

    train_data = process_manifest(
        train_manifest,
        "training",
        feature_extractor,
    )

    # --------------------------------------------------------
    # Validation features
    # --------------------------------------------------------

    validation_data = process_manifest(
        validation_manifest,
        "validation",
        feature_extractor,
    )

    # --------------------------------------------------------
    # Save using joblib
    # --------------------------------------------------------

    joblib.dump(
        train_data,
        train_output,
    )

    joblib.dump(
        validation_data,
        validation_output,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAVED FEATURES")
    print("=" * 70)

    print()
    print(
        f"Training features:\n"
        f"{train_output}"
    )

    print()
    print(
        f"Validation features:\n"
        f"{validation_output}"
    )

    print()
    print("=" * 70)
    print("SUCCESS")
    print("=" * 70)

    print(
        f"{modality.upper()} feature extraction "
        f"completed successfully."
    )

    print(
        "Final test data was NOT loaded."
    )


if __name__ == "__main__":
    main()