from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_MANIFEST = (
    PROJECT_ROOT
    / "training_data"
    / "ct_train_manifest.csv"
)

VALIDATION_MANIFEST = (
    PROJECT_ROOT
    / "training_data"
    / "ct_validation_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "training_data"
)

IMAGE_ROOT = (
    PROJECT_ROOT
    / "test_data"
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
    # ResNet18 then produces a 512-dimensional embedding.
    feature_extractor = nn.Sequential(
        *list(model.children())[:-1]
    )

    feature_extractor.eval()

    print("Feature extractor ready.")

    return feature_extractor


# ============================================================
# EXTRACT ONE IMAGE FEATURE
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
        )

        return feature.astype(
            np.float32
        )

    except Exception as error:

        print(
            f"ERROR processing image: "
            f"{image_path}"
        )

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
                f"MISSING IMAGE: "
                f"{image_path}"
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
            f"{split_name}"
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

    print()
    print("=" * 70)
    print("CT FEATURE EXTRACTION")
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "Only CT training and validation "
        "manifests will be loaded."
    )
    print(
        "Final test data will NOT be loaded."
    )

    if not TRAIN_MANIFEST.exists():

        raise FileNotFoundError(
            f"Training manifest not found: "
            f"{TRAIN_MANIFEST}"
        )

    if not VALIDATION_MANIFEST.exists():

        raise FileNotFoundError(
            f"Validation manifest not found: "
            f"{VALIDATION_MANIFEST}"
        )

    feature_extractor = (
        load_feature_extractor()
    )

    train_data = process_manifest(
        TRAIN_MANIFEST,
        "training",
        feature_extractor,
    )

    validation_data = process_manifest(
        VALIDATION_MANIFEST,
        "validation",
        feature_extractor,
    )

    train_output = (
        OUTPUT_DIR
        / "ct_train_features.pkl"
    )

    validation_output = (
        OUTPUT_DIR
        / "ct_validation_features.pkl"
    )

    joblib.dump(
        train_data,
        train_output,
    )

    joblib.dump(
        validation_data,
        validation_output,
    )

    print()
    print("=" * 70)
    print("SAVED FEATURES")
    print("=" * 70)

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
        "Final test data was NOT loaded."
    )


if __name__ == "__main__":
    main()