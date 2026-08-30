from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocess import load_representative_views


def extract_features(image):

    image = image.convert("RGB")

    array = (
        np.asarray(image, dtype=np.float32)
        / 255.0
    )

    gray = (
        0.299 * array[:, :, 0]
        + 0.587 * array[:, :, 1]
        + 0.114 * array[:, :, 2]
    )

    height, width = gray.shape

    saturation = (
        np.max(array, axis=2)
        - np.min(array, axis=2)
    )

    # =====================================
    # BASIC PROFILES
    # =====================================

    horizontal_profile = gray.mean(axis=1)
    vertical_profile = gray.mean(axis=0)

    # =====================================
    # TEXTURE
    # =====================================

    diff_vertical = np.diff(
        gray,
        axis=0,
    )

    diff_horizontal = np.diff(
        gray,
        axis=1,
    )

    texture_vertical = float(
        np.mean(
            np.abs(diff_vertical)
        )
    )

    texture_horizontal = float(
        np.mean(
            np.abs(diff_horizontal)
        )
    )

    # =====================================
    # GRADIENT FEATURES
    # =====================================

    grad_y, grad_x = np.gradient(
        gray
    )

    gradient_magnitude = np.sqrt(
        grad_x ** 2
        + grad_y ** 2
    )

    gradient_mean = float(
        gradient_magnitude.mean()
    )

    gradient_std = float(
        gradient_magnitude.std()
    )

    gradient_max = float(
        gradient_magnitude.max()
    )

    # =====================================
    # EDGE DENSITY
    # =====================================

    edge_threshold = (
        gradient_mean
        + gradient_std
    )

    edge_density = float(
        np.mean(
            gradient_magnitude
            > edge_threshold
        )
    )

    # =====================================
    # HISTOGRAM / ENTROPY
    # =====================================

    histogram, _ = np.histogram(

        gray,

        bins=64,

        range=(0.0, 1.0),

        density=True,
    )

    histogram = (
        histogram
        + 1e-10
    )

    histogram = (
        histogram
        / histogram.sum()
    )

    entropy = float(
        -np.sum(
            histogram
            * np.log2(histogram)
        )
    )

    # =====================================
    # INTENSITY PERCENTILES
    # =====================================

    p05 = float(
        np.percentile(gray, 5)
    )

    p25 = float(
        np.percentile(gray, 25)
    )

    p50 = float(
        np.percentile(gray, 50)
    )

    p75 = float(
        np.percentile(gray, 75)
    )

    p95 = float(
        np.percentile(gray, 95)
    )

    # =====================================
    # CENTER VS BORDER
    # =====================================

    center_h_start = height // 4
    center_h_end = (
        height
        - center_h_start
    )

    center_w_start = width // 4
    center_w_end = (
        width
        - center_w_start
    )

    center_region = gray[
        center_h_start:center_h_end,
        center_w_start:center_w_end,
    ]

    center_mean = float(
        center_region.mean()
    )

    border_mask = np.ones(
        gray.shape,
        dtype=bool,
    )

    border_mask[
        center_h_start:center_h_end,
        center_w_start:center_w_end,
    ] = False

    border_mean = float(
        gray[border_mask].mean()
    )

    center_border_diff = float(
        center_mean
        - border_mean
    )

    # =====================================
    # SYMMETRY
    # =====================================

    flipped_horizontal = np.fliplr(
        gray
    )

    horizontal_symmetry = float(
        np.mean(
            np.abs(
                gray
                - flipped_horizontal
            )
        )
    )

    flipped_vertical = np.flipud(
        gray
    )

    vertical_symmetry = float(
        np.mean(
            np.abs(
                gray
                - flipped_vertical
            )
        )
    )

    # =====================================
    # ASPECT RATIO
    # =====================================

    aspect_ratio = float(
        width / height
    )

    # =====================================
    # RETURN FEATURES
    # =====================================

    return {

        # Basic intensity

        "mean": float(
            gray.mean()
        ),

        "std": float(
            gray.std()
        ),

        "dark": float(
            np.mean(
                gray < 0.15
            )
        ),

        "bright": float(
            np.mean(
                gray > 0.85
            )
        ),

        # Color

        "colored": float(
            np.mean(
                saturation > 0.15
            )
        ),

        "red": float(
            array[:, :, 0].mean()
        ),

        "green": float(
            array[:, :, 1].mean()
        ),

        "blue": float(
            array[:, :, 2].mean()
        ),

        # Profiles

        "horizontal_var": float(
            horizontal_profile.std()
        ),

        "vertical_var": float(
            vertical_profile.std()
        ),

        # Texture

        "texture_v": texture_vertical,

        "texture_h": texture_horizontal,

        # New structural features

        "aspect_ratio": aspect_ratio,

        "entropy": entropy,

        "p05": p05,

        "p25": p25,

        "p50": p50,

        "p75": p75,

        "p95": p95,

        "gradient_mean": gradient_mean,

        "gradient_std": gradient_std,

        "gradient_max": gradient_max,

        "edge_density": edge_density,

        "center_mean": center_mean,

        "border_mean": border_mean,

        "center_border_diff": center_border_diff,

        "horizontal_symmetry": horizontal_symmetry,

        "vertical_symmetry": vertical_symmetry,
    }


def main():

    print()
    print("=" * 90)
    print("OMNI MODALITY FEATURE ANALYSIS V2")
    print("=" * 90)

    metadata = pd.read_csv(
        "test_data/test_metadata.csv"
    )

    answers = pd.read_csv(
        "test_data/answer_key.csv"
    )

    df = metadata.merge(

        answers[
            [
                "query_id",
                "modality",
            ]
        ],

        on="query_id",
    )

    all_results = []

    modalities = sorted(
        df["modality"].unique()
    )

    for modality in modalities:

        print()
        print("-" * 90)
        print(f"Analyzing {modality}")
        print("-" * 90)

        subset = df[
            df["modality"] == modality
        ].head(100)

        modality_results = []

        for index, (_, row) in enumerate(

            subset.iterrows(),

            start=1,
        ):

            image_path = (
                Path("test_data")
                / row["image"]
            )

            try:

                images = (
                    load_representative_views(
                        image_path
                    )
                )

                if not images:
                    continue

                features = extract_features(
                    images[0]
                )

                features["modality"] = modality

                features["query_id"] = (
                    row["query_id"]
                )

                modality_results.append(
                    features
                )

                print(

                    f"\rProcessed "
                    f"{index}/{len(subset)}",

                    end="",
                )

            except Exception as error:

                print(

                    f"\nError: "
                    f"{row['query_id']} "
                    f"{error}"
                )

        print()

        all_results.extend(
            modality_results
        )

    results = pd.DataFrame(
        all_results
    )

    output = (
        Path("test_data")
        / "modality_features.csv"
    )

    results.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 90)
    print(
        f"Saved features to: {output}"
    )

    print(
        f"Total images analyzed: "
        f"{len(results)}"
    )

    print(
        f"Total features: "
        f"{len(results.columns) - 2}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()