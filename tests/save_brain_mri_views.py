from pathlib import Path

from src.preprocess import load_representative_views


def save_views(volume_path, prefix):

    images = load_representative_views(
        Path(volume_path)
    )

    print(f"\n{volume_path}")
    print(f"Number of views: {len(images)}")

    for i, image in enumerate(images):

        output = f"{prefix}_view_{i}.png"

        image.save(output)

        print(
            f"Saved {output} "
            f"| size={image.size}"
        )


def main():

    save_views(
        "dev/images/0006.nii",
        "mri_0006",
    )

    save_views(
        "dev/images/0007.nii",
        "mri_0007",
    )


if __name__ == "__main__":
    main()
