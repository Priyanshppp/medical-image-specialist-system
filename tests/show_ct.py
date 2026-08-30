from src.preprocess import load_representative_views


def main():

    paths = [
        "dev/images/0004",
        "dev/images/0005",
    ]

    for path in paths:

        print()
        print("=" * 70)
        print(f"LOADING: {path}")

        images = load_representative_views(path)

        for i, image in enumerate(images):

            output_path = (
                f"ct_{path.split('/')[-1]}_view_{i}.png"
            )

            image.save(output_path)

            print(
                f"Saved: {output_path} "
                f"| size={image.size}"
            )


if __name__ == "__main__":
    main()
    
