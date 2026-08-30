from src.metadata import load_queries
from src.preprocess import load_representative_views


df, root = load_queries("dev")

print("\nFULL DATASET PREPROCESSING TEST")
print("=" * 75)

for _, row in df.iterrows():

    query_id = row["query_id"]
    path = root / row["image"]

    try:
        views = load_representative_views(path)

        print(
            f"Query {query_id:>2} | "
            f"{row['image']:<20} | "
            f"{len(views)} view(s)"
        )

        for index, image in enumerate(views):
            print(
                f"          view {index}: "
                f"size={image.size}, "
                f"mode={image.mode}"
            )

    except Exception as error:

        print(
            f"Query {query_id:>2} FAILED | "
            f"{row['image']} | "
            f"{repr(error)}"
        )
