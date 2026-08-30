from pathlib import Path

from src.metadata import load_queries
from src.input_router import detect_input_type


df, root = load_queries("dev")

print("\nINPUT ROUTING RESULTS")
print("=" * 60)

for _, row in df.iterrows():
    path = root / row["image"]

    try:
        input_type = detect_input_type(path)

        print(
            f"Query {row['query_id']:>2} | "
            f"{str(row['image']):<20} | "
            f"{input_type}"
        )

    except Exception as error:
        print(
            f"Query {row['query_id']:>2} | "
            f"{str(row['image']):<20} | "
            f"FAILED: {error}"
        )
