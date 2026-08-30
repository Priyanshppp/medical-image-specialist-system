from pathlib import Path

from src.metadata import load_queries


df, root = load_queries("dev")

print("\nDIRECTORY INSPECTION")
print("=" * 70)

for _, row in df.iterrows():

    path = root / row["image"]

    if not path.is_dir():
        continue

    print(f"\nQuery {row['query_id']}")
    print(f"Path: {path}")

    all_items = list(path.rglob("*"))

    files = [p for p in all_items if p.is_file()]
    directories = [p for p in all_items if p.is_dir()]

    print(f"Subdirectories: {len(directories)}")
    print(f"Files: {len(files)}")

    print("\nFirst files:")

    for file_path in files[:20]:
        print(
            f"  {file_path.relative_to(path)} "
            f"| suffix: {file_path.suffix}"
        )
