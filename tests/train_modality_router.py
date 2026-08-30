from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib


DATA_PATH = Path("test_data/modality_features.csv")
MODEL_PATH = Path("src/models/modality_router.pkl")


def main():

    print("=" * 70)
    print("TRAINING MODALITY ROUTER")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    print(f"Samples: {len(df)}")
    print()
    print(df["modality"].value_counts())

    # Remove non-feature columns
    feature_columns = [
        "mean",
        "std",
        "dark",
        "bright",
        "colored",
        "red",
        "green",
        "blue",
        "horizontal_var",
        "vertical_var",
        "texture_v",
        "texture_h",
    ]

    # Keep only columns that actually exist
    feature_columns = [
        col for col in feature_columns
        if col in df.columns
    ]

    print()
    print("Features:")
    for col in feature_columns:
        print(f" - {col}")

    X = df[feature_columns]
    y = df["modality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
    )

    print()
    print("Training...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print()
    print("=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(
        classification_report(
            y_test,
            predictions,
            digits=4,
        )
    )

    accuracy = model.score(X_test, y_test)

    print(f"Validation accuracy: {accuracy:.4%}")

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "features": feature_columns,
        },
        MODEL_PATH,
    )

    print()
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
