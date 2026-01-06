import pandas as pd
import numpy as np
import joblib
import re
import json
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = "data/resale_raw.csv"

MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "price_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


def parse_remaining_lease(lease):
    if pd.isna(lease):
        return np.nan

    years = 0
    months = 0

    year_match = re.search(r"(\d+)\s+years?", str(lease))
    month_match = re.search(r"(\d+)\s+months?", str(lease))

    if year_match:
        years = int(year_match.group(1))
    if month_match:
        months = int(month_match.group(1))

    return years + months / 12.0


def clean_data(df):
    # Convert month to datetime, then extract numeric time features
    df["month"] = pd.to_datetime(df["month"])
    df["year"] = df["month"].dt.year
    df["month_num"] = df["month"].dt.month

    # Feature engineering
    df["remaining_lease_years"] = df["remaining_lease"].apply(parse_remaining_lease)
    df["storey_mid"] = df["storey_range"].apply(
        lambda x: np.mean([int(i) for i in str(x).split(" TO ")])
    )

    # Drop columns we don't want to model on
    df = df.drop(
        columns=[
            "month",
            "remaining_lease",
            "storey_range",
            "block",
            "street_name",
        ],
        errors="ignore",
    )

    # Drop missing rows (simple v1)
    df = df.dropna()

    return df


def split_data_timebased(df):
    # Sort chronologically so test is "future" relative to train
    df = df.sort_values(by=["year", "month_num"])

    X = df.drop("resale_price", axis=1)
    y = df["resale_price"]

    split_idx = int(len(df) * 0.8)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def build_pipeline(X):
    categorical_features = X.select_dtypes(include="object").columns
    numerical_features = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numerical_features),
        ]
    )

    # Keep it reasonable (faster than huge forests, still strong baseline)
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=18,
        min_samples_leaf=3,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def main():
    print("Loading data...")
    df_raw = pd.read_csv(DATA_PATH)

    # Track dataset month range (for honesty + README)
    month_min = str(df_raw["month"].min())
    month_max = str(df_raw["month"].max())

    print("Cleaning data...")
    df = clean_data(df_raw)

    print("Splitting train/test (time-based)...")
    X_train, X_test, y_train, y_test = split_data_timebased(df)

    print("Building pipeline...")
    pipeline = build_pipeline(X_train)

    print("Training model...")
    pipeline.fit(X_train, y_train)

    print("Evaluating model...")
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    abs_err = np.abs(y_test.to_numpy() - preds)

    # 80% prediction range based on absolute error distribution on test set
    p80_abs_error = float(np.quantile(abs_err, 0.80))

    print(f"Test MAE: ${mae:,.0f}")
    print(f"80% error band (p80 abs error): ±${p80_abs_error:,.0f}")

    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(pipeline, MODEL_PATH)
    print("Model saved to:", MODEL_PATH)

    metadata = {
        "mae": float(mae),
        "p80_abs_error": p80_abs_error,
        "data_month_min": month_min,
        "data_month_max": month_max,
    }

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Metadata saved to:", METADATA_PATH)


if __name__ == "__main__":
    main()
