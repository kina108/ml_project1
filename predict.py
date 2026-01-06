import pandas as pd
import joblib
import json
import zipfile
from pathlib import Path

# Paths
MODEL_DIR = Path("model")
MODEL_ZIP_PATH = MODEL_DIR / "price_model.zip"           # contains price_model.joblib
MODEL_PATH = MODEL_DIR / "price_model.joblib"      # extracted model file
METADATA_PATH = MODEL_DIR / "metadata.json"        # stays normal


def _ensure_model_unzipped():
    """
    Ensures price_model.joblib exists.
    If missing, extract it from model.zip.
    """
    if MODEL_PATH.exists():
        return  # already extracted

    if MODEL_ZIP_PATH.exists():
        with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as zf:
            zf.extractall(MODEL_DIR)
    else:
        raise FileNotFoundError(
            "Model file missing. Expected either:\n"
            f"- {MODEL_PATH} (after unzip), or\n"
            f"- {MODEL_ZIP_PATH} (zip file containing the model)"
        )


def load_model():
    _ensure_model_unzipped()
    return joblib.load(MODEL_PATH)


def load_metadata():
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def predict_price_with_range(input_data: dict):
    model = load_model()
    meta = load_metadata()

    df = pd.DataFrame([input_data])
    pred = float(model.predict(df)[0])

    band = float(meta["p80_abs_error"])  # prediction range band
    low = max(0.0, pred - band)
    high = pred + band

    return pred, low, high, band


