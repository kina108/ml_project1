import pandas as pd
import joblib
import json
import zipfile
from pathlib import Path


MODEL_DIR = Path("model")
MODEL_ZIP_PATH = MODEL_DIR / "price_model.zip"     
MODEL_PATH = MODEL_DIR / "price_model.joblib"       
METADATA_PATH = MODEL_DIR / "metadata.json"        


def _ensure_model_unzipped():

    if MODEL_PATH.exists():
        return 

    if MODEL_ZIP_PATH.exists():
        with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as zf:
            zf.extractall(MODEL_DIR)
    else:
        raise FileNotFoundError("Model ZIP file not found.")


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

    band = float(meta["p80_abs_error"])
    low = max(0.0, pred - band)
    high = pred + band

    return pred, low, high, band



