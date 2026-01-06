import pandas as pd
import joblib
import json

MODEL_PATH = "model/price_model.joblib"
METADATA_PATH = "model/metadata.json"


def load_model():
    return joblib.load(MODEL_PATH)


def load_metadata():
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


def predict_price_with_range(input_data: dict):
    model = load_model()
    meta = load_metadata()

    df = pd.DataFrame([input_data])
    pred = float(model.predict(df)[0])

    band = float(meta["p80_abs_error"])  # ± band for ~80% range
    low = max(0.0, pred - band)
    high = pred + band

    return pred, low, high, band
