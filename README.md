# HDB Resale Price Predictor

A simple machine learning app that estimates **HDB resale prices in Singapore** based on historical transaction data.

The goal of this project is to provide a **reasonable price estimate with uncertainty**, not an exact valuation. It is intended as an educational and exploratory tool.

**Live demo:** _Streamlit Community Cloud link_

---

## What it does

Given basic flat and transaction details, the app:
- Predicts an estimated resale price
- Displays an **approximate 80% price range** based on historical model error

This helps users understand both the prediction **and** its uncertainty.

---

## How to use (for users)

1. Select the flat details:
   - Town
   - Flat type and model
   - Floor area
   - Storey level
   - Remaining lease
2. Enter the transaction year and month
3. Click **Predict Price**

The app will display:
- **Estimated resale price**
- **Approximate 80% range**, showing likely variation based on past data

No login or setup is required to use the deployed app.

---

## Model overview (brief)

- Model type: **Random Forest Regressor**
- Trained on historical HDB resale transactions
- Uses both categorical and numerical features:
  - Town, flat type, flat model
  - Floor area, storey level
  - Remaining lease
  - Transaction timing

Categorical features are one-hot encoded, and the full preprocessing + model pipeline is trained end-to-end.

---

## Interpreting the price range

Alongside the point prediction, the app shows an **approximate 80% range**.

This range is derived from the model’s historical absolute errors on a held-out test set (80th percentile). It is meant to reflect typical uncertainty, not worst-case bounds.

---

## Intended scope & limitations

- Designed for **educational and exploratory use**
- Predictions are based on historical data patterns
- Does not account for:
  - Renovation quality
  - Unit-specific features (view, orientation)
  - Market shocks or policy changes

The output should not be treated as a professional valuation.

---

## Tech stack

- Python  
- pandas, NumPy  
- scikit-learn  
- Streamlit  
- joblib  

The trained model and metadata are loaded at runtime.

## Project structure

## Project structure

```text
hdb-price-predictor/
├── app.py                 - Streamlit user interface
├── predict.py             - Model loading and price prediction logic
├── train.py               - Data cleaning, training, and evaluation
├── requirements.txt
│
├── model/
│   ├── price_model.joblib # Trained model pipeline
│   ├── price_model.zip    -Zipped model for deployment
│   └── metadata.json      -MAE and error band information
│
└── data/
    └── resale_raw.csv     - Raw HDB resale dataset used for training


## Running Locally
1. pip install -r requirements.txt
2. python train.py
3. python -m streamlit run app.py


