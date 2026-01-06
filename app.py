import streamlit as st
from predict import predict_price_with_range

st.set_page_config(
    page_title="HDB Resale Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("HDB Resale Price Predictor")
st.write("Estimate HDB resale prices using historical data (with an approximate 80% range).")

town = st.selectbox(
    "Town",
    [
        "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH",
        "BUKIT PANJANG", "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG",
        "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST",
        "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL",
        "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON", "TAMPINES",
        "TOA PAYOH", "WOODLANDS", "YISHUN"
    ]
)

flat_type = st.selectbox("Flat Type", ["2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE"])

flat_model = st.selectbox(
    "Flat Model",
    ["Improved", "New Generation", "Model A", "Standard", "Premium Apartment"]
)

floor_area_sqm = st.number_input("Floor Area (sqm)", 30.0, 200.0, 90.0)
storey_mid = st.number_input("Storey Level (approx midpoint)", 1.0, 50.0, 10.0)
remaining_lease_years = st.number_input("Remaining Lease (years)", 1.0, 99.0, 60.0)

year = st.number_input("Transaction Year", 2017, 2026, 2024)
month_num = st.number_input("Transaction Month", 1, 12, 6)

lease_commence_date = int(round(year - (99 - remaining_lease_years)))

if st.button("Predict Price"):
    input_data = {
        "town": town,
        "flat_type": flat_type,
        "flat_model": flat_model,
        "floor_area_sqm": float(floor_area_sqm),
        "storey_mid": float(storey_mid),
        "remaining_lease_years": float(remaining_lease_years),
        "lease_commence_date": int(lease_commence_date),
        "year": int(year),
        "month_num": int(month_num),
    }

    pred, low, high, band = predict_price_with_range(input_data)

    st.success(f"Estimated Resale Price: ${pred:,.0f}")
    st.markdown(
    f"""
**Approximate 80% range:**  
\\${low:,.0f} to \\${high:,.0f} (±\\${band:,.0f})
"""
)



