# Streamlit front end for the SuperKart sales forecasting API
import os
import requests
import pandas as pd
import streamlit as st

# Backend URL: service name on the Docker network (override with BACKEND_URL)
BACKEND_URL = os.getenv("BACKEND_URL", "http://superkart-backend:7860").rstrip("/")
PREDICT_URL = f"{BACKEND_URL}/v1/predict"
BATCH_URL = f"{BACKEND_URL}/v1/predictbatch"

st.set_page_config(page_title="SuperKart Sales Forecaster", page_icon="🛒", layout="centered")
st.title("🛒 SuperKart Sales Forecaster")
st.caption("Predict Product_Store_Sales_Total for a product in a store.")

tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction (CSV)"])

# ------------------------------------------------------------------ single
with tab_single:
    st.subheader("Product details")
    c1, c2 = st.columns(2)
    product_weight = c1.number_input("Product weight", 1.0, 30.0, 12.66, 0.01)
    sugar = c2.selectbox("Sugar content", ["Low Sugar", "Regular", "No Sugar"])
    area = c1.number_input("Allocated display area (ratio)", 0.001, 0.400, 0.027, 0.001, format="%.3f")
    mrp = c2.number_input("Product MRP", 10.0, 400.0, 117.08, 0.01)
    pid = c1.selectbox("Product ID prefix", ["FD", "NC", "DR"],
                       help="FD = Food, NC = Non-consumable, DR = Drinks")
    ptype = c2.selectbox("Product category", ["Perishables", "Non Perishables"])

    st.subheader("Store details")
    c3, c4 = st.columns(2)
    store_size = c3.selectbox("Store size", ["Small", "Medium", "High"])
    city = c4.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"])
    store_type = c3.selectbox("Store type", ["Supermarket Type1", "Supermarket Type2",
                                             "Departmental Store", "Food Mart"])
    age = c4.number_input("Store age (years)", 0, 60, 16, 1)

    payload = {
        "Product_Weight": product_weight,
        "Product_Sugar_Content": sugar,
        "Product_Allocated_Area": area,
        "Product_MRP": mrp,
        "Store_Size": store_size,
        "Store_Location_City_Type": city,
        "Store_Type": store_type,
        "Product_Id_char": pid,
        "Store_Age_Years": int(age),
        "Product_Type_Category": ptype,
    }

    if st.button("Predict sales", type="primary"):
        try:
            r = requests.post(PREDICT_URL, json=payload, timeout=30)
            if r.status_code == 200:
                value = r.json()["Predicted_Product_Store_Sales_Total"]
                st.success(f"Predicted sales total: **{value:,.2f}**")
            else:
                st.error(f"API error {r.status_code}: {r.text}")
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach the backend at {BACKEND_URL}: {exc}")

# ------------------------------------------------------------------- batch
with tab_batch:
    st.subheader("Upload a CSV")
    st.write("Required columns: `Product_Weight`, `Product_Sugar_Content`, `Product_Allocated_Area`, "
             "`Product_MRP`, `Store_Size`, `Store_Location_City_Type`, `Store_Type`, "
             "`Product_Id_char`, `Store_Age_Years`, `Product_Type_Category`.")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview", df.head())
        if st.button("Predict for all rows"):
            try:
                r = requests.post(BATCH_URL, files={"file": df.to_csv(index=False).encode("utf-8")},
                                  timeout=120)
                if r.status_code == 200:
                    preds = r.json()
                    out = df.copy()
                    out["Predicted_Sales"] = [preds[str(i)] for i in range(len(df))]
                    st.success(f"Predicted {len(out)} rows.")
                    st.dataframe(out)
                    st.download_button("Download predictions", out.to_csv(index=False).encode("utf-8"),
                                       "superkart_predictions.csv", "text/csv")
                else:
                    st.error(f"API error {r.status_code}: {r.text}")
            except requests.exceptions.RequestException as exc:
                st.error(f"Could not reach the backend at {BACKEND_URL}: {exc}")
