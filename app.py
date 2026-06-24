import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib

st.set_page_config(page_title="Superstore Sales Predictor", layout="centered")
st.title("🛍️ Superstore Sales Predictor - ANN Model")
st.write("Aplikasi untuk memprediksi besaran nilai transaksi penjualan berdasarkan profil order.")

# 1. Load Model ANN dan Preprocessing Objects
@st.cache_resource
def load_assets():
    try:
        # LOAD MODEL TANPA COMPILE DULU
        model = keras.models.load_model('Model/model_ann_sales.h5', compile=False)
        
        # COMPILE ULANG DENGAN SETTING YANG SAMA
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        scaler = joblib.load('Model/scaler.pkl')
        model_columns = joblib.load('Model/model_columns.pkl')
        return model, scaler, model_columns
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

try:
    model, scaler, model_columns = load_assets()
    
    if model is not None:
        # 2. Membuat Form Input Pengguna
        col1, col2 = st.columns(2)
        with col1:
            segment = st.selectbox("Customer Segment", ['Consumer', 'Corporate', 'Home Office'])
            region = st.selectbox("Region", ['West', 'East', 'Central', 'South'])
        with col2:
            category = st.selectbox("Product Category", ['Office Supplies', 'Furniture', 'Technology'])
            sub_category = st.selectbox("Sub-Category", [
                'Paper', 'Binders', 'Art', 'Storage', 'Phones', 'Furnishings',
                'Accessories', 'Labels', 'Chairs', 'Appliances', 'Fasteners',
                'Envelopes', 'Bookcases', 'Tables', 'Supplies', 'Machines', 'Copiers'
            ])

        # 3. Tombol Eksekusi Prediksi
        if st.button("🚀 Hitung Estimasi Sales"):
            input_data = pd.DataFrame([{
                'Ship Mode': 'Standard Class',
                'Segment': segment,
                'Region': region,
                'Category': category,
                'Sub-Category': sub_category
            }])

            input_encoded = pd.get_dummies(input_data).astype(float)
            input_final = input_encoded.reindex(columns=model_columns, fill_value=0)

            input_scaled = scaler.transform(input_final)

            prediction = model.predict(input_scaled)
            predicted_sales = float(prediction[0][0])

            if predicted_sales < 0:
                predicted_sales = 0.0

            st.success(f"### 🎯 Hasil Estimasi Nilai Sales: **${predicted_sales:,.2f}**")
    else:
        st.error("❌ Model gagal dimuat. Periksa file model.")

except FileNotFoundError:
    st.error("⚠️ Error: Berkas komponen model ('model_ann_sales.h5', 'scaler.pkl', 'model_columns.pkl') belum lengkap di folder aplikasi. Pastikan kamu sudah mengekspornya dari notebook.")
except Exception as e:
    st.error(f"❌ Terjadi error: {e}")