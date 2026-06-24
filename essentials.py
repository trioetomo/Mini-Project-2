# ==========================================
# essentials.py
# ==========================================

import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import joblib
import numpy as np

# ==========================================
# FUNGSI LOAD DATA
# ==========================================
@st.cache_data
def load_data():
    """Load dataset Superstore dari GitHub"""
    try:
        df = pd.read_csv(
            "https://raw.githubusercontent.com/trioetomo/Sales_forcasting_Data/refs/heads/main/sales-forecasting.csv"
        )
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# ==========================================
# FUNGSI LOAD MODEL ANN
# ==========================================
@st.cache_resource
def load_model_ann():
    """Load model ANN, scaler, dan daftar kolom"""
    try:
        # Coba load dari folder Model/
        model = keras.models.load_model('Model/model_ann_sales.keras')
        scaler = joblib.load('Model/scaler.pkl')
        model_columns = joblib.load('Model/model_columns.pkl')
        return model, scaler, model_columns
    except:
        try:
            # Coba load dari folder yang sama
            model = keras.models.load_model('Model/model_ann_sales.keras')
            scaler = joblib.load('Model/scaler.pkl')
            model_columns = joblib.load('Model/model_columns.pkl')
            return model, scaler, model_columns
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None, None, None

# ==========================================
# FUNGSI PREDIKSI
# ==========================================
def predict_sales(input_data, model, scaler, model_columns):
    try:
        input_encoded = pd.get_dummies(input_data).astype(float)
        input_final = input_encoded.reindex(columns=model_columns, fill_value=0)
        input_scaled = scaler.transform(input_final)
        prediction = model.predict(input_scaled)
        predicted_sales = float(prediction[0][0])
        if predicted_sales < 0:
            predicted_sales = 0.0
        return predicted_sales
    except Exception as e:
        st.error(f"Error dalam prediksi: {e}")
        return None