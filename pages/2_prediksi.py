import streamlit as st
import pandas as pd
import numpy as np
from essentials import load_data, load_model_ann, predict_sales

import sys
import os

# Tambahkan path parent folder agar bisa import essentials


# ==========================================
# SETTING HALAMAN
# ==========================================
st.set_page_config(
    page_title="Prediksi Sales",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Prediksi Penjualan Superstore")
st.markdown("**Masukkan data transaksi untuk memprediksi nilai sales**")

# ==========================================
# LOAD DATA & MODEL
# ==========================================
df = load_data()
model, scaler, model_columns = load_model_ann()

# ==========================================
# FORM INPUT
# ==========================================
st.subheader("📋 Input Data Transaksi")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        ship_mode = st.selectbox(
            "Ship Mode",
            ['Standard Class', 'Second Class', 'First Class', 'Same Day']
        )
        
        segment = st.selectbox(
            "Segment Pelanggan",
            ['Consumer', 'Corporate', 'Home Office']
        )
        
        region = st.selectbox(
            "Region",
            ['West', 'East', 'Central', 'South']
        )
    
    with col2:
        category = st.selectbox(
            "Kategori Produk",
            ['Office Supplies', 'Furniture', 'Technology']
        )
        
        sub_category = st.selectbox(
            "Sub-Kategori",
            ['Paper', 'Binders', 'Art', 'Storage', 'Phones', 'Furnishings',
             'Accessories', 'Labels', 'Chairs', 'Appliances', 'Fasteners',
             'Envelopes', 'Bookcases', 'Tables', 'Supplies', 'Machines', 'Copiers']
        )
    
    # Tombol submit
    submitted = st.form_submit_button("🚀 Prediksi Sales", use_container_width=True)

# ==========================================
# PROSES PREDIKSI
# ==========================================
if submitted:
    if model is None:
        st.error("❌ Model tidak tersedia. Pastikan file model ada di folder Model/")
    else:
        with st.spinner("⏳ Memproses prediksi..."):
            # Buat DataFrame input
            input_data = pd.DataFrame([{
                'Ship Mode': ship_mode,
                'Segment': segment,
                'Region': region,
                'Category': category,
                'Sub-Category': sub_category
            }])
            
            # Prediksi
            result = predict_sales(input_data, model, scaler, model_columns)
            
            if result is not None:
                predicted_sales = result
                
                # Tampilkan hasil
                st.success("✅ Prediksi berhasil!")
                
                # Tampilkan hasil besar
                st.metric(
                    label="💰 Estimasi Nilai Sales",
                    value=f"${predicted_sales:,.2f}",
                    help="Nilai prediksi berdasarkan input yang diberikan"
                )
                
                # Tampilkan detail input
                with st.expander("📝 Detail Input"):
                    st.json({
                        "Ship Mode": ship_mode,
                        "Segment": segment,
                        "Region": region,
                        "Category": category,
                        "Sub-Category": sub_category
                    })
                
                # Tampilkan informasi tambahan
                st.info("""
                    💡 **Catatan:**
                    - Prediksi berdasarkan model ANN yang dilatih dengan data Superstore
                    - Hasil prediksi ini adalah estimasi, bukan nilai pasti
                """)

# ==========================================
# RIWAYAT PREDIKSI (Opsional)
# ==========================================
st.subheader("📜 Riwayat Prediksi")

if 'history' not in st.session_state:
    st.session_state.history = []

if submitted and result is not None:
    st.session_state.history.append({
        'Input': {
            'Ship Mode': ship_mode,
            'Segment': segment,
            'Region': region,
            'Category': category,
            'Sub-Category': sub_category
        },
        'Predicted Sales': predicted_sales
    })

if st.session_state.history:
    history_df = pd.DataFrame([
        {
            'Ship Mode': h['Input']['Ship Mode'],
            'Segment': h['Input']['Segment'],
            'Region': h['Input']['Region'],
            'Category': h['Input']['Category'],
            'Sub-Category': h['Input']['Sub-Category'],
            'Predicted Sales': f"${h['Predicted Sales']:,.2f}"
        }
        for h in st.session_state.history
    ])
    st.dataframe(history_df, use_container_width=True)
    
    if st.button("🗑️ Hapus Riwayat"):
        st.session_state.history = []
        st.rerun()
else:
    st.caption("Belum ada riwayat prediksi")

st.markdown("---")
st.caption("Dibuat dengan Streamlit | Mini Project 2")