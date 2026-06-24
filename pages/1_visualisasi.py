import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from essentials import load_data
import sys
import os

# Tambahkan path parent folder agar bisa import essentials
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from essentials import load_data

# ==========================================
# SETTING HALAMAN
# ==========================================
st.set_page_config(
    page_title="Visualisasi Data",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Visualisasi Data Superstore")
st.markdown("**Eksplorasi data penjualan Superstore**")

# ==========================================
# LOAD DATA
# ==========================================
df = load_data()

# ==========================================
# SIDEBAR FILTER
# ==========================================
st.sidebar.header("🔍 Filter Data")

# Pilih kategori
categories = ['Semua'] + df['Category'].unique().tolist()
selected_category = st.sidebar.selectbox("Pilih Kategori", categories)

# Filter data
if selected_category != 'Semua':
    df_filtered = df[df['Category'] == selected_category]
else:
    df_filtered = df

st.sidebar.caption(f"Menampilkan {len(df_filtered)} baris data")

# ==========================================
# VISUALISASI 1: DISTRIBUSI SALES
# ==========================================
st.subheader("📈 Distribusi Nilai Sales")

fig1, ax1 = plt.subplots(figsize=(10, 5))
sns.histplot(df_filtered['Sales'], bins=50, kde=True, color='royalblue')
ax1.set_title('Distribusi Nilai Sales')
ax1.set_xlabel('Sales')
ax1.set_ylabel('Frekuensi')
st.pyplot(fig1)

# ==========================================
# VISUALISASI 2: SALES PER KATEGORI
# ==========================================
st.subheader("📊 Rata-rata Sales per Kategori")

fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.barplot(x='Category', y='Sales', data=df_filtered, palette='Set2')
ax2.set_title('Rata-rata Sales per Kategori')
ax2.set_xlabel('Kategori')
ax2.set_ylabel('Rata-rata Sales')
st.pyplot(fig2)

# ==========================================
# VISUALISASI 3: SALES PER SEGMEN
# ==========================================
st.subheader("📊 Rata-rata Sales per Segmen Pelanggan")

fig3, ax3 = plt.subplots(figsize=(10, 5))
sns.barplot(x='Segment', y='Sales', data=df_filtered, palette='Pastel1')
ax3.set_title('Rata-rata Sales per Segmen Pelanggan')
ax3.set_xlabel('Segmen')
ax3.set_ylabel('Rata-rata Sales')
st.pyplot(fig3)

# ==========================================
# VISUALISASI 4: SALES PER REGION
# ==========================================
st.subheader("📦 Sebaran Sales per Wilayah")

fig4, ax4 = plt.subplots(figsize=(10, 5))
sns.boxplot(x='Region', y='Sales', data=df_filtered, palette='Set3')
ax4.set_title('Sebaran Sales per Wilayah')
ax4.set_xlabel('Region')
ax4.set_ylabel('Sales')
st.pyplot(fig4)

# ==========================================
# VISUALISASI 5: KORELASI (Jika ada fitur numerik)
# ==========================================
st.subheader("🔗 Korelasi Antar Fitur Numerik")

# Pilih kolom numerik
numeric_cols = df_filtered.select_dtypes(include=['float64', 'int64']).columns.tolist()

if len(numeric_cols) > 1:
    fig5, ax5 = plt.subplots(figsize=(10, 8))
    corr_matrix = df_filtered[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    ax5.set_title('Matriks Korelasi')
    st.pyplot(fig5)
else:
    st.info("Tidak ada cukup kolom numerik untuk menampilkan korelasi")

st.markdown("---")
st.caption("Dibuat dengan Streamlit | Mini Project 2")