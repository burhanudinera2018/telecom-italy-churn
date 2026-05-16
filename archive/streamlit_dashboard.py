#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STREAMLIT DASHBOARD - TELECOM DATA ANALYSIS
Versi Bersih (Tanpa 0 Palsu)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Konfigurasi halaman
st.set_page_config(
    page_title="Telecom Data Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📱 Telecom Milan Network Traffic Analysis")
st.markdown("**Data periode: 1-3 November 2013 | Versi Bersih (Tanpa 0 Palsu)**")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('parsed_sample.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['date'] = df['datetime'].dt.date
    return df

df = load_data()

# Sidebar filters
st.sidebar.header("🔍 Filter Data")
selected_date = st.sidebar.multiselect(
    "Pilih Tanggal",
    options=sorted(df['date'].unique()),
    default=sorted(df['date'].unique())[:1]
)

selected_service = st.sidebar.multiselect(
    "Pilih Service Type",
    options=sorted(df['service_type'].unique()),
    default=sorted(df['service_type'].unique())[:3]
)

# Filter data
filtered_df = df.copy()
if selected_date:
    filtered_df = filtered_df[filtered_df['date'].isin(selected_date)]
if selected_service:
    filtered_df = filtered_df[filtered_df['service_type'].isin(selected_service)]

# ============================================================
# METRICS
# ============================================================
st.subheader("📊 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Records", f"{len(filtered_df):,}")
with col2:
    st.metric("Unique Square IDs", filtered_df['square_id'].nunique())
with col3:
    st.metric("Unique Service Types", filtered_df['service_type'].nunique())
with col4:
    st.metric("Missing Values (NaN)", filtered_df[['value_1', 'value_2', 'value_3']].isna().sum().sum())

# ============================================================
# VISUALISASI 1: Hourly Pattern
# ============================================================
st.subheader("📈 Pola Aktivitas per Jam")

fig1, ax1 = plt.subplots(figsize=(12, 5))
hourly_data = filtered_df.groupby(['hour', 'service_type'])['value_1'].mean().reset_index()

for st_type in hourly_data['service_type'].unique():
    st_data = hourly_data[hourly_data['service_type'] == st_type]
    ax1.plot(st_data['hour'], st_data['value_1'], 'o-', label=f'Service {int(st_type)}', linewidth=2)

ax1.set_xlabel('Jam')
ax1.set_ylabel('Rata-rata Value_1')
ax1.set_title('Pola Aktivitas per Jam')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
st.pyplot(fig1)

# ============================================================
# VISUALISASI 2: Top Service Types
# ============================================================
st.subheader("🏆 Top Service Types")

col1, col2 = st.columns(2)

with col1:
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    top_services = filtered_df['service_type'].value_counts().head(10)
    ax2.barh([str(int(s)) for s in top_services.index], top_services.values, color='steelblue')
    ax2.set_xlabel('Jumlah Record')
    ax2.set_title('Top 10 Service Types')
    ax2.invert_yaxis()
    st.pyplot(fig2)

with col2:
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    vc_counts = filtered_df['values_count'].value_counts().sort_index()
    ax3.bar(vc_counts.index, vc_counts.values, color='coral', edgecolor='black')
    ax3.set_xlabel('Jumlah Values per Record')
    ax3.set_ylabel('Jumlah Record')
    ax3.set_title('Distribusi Values Count')
    ax3.set_xticks(vc_counts.index)
    st.pyplot(fig3)

# ============================================================
# TABEL DATA
# ============================================================
st.subheader("📋 Sample Data")
st.dataframe(filtered_df.head(100), use_container_width=True)

# ============================================================
# DOWNLOAD BUTTON
# ============================================================
st.subheader("💾 Download Data")
csv = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Data sebagai CSV",
    data=csv,
    file_name="telecom_filtered_data.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("✅ **Data dalam kondisi BERSIH (tanpa 0 palsu)** | Analisis valid dan dapat diandalkan")