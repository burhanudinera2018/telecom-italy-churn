#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
STREAMLIT DASHBOARD - HASIL CLUSTERING TELECOM MILAN
Dashboard interaktif untuk eksplorasi hasil clustering
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Konfigurasi halaman
st.set_page_config(
    page_title="Telecom Milan Cluster Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📱 Telecom Milan Network Traffic Analysis")
st.markdown("**Dashboard Interaktif Hasil Clustering - Data 1-3 November 2013**")

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    # Load hasil clustering
    clusters = pd.read_csv('data/square_clusters_full.csv')
    
    # Load hourly data
    hourly = pd.read_csv('data/hourly_aggregated_full.csv')
    
    # Load square_hour untuk time series
    square_hour = pd.read_csv('data/square_hour_aggregated_full.csv')
    
    return clusters, hourly, square_hour

clusters, hourly, square_hour = load_data()

# Sidebar
st.sidebar.header("🔍 Filter")

# Filter cluster
selected_clusters = st.sidebar.multiselect(
    "Pilih Cluster",
    options=sorted(clusters['cluster'].unique()),
    default=sorted(clusters['cluster'].unique())
)

# Filter berdasarkan value_mean
value_min, value_max = st.sidebar.slider(
    "Rentang Value Mean",
    min_value=float(clusters['value_mean'].min()),
    max_value=float(clusters['value_mean'].max()),
    value=(float(clusters['value_mean'].min()), float(clusters['value_mean'].max()))
)

# Filter data
filtered_clusters = clusters[
    (clusters['cluster'].isin(selected_clusters)) &
    (clusters['value_mean'] >= value_min) &
    (clusters['value_mean'] <= value_max)
]

# ============================================================
# METRICS
# ============================================================
st.subheader("📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Square ID", f"{len(clusters):,}")
with col2:
    st.metric("Total Clusters", clusters['cluster'].nunique())
with col3:
    st.metric("Rata-rata Value Mean", f"{clusters['value_mean'].mean():.4f}")
with col4:
    st.metric("Value Mean Maks", f"{clusters['value_mean'].max():.4f}")

# ============================================================
# VISUALISASI 1: Distribusi Cluster
# ============================================================
st.subheader("📊 Distribusi Cluster")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    cluster_sizes = filtered_clusters['cluster'].value_counts().sort_index()
    ax1.pie(cluster_sizes, labels=[f'Cluster {c}' for c in cluster_sizes.index], 
            autopct='%1.1f%%', startangle=90)
    ax1.set_title('Proporsi Cluster (Setelah Filter)')
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    cluster_stats = filtered_clusters.groupby('cluster')['value_mean'].mean().sort_index()
    ax2.bar([f'Cluster {c}' for c in cluster_stats.index], cluster_stats.values, color='steelblue')
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel('Rata-rata Value Mean')
    ax2.set_title('Rata-rata Value Mean per Cluster')
    st.pyplot(fig2)

# ============================================================
# VISUALISASI 2: Scatter Plot Cluster
# ============================================================
st.subheader("🎯 Visualisasi Cluster (Value Mean vs Value Std)")

fig3 = px.scatter(
    filtered_clusters,
    x='value_mean',
    y='value_std',
    color='cluster',
    hover_data=['square_id', 'hour_count', 'value_range'],
    title='Persebaran Square ID Berdasarkan Cluster',
    labels={'value_mean': 'Value Mean', 'value_std': 'Standard Deviation'},
    opacity=0.6
)
st.plotly_chart(fig3, use_container_width=True)

# ============================================================
# VISUALISASI 3: Hourly Pattern
# ============================================================
st.subheader("⏰ Pola Aktivitas per Jam")

fig4, ax4 = plt.subplots(figsize=(12, 5))
ax4.plot(hourly['hour'], hourly['value_mean'], 'bo-', linewidth=2, markersize=8)
ax4.set_xlabel('Jam')
ax4.set_ylabel('Rata-rata Value Mean')
ax4.set_title('Pola Aktivitas Harian (0-23 Jam)')
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(0, 24, 2))
st.pyplot(fig4)

# ============================================================
# VISUALISASI 4: Top Square ID
# ============================================================
st.subheader("🏆 Top 10 Square ID dengan Value Mean Tertinggi")

top_squares = filtered_clusters.nlargest(10, 'value_mean')[['square_id', 'value_mean', 'value_std', 'cluster', 'hour_count']]
st.dataframe(top_squares, use_container_width=True)

# ============================================================
# TABEL DATA
# ============================================================
st.subheader("📋 Data Hasil Clustering (Sample 100 baris)")
st.dataframe(filtered_clusters.head(100), use_container_width=True)

# ============================================================
# DOWNLOAD BUTTON
# ============================================================
st.subheader("💾 Download Data")

csv = filtered_clusters.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Data Clustering sebagai CSV",
    data=csv,
    file_name="cluster_results.csv",
    mime="text/csv"
)

st.markdown("---")
st.markdown("✅ **Dashboard ini menampilkan hasil clustering dari data Telecom Milan (1-3 November 2013)**")