#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLUSTERING SQUARE_ID BERDASARKAN POLA PENGGUNAAN
Mengelompokkan area yang memiliki perilaku serupa
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('parsed_sample.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour

print("=" * 70)
print(" CLUSTERING SQUARE_ID BERDASARKAN POLA PENGGUNAAN")
print("=" * 70)

# ============================================================
# 1. AGREGASI DATA PER SQUARE_ID
# ============================================================
print("\n📊 1. AGREGASI DATA PER SQUARE_ID...")

# Fitur untuk clustering
square_features = df.groupby('square_id').agg({
    'value_1': ['mean', 'std', 'count'],
    'service_type': lambda x: x.nunique(),
    'hour': lambda x: x.mode()[0] if len(x.mode()) > 0 else 0  # jam tersibuk
}).reset_index()

# Flatten column names
square_features.columns = ['square_id', 'value_mean', 'value_std', 'record_count', 
                           'unique_services', 'peak_hour']

print(f"   Data shape: {square_features.shape}")
print(f"   Jumlah unique square_id: {len(square_features)}")

# ============================================================
# 2. SCALING FITUR
# ============================================================
print("\n📐 2. SCALING FITUR...")

feature_cols = ['value_mean', 'value_std', 'record_count', 'unique_services', 'peak_hour']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(square_features[feature_cols])

# ============================================================
# 3. MENENTUKAN JUMLAH CLUSTER (Elbow Method)
# ============================================================
print("\n🔍 3. MENENTUKAN JUMLAH CLUSTER...")

inertias = []
K_range = range(2, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range, inertias, 'bo-')
ax.set_xlabel('Jumlah Cluster (k)')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method untuk Optimal k')
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=150)
print("   ✅ Elbow plot disimpan ke: elbow_method.png")

# Pilih k (misal 4 cluster)
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
square_features['cluster'] = kmeans.fit_predict(X_scaled)

print(f"\n   ✅ Menggunakan {k} cluster")

# ============================================================
# 4. ANALISIS KARAKTERISTIK CLUSTER
# ============================================================
print("\n📊 4. KARAKTERISTIK SETIAP CLUSTER:")

cluster_summary = square_features.groupby('cluster')[feature_cols].mean()
for cluster_id in range(k):
    print(f"\n   🔹 Cluster {cluster_id}:")
    cluster_data = cluster_summary.loc[cluster_id]
    print(f"      Rata-rata value: {cluster_data['value_mean']:.4f}")
    print(f"      Rata-rata record: {cluster_data['record_count']:.0f}")
    print(f"      Peak hour dominan: {cluster_data['peak_hour']:.0f}:00")

# ============================================================
# 5. VISUALISASI CLUSTER
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter plot: value_mean vs record_count
ax1 = axes[0]
scatter = ax1.scatter(square_features['value_mean'], square_features['record_count'], 
                      c=square_features['cluster'], cmap='viridis', s=50, alpha=0.7)
ax1.set_xlabel('Rata-rata Value_1')
ax1.set_ylabel('Jumlah Record')
ax1.set_title('Cluster: Value Mean vs Record Count')
plt.colorbar(scatter, ax=ax1)

# Scatter plot: value_mean vs peak_hour
ax2 = axes[1]
scatter2 = ax2.scatter(square_features['value_mean'], square_features['peak_hour'], 
                       c=square_features['cluster'], cmap='plasma', s=50, alpha=0.7)
ax2.set_xlabel('Rata-rata Value_1')
ax2.set_ylabel('Peak Hour')
ax2.set_title('Cluster: Value Mean vs Peak Hour')
plt.colorbar(scatter2, ax=ax2)

plt.tight_layout()
plt.savefig('clustering_results.png', dpi=150)
print("\n   ✅ Visualisasi cluster disimpan ke: clustering_results.png")

# ============================================================
# 6. SIMPAN HASIL CLUSTERING
# ============================================================
square_features.to_csv('square_clusters.csv', index=False)
print("\n   ✅ Hasil clustering disimpan ke: square_clusters.csv")

print("\n" + "=" * 70)
print(" ✅ CLUSTERING ANALYSIS SELESAI!")
print("=" * 70)