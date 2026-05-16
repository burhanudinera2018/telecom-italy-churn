#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLUSTERING SQUARE_ID - FULL DATASET
Mengelompokkan area grid berdasarkan pola penggunaan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

print("=" * 70)
print(" CLUSTERING SQUARE_ID (FULL DATASET)")
print("=" * 70)

# ============================================================
# 1. LOAD DATA SQUARE HOUR
# ============================================================
print("\n📂 1. LOAD DATA square_hour_aggregated_full.csv...")

df = pd.read_csv('data/square_hour_aggregated_full.csv')
print(f"   ✅ Loaded: {len(df):,} record")

# ============================================================
# 2. BUAT FITUR PER SQUARE_ID
# ============================================================
print("\n🔧 2. MEMBUAT FITUR UNTUK CLUSTERING...")

# Agregasi per square_id
square_features = df.groupby('square_id').agg({
    'value_mean': ['mean', 'std', 'min', 'max', 'count'],
    'avg_values_count': 'mean'
}).reset_index()

# Flatten column names
square_features.columns = [
    'square_id', 
    'value_mean', 'value_std', 'value_min', 'value_max', 'hour_count',
    'avg_values_count'
]

# Hitung range (max - min) sebagai fitur tambahan
square_features['value_range'] = square_features['value_max'] - square_features['value_min']

# Hitung coefficient of variation (std/mean) untuk stabilitas
square_features['value_cv'] = square_features['value_std'] / square_features['value_mean'].replace(0, np.nan)

print(f"   Fitur yang tersedia: {square_features.columns.tolist()}")
print(f"   Total square_id: {len(square_features):,}")

# Bersihkan NaN
square_features = square_features.fillna(0)

# ============================================================
# 3. SCALING FITUR
# ============================================================
print("\n📐 3. SCALING FITUR...")

feature_cols = ['value_mean', 'value_std', 'value_range', 'avg_values_count', 'hour_count']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(square_features[feature_cols])

print(f"   Fitur yang digunakan: {feature_cols}")

# ============================================================
# 4. MENENTUKAN JUMLAH CLUSTER (Elbow Method)
# ============================================================
print("\n🔍 4. MENENTUKAN JUMLAH CLUSTER OPTIMAL...")

inertias = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    print(f"   k={k}: inertia={kmeans.inertia_:.0f}")

# Plot elbow
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
ax.set_xlabel('Jumlah Cluster (k)')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method untuk Menentukan k Optimal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elbow_method_full.png', dpi=150)
print("\n   ✅ Elbow plot disimpan ke: elbow_method_full.png")

# Pilih k (berdasarkan elbow, biasanya k=4 atau k=5)
k = 5
print(f"\n   📌 Memilih k = {k} clusters")

# ============================================================
# 5. LAKUKAN CLUSTERING
# ============================================================
print("\n🤖 5. MELAKUKAN CLUSTERING...")

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
square_features['cluster'] = kmeans.fit_predict(X_scaled)

print(f"   ✅ {len(square_features)} square_id dikelompokkan ke {k} cluster")

# ============================================================
# 6. ANALISIS KARAKTERISTIK CLUSTER
# ============================================================
print("\n📊 6. KARAKTERISTIK SETIAP CLUSTER:")

cluster_summary = square_features.groupby('cluster')[feature_cols].mean()
for cluster_id in range(k):
    print(f"\n   🔹 CLUSTER {cluster_id}: ({len(square_features[square_features['cluster']==cluster_id]):,} square_id)")
    row = cluster_summary.loc[cluster_id]
    print(f"      • Rata-rata value: {row['value_mean']:.4f}")
    print(f"      • Standar deviasi: {row['value_std']:.4f}")
    print(f"      • Range value: {row['value_range']:.4f}")
    print(f"      • Avg values count: {row['avg_values_count']:.2f}")
    print(f"      • Jumlah jam terekam: {row['hour_count']:.0f}")

# ============================================================
# 7. VISUALISASI CLUSTER
# ============================================================
print("\n📊 7. MEMBUAT VISUALISASI CLUSTER...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Value Mean vs Value Std
ax1 = axes[0, 0]
scatter1 = ax1.scatter(square_features['value_mean'], square_features['value_std'], 
                       c=square_features['cluster'], cmap='viridis', alpha=0.6, s=20)
ax1.set_xlabel('Value Mean')
ax1.set_ylabel('Value Std')
ax1.set_title('Cluster: Value Mean vs Standard Deviation')
plt.colorbar(scatter1, ax=ax1)

# Plot 2: Value Mean vs Hour Count
ax2 = axes[0, 1]
scatter2 = ax2.scatter(square_features['value_mean'], square_features['hour_count'], 
                       c=square_features['cluster'], cmap='plasma', alpha=0.6, s=20)
ax2.set_xlabel('Value Mean')
ax2.set_ylabel('Hour Count')
ax2.set_title('Cluster: Value Mean vs Hours Recorded')
plt.colorbar(scatter2, ax=ax2)

# Plot 3: Distribution of each cluster (boxplot)
ax3 = axes[1, 0]
cluster_data = [square_features[square_features['cluster'] == c]['value_mean'].values for c in range(k)]
bp = ax3.boxplot(cluster_data, labels=[f'Cluster {c}' for c in range(k)], patch_artist=True)
for patch, color in zip(bp['boxes'], sns.color_palette("husl", k)):
    patch.set_facecolor(color)
ax3.set_xlabel('Cluster')
ax3.set_ylabel('Value Mean')
ax3.set_title('Distribusi Value Mean per Cluster')

# Plot 4: Cluster size pie chart
ax4 = axes[1, 1]
cluster_sizes = square_features['cluster'].value_counts().sort_index()
ax4.pie(cluster_sizes, labels=[f'Cluster {c}' for c in cluster_sizes.index], 
        autopct='%1.1f%%', startangle=90)
ax4.set_title('Proporsi Ukuran Cluster')

plt.tight_layout()
plt.savefig('clustering_full.png', dpi=150)
print("   ✅ Visualisasi disimpan ke: clustering_full.png")

# ============================================================
# 8. SIMPAN HASIL CLUSTERING
# ============================================================
square_features.to_csv('data/square_clusters_full.csv', index=False)
print("\n💾 Hasil clustering disimpan ke: data/square_clusters_full.csv")

print("\n" + "=" * 70)
print(" ✅ CLUSTERING SELESAI!")
print("=" * 70)

print("\n💡 INSIGHT DARI CLUSTERING:")
print("   • Cluster dengan value_mean tinggi → area padat aktivitas")
print("   • Cluster dengan value_std tinggi → area dengan variasi aktivitas besar")
print("   • Cluster dengan hour_count sedikit → area yang kurang terpantau")