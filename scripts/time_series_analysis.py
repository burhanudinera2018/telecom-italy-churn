#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TIME SERIES ANALYSIS - TELECOM MILAN (FIXED)
Analisis pola temporal per jam dan per cluster
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print(" TIME SERIES ANALYSIS - TELECOM MILAN (FIXED)")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n📂 1. LOAD DATA...")

# Load hourly data
hourly = pd.read_csv('data/hourly_aggregated_full.csv')
print(f"   Hourly data: {len(hourly)} jam")

# Load square_hour
square_hour = pd.read_csv('data/square_hour_aggregated_full.csv')
print(f"   Square-hour data: {len(square_hour):,} record")

# Load cluster hasil
clusters = pd.read_csv('data/square_clusters_full.csv')
print(f"   Cluster data: {len(clusters):,} square_id")

# ============================================================
# 2. GABUNGKAN SQUARE_HOUR DENGAN CLUSTER
# ============================================================
print("\n🔗 2. MENGGABUNGKAN DATA SQUARE_HOUR DENGAN CLUSTER...")

# Merge
square_hour_cluster = square_hour.merge(
    clusters[['square_id', 'cluster']], 
    on='square_id', 
    how='left'
)

# HANDLING NaN: isi cluster yang NaN dengan -1 (artinya unassigned)
square_hour_cluster['cluster'] = square_hour_cluster['cluster'].fillna(-1).astype(int)

# Hapus baris dengan value_mean NaN
before_count = len(square_hour_cluster)
square_hour_cluster = square_hour_cluster.dropna(subset=['value_mean'])
after_count = len(square_hour_cluster)

print(f"   Hasil merge: {before_count:,} record")
print(f"   Setelah drop NaN value_mean: {after_count:,} record")
print(f"   Unique cluster: {sorted(square_hour_cluster['cluster'].unique())}")

# ============================================================
# 3. ANALISIS PER JAM UNTUK SETIAP CLUSTER
# ============================================================
print("\n📊 3. ANALISIS POLA PER JAM PER CLUSTER...")

hourly_by_cluster = square_hour_cluster.groupby(['cluster', 'hour']).agg({
    'value_mean': 'mean',
    'avg_values_count': 'mean'
}).reset_index()

print(f"\n   Rata-rata Value Mean per Jam untuk Setiap Cluster:")

for cluster in sorted(hourly_by_cluster['cluster'].unique()):
    if cluster >= 0:  # hanya cluster valid (0-4)
        cluster_data = hourly_by_cluster[hourly_by_cluster['cluster'] == cluster]
        # Cari peak hour (pastikan tidak ada NaN)
        if not cluster_data['value_mean'].isna().all():
            max_idx = cluster_data['value_mean'].idxmax()
            peak_hour = cluster_data.loc[max_idx, 'hour']
            peak_value = cluster_data.loc[max_idx, 'value_mean']
            print(f"      Cluster {int(cluster)}: puncak jam {int(peak_hour):02d}:00 (value={peak_value:.4f})")
        else:
            print(f"      Cluster {int(cluster)}: semua nilai NaN")

# ============================================================
# 4. CEK EXISTING KOLOM UNTUK ANALISIS HARIAN
# ============================================================
print("\n📅 4. CEK KETERSEDIAAN DATA HARIAN...")

# Cek apakah ada kolom datetime di square_hour_cluster
if 'datetime' in square_hour_cluster.columns:
    print("   ✅ Kolom 'datetime' tersedia")
    square_hour_cluster['datetime'] = pd.to_datetime(square_hour_cluster['datetime'])
    square_hour_cluster['date'] = square_hour_cluster['datetime'].dt.date
    
    # Analisis per hari
    daily_by_cluster = square_hour_cluster.groupby(['cluster', 'date'])['value_mean'].mean().reset_index()
    print(f"\n   Rata-rata Value Mean per Hari per Cluster:")
    for cluster in sorted(daily_by_cluster['cluster'].unique()):
        if cluster >= 0:
            cluster_daily = daily_by_cluster[daily_by_cluster['cluster'] == cluster]
            for _, row in cluster_daily.iterrows():
                print(f"      Cluster {int(cluster)} - {row['date']}: {row['value_mean']:.4f}")
else:
    print("   ⚠️ Kolom 'datetime' tidak tersedia")
    print("   Analisis harian dilewati (tidak ada data per hari)")

# ============================================================
# 5. VISUALISASI TIME SERIES
# ============================================================
print("\n📊 5. MEMBUAT VISUALISASI TIME SERIES...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Time Series Analysis - Telecom Milan', fontsize=14, fontweight='bold')

# Plot 1: Hourly pattern (global)
ax1 = axes[0, 0]
ax1.plot(hourly['hour'], hourly['value_mean'], 'bo-', linewidth=2, markersize=6)
ax1.set_xlabel('Jam')
ax1.set_ylabel('Value Mean (Global)')
ax1.set_title('Pola Aktivitas Global per Jam')
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(0, 24, 2))

# Plot 2: Hourly pattern per cluster (hanya cluster 0-4)
ax2 = axes[0, 1]
valid_clusters = [c for c in sorted(hourly_by_cluster['cluster'].unique()) if c >= 0 and c <= 4]
for cluster in valid_clusters:
    data = hourly_by_cluster[hourly_by_cluster['cluster'] == cluster]
    if not data.empty and not data['value_mean'].isna().all():
        ax2.plot(data['hour'], data['value_mean'], 'o-', linewidth=2, label=f'Cluster {int(cluster)}')
ax2.set_xlabel('Jam')
ax2.set_ylabel('Value Mean')
ax2.set_title('Pola Aktivitas per Jam per Cluster')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(0, 24, 2))

# Plot 3: Heatmap hour vs cluster
ax3 = axes[1, 0]
pivot_heatmap = hourly_by_cluster[hourly_by_cluster['cluster'].isin(valid_clusters)].pivot(
    index='cluster', columns='hour', values='value_mean'
)
if not pivot_heatmap.empty:
    sns.heatmap(pivot_heatmap, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax3)
    ax3.set_title('Heatmap: Value Mean (Cluster vs Hour)')
else:
    ax3.text(0.5, 0.5, 'Tidak ada data untuk heatmap', ha='center', va='center')
    ax3.set_title('Heatmap: No Data')

# Plot 4: Distribution of value_mean per cluster
ax4 = axes[1, 1]
cluster_data_box = []
cluster_labels = []
for cluster in valid_clusters:
    data = square_hour_cluster[square_hour_cluster['cluster'] == cluster]['value_mean'].dropna().values
    if len(data) > 0:
        cluster_data_box.append(data)
        cluster_labels.append(f'Cluster {cluster}')

if cluster_data_box:
    ax4.boxplot(cluster_data_box, labels=cluster_labels)
    ax4.set_xlabel('Cluster')
    ax4.set_ylabel('Value Mean')
    ax4.set_title('Distribusi Value Mean per Cluster')
    ax4.tick_params(axis='x', rotation=45)
else:
    ax4.text(0.5, 0.5, 'Tidak ada data untuk boxplot', ha='center', va='center')

plt.tight_layout()
plt.savefig('time_series_analysis.png', dpi=150)
print("   ✅ Visualisasi disimpan ke: time_series_analysis.png")

# ============================================================
# 6. IDENTIFIKASI PEAK DAN OFF-PEAK HOUR
# ============================================================
print("\n⏰ 6. IDENTIFIKASI PEAK DAN OFF-PEAK HOUR...")

# Peak hour global
if not hourly['value_mean'].isna().all():
    peak_hour_global = hourly.loc[hourly['value_mean'].idxmax(), 'hour']
    peak_value_global = hourly['value_mean'].max()
    print(f"\n   🌍 GLOBAL:")
    print(f"      Peak hour: {int(peak_hour_global):02d}:00 (value={peak_value_global:.4f})")
    
    off_peak_hour_global = hourly.loc[hourly['value_mean'].idxmin(), 'hour']
    off_peak_value_global = hourly['value_mean'].min()
    print(f"      Off-peak hour: {int(off_peak_hour_global):02d}:00 (value={off_peak_value_global:.4f})")

# Peak hour per cluster
print(f"\n   📍 PER CLUSTER:")
for cluster in valid_clusters:
    data = hourly_by_cluster[hourly_by_cluster['cluster'] == cluster]
    if not data.empty and not data['value_mean'].isna().all():
        max_idx = data['value_mean'].idxmax()
        peak_hour = data.loc[max_idx, 'hour']
        peak_value = data.loc[max_idx, 'value_mean']
        min_idx = data['value_mean'].idxmin()
        off_hour = data.loc[min_idx, 'hour']
        print(f"      Cluster {int(cluster)}: peak jam {int(peak_hour):02d}:00 ({peak_value:.4f}) | "
              f"off-peak jam {int(off_hour):02d}:00")

# ============================================================
# 7. SIMPAN HASIL
# ============================================================
hourly_by_cluster.to_csv('data/hourly_by_cluster.csv', index=False)
print("\n💾 Hasil time series disimpan ke: data/hourly_by_cluster.csv")

print("\n" + "=" * 70)
print(" ✅ TIME SERIES ANALYSIS SELESAI!")
print("=" * 70)

print("\n💡 INSIGHT TIME SERIES:")
print("   • Pola aktivitas global menunjukkan puncak di jam-jam tertentu")
print("   • Setiap cluster memiliki pola jam sibuk yang berbeda")
print("   • Cluster dengan aktivitas tinggi cenderung memiliki peak hour berbeda")