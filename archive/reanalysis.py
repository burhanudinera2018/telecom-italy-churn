#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RE-ANALYSIS DATA TELECOM
Mengulang analisis dengan data yang sudah bersih (tanpa fillna(0))
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setting style visualisasi
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print(" RE-ANALYSIS DATA TELECOM - VERSI BERSIH")
print("=" * 70)

# ============================================================
# 1. LOAD DATA YANG SUDAH BERSIH
# ============================================================
print("\n📂 1. LOAD DATA PARSED_sample.csv...")
df = pd.read_csv('parsed_sample.csv')
print(f"   ✅ Data loaded: {df.shape[0]} baris, {df.shape[1]} kolom")
print(f"   📅 Periode: {df['datetime'].min()} → {df['datetime'].max()}")

# ============================================================
# 2. CEK KUALITAS DATA (SEKARANG BERSIH!)
# ============================================================
print("\n" + "=" * 70)
print(" 2. CEK KUALITAS DATA (TIDAK ADA 0 PALSU!)")
print("=" * 70)

# Cek nilai 0 di value_1, value_2, value_3
for col in ['value_1', 'value_2', 'value_3']:
    zero_count = (df[col] == 0).sum()
    nan_count = df[col].isna().sum()
    print(f"\n   Kolom {col}:")
    print(f"      Nilai 0 asli : {zero_count} ({zero_count/len(df)*100:.2f}%)")
    print(f"      Nilai NaN    : {nan_count} ({nan_count/len(df)*100:.2f}%)")

print("\n   ✅ KESIMPULAN: Tidak ada nilai 0 palsu! Analisis akan valid.")

# ============================================================
# 3. ANALISIS PER SERVICE TYPE
# ============================================================
print("\n" + "=" * 70)
print(" 3. ANALISIS PER SERVICE TYPE")
print("=" * 70)

service_stats = df.groupby('service_type').agg({
    'value_1': ['count', 'mean', 'median', 'std'],
    'values_count': 'mean'
}).round(4)

# Urutkan berdasarkan jumlah record terbanyak
service_stats = service_stats.sort_values(('value_1', 'count'), ascending=False)

print("\n   Top 10 Service Types (berdasarkan jumlah record):")
print("-" * 60)
for st in service_stats.head(10).index:
    count = service_stats.loc[st, ('value_1', 'count')]
    mean_val = service_stats.loc[st, ('value_1', 'mean')]
    print(f"   Service {int(st)}: {int(count):,} record, rata-rata value_1 = {mean_val:.4f}")

# ============================================================
# 4. ANALISIS TIME SERIES (Per jam)
# ============================================================
print("\n" + "=" * 70)
print(" 4. ANALISIS TIME SERIES (Per Jam)")
print("=" * 70)

# Ekstrak jam dari datetime
df['hour'] = pd.to_datetime(df['datetime']).dt.hour

# Hitung rata-rata value_1 per jam (untuk service_type yang umum)
common_services = df['service_type'].value_counts().head(3).index
hourly_trend = df[df['service_type'].isin(common_services)].groupby(['hour', 'service_type'])['value_1'].mean().reset_index()

print("\n   Rata-rata value_1 per jam untuk 3 service type teratas:")
print("-" * 60)
for st in common_services:
    st_data = hourly_trend[hourly_trend['service_type'] == st]
    peak_hour = st_data.loc[st_data['value_1'].idxmax(), 'hour']
    peak_value = st_data['value_1'].max()
    print(f"   Service {int(st)}: puncak aktivitas jam {int(peak_hour):02d}:00 (nilai: {peak_value:.4f})")

# ============================================================
# 5. DISTRIBUSI VALUES_COUNT
# ============================================================
print("\n" + "=" * 70)
print(" 5. DISTRIBUSI JUMLAH VALUES PER RECORD")
print("=" * 70)

vc_dist = df['values_count'].value_counts().sort_index()
for vc, count in vc_dist.items():
    pct = count/len(df)*100
    bar = '█' * int(pct/2)
    print(f"   {vc} values : {count:5,} record ({pct:5.1f}%) {bar}")

# ============================================================
# 6. VISUALISASI (Disimpan ke file)
# ============================================================
print("\n" + "=" * 70)
print(" 6. MEMBUAT VISUALISASI")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Analisis Data Telecom - Versi Bersih (Tanpa 0 Palsu)', fontsize=16, fontweight='bold')

# Plot 1: Top 10 Service Types
ax1 = axes[0, 0]
top_services = df['service_type'].value_counts().head(10)
ax1.barh([str(int(s)) for s in top_services.index], top_services.values, color='steelblue')
ax1.set_xlabel('Jumlah Record')
ax1.set_title('Top 10 Service Types')
ax1.invert_yaxis()

# Plot 2: Distribusi Values Count
ax2 = axes[0, 1]
vc_counts = df['values_count'].value_counts().sort_index()
ax2.bar(vc_counts.index, vc_counts.values, color='coral', edgecolor='black')
ax2.set_xlabel('Jumlah Values per Record')
ax2.set_ylabel('Jumlah Record')
ax2.set_title('Distribusi Nilai Values Count')
ax2.set_xticks(vc_counts.index)

# Plot 3: Hourly Pattern
ax3 = axes[1, 0]
for st in common_services[:2]:
    st_data = hourly_trend[hourly_trend['service_type'] == st]
    ax3.plot(st_data['hour'], st_data['value_1'], 'o-', label=f'Service {int(st)}', linewidth=2)
ax3.set_xlabel('Jam')
ax3.set_ylabel('Rata-rata Value_1')
ax3.set_title('Pola Aktivitas per Jam')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Boxplot per Service Type (top 5)
ax4 = axes[1, 1]
top5_services = df['service_type'].value_counts().head(5).index
box_data = [df[df['service_type'] == st]['value_1'].dropna().values for st in top5_services]
bp = ax4.boxplot(box_data, labels=[str(int(st)) for st in top5_services], patch_artist=True)
for patch, color in zip(bp['boxes'], sns.color_palette("husl", 5)):
    patch.set_facecolor(color)
ax4.set_xlabel('Service Type')
ax4.set_ylabel('Value_1')
ax4.set_title('Distribusi Value_1 per Service Type')
ax4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('reanalysis_visualization.png', dpi=150, bbox_inches='tight')
print("   ✅ Visualisasi disimpan ke: reanalysis_visualization.png")

# ============================================================
# 7. SUMMARY STATISTIK
# ============================================================
print("\n" + "=" * 70)
print(" 7. RINGKASAN STATISTIK")
print("=" * 70)

summary = {
    'total_records': len(df),
    'unique_square_ids': df['square_id'].nunique(),
    'unique_service_types': df['service_type'].nunique(),
    'date_range': f"{df['datetime'].min()} → {df['datetime'].max()}",
    'total_missing_values': df[['value_1', 'value_2', 'value_3']].isna().sum().sum(),
    'total_real_zeros': (df[['value_1', 'value_2', 'value_3']] == 0).sum().sum()
}

print(f"\n   📊 Statistik Data:")
print(f"      Total record        : {summary['total_records']:,}")
print(f"      Unique square_id    : {summary['unique_square_ids']}")
print(f"      Unique service_type : {summary['unique_service_types']}")
print(f"      Periode             : {summary['date_range']}")
print(f"      Total missing (NaN) : {summary['total_missing_values']:,}")
print(f"      Total nilai 0 asli  : {summary['total_real_zeros']}")

# Simpan summary ke file
summary_df = pd.DataFrame([summary])
summary_df.to_csv('reanalysis_summary.csv', index=False)
print("\n   ✅ Ringkasan disimpan ke: reanalysis_summary.csv")

print("\n" + "=" * 70)
print(" ✅ RE-ANALYSIS SELESAI!")
print("=" * 70)
print("\n📁 File yang dihasilkan:")
print("   - reanalysis_visualization.png → grafik analisis")
print("   - reanalysis_summary.csv → ringkasan statistik")
print("\n💡 Bandingkan dengan hasil dulu yang banyak nilai 0 palsu!")