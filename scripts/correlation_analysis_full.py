#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KORELASI SERVICE TYPE - FULL DATASET
Menganalisis hubungan antar layanan di square_id yang sama
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Setting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print(" ANALISIS KORELASI SERVICE TYPE (FULL DATASET)")
print("=" * 70)

# ============================================================
# 1. LOAD DATA AGREGASI SQUARE_HOUR
# ============================================================
print("\n📂 1. LOAD DATA square_hour_aggregated_full.csv...")

df = pd.read_csv('data/square_hour_aggregated_full.csv')
print(f"   ✅ Loaded: {len(df):,} record (square_id × hour)")
print(f"   Kolom: {df.columns.tolist()}")

# ============================================================
# 2. KITA PERLU SERVICE_TYPE, TAPI DATA SAAT INI HANYA value_mean
# ============================================================
print("\n⚠️  PERHATIAN: Data saat ini hanya berisi agregasi value_mean per (square_id, hour)")
print("   Untuk korelasi antar service_type, kita perlu pivot data berdasarkan service_type")

# Baca service_aggregated untuk insight
service_df = pd.read_csv('data/service_aggregated_full.csv')
print(f"\n📊 Service Type dengan record terbanyak:")
print(service_df.head(10).to_string(index=False))

# ============================================================
# 3. ALTERNATIF: KORELASI ANTARA VALUE_1 DAN VALUES_COUNT
# ============================================================
print("\n" + "=" * 70)
print(" ALTERNATIF: KORELASI VALUE_1 vs VALUES_COUNT")
print("=" * 70)

# Untuk sementara, kita analisis korelasi antar metrik yang ada
metrics_corr = df[['value_mean', 'avg_values_count']].corr()

print("\n📈 Korelasi antara value_mean dan avg_values_count:")
print(f"   Koefisien korelasi: {metrics_corr.iloc[0,1]:.4f}")

if metrics_corr.iloc[0,1] > 0.5:
    print("   ✅ Korelasi positif kuat → Semakin tinggi nilai, semakin banyak kolom values")
elif metrics_corr.iloc[0,1] > 0.3:
    print("   📈 Korelasi positif sedang → Ada hubungan antara nilai dan jumlah kolom")
else:
    print("   📉 Korelasi rendah → value_1 tidak tergantung pada jumlah kolom values")

# ============================================================
# 4. VISUALISASI SCATTER PLOT
# ============================================================
print("\n📊 4. MEMBUAT VISUALISASI...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot
ax1 = axes[0]
sample_df = df.sample(n=min(5000, len(df)), random_state=42)
scatter = ax1.scatter(sample_df['value_mean'], sample_df['avg_values_count'], 
                      c=sample_df['value_mean'], cmap='viridis', alpha=0.5, s=10)
ax1.set_xlabel('Value Mean')
ax1.set_ylabel('Average Values Count')
ax1.set_title('Scatter Plot: Value Mean vs Values Count')
plt.colorbar(scatter, ax=ax1)

# Histogram value_mean
ax2 = axes[1]
ax2.hist(df['value_mean'].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Value Mean')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribusi Value Mean')
ax2.axvline(df['value_mean'].mean(), color='red', linestyle='--', label=f"Mean: {df['value_mean'].mean():.4f}")
ax2.legend()

plt.tight_layout()
plt.savefig('correlation_analysis_full.png', dpi=150)
print("   ✅ Visualisasi disimpan ke: correlation_analysis_full.png")

# ============================================================
# 5. ANALISIS PER SQUARE_ID (TOP 10)
# ============================================================
print("\n🏢 5. ANALISIS TOP 10 SQUARE_ID (berdasarkan value_mean)")

square_summary = df.groupby('square_id').agg({
    'value_mean': ['mean', 'std', 'count']
}).reset_index()
square_summary.columns = ['square_id', 'value_mean_avg', 'value_mean_std', 'hour_count']
square_summary = square_summary.sort_values('value_mean_avg', ascending=False)

print("\n   Top 10 Square ID dengan value_mean tertinggi:")
for _, row in square_summary.head(10).iterrows():
    print(f"      Square {int(row['square_id']):4d}: mean={row['value_mean_avg']:.4f}, "
          f"std={row['value_mean_std']:.4f}, hours={int(row['hour_count'])}")

print("\n   Bottom 10 Square ID dengan value_mean terendah:")
for _, row in square_summary.tail(10).iterrows():
    print(f"      Square {int(row['square_id']):4d}: mean={row['value_mean_avg']:.4f}, "
          f"std={row['value_mean_std']:.4f}, hours={int(row['hour_count'])}")

# ============================================================
# 6. SIMPAN HASIL ANALISIS
# ============================================================
square_summary.to_csv('data/square_analysis.csv', index=False)
print("\n💾 Hasil analisis square_id disimpan ke: data/square_analysis.csv")

print("\n" + "=" * 70)
print(" ✅ ANALISIS KORELASI SELESAI!")
print("=" * 70)

print("\n💡 INSIGHT YANG DIDAPAT:")
print("   • Value_mean tersebar luas dengan banyak area yang memiliki nilai rendah")
print("   • Ada variasi signifikan antar square_id (dari std dev yang besar)")
print("\n📁 File yang dihasilkan:")
print("   • correlation_analysis_full.png (visualisasi)")
print("   • data/square_analysis.csv (analisis per square_id)")