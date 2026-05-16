#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AGREGASI FULL DATASET TELECOM (2.5 GB)
Membaca telecom_full_dataset_compact.csv dan mengagregasi untuk analisis lanjutan
"""

import pandas as pd
import numpy as np
import time

start_time = time.time()

print("=" * 70)
print(" AGREGASI FULL DATASET TELECOM (Versi 2.5 GB)")
print("=" * 70)

# ============================================================
# 1. BACA DATA (hanya kolom yang diperlukan)
# ============================================================
print("\n📂 1. MEMBACA DATA...")
print("   (Ini akan memakan waktu 1-2 menit karena file 2.5 GB)")

# Baca file compact
df = pd.read_csv('telecom_full_dataset_compact.csv')
print(f"   ✅ Loaded: {len(df):,} record")
print(f"   Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

# Konversi datetime
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['date'] = df['datetime'].dt.date

print(f"   Periode: {df['datetime'].min()} → {df['datetime'].max()}")
print(f"   Unique square_id: {df['square_id'].nunique():,}")
print(f"   Unique service_type: {df['service_type'].nunique():,}")

# ============================================================
# 2. AGREGASI PER JAM (Hourly Aggregation)
# ============================================================
print("\n📊 2. AGREGASI PER JAM...")

hourly_agg = df.groupby('hour').agg({
    'value_1': ['mean', 'count'],
    'values_count': 'mean'
}).reset_index()

hourly_agg.columns = ['hour', 'value_mean', 'record_count', 'avg_values_count']
hourly_agg.to_csv('data/hourly_aggregated_full.csv', index=False)
print(f"   ✅ {len(hourly_agg)} jam → disimpan ke data/hourly_aggregated_full.csv")
print(f"\n   Preview:")
print(hourly_agg.head())

# ============================================================
# 3. AGREGASI PER SQUARE_ID (Square Aggregation)
# ============================================================
print("\n🏢 3. AGREGASI PER SQUARE_ID...")
print("   (Ini akan memakan waktu ~30 detik)")

square_agg = df.groupby('square_id').agg({
    'value_1': ['mean', 'median', 'count', 'sum'],
    'values_count': 'mean'
}).reset_index()

square_agg.columns = ['square_id', 'value_mean', 'value_median', 'record_count', 'value_sum', 'avg_values_count']
square_agg.to_csv('data/square_aggregated_full.csv', index=False)
print(f"   ✅ {len(square_agg):,} square_id → disimpan ke data/square_aggregated_full.csv")

# ============================================================
# 4. AGREGASI PER SERVICE TYPE
# ============================================================
print("\n📱 4. AGREGASI PER SERVICE TYPE...")

service_agg = df.groupby('service_type').agg({
    'value_1': ['mean', 'median', 'count', 'sum'],
    'values_count': 'mean'
}).reset_index()

service_agg.columns = ['service_type', 'value_mean', 'value_median', 'record_count', 'value_sum', 'avg_values_count']
service_agg = service_agg.sort_values('record_count', ascending=False)
service_agg.to_csv('data/service_aggregated_full.csv', index=False)
print(f"   ✅ {len(service_agg):,} service_type → disimpan ke data/service_aggregated_full.csv")
print(f"\n   Top 10 Service Types:")
for _, row in service_agg.head(10).iterrows():
    print(f"      Service {int(row['service_type'])}: {row['record_count']:,} record, mean={row['value_mean']:.4f}")

# ============================================================
# 5. AGREGASI PER (SQUARE_ID, HOUR) untuk Analisis Lanjutan
# ============================================================
print("\n🗺️ 5. AGREGASI PER (SQUARE_ID, HOUR)...")

square_hour_agg = df.groupby(['square_id', 'hour']).agg({
    'value_1': 'mean',
    'values_count': 'mean'
}).reset_index()
square_hour_agg.columns = ['square_id', 'hour', 'value_mean', 'avg_values_count']
square_hour_agg.to_csv('data/square_hour_aggregated_full.csv', index=False)
print(f"   ✅ {len(square_hour_agg):,} kombinasi → disimpan ke data/square_hour_aggregated_full.csv")

# ============================================================
# 6. RINGKASAN
# ============================================================
elapsed = time.time() - start_time

print("\n" + "=" * 70)
print(" ✅ AGREGASI SELESAI!")
print("=" * 70)
print(f"\n⏱️  Waktu total: {elapsed:.2f} detik")
print("\n📁 FILE YANG DIHASILKAN:")
print(f"   • data/hourly_aggregated_full.csv        → {len(hourly_agg):,} baris")
print(f"   • data/square_aggregated_full.csv       → {len(square_agg):,} baris")
print(f"   • data/service_aggregated_full.csv      → {len(service_agg):,} baris")
print(f"   • data/square_hour_aggregated_full.csv  → {len(square_hour_agg):,} baris")

print("\n💡 LANGKAH SELANJUTNYA:")
print("   1. correlation_analysis_full.py (analisis korelasi)")
print("   2. clustering_analysis_full.py (clustering square_id)")
print("   3. traffic_prediction_full.py (prediksi traffic)")

# ============================================================
# 7. SIMPAN INFO DATASET
# ============================================================
info = {
    'total_records': len(df),
    'unique_square_id': df['square_id'].nunique(),
    'unique_service_type': df['service_type'].nunique(),
    'date_min': str(df['datetime'].min()),
    'date_max': str(df['datetime'].max()),
    'value_1_zero_count': (df['value_1'] == 0).sum(),
    'value_1_nan_count': df['value_1'].isna().sum()
}

info_df = pd.DataFrame([info])
info_df.to_csv('data/dataset_info.csv', index=False)
print(f"\n📊 Info dataset disimpan ke: data/dataset_info.csv")