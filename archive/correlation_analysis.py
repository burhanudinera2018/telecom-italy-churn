#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KORELASI ANTAR SERVICE TYPE
Menganalisis hubungan antar layanan di square_id yang sama
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('parsed_sample.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour

print("=" * 70)
print(" ANALISIS KORELASI ANTAR SERVICE TYPE")
print("=" * 70)

# ============================================================
# 1. PIVOT DATA (square_id, hour, service_type) -> value_1
# ============================================================
print("\n📊 1. MEMBANGUN PIVOT TABLE...")

# Ambil hanya service_type yang memiliki cukup data
service_counts = df['service_type'].value_counts()
major_services = service_counts[service_counts > 1000].index.tolist()
print(f"   Service type dengan >1000 record: {major_services}")

# Pivot data
pivot_df = df[df['service_type'].isin(major_services)].pivot_table(
    index=['square_id', 'hour'],
    columns='service_type',
    values='value_1',
    aggfunc='mean'
).reset_index()

print(f"   Shape pivot table: {pivot_df.shape}")

# ============================================================
# 2. HITUNG KORELASI
# ============================================================
print("\n📈 2. MENGHITUNG KORELASI...")

# Ambil kolom service_type saja (hapus square_id, hour)
service_cols = [col for col in pivot_df.columns if col not in ['square_id', 'hour']]
corr_matrix = pivot_df[service_cols].corr()

print("\n   Matriks Korelasi (top 5):")
print(corr_matrix.head())

# ============================================================
# 3. VISUALISASI HEATMAP KORELASI
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, square=True, ax=ax)
ax.set_title('Korelasi Antar Service Type', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=150)
print("\n   ✅ Heatmap disimpan ke: correlation_heatmap.png")

# ============================================================
# 4. INTERPRETASI
# ============================================================
print("\n💡 4. INTERPRETASI KORELASI:")
print("-" * 50)

# Cari korelasi tertinggi (positif)
corr_flat = corr_matrix.unstack().sort_values(ascending=False)
corr_flat = corr_flat[corr_flat < 1]  # hapus korelasi dengan diri sendiri

print("\n   🔗 Korelasi Positif Tertinggi:")
for (s1, s2), val in corr_flat.head(5).items():
    print(f"      Service {s1} ↔ Service {s2}: {val:.3f}")

print("\n   🔗 Korelasi Negatif Tertinggi (saling menggantikan):")
for (s1, s2), val in corr_flat.tail(5).items():
    print(f"      Service {s1} ↔ Service {s2}: {val:.3f}")

print("\n" + "=" * 70)
print(" ✅ ANALISIS KORELASI SELESAI!")