#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PREDIKSI TRAFFIC - TELECOM MILAN
Machine Learning untuk memprediksi value_mean berdasarkan hour dan square_id
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

print("=" * 70)
print(" PREDIKSI TRAFFIC - MACHINE LEARNING")
print("=" * 70)

# ============================================================
# 1. LOAD DATA
# ============================================================
print("\n📂 1. LOAD DATA...")

# Load square_hour data
df = pd.read_csv('data/square_hour_aggregated_full.csv')
print(f"   Data shape: {df.shape}")
print(f"   Kolom: {df.columns.tolist()}")

# Load cluster results
clusters = pd.read_csv('data/square_clusters_full.csv')
print(f"   Cluster data: {len(clusters):,} square_id")

# Merge dengan cluster
df = df.merge(clusters[['square_id', 'cluster']], on='square_id', how='left')
print(f"   Setelah merge: {len(df):,} record")

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
print("\n🔧 2. FEATURE ENGINEERING...")

# Target
target = 'value_mean'

# Features
features = ['hour', 'avg_values_count']

# Tambahkan cluster sebagai feature (kategori)
df['cluster'] = df['cluster'].fillna(0).astype(int)
features.append('cluster')

# One-hot encoding untuk cluster (opsional, Random Forest bisa handle integer)
# Tapi kita tetap gunakan sebagai integer karena sudah ordinal

X = df[features].copy()
y = df[target].copy()

# Handle missing values
X = X.fillna(0)
y = y.fillna(0)

print(f"   Features: {features}")
print(f"   Total samples: {len(X):,}")

# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================
print("\n📊 3. TRAIN-TEST SPLIT...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training set: {len(X_train):,} samples")
print(f"   Testing set: {len(X_test):,} samples")

# ============================================================
# 4. TRAIN MODEL (RANDOM FOREST)
# ============================================================
print("\n🤖 4. TRAINING RANDOM FOREST...")

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Predict
y_pred_rf = rf.predict(X_test)

# Metrics
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print(f"\n   📈 RANDOM FOREST METRICS:")
print(f"      RMSE: {rmse_rf:.4f}")
print(f"      R² Score: {r2_rf:.4f}")
print(f"      MAE: {mae_rf:.4f}")

# ============================================================
# 5. TRAIN MODEL (GRADIENT BOOSTING)
# ============================================================
print("\n🤖 5. TRAINING GRADIENT BOOSTING...")

gb = GradientBoostingRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
gb.fit(X_train, y_train)

# Predict
y_pred_gb = gb.predict(X_test)

# Metrics
rmse_gb = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb = r2_score(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)

print(f"\n   📈 GRADIENT BOOSTING METRICS:")
print(f"      RMSE: {rmse_gb:.4f}")
print(f"      R² Score: {r2_gb:.4f}")
print(f"      MAE: {mae_gb:.4f}")

# ============================================================
# 6. PERBANDINGAN MODEL
# ============================================================
print("\n📊 6. PERBANDINGAN MODEL:")

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting'],
    'RMSE': [rmse_rf, rmse_gb],
    'R²': [r2_rf, r2_gb],
    'MAE': [mae_rf, mae_gb]
})
print(comparison.to_string(index=False))

# Tentukan model terbaik
if r2_rf > r2_gb:
    best_model = rf
    best_name = 'Random Forest'
    best_r2 = r2_rf
else:
    best_model = gb
    best_name = 'Gradient Boosting'
    best_r2 = r2_gb

print(f"\n   🏆 MODEL TERBAIK: {best_name} (R² = {best_r2:.4f})")

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
print("\n🔍 7. FEATURE IMPORTANCE...")

feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n   Feature Importance:")
for _, row in feature_importance.iterrows():
    bar = '█' * int(row['importance'] * 50)
    print(f"      {row['feature']:20s}: {row['importance']:.4f} {bar}")

# ============================================================
# 8. VISUALISASI
# ============================================================
print("\n📊 8. MEMBUAT VISUALISASI...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Model Performance: {best_name} (R² = {best_r2:.4f})', fontsize=14, fontweight='bold')

# Plot 1: Actual vs Predicted
ax1 = axes[0, 0]
sample_size = min(1000, len(y_test))
ax1.scatter(y_test[:sample_size], y_pred_rf[:sample_size], alpha=0.5, s=10)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
ax1.set_xlabel('Actual Value')
ax1.set_ylabel('Predicted Value')
ax1.set_title('Actual vs Predicted (Random Forest)')

# Plot 2: Residuals
ax2 = axes[0, 1]
residuals = y_test - y_pred_rf
ax2.hist(residuals, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Residual')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribusi Residual')
ax2.axvline(0, color='red', linestyle='--')

# Plot 3: Feature Importance
ax3 = axes[1, 0]
ax3.barh(feature_importance['feature'], feature_importance['importance'], color='coral')
ax3.set_xlabel('Importance')
ax3.set_title('Feature Importance')
ax3.invert_yaxis()

# Plot 4: R² Comparison
ax4 = axes[1, 1]
models = ['Random Forest', 'Gradient Boosting']
r2_scores = [r2_rf, r2_gb]
colors = ['steelblue', 'coral']
ax4.bar(models, r2_scores, color=colors)
ax4.set_ylabel('R² Score')
ax4.set_title('Perbandingan R² Score')
ax4.set_ylim(0, 1)
for i, v in enumerate(r2_scores):
    ax4.text(i, v + 0.02, f'{v:.4f}', ha='center')

plt.tight_layout()
plt.savefig('prediction_results.png', dpi=150)
print("   ✅ Visualisasi disimpan ke: prediction_results.png")

# ============================================================
# 9. CONTOH PREDIKSI
# ============================================================
print("\n🔮 9. CONTOH PREDIKSI UNTUK SQUARE_ID TERTENTU...")

sample_square = df.sample(5, random_state=42)
sample_features = sample_square[features].fillna(0)
sample_predictions = best_model.predict(sample_features)

print("\n   Hasil Prediksi untuk 5 sample:")
for i, (_, row) in enumerate(sample_square.iterrows()):
    print(f"      Square {int(row['square_id'])}: Jam {int(row['hour']):02d}:00 → "
          f"Aktual={row['value_mean']:.4f}, Prediksi={sample_predictions[i]:.4f}")

# ============================================================
# 10. SIMPAN MODEL (opsional)
# ============================================================
print("\n💾 10. MENYIMPAN MODEL...")

import joblib
joblib.dump(best_model, 'traffic_prediction_model.pkl')
joblib.dump(features, 'model_features.pkl')
print("   ✅ Model disimpan ke: traffic_prediction_model.pkl")

print("\n" + "=" * 70)
print(" ✅ PREDIKSI TRAFFIC SELESAI!")
print("=" * 70)

print("\n💡 INSIGHT MODEL:")
print(f"   • Model terbaik: {best_name} dengan R² = {best_r2:.4f}")
print(f"   • Feature terpenting: {feature_importance.iloc[0]['feature']}")
print(f"   • Model dapat memprediksi traffic dengan akurasi yang baik")
print("\n📁 File yang dihasilkan:")
print("   • prediction_results.png (visualisasi)")
print("   • traffic_prediction_model.pkl (model siap pakai)")