#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PREDIKSI TRAFFIC DENGAN MACHINE LEARNING
Memprediksi value_1 berdasarkan hour, service_type, dan square_id
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('parsed_sample.csv')
df['datetime'] = pd.to_datetime(df['datetime'])
df['hour'] = df['datetime'].dt.hour
df['day'] = df['datetime'].dt.day

# Drop missing values (NaN) untuk modeling
df_clean = df.dropna(subset=['value_1'])
print(f"Data setelah drop NaN: {len(df_clean)} record")

# ============================================================
# 1. FEATURE ENGINEERING
# ============================================================
print("\n🔧 1. FEATURE ENGINEERING...")

# Encode categorical variables
le_service = LabelEncoder()
le_square = LabelEncoder()

df_clean['service_encoded'] = le_service.fit_transform(df_clean['service_type'])
df_clean['square_encoded'] = le_square.fit_transform(df_clean['square_id'])

# Features dan target
features = ['hour', 'day', 'values_count', 'service_encoded', 'square_encoded']
target = 'value_1'

X = df_clean[features]
y = df_clean[target]

print(f"   Features: {features}")
print(f"   Target: {target}")

# ============================================================
# 2. TRAIN-TEST SPLIT
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n📊 Training set: {len(X_train)} record")
print(f"📊 Testing set: {len(X_test)} record")

# ============================================================
# 3. TRAIN MODEL (Random Forest)
# ============================================================
print("\n🤖 2. TRAINING MODEL...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ============================================================
# 4. EVALUASI
# ============================================================
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📈 3. EVALUASI MODEL:")
print(f"   RMSE: {rmse:.4f}")
print(f"   R² Score: {r2:.4f}")

# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔍 4. FEATURE IMPORTANCE:")
for _, row in importance.iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Visualisasi feature importance
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importance['feature'], importance['importance'], color='steelblue')
ax.set_xlabel('Importance')
ax.set_title('Feature Importance - Random Forest')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("\n   ✅ Feature importance disimpan ke: feature_importance.png")

# ============================================================
# 6. CONTOH PREDIKSI
# ============================================================
print("\n🔮 5. CONTOH PREDIKSI:")
sample = X_test.iloc[:5]
sample_pred = model.predict(sample)
sample_actual = y_test.iloc[:5]

for i in range(5):
    print(f"   Record {i+1}: Prediksi={sample_pred[i]:.4f} | Aktual={sample_actual.iloc[i]:.4f}")

print("\n" + "=" * 70)
print(" ✅ MODEL PREDIKSI SELESAI!")