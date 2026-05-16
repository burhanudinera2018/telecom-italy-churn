# investigate_data.py
import pandas as pd
import os

print("=" * 60)
print("INVESTIGASI DATA ASLI")
print("=" * 60)

# Cek file yang ada
print("\n📁 Files in data folder:")
for file in os.listdir('data'):
    if file.endswith('.csv'):
        size = os.path.getsize(f'data/{file}') / (1024*1024)
        print(f"   - {file} ({size:.2f} MB)")

# Coba load data asli (tanpa parse_dates dulu)
print("\n📂 Loading raw CSV (first 5 rows)...")
df_raw = pd.read_csv('data/telecom_milan_parsed.csv', nrows=5)

print("\n✅ COLUMNS FOUND:")
for i, col in enumerate(df_raw.columns):
    print(f"   {i+1}. '{col}'")

print("\n📊 FIRST 5 ROWS:")
print(df_raw)

print("\n📊 DATA TYPES:")
print(df_raw.dtypes)

print("\n📊 SAMPLE VALUES:")
for col in df_raw.columns:
    print(f"\n{col}:")
    print(f"   Sample values: {df_raw[col].iloc[:3].tolist()}")
    print(f"   Unique values: {df_raw[col].nunique()}")