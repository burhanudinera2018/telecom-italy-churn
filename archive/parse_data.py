# parse_data.py
import pandas as pd
import numpy as np
from pathlib import Path
import glob

print("=" * 60)
print("📡 PARSING REAL TELECOM ITALY DATASET")
print("Data: Milan, Italy - November 2013 (3 hari pertama)")
print("=" * 60)

# Lokasi file TXT hasil download
data_folder = Path("data/telecom-italia-sample")
txt_files = sorted(glob.glob(str(data_folder / "*.txt")))

print(f"\n📁 Ditemukan {len(txt_files)} file:")
for f in txt_files:
    size_mb = Path(f).stat().st_size / (1024 * 1024)
    print(f"   📄 {Path(f).name} ({size_mb:.1f} MB)")

# ============================================
# Parsing setiap file TXT
# ============================================

def parse_telecom_file(filepath):
    """
    Parse file TXT dari Telecom Italia.
    Format: square_id;timestamp;sms_in;sms_out;call_in;call_out;internet_traffic
    """
    print(f"\n📖 Parsing: {Path(filepath).name}...")
    
    # Baca file dengan separator ;
    df = pd.read_csv(
        filepath, 
        sep=';',
        header=None,
        names=['square_id', 'timestamp', 'sms_in', 'sms_out', 
               'call_in', 'call_out', 'internet_traffic'],
        dtype={
            'square_id': str,
            'timestamp': str,
            'sms_in': float,
            'sms_out': float,
            'call_in': float,
            'call_out': float,
            'internet_traffic': float
        }
    )
    
    # Konversi timestamp ke datetime
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    
    # Hitung total traffic
    df['total_calls'] = df['call_in'] + df['call_out']
    df['total_sms'] = df['sms_in'] + df['sms_out']
    df['total_traffic'] = df['total_calls'] + df['total_sms'] + df['internet_traffic']
    
    print(f"   ✅ {len(df):,} records, {df['square_id'].nunique()} unique grid cells")
    
    return df

# Parse semua file
all_dfs = []
for filepath in txt_files:
    df = parse_telecom_file(filepath)
    all_dfs.append(df)

# Gabungkan semua data
df_combined = pd.concat(all_dfs, ignore_index=True)

print("\n" + "=" * 60)
print("📊 DATA SUMMARY")
print("=" * 60)
print(f"✅ Total records: {len(df_combined):,}")
print(f"✅ Time range: {df_combined['datetime'].min()} → {df_combined['datetime'].max()}")
print(f"✅ Unique grid cells: {df_combined['square_id'].nunique():,}")
print(f"✅ Hours covered: {df_combined['hour'].nunique()} jam")

# Simpan ke CSV untuk analisis selanjutnya
output_path = "data/telecom_milan_parsed.csv"
df_combined.to_csv(output_path, index=False)
print(f"\n💾 Data saved: {output_path} ({Path(output_path).stat().st_size / (1024*1024):.1f} MB)")

# Preview data
print("\n📋 PREVIEW (5 rows):")
print(df_combined.head())

print("\n📊 STATISTIK:")
print(df_combined[['sms_in', 'sms_out', 'call_in', 'call_out', 'internet_traffic']].describe())