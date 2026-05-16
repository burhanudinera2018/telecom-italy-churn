#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PARSE FULL DATASET TELECOM
Menggabungkan 3 file txt (1-3 November 2013) menjadi satu CSV lengkap
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time

def parse_telecom_file_to_df(file_path, file_date):
    """
    Parse file txt telekomunikasi menjadi dataframe
    
    Format: square_id | timestamp_ms | service_type | value1 | value2 | ...
    """
    records = []
    error_count = 0
    
    print(f"   Membaca file: {file_path.name}...")
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        total_lines = len(lines)
        
        for idx, line in enumerate(lines):
            # Progress indicator setiap 100k baris
            if idx % 100000 == 0 and idx > 0:
                print(f"      Proses: {idx:,} / {total_lines:,} baris ({idx/total_lines*100:.1f}%)")
            
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            
            # Validasi minimal 4 kolom (square_id, timestamp, service_type, setidaknya 1 value)
            if len(parts) < 4:
                error_count += 1
                continue
            
            try:
                square_id = int(parts[0])
                timestamp_ms = int(parts[1])
                service_type = int(parts[2])
                raw_values = parts[3:]
                
                # Bersihkan values (string kosong -> NaN)
                clean_values = []
                for v in raw_values:
                    if v == '' or v.strip() == '':
                        clean_values.append(np.nan)
                    else:
                        try:
                            clean_values.append(float(v))
                        except ValueError:
                            clean_values.append(np.nan)
                
                # Buat record dengan maksimal 5 values (sesuai struktur data)
                record = {
                    'square_id': square_id,
                    'timestamp_ms': timestamp_ms,
                    'service_type': service_type,
                    'values_count': len(clean_values),
                    'file_date': file_date
                }
                
                # Tambahkan value_1 sampai value_5
                for i in range(1, 6):
                    record[f'value_{i}'] = clean_values[i-1] if len(clean_values) >= i else np.nan
                
                records.append(record)
                
            except (ValueError, IndexError) as e:
                error_count += 1
                continue
    
    print(f"      ✅ Selesai: {len(records):,} record valid, {error_count} baris error")
    
    # Konversi ke DataFrame
    df = pd.DataFrame(records)
    
    # Konversi timestamp ke datetime
    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    
    return df


def main():
    print("=" * 70)
    print(" PARSE FULL DATASET TELECOM - 3 November 2013")
    print("=" * 70)
    
    # Lokasi file
    base_path = Path('./data/telecom-italia-sample/')
    
    # Cek apakah folder ada
    if not base_path.exists():
        print(f"\n❌ Folder tidak ditemukan: {base_path}")
        print("   Pastikan Anda menjalankan script dari folder yang benar.")
        print(f"   Current directory: {Path.cwd()}")
        return
    
    # Daftar file
    files = [
        ('sms-call-internet-mi-2013-11-01.txt', '2013-11-01'),
        ('sms-call-internet-mi-2013-11-02.txt', '2013-11-02'),
        ('sms-call-internet-mi-2013-11-03.txt', '2013-11-03')
    ]
    
    print("\n📂 MEMPROSES FILE:")
    print("-" * 50)
    
    all_dfs = []
    start_time = time.time()
    
    for filename, file_date in files:
        file_path = base_path / filename
        
        if file_path.exists():
            print(f"\n📄 {filename}")
            df = parse_telecom_file_to_df(file_path, file_date)
            all_dfs.append(df)
        else:
            print(f"\n❌ File tidak ditemukan: {file_path}")
    
    if not all_dfs:
        print("\n❌ Tidak ada file yang diproses. Pastikan file ada di folder yang benar.")
        return
    
    # Gabungkan semua dataframe
    print("\n" + "=" * 70)
    print(" MENGABUNGKAN DATA...")
    print("=" * 70)
    
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    elapsed_time = time.time() - start_time
    
    print(f"\n✅ HASIL PARSING:")
    print(f"   Total record      : {len(full_df):,}")
    print(f"   Total kolom       : {len(full_df.columns)}")
    print(f"   Unique square_id  : {full_df['square_id'].nunique():,}")
    print(f"   Unique service_type: {full_df['service_type'].nunique():,}")
    print(f"   Periode           : {full_df['datetime'].min()} → {full_df['datetime'].max()}")
    print(f"   Waktu parsing     : {elapsed_time:.2f} detik")
    
    # ============================================================
    # CEK KUALITAS DATA
    # ============================================================
    print("\n" + "=" * 70)
    print(" CEK KUALITAS DATA")
    print("=" * 70)
    
    # Cek nilai 0 asli
    for col in ['value_1', 'value_2', 'value_3']:
        zero_count = (full_df[col] == 0).sum()
        nan_count = full_df[col].isna().sum()
        print(f"\n   Kolom {col}:")
        print(f"      Nilai 0 asli : {zero_count:,} ({zero_count/len(full_df)*100:.2f}%)")
        print(f"      Nilai NaN    : {nan_count:,} ({nan_count/len(full_df)*100:.2f}%)")
    
    # ============================================================
    # DISTRIBUSI SERVICE TYPE
    # ============================================================
    print("\n" + "=" * 70)
    print(" DISTRIBUSI SERVICE TYPE (TOP 15)")
    print("=" * 70)
    
    service_counts = full_df['service_type'].value_counts()
    for st, count in service_counts.head(15).items():
        pct = count / len(full_df) * 100
        bar = '█' * int(pct / 2)
        print(f"   Service {int(st):5d}: {count:8,} record ({pct:5.2f}%) {bar}")
    
    # ============================================================
    # SIMPAN KE CSV
    # ============================================================
    print("\n" + "=" * 70)
    print(" MENYIMPAN DATA")
    print("=" * 70)
    
    output_file = 'telecom_full_dataset.csv'
    full_df.to_csv(output_file, index=False)
    print(f"\n   ✅ Full dataset disimpan ke: {output_file}")
    print(f"      Ukuran file: ~{full_df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    # Juga simpan versi ringkas (tanpa raw values)
    compact_df = full_df[['square_id', 'datetime', 'service_type', 'value_1', 'values_count']]
    compact_file = 'telecom_full_dataset_compact.csv'
    compact_df.to_csv(compact_file, index=False)
    print(f"   ✅ Versi ringkas disimpan ke: {compact_file}")
    
    print("\n" + "=" * 70)
    print(" ✅ PARSING FULL DATASET SELESAI!")
    print("=" * 70)
    
    print("\n💡 LANGKAH SELANJUTNYA:")
    print("   1. Buka telecom_full_dataset.csv dengan Excel untuk eksplorasi")
    print("   2. Jalankan correlation_analysis.py (baca file ini)")
    print("   3. Jalankan traffic_prediction.py")
    print("   4. Jalankan clustering_analysis.py")
    
    return full_df

if __name__ == "__main__":
    df = main()