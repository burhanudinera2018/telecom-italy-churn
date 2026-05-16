#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Parsing dataset telecom log dari Harvard Dataverse
Mengubah format sparse menjadi struktur yang rapi
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

def parse_telecom_file(file_path):
    """
    Parse sparse telecom log file menjadi list of dictionaries
    
    Format input: 
    square_id | timestamp_ms | service_type | value1 | value2 | ...
    
    Parameter:
        file_path (str/Path): Path ke file .txt
    
    Returns:
        list: List of dictionary dengan keys: square_id, timestamp, service_type, values
    """
    records = []
    error_lines = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            
            # Validasi minimal harus punya 4 kolom
            if len(parts) < 4:
                error_lines.append({
                    'line': line_num,
                    'reason': f'Only {len(parts)} columns (need at least 4)',
                    'content': line[:100]
                })
                continue
            
            try:
                # Ekstrak kolom wajib
                square_id = int(parts[0])
                timestamp_ms = int(parts[1])
                service_type = int(parts[2])
                
                # Kolom values (mulai kolom ke 4/index 3)
                # Nilai kosong '' menjadi None (akan jadi NaN nanti)
                raw_values = parts[3:]
                values = []
                for v in raw_values:
                    if v == '' or v.strip() == '':
                        values.append(np.nan)
                    else:
                        try:
                            values.append(float(v))
                        except ValueError:
                            values.append(np.nan)
                
                records.append({
                    'square_id': square_id,
                    'timestamp_ms': timestamp_ms,
                    'datetime': datetime.fromtimestamp(timestamp_ms / 1000).strftime('%Y-%m-%d %H:%M:%S'),
                    'service_type': service_type,
                    'values': values,
                    'num_values': len(values)
                })
                
            except ValueError as e:
                error_lines.append({
                    'line': line_num,
                    'reason': str(e),
                    'content': line[:100]
                })
                continue
    
    return records, error_lines


def analyze_service_types(records):
    """Analisis distribusi service_type"""
    service_counts = {}
    service_examples = {}
    
    for rec in records:
        st = rec['service_type']
        service_counts[st] = service_counts.get(st, 0) + 1
        
        if st not in service_examples:
            service_examples[st] = {
                'sample_values': rec['values'][:5],
                'sample_timestamp': rec['datetime'],
                'num_values': rec['num_values']
            }
    
    return service_counts, service_examples


def analyze_values_structure(records, sample_size=100):
    """Analisis struktur kolom values"""
    value_lengths = {}
    
    for rec in records[:sample_size]:
        vl = rec['num_values']
        value_lengths[vl] = value_lengths.get(vl, 0) + 1
    
    return value_lengths


def main():
    print("=" * 70)
    print(" PARSING DATASET TELECOM - HARVARD DATAVERSE")
    print("=" * 70)
    print()
    
    # Lokasi file
    base_path = Path('./data/telecom-italia-sample/')
    
    # Cek apakah folder ada
    if not base_path.exists():
        print(f"❌ Folder tidak ditemukan: {base_path}")
        print("   Pastikan Anda menjalankan script dari folder yang benar.")
        print(f"   Current directory: {Path.cwd()}")
        return
    
    files = [
        'sms-call-internet-mi-2013-11-01.txt',
        'sms-call-internet-mi-2013-11-02.txt',
        'sms-call-internet-mi-2013-11-03.txt'
    ]
    
    all_results = {}
    all_errors = {}
    
    # Parse setiap file
    for f in files:
        file_path = base_path / f
        print(f"\n📂 MEMPROSES: {f}")
        print("-" * 50)
        
        if not file_path.exists():
            print(f"   ❌ File tidak ditemukan!")
            continue
        
        # Parse file
        records, errors = parse_telecom_file(file_path)
        all_results[f] = records
        all_errors[f] = errors
        
        print(f"   ✅ Berhasil parsing: {len(records):,} record")
        print(f"   ⚠️  Baris error: {len(errors)}")
        
        if records:
            # Sample record pertama
            sample = records[0]
            print(f"\n   📋 CONTOH RECORD PERTAMA:")
            print(f"      square_id     : {sample['square_id']}")
            print(f"      timestamp     : {sample['timestamp_ms']}")
            print(f"      datetime      : {sample['datetime']}")
            print(f"      service_type  : {sample['service_type']}")
            print(f"      jumlah values : {sample['num_values']}")
            print(f"      values (3)    : {sample['values'][:3]}")
    
    # ==================== ANALISIS ====================
    print("\n" + "=" * 70)
    print(" HASIL ANALISIS")
    print("=" * 70)
    
    # Gabungkan semua record untuk analisis
    all_records = []
    for records in all_results.values():
        all_records.extend(records)
    
    print(f"\n📊 TOTAL RECORD: {len(all_records):,}")
    
    # 1. Service Type Distribution
    print("\n" + "-" * 50)
    print("1. DISTRIBUSI SERVICE TYPE")
    print("-" * 50)
    
    service_counts, service_examples = analyze_service_types(all_records)
    
    for st, count in sorted(service_counts.items()):
        percentage = (count / len(all_records)) * 100
        example = service_examples[st]
        print(f"\n   🔹 Service Type {st}: {count:,} record ({percentage:.1f}%)")
        print(f"      Contoh values: {example['sample_values']}")
        print(f"      Jumlah kolom values: {example['num_values']}")
    
    # 2. Struktur Values
    print("\n" + "-" * 50)
    print("2. STRUKTUR KOLOM VALUES (sample 100 record)")
    print("-" * 50)
    
    value_lengths = analyze_values_structure(all_records, sample_size=100)
    for vl, count in sorted(value_lengths.items()):
        print(f"   Jumlah values = {vl}: {count} record")
    
    # 3. Cek Nilai NaN dan 0
    print("\n" + "-" * 50)
    print("3. CEK NILAI NaN DAN 0 (sample 500 record)")
    print("-" * 50)
    
    nan_count = 0
    zero_count = 0
    total_values = 0
    
    for rec in all_records[:500]:
        for v in rec['values']:
            total_values += 1
            if pd.isna(v):
                nan_count += 1
            elif v == 0:
                zero_count += 1
    
    print(f"   Total nilai yang diperiksa: {total_values:,}")
    print(f"   Nilai NaN (missing)      : {nan_count} ({nan_count/total_values*100:.1f}%)")
    print(f"   Nilai 0 (asli data)     : {zero_count} ({zero_count/total_values*100:.1f}%)")
    
    if zero_count > 0:
        print("\n   ℹ️  Ada nilai 0 dalam data asli — ini WAJAR untuk data telekomunikasi")
        print("      (panggilan gagal, SMS tidak terkirim, internet 0 byte)")
    
    # 4. Simpan hasil ke file
    print("\n" + "-" * 50)
    print("4. MENYIMPAN HASIL")
    print("-" * 50)
    
    # Simpan summary ke JSON
    summary = {
        'total_records': len(all_records),
        'service_types': {str(k): v for k, v in service_counts.items()},
        'file_stats': {
            f: {
                'records': len(all_results.get(f, [])),
                'errors': len(all_errors.get(f, []))
            } for f in files
        }
    }
    
    output_file = 'parsing_summary.json'
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✅ Summary disimpan ke: {output_file}")
    
    # Simpan sample data ke CSV untuk dilihat
    sample_data = []
    for rec in all_records[:1000]:
        sample_data.append({
            'square_id': rec['square_id'],
            'datetime': rec['datetime'],
            'service_type': rec['service_type'],
            'values_count': rec['num_values'],
            'value_1': rec['values'][0] if len(rec['values']) > 0 else None,
            'value_2': rec['values'][1] if len(rec['values']) > 1 else None,
            'value_3': rec['values'][2] if len(rec['values']) > 2 else None,
        })
    
    df_sample = pd.DataFrame(sample_data)
    csv_file = 'parsed_sample.csv'
    df_sample.to_csv(csv_file, index=False)
    print(f"   ✅ Sample 1000 record disimpan ke: {csv_file}")
    
    print("\n" + "=" * 70)
    print(" ✅ PARSING SELESAI!")
    print("=" * 70)
    print("\n📁 File yang dihasilkan:")
    print(f"   - {output_file} → ringkasan data")
    print(f"   - {csv_file} → sample 1000 record (buka dengan Excel)")
    print("\n💡 Next step: Buka file CSV dengan Excel untuk melihat strukturnya")


if __name__ == "__main__":
    main()