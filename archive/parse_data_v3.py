# parse_data_v3.py
import pandas as pd
import numpy as np
from pathlib import Path
import glob
from datetime import datetime

print("=" * 60)
print("📡 PARSING TELECOM ITALIA DATASET - VERSION 3")
print("Format: TAB-separated, variable columns per row")
print("=" * 60)

data_folder = Path("data/telecom-italia-sample")
txt_files = sorted(glob.glob(str(data_folder / "*.txt")))

print(f"\n📁 Found {len(txt_files)} files:")
for f in txt_files:
    size_mb = Path(f).stat().st_size / (1024 * 1024)
    print(f"   📄 {Path(f).name} ({size_mb:.1f} MB)")

# Mapping berdasarkan nilai kolom ke-3 (lambda indicator)
# Berdasarkan sample data:
#   - value 0 = SMS? (nilai desimal kecil)
#   - value 33 = Call? (nilai desimal sedang)
#   - value 39 = Internet? (nilai desimal dan integer traffic)

def parse_row_parts(parts):
    """
    Parse row dengan panjang variabel
    Berdasarkan dokumentasi Telecom Italia:
    Format: square_id;timestamp;type;value1;value2;...
    """
    if len(parts) < 3:
        return None
    
    square_id = parts[0].strip()
    timestamp_ms = int(parts[1].strip())
    
    # Konversi timestamp milliseconds ke datetime
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0)
    
    data_type = int(parts[2].strip())
    
    result = {
        'square_id': square_id,
        'timestamp': timestamp_ms,
        'datetime': dt,
        'date': dt.date(),
        'hour': dt.hour,
        'data_type': data_type
    }
    
    # Parse values berdasarkan tipe
    if data_type == 0:  # SMS related
        if len(parts) > 3:
            result['sms_value'] = float(parts[3]) if parts[3].strip() else np.nan
        if len(parts) > 4:
            result['sms_value2'] = float(parts[4]) if parts[4].strip() else np.nan
            
    elif data_type == 33:  # Call related
        if len(parts) > 3:
            result['call_value'] = float(parts[3]) if parts[3].strip() else np.nan
        if len(parts) > 4:
            result['call_value2'] = float(parts[4]) if parts[4].strip() else np.nan
        if len(parts) > 5:
            result['call_value3'] = float(parts[5]) if parts[5].strip() else np.nan
            
    elif data_type == 39:  # Internet traffic
        if len(parts) > 3:
            result['internet_traffic'] = float(parts[3]) if parts[3].strip() else np.nan
        if len(parts) > 4:
            result['internet_value2'] = float(parts[4]) if parts[4].strip() else np.nan
        if len(parts) > 5:
            result['internet_value3'] = float(parts[5]) if parts[5].strip() else np.nan
        if len(parts) > 6:
            result['internet_value4'] = float(parts[6]) if parts[6].strip() else np.nan
        if len(parts) > 7:
            result['internet_total'] = float(parts[7]) if parts[7].strip() else np.nan
    
    return result

def parse_telecom_file(filepath):
    """Parse file Telecom Italia dengan format TAB separator"""
    print(f"\n📖 Parsing: {Path(filepath).name}...")
    
    rows = []
    total_lines = 0
    error_lines = 0
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f):
            total_lines += 1
            
            if total_lines % 500000 == 0:
                print(f"   Processed {total_lines:,} lines, {len(rows):,} valid records...")
            
            # Split dengan TAB
            parts = line.strip().split('\t')
            
            if len(parts) < 3:
                error_lines += 1
                continue
            
            try:
                parsed = parse_row_parts(parts)
                if parsed:
                    rows.append(parsed)
                else:
                    error_lines += 1
            except Exception as e:
                error_lines += 1
                continue
    
    df = pd.DataFrame(rows)
    print(f"   ✅ Total lines: {total_lines:,}")
    print(f"   ✅ Valid records: {len(df):,}")
    print(f"   ⚠️ Skipped: {error_lines:,}")
    
    return df

# Parse semua file
all_dfs = []
for filepath in txt_files:
    df = parse_telecom_file(filepath)
    if len(df) > 0:
        all_dfs.append(df)
        print(f"   📊 Unique square_ids: {df['square_id'].nunique()}")
        print(f"   📊 Data types found: {sorted(df['data_type'].unique())}")

if all_dfs:
    df_combined = pd.concat(all_dfs, ignore_index=True)
    
    print("\n" + "=" * 60)
    print("📊 DATA SUMMARY")
    print("=" * 60)
    print(f"✅ Total records: {len(df_combined):,}")
    print(f"✅ Time range: {df_combined['datetime'].min()} → {df_combined['datetime'].max()}")
    print(f"✅ Unique grid cells: {df_combined['square_id'].nunique():,}")
    print(f"✅ Data types: {sorted(df_combined['data_type'].unique())}")
    
    # Aggregasi per square_id per hour untuk traffic analysis
    print("\n📊 Aggregating network traffic...")
    
    # Pisahkan berdasarkan tipe data
    sms_data = df_combined[df_combined['data_type'] == 0].groupby(['square_id', 'hour']).agg({
        'sms_value': 'sum'
    }).rename(columns={'sms_value': 'total_sms'})
    
    call_data = df_combined[df_combined['data_type'] == 33].groupby(['square_id', 'hour']).agg({
        'call_value': 'sum'
    }).rename(columns={'call_value': 'total_calls'})
    
    internet_data = df_combined[df_combined['data_type'] == 39].groupby(['square_id', 'hour']).agg({
        'internet_total': 'sum'
    }).rename(columns={'internet_total': 'internet_traffic_mb'})
    
    # Gabungkan
    traffic_df = sms_data.join(call_data, how='outer').join(internet_data, how='outer')
    traffic_df = traffic_df.fillna(0).reset_index()
    
    traffic_df['total_traffic'] = (
        traffic_df['total_sms'] + 
        traffic_df['total_calls'] + 
        traffic_df['internet_traffic_mb']
    )
    
    print(f"✅ Aggregated data: {len(traffic_df):,} records (grid_cell x hour)")
    print(f"   Unique cells: {traffic_df['square_id'].nunique()}")
    print(f"   Hours range: {traffic_df['hour'].min()} - {traffic_df['hour'].max()}")
    
    # Preview
    print("\n📋 PREVIEW aggregated data (10 rows):")
    print(traffic_df.head(10))
    
    # Simpan hasil parsing
    output_path = "data/telecom_milan_parsed_v3.csv"
    df_combined.to_csv(output_path, index=False)
    print(f"\n💾 Raw parsed data saved: {output_path}")
    
    output_agg_path = "data/telecom_milan_traffic.csv"
    traffic_df.to_csv(output_agg_path, index=False)
    print(f"💾 Aggregated traffic data saved: {output_agg_path}")
    
    print("\n" + "=" * 60)
    print("✅ PARSING COMPLETE!")
    print("=" * 60)
    
    # Statistik ringkas
    print("\n📊 QUICK STATISTICS:")
    print(f"   Total SMS volume: {traffic_df['total_sms'].sum():,.0f}")
    print(f"   Total Call volume: {traffic_df['total_calls'].sum():,.0f}")
    print(f"   Total Internet: {traffic_df['internet_traffic_mb'].sum():,.0f} MB")
    
else:
    print("\n❌ No data successfully parsed!")