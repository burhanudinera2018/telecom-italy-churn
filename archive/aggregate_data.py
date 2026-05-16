# aggregate_data.py
import pandas as pd
import time

start_time = time.time()

print("=" * 50)
print("DATA AGGREGATION FOR STREAMLIT")
print("=" * 50)

# Load data
print("\n📂 Loading data...")
df = pd.read_csv('data/telecom_milan_parsed.csv', parse_dates=['datetime'])
print(f"✅ Loaded {len(df):,} rows")

# Clean column names
df.columns = df.columns.str.lower()
print(f"📊 Columns: {df.columns.tolist()}")

# Create hour column if not exists
if 'hour' not in df.columns:
    print("\n⏰ Creating hour column...")
    df['hour'] = pd.to_datetime(df['datetime']).dt.hour

# Create total columns if needed
if 'total_calls' not in df.columns and 'call_in' in df.columns and 'call_out' in df.columns:
    print("📞 Creating total_calls from call_in + call_out...")
    df['total_calls'] = df['call_in'] + df['call_out']

if 'total_sms' not in df.columns and 'sms_in' in df.columns and 'sms_out' in df.columns:
    print("💬 Creating total_sms from sms_in + sms_out...")
    df['total_sms'] = df['sms_in'] + df['sms_out']

if 'internet_traffic' not in df.columns and 'internet_traffic_mb' in df.columns:
    print("🌐 Renaming internet_traffic_mb to internet_traffic...")
    df['internet_traffic'] = df['internet_traffic_mb']

# 1. Hourly aggregation
print("\n📊 Aggregating by hour...")
hourly_agg = df.groupby('hour').agg({
    'total_calls': 'sum',
    'total_sms': 'sum',
    'internet_traffic': 'sum'
}).reset_index()

hourly_agg.to_csv('data/hourly_aggregated.csv', index=False)
print(f"✅ Hourly data saved: {len(hourly_agg)} rows (from 24 hours)")

# 2. Square ID aggregation (for clusters)
print("\n🏢 Aggregating by square_id...")
square_agg = df.groupby('square_id').agg({
    'total_calls': 'mean',
    'total_sms': 'mean',
    'internet_traffic': 'mean'
}).reset_index()

square_agg.to_csv('data/square_aggregated.csv', index=False)
print(f"✅ Square data saved: {len(square_agg):,} rows")

# 3. Create sample data for quick testing (optional)
print("\n🎲 Creating sample data (100k rows for testing)...")
sample_df = df.sample(n=min(100000, len(df)), random_state=42)
sample_df.to_csv('data/sample_data.csv', index=False)
print(f"✅ Sample data saved: {len(sample_df):,} rows")

# Summary
end_time = time.time()
print("\n" + "=" * 50)
print("AGGREGATION COMPLETE!")
print("=" * 50)
print(f"⏱️  Time taken: {(end_time - start_time):.2f} seconds")
print(f"📁 Files created:")
print(f"   - data/hourly_aggregated.csv ({len(hourly_agg)} rows)")
print(f"   - data/square_aggregated.csv ({len(square_agg):,} rows)")
print(f"   - data/sample_data.csv ({len(sample_df):,} rows)")
print("\n💡 You can now run the optimized Streamlit app!")