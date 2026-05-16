# fix_parse_data.py
import pandas as pd
import numpy as np
from datetime import datetime
import time

print("=" * 60)
print("FIXED PARSING FOR TELECOM MILAN DATA (TSV FORMAT)")
print("=" * 60)

start_time = time.time()

# Load data dengan separator tab (bukan koma)
print("\n1. Loading data with tab separator...")
df = pd.read_csv('data/telecom_milan_parsed.csv', 
                 sep='\t',  # Kunci utama: pakai tab separator!
                 header=0,
                 low_memory=False)

print(f"   ✓ Loaded {len(df):,} rows")
print(f"   ✓ Columns: {len(df.columns)}")

# Rename columns (remove any whitespace)
df.columns = df.columns.str.strip().str.lower()
print(f"\n2. Columns found: {df.columns.tolist()}")

# Check if data is loaded correctly
print("\n3. Data preview (first 3 rows):")
print(df.head(3))

# Convert timestamp to datetime
print("\n4. Converting timestamp to datetime...")
if 'timestamp' in df.columns:
    # Timestamp appears to be in milliseconds
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    print(f"   ✓ Converted timestamp to datetime")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")

# Convert numeric columns
print("\n5. Converting numeric columns...")
numeric_cols = ['sms_in', 'sms_out', 'call_in', 'call_out', 'internet_traffic']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"   ✓ {col}")

# Create total metrics
print("\n6. Creating total metrics...")
if 'call_in' in df.columns and 'call_out' in df.columns:
    df['total_calls'] = df['call_in'] + df['call_out']
    print("   ✓ total_calls created")

if 'sms_in' in df.columns and 'sms_out' in df.columns:
    df['total_sms'] = df['sms_in'] + df['sms_out']
    print("   ✓ total_sms created")

# Fill missing values
print("\n7. Handling missing values...")
print(f"   Missing before: {df.isnull().sum().sum():,}")
df = df.fillna(0)
print(f"   Missing after: {df.isnull().sum().sum():,}")

# Extract hour
if 'datetime' in df.columns:
    df['hour'] = df['datetime'].dt.hour
    print(f"\n8. Hour extracted: {df['hour'].min()} to {df['hour'].max()}")

# Verify data
print("\n9. Data verification:")
print(f"   Total calls: {df['total_calls'].sum():,.0f}")
print(f"   Total SMS: {df['total_sms'].sum():,.0f}")
print(f"   Total internet: {df['internet_traffic'].sum():,.0f} MB")

# Check if data is valid
if df['total_calls'].sum() > 0:
    print("\n   ✅ DATA IS VALID!")
else:
    print("\n   ❌ Data still zeros, checking column values...")
    for col in ['total_calls', 'total_sms', 'internet_traffic']:
        if col in df.columns:
            print(f"   {col} sample: {df[col].head(10).tolist()}")

# Create aggregated data for Streamlit
print("\n10. Creating aggregated files for Streamlit...")

# Hourly aggregation
hourly_agg = df.groupby('hour').agg({
    'total_calls': 'sum',
    'total_sms': 'sum',
    'internet_traffic': 'sum'
}).reset_index()

# Sort by hour
hourly_agg = hourly_agg.sort_values('hour')

hourly_agg.to_csv('data/hourly_aggregated.csv', index=False)
print(f"   ✓ hourly_aggregated.csv ({len(hourly_agg)} rows)")
print("\n   Hourly data preview:")
print(hourly_agg)

# Square aggregation (take sample for performance)
print("\n11. Creating square aggregation...")
if len(df) > 500000:
    df_sample = df.sample(n=500000, random_state=42)
    print(f"   Using {len(df_sample):,} sample rows")
else:
    df_sample = df

square_agg = df_sample.groupby('square_id').agg({
    'total_calls': 'mean',
    'total_sms': 'mean',
    'internet_traffic': 'mean'
}).reset_index()

square_agg.to_csv('data/square_aggregated.csv', index=False)
print(f"   ✓ square_aggregated.csv ({len(square_agg):,} rows)")

# Create clusters
print("\n12. Creating cluster analysis...")
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Prepare features
feature_cols = ['total_calls', 'total_sms', 'internet_traffic']
feature_data = square_agg[feature_cols].fillna(0)

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_data)

# K-Means clustering
n_clusters = min(5, len(square_agg))
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
square_agg['cluster'] = kmeans.fit_predict(scaled_features)

# Add cluster labels (interpretation)
cluster_names = {
    0: 'Low Activity',
    1: 'Medium Activity', 
    2: 'High Activity',
    3: 'Very High Activity',
    4: 'Hotspot'
}

square_agg['cluster_name'] = square_agg['cluster'].map(cluster_names)

square_agg.to_csv('data/telecom_milan_clustered.csv', index=False)
print(f"   ✓ telecom_milan_clustered.csv ({len(square_agg):,} rows, {n_clusters} clusters)")

# Print cluster profiles
print("\n   Cluster profiles:")
cluster_profile = square_agg.groupby('cluster')[feature_cols].mean().round(2)
print(cluster_profile)

# Final summary
end_time = time.time()
print("\n" + "=" * 60)
print("✅ PARSING COMPLETE!")
print("=" * 60)
print(f"⏱️  Time taken: {(end_time - start_time):.2f} seconds")

print("\n📊 FINAL VERIFICATION:")
hourly_check = pd.read_csv('data/hourly_aggregated.csv')
print(f"   Hourly data shape: {hourly_check.shape}")
print(f"   Total calls: {hourly_check['total_calls'].sum():,.0f}")
print(f"   Total SMS: {hourly_check['total_sms'].sum():,.0f}")
print(f"   Total internet: {hourly_check['internet_traffic'].sum():,.0f} MB")

print("\n💡 NEXT STEP:")
print("   Run: streamlit run streamlit_app_optimized.py")