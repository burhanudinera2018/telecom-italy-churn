# aggregate_data_fixed.py
import pandas as pd
import numpy as np
import time

start_time = time.time()

print("=" * 50)
print("DATA AGGREGATION FOR STREAMLIT - FIXED")
print("=" * 50)

# Load data
print("\n📂 Loading data...")
df = pd.read_csv('data/telecom_milan_parsed.csv', parse_dates=['datetime'])
print(f"✅ Loaded {len(df):,} rows")

# Clean column names
df.columns = df.columns.str.lower()
print(f"📊 Columns: {df.columns.tolist()}")

# Fill missing values first
print("\n🔧 Handling missing values...")
initial_nulls = df.isnull().sum().sum()
if initial_nulls > 0:
    print(f"   Found {initial_nulls} missing values, filling with 0...")
    df = df.fillna(0)
else:
    print("   No missing values found")

# 1. Hourly aggregation - FIXED
print("\n📊 Aggregating by hour...")

# Create hour column safely
if 'hour' in df.columns:
    # Handle NaN values and convert to int safely
    df['hour'] = df['hour'].fillna(0)  # Replace NaN with 0
    df['hour'] = df['hour'].astype(int)  # Now safe to convert to int
    print(f"   Using existing hour column")
else:
    df['hour'] = df['datetime'].dt.hour
    print(f"   Created hour column from datetime")

print(f"   Hours range: {df['hour'].min()} to {df['hour'].max()}")

# Check what columns are available for aggregation
agg_columns = {}
for col in ['total_calls', 'total_sms', 'internet_traffic', 'call_in', 'call_out', 'sms_in', 'sms_out']:
    if col in df.columns:
        agg_columns[col] = 'sum'

print(f"   Aggregating columns: {list(agg_columns.keys())}")

# Perform aggregation
hourly_agg = df.groupby('hour').agg(agg_columns).reset_index()

# Create total_calls and total_sms if they don't exist
if 'total_calls' not in hourly_agg.columns and 'call_in' in hourly_agg.columns and 'call_out' in hourly_agg.columns:
    hourly_agg['total_calls'] = hourly_agg['call_in'] + hourly_agg['call_out']
    print("   Created total_calls from call_in + call_out")
    
if 'total_sms' not in hourly_agg.columns and 'sms_in' in hourly_agg.columns and 'sms_out' in hourly_agg.columns:
    hourly_agg['total_sms'] = hourly_agg['sms_in'] + hourly_agg['sms_out']
    print("   Created total_sms from sms_in + sms_out")

# Keep only needed columns
keep_cols = ['hour']
if 'total_calls' in hourly_agg.columns:
    keep_cols.append('total_calls')
else:
    # If no calls data, create dummy
    hourly_agg['total_calls'] = 0
    keep_cols.append('total_calls')
    print("   WARNING: No calls data found, using zeros")
    
if 'total_sms' in hourly_agg.columns:
    keep_cols.append('total_sms')
else:
    hourly_agg['total_sms'] = 0
    keep_cols.append('total_sms')
    print("   WARNING: No SMS data found, using zeros")
    
if 'internet_traffic' in hourly_agg.columns:
    keep_cols.append('internet_traffic')
else:
    hourly_agg['internet_traffic'] = 0
    keep_cols.append('internet_traffic')
    print("   WARNING: No internet data found, using zeros")

hourly_agg = hourly_agg[keep_cols]

print(f"\n✅ Hourly aggregation complete: {len(hourly_agg)} rows (hours 0-23)")
print("\n📊 Hourly Data Preview:")
print(hourly_agg.to_string(index=False))

# Save hourly aggregation
hourly_agg.to_csv('data/hourly_aggregated.csv', index=False)
print(f"\n💾 Saved to: data/hourly_aggregated.csv")

# 2. Square ID aggregation (using sampling for performance)
print("\n🏢 Aggregating by square_id...")

# Use sampling for square aggregation (500k rows is enough)
if len(df) > 500000:
    df_sample = df.sample(n=500000, random_state=42)
    print(f"   Using sample: {len(df_sample):,} rows for square aggregation")
else:
    df_sample = df
    print(f"   Using all {len(df_sample):,} rows")

# Prepare aggregation columns for square
square_agg_cols = {}
for col in ['total_calls', 'total_sms', 'internet_traffic']:
    if col in df_sample.columns:
        square_agg_cols[col] = 'mean'

if not square_agg_cols:
    # Fallback to original columns
    for col in ['call_in', 'call_out', 'sms_in', 'sms_out', 'internet_traffic']:
        if col in df_sample.columns:
            square_agg_cols[col] = 'mean'

square_agg = df_sample.groupby('square_id').agg(square_agg_cols).reset_index()

# Create total columns if needed
if 'total_calls' not in square_agg.columns and 'call_in' in square_agg.columns and 'call_out' in square_agg.columns:
    square_agg['total_calls'] = square_agg['call_in'] + square_agg['call_out']
    
if 'total_sms' not in square_agg.columns and 'sms_in' in square_agg.columns and 'sms_out' in square_agg.columns:
    square_agg['total_sms'] = square_agg['sms_in'] + square_agg['sms_out']

square_agg.to_csv('data/square_aggregated.csv', index=False)
print(f"✅ Square data saved: {len(square_agg):,} unique grid cells")

# 3. Create clustered data for Streamlit
print("\n🎯 Creating cluster-ready data...")

# Make a copy for clustering
cluster_data = square_agg.copy()

# Check if we have the necessary columns
feature_cols = []
for col in ['total_calls', 'total_sms', 'internet_traffic']:
    if col in cluster_data.columns:
        feature_cols.append(col)

if len(feature_cols) >= 2:  # Need at least 2 features for clustering
    print(f"   Clustering with features: {feature_cols}")
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Handle any remaining NaN values
    cluster_data[feature_cols] = cluster_data[feature_cols].fillna(0)
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_data[feature_cols])
    
    # Perform K-Means clustering
    n_clusters = min(5, len(cluster_data))  # Max 5 clusters or fewer if less data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_data['cluster'] = kmeans.fit_predict(scaled_features)
    
    print(f"   Created {n_clusters} clusters")
else:
    print("   WARNING: Not enough features for clustering, creating dummy clusters")
    # Create dummy clusters based on data distribution
    if 'total_calls' in cluster_data.columns:
        cluster_data['cluster'] = pd.qcut(cluster_data['total_calls'].rank(method='first'), 
                                           q=5, labels=False)
    else:
        cluster_data['cluster'] = 0

# Save clustered data
cluster_data.to_csv('data/telecom_milan_clustered.csv', index=False)
print(f"✅ Cluster data saved: {len(cluster_data):,} rows with {cluster_data['cluster'].nunique()} clusters")

# 4. Verify files
print("\n" + "=" * 50)
print("📁 VERIFICATION")
print("=" * 50)

# Check hourly file
hourly_check = pd.read_csv('data/hourly_aggregated.csv')
print(f"\n✓ hourly_aggregated.csv: {len(hourly_check)} rows")
print(f"  Columns: {hourly_check.columns.tolist()}")
print(f"  Sample:")
print(hourly_check.head())

# Check cluster file
cluster_check = pd.read_csv('data/telecom_milan_clustered.csv')
print(f"\n✓ telecom_milan_clustered.csv: {len(cluster_check):,} rows")
print(f"  Columns: {cluster_check.columns.tolist()}")
print(f"  Clusters: {sorted(cluster_check['cluster'].unique())}")

# 5. Create a simple test script
print("\n" + "=" * 50)
print("✅ AGGREGATION COMPLETE!")
print("=" * 50)
print(f"⏱️  Time taken: {(time.time() - start_time):.2f} seconds")
print("\n📁 Files created:")
print("   ✓ data/hourly_aggregated.csv")
print("   ✓ data/square_aggregated.csv")
print("   ✓ data/telecom_milan_clustered.csv")

print("\n💡 NEXT STEPS:")
print("1. Run the Streamlit app:")
print("   streamlit run streamlit_app_optimized.py")
print("\n2. Or test with this quick check:")
print("   python -c \"import pandas as pd; print(pd.read_csv('data/hourly_aggregated.csv'))\"")