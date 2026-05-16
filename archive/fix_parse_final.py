# fix_parse_final.py
import pandas as pd
import numpy as np
import re
from datetime import datetime
import time

print("=" * 60)
print("FINAL FIXED PARSING - TELECOM MILAN DATA")
print("=" * 60)

start_time = time.time()

# Baca file dengan separator yang benar (whitespace)
print("\n1. Loading data with proper separator...")
df = pd.read_csv('data/telecom_milan_parsed.csv', 
                 sep='\s+',  # Whitespace separator (space/tab)
                 header=0,
                 engine='python',
                 encoding='utf-8')

print(f"   ✓ Loaded {len(df):,} rows")
print(f"   ✓ Columns: {len(df.columns)}")

# Cek nama kolom
print(f"\n2. Columns found: {df.columns.tolist()}")

# Rename columns if needed
if len(df.columns) == 13:
    df.columns = ['square_id', 'timestamp', 'sms_in', 'sms_out', 'call_in', 
                  'call_out', 'internet_traffic', 'datetime', 'date', 'hour', 
                  'total_calls', 'total_sms', 'total_traffic']
    print("   ✓ Columns renamed successfully")
else:
    # If we have only one column, need to split manually
    if len(df.columns) == 1:
        print("   ⚠ Data in single column, splitting manually...")
        
        # Get the column name
        col_name = df.columns[0]
        
        # Split the single column into multiple columns
        # The data appears to be space-separated
        split_data = df[col_name].str.split('\s+', expand=True)
        
        # Assign column names
        column_names = ['square_id', 'timestamp', 'sms_in', 'sms_out', 'call_in', 
                       'call_out', 'internet_traffic', 'datetime', 'date', 'hour', 
                       'total_calls', 'total_sms', 'total_traffic']
        
        # Make sure we have the right number of columns
        if split_data.shape[1] >= len(column_names):
            df = split_data.iloc[:, :len(column_names)]
            df.columns = column_names
            print(f"   ✓ Split into {len(df.columns)} columns")
        else:
            print(f"   ⚠ Got {split_data.shape[1]} columns, expected 13")

# Convert data types
print("\n3. Converting data types...")

# Convert square_id to string
df['square_id'] = df['square_id'].astype(str)

# Convert timestamp to datetime (assuming milliseconds)
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
    print("   ✓ Timestamp converted to datetime")

# Convert numeric columns
numeric_cols = ['sms_in', 'sms_out', 'call_in', 'call_out', 'internet_traffic', 
                'total_calls', 'total_sms', 'total_traffic']

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        print(f"   ✓ {col}")

# If total_calls doesn't exist, create it
if 'total_calls' not in df.columns or df['total_calls'].isnull().all():
    if 'call_in' in df.columns and 'call_out' in df.columns:
        df['total_calls'] = df['call_in'] + df['call_out']
        print("   ✓ Created total_calls from call_in + call_out")

if 'total_sms' not in df.columns or df['total_sms'].isnull().all():
    if 'sms_in' in df.columns and 'sms_out' in df.columns:
        df['total_sms'] = df['sms_in'] + df['sms_out']
        print("   ✓ Created total_sms from sms_in + sms_out")

# Handle missing values
print("\n4. Handling missing values...")
print(f"   Missing before: {df.isnull().sum().sum():,}")
df = df.fillna(0)
print(f"   Missing after: {df.isnull().sum().sum():,}")

# Extract hour if not already present
if 'hour' in df.columns:
    # Convert hour to int if it's numeric
    df['hour'] = pd.to_numeric(df['hour'], errors='coerce').fillna(0).astype(int)
else:
    if 'datetime' in df.columns:
        df['hour'] = df['datetime'].dt.hour.fillna(0).astype(int)
    else:
        df['hour'] = 0

print(f"\n5. Hour range: {df['hour'].min()} to {df['hour'].max()}")

# Verify data
print("\n6. Data verification:")
print(f"   Total calls: {df['total_calls'].sum():,.0f}")
print(f"   Total SMS: {df['total_sms'].sum():,.0f}")
print(f"   Total internet: {df['internet_traffic'].sum():,.0f} MB")

# Check if data is valid
if df['total_calls'].sum() > 0:
    print("\n   ✅ DATA IS VALID!")
    
    # Show sample
    print("\n   Sample data (first 5 rows):")
    print(df[['square_id', 'total_calls', 'total_sms', 'internet_traffic']].head())
    
else:
    print("\n   ⚠ Data might still have issues. Showing raw sample:")
    print(df.head(10))
    
    # Try to understand the data format better
    print("\n   First row raw values:")
    for col in df.columns[:7]:  # First 7 columns
        print(f"   {col}: {df[col].iloc[0]}")

# Create aggregated data for Streamlit
print("\n7. Creating aggregated files for Streamlit...")

# Hourly aggregation
if df['total_calls'].sum() > 0:
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
    
    # Square aggregation (use sample for performance)
    print("\n8. Creating square aggregation...")
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
    print("\n9. Creating cluster analysis...")
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
    
    # Add cluster labels
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
    
else:
    print("\n   ⚠ No valid data found. Please check the input file format.")
    print("   Trying alternative parsing method...")
    
    # Alternative: read as CSV with different parameters
    df_alt = pd.read_csv('data/telecom_milan_parsed.csv', 
                         sep=None,  # Auto-detect separator
                         engine='python',
                         header=0)
    
    print(f"\n   Alternative parse result:")
    print(f"   Shape: {df_alt.shape}")
    print(f"   Columns: {df_alt.columns.tolist()}")
    print(f"   First row: {df_alt.iloc[0].tolist()}")

# Final summary
end_time = time.time()
print("\n" + "=" * 60)
print("✅ PROCESSING COMPLETE!")
print("=" * 60)
print(f"⏱️  Time taken: {(end_time - start_time):.2f} seconds")

if 'hourly_agg' in locals() and len(hourly_agg) > 0 and hourly_agg['total_calls'].sum() > 0:
    print("\n📊 FINAL VERIFICATION:")
    print(f"   Hourly data shape: {hourly_agg.shape}")
    print(f"   Total calls: {hourly_agg['total_calls'].sum():,.0f}")
    print(f"   Total SMS: {hourly_agg['total_sms'].sum():,.0f}")
    print(f"   Total internet: {hourly_agg['internet_traffic'].sum():,.0f} MB")
    print(f"   Peak hour: {hourly_agg.loc[hourly_agg['total_calls'].idxmax(), 'hour']}:00")
    
    print("\n💡 NEXT STEP:")
    print("   Run: streamlit run streamlit_app_optimized.py")
else:
    print("\n❌ DATA PARSING FAILED")
    print("Please check the file format using:")
    print("   head -20 data/telecom_milan_parsed.csv | cat -A")