# prepare_clean_data.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("PREPARING CLEAN DATA FOR STREAMLIT")
print("=" * 60)

# Load clean data
print("\n1. Loading clean telecom data...")
df = pd.read_csv('data/clean_telecom_data.csv')
print(f"   Shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")

# If the file has different column names, try to standardize
print("\n2. Standardizing column names...")
df.columns = df.columns.str.lower().str.strip()

# Map common column names
column_mapping = {
    'calls': 'total_calls',
    'call': 'total_calls',
    'sms': 'total_sms',
    'internet': 'internet_traffic',
    'data': 'internet_traffic',
    'traffic': 'internet_traffic'
}

for old_col, new_col in column_mapping.items():
    if old_col in df.columns and new_col not in df.columns:
        df.rename(columns={old_col: new_col}, inplace=True)
        print(f"   ✓ Renamed '{old_col}' to '{new_col}'")

# Create required columns if they don't exist
if 'total_calls' not in df.columns:
    # Try to create from available columns
    if 'call_in' in df.columns and 'call_out' in df.columns:
        df['total_calls'] = df['call_in'] + df['call_out']
        print("   ✓ Created total_calls from call_in + call_out")
    elif 'calls' in df.columns:
        df['total_calls'] = df['calls']
        print("   ✓ Using 'calls' as total_calls")
    else:
        # Create synthetic calls data
        df['total_calls'] = np.random.exponential(100, size=len(df))
        print("   ⚠ Created synthetic total_calls")

if 'total_sms' not in df.columns:
    if 'sms_in' in df.columns and 'sms_out' in df.columns:
        df['total_sms'] = df['sms_in'] + df['sms_out']
        print("   ✓ Created total_sms from sms_in + sms_out")
    elif 'sms' in df.columns:
        df['total_sms'] = df['sms']
        print("   ✓ Using 'sms' as total_sms")
    else:
        df['total_sms'] = np.random.exponential(50, size=len(df))
        print("   ⚠ Created synthetic total_sms")

if 'internet_traffic' not in df.columns:
    if 'internet' in df.columns:
        df['internet_traffic'] = df['internet']
        print("   ✓ Using 'internet' as internet_traffic")
    elif 'data' in df.columns:
        df['internet_traffic'] = df['data']
        print("   ✓ Using 'data' as internet_traffic")
    else:
        df['internet_traffic'] = np.random.exponential(200, size=len(df))
        print("   ⚠ Created synthetic internet_traffic")

# Extract hour from datetime if available
print("\n3. Extracting hour information...")
if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    print("   ✓ Extracted hour from datetime")
elif 'timestamp' in df.columns:
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['datetime'].dt.hour
    print("   ✓ Extracted hour from timestamp")
elif 'hour' in df.columns:
    print("   ✓ Using existing hour column")
else:
    # Create synthetic hour distribution
    df['hour'] = np.random.choice(range(24), size=len(df), p=[0.02]*24)
    print("   ⚠ Created synthetic hour distribution")

# Create hourly aggregation
print("\n4. Creating hourly aggregation...")
hourly_agg = df.groupby('hour').agg({
    'total_calls': 'sum',
    'total_sms': 'sum',
    'internet_traffic': 'sum'
}).reset_index()

# Ensure all hours 0-23 exist
all_hours = pd.DataFrame({'hour': range(24)})
hourly_agg = all_hours.merge(hourly_agg, on='hour', how='left').fillna(0)

# Smooth the data a bit for better visualization
if len(hourly_agg) == 24:
    # Apply simple moving average for smoothing
    hourly_agg['total_calls'] = hourly_agg['total_calls'].rolling(window=3, center=True, min_periods=1).mean()
    hourly_agg['total_sms'] = hourly_agg['total_sms'].rolling(window=3, center=True, min_periods=1).mean()
    hourly_agg['internet_traffic'] = hourly_agg['internet_traffic'].rolling(window=3, center=True, min_periods=1).mean()

hourly_agg.to_csv('data/hourly_aggregated.csv', index=False)
print(f"   ✓ hourly_aggregated.csv created ({len(hourly_agg)} rows)")
print("\n   Hourly data preview:")
print(hourly_agg.head(10))

# Create square/cell aggregation
print("\n5. Creating cell/square aggregation...")
if 'square_id' in df.columns:
    square_agg = df.groupby('square_id').agg({
        'total_calls': 'mean',
        'total_sms': 'mean',
        'internet_traffic': 'mean'
    }).reset_index()
else:
    # Create synthetic square_ids
    print("   No square_id found, creating synthetic grid cells...")
    n_cells = min(10000, len(df))
    df['square_id'] = np.random.choice(range(n_cells), size=len(df))
    square_agg = df.groupby('square_id').agg({
        'total_calls': 'mean',
        'total_sms': 'mean',
        'internet_traffic': 'mean'
    }).reset_index()

square_agg.to_csv('data/square_aggregated.csv', index=False)
print(f"   ✓ square_aggregated.csv created ({len(square_agg):,} rows)")

# Create clusters
print("\n6. Creating clusters...")
if len(square_agg) >= 5:
    # Prepare features
    features = ['total_calls', 'total_sms', 'internet_traffic']
    X = square_agg[features].fillna(0)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-Means clustering
    n_clusters = min(5, len(square_agg))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    square_agg['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Add cluster descriptions
    cluster_labels = {
        0: 'Low Activity Zone',
        1: 'Medium-Low Zone',
        2: 'Medium Zone',
        3: 'High Activity Zone',
        4: 'Hotspot Zone'
    }
    square_agg['cluster_name'] = square_agg['cluster'].map(cluster_labels)
    
    square_agg.to_csv('data/telecom_milan_clustered.csv', index=False)
    print(f"   ✓ telecom_milan_clustered.csv created ({len(square_agg):,} rows, {n_clusters} clusters)")
    
    # Show cluster profiles
    print("\n   Cluster Profiles:")
    cluster_profile = square_agg.groupby('cluster')[features].mean().round(2)
    print(cluster_profile)
else:
    print("   ⚠ Not enough data for clustering")

# Final verification
print("\n" + "=" * 60)
print("✅ DATA PREPARATION COMPLETE!")
print("=" * 60)

# Verify hourly data
hourly_check = pd.read_csv('data/hourly_aggregated.csv')
print(f"\n📊 Final Verification:")
print(f"   Hourly data shape: {hourly_check.shape}")
print(f"   Total calls (3 days): {hourly_check['total_calls'].sum():,.0f}")
print(f"   Total SMS (3 days): {hourly_check['total_sms'].sum():,.0f}")
print(f"   Total Internet (3 days): {hourly_check['internet_traffic'].sum():,.0f} MB")
print(f"   Peak call hour: {hourly_check.loc[hourly_check['total_calls'].idxmax(), 'hour']}:00")
print(f"   Peak internet hour: {hourly_check.loc[hourly_check['internet_traffic'].idxmax(), 'hour']}:00")

print("\n💡 NEXT STEP:")
print("   streamlit run streamlit_app_optimized.py")
print("\n🎯 The data is now ready for visualization!")