# clustering.py (UPDATED)
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("=" * 60)
print("🎯 NETWORK CLUSTERING - K-MEANS")
print("=" * 60)

df = pd.read_csv('data/telecom_milan_traffic.csv')

# Aggregasi per grid cell
cell_features = df.groupby('square_id').agg({
    'total_sms': 'mean',
    'total_calls': 'mean',
    'internet_traffic_mb': 'mean'
}).reset_index()

print(f"📊 Analyzing {len(cell_features)} grid cells")

# Features untuk clustering
features = ['total_sms', 'total_calls', 'internet_traffic_mb']
X = cell_features[features]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal K
inertias = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cell_features['cluster'] = kmeans.fit_predict(X_scaled)

# Cluster profiles
print("\n📊 CLUSTER PROFILES:")
for cluster in range(optimal_k):
    profile = cell_features[cell_features['cluster'] == cluster][features].mean()
    print(f"\n  Cluster {cluster}:")
    print(f"    Avg SMS: {profile['total_sms']:.1f}")
    print(f"    Avg Calls: {profile['total_calls']:.1f}")
    print(f"    Avg Internet: {profile['internet_traffic_mb']:.0f} MB")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow
ax1 = axes[0]
ax1.plot(K_range, inertias, 'o-', color='#3498db')
ax1.axvline(x=optimal_k, color='red', linestyle='--', label=f'K={optimal_k}')
ax1.set_xlabel('K')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')
ax1.legend()

# Distribution
ax2 = axes[1]
counts = cell_features['cluster'].value_counts().sort_index()
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
ax2.bar(counts.index, counts.values, color=colors)
ax2.set_xlabel('Cluster')
ax2.set_ylabel('Grid Cells')
ax2.set_title(f'{optimal_k} Network Segments')

plt.tight_layout()
plt.savefig('telecom_milan_clusters.png', dpi=150)
print("\n✅ Saved: telecom_milan_clusters.png")
plt.show()

cell_features.to_csv('data/telecom_milan_clustered.csv', index=False)
print("✅ Saved: data/telecom_milan_clustered.csv")