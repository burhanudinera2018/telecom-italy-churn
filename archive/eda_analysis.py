# eda_analysis.py (UPDATED untuk data yang sudah siap)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("📊 EXPLORATORY DATA ANALYSIS - TELECOM MILAN")
print("Dataset: 3 days traffic (10,000 grid cells, 24 hours)")
print("=" * 60)

# Load aggregated traffic data
df = pd.read_csv('data/telecom_milan_traffic.csv')
print(f"✅ Data loaded: {len(df):,} records")
print(f"   Unique cells: {df['square_id'].nunique():,}")
print(f"   Hours: {df['hour'].min()}-{df['hour'].max()}")

# ============================================
# 1. HOURLY TRAFFIC PATTERNS
# ============================================
print("\n" + "=" * 40)
print("1. HOURLY TRAFFIC PATTERNS")
print("=" * 40)

hourly = df.groupby('hour').agg({
    'total_sms': 'sum',
    'total_calls': 'sum',
    'internet_traffic_mb': 'sum'
}).reset_index()

print("Peak hours:")
peak_sms = hourly['total_sms'].idxmax()
peak_calls = hourly['total_calls'].idxmax()
peak_internet = hourly['internet_traffic_mb'].idxmax()
print(f"  📱 Peak SMS: jam {peak_sms}:00 ({hourly.loc[peak_sms, 'total_sms']:,.0f} SMS)")
print(f"  📞 Peak Calls: jam {peak_calls}:00 ({hourly.loc[peak_calls, 'total_calls']:,.0f} calls)")
print(f"  🌐 Peak Internet: jam {peak_internet}:00 ({hourly.loc[peak_internet, 'internet_traffic_mb']:,.0f} MB)")

# ============================================
# 2. TOP BUSIEST GRID CELLS
# ============================================
print("\n" + "=" * 40)
print("2. BUSIEST LOCATIONS (Top 10 Grid Cells)")
print("=" * 40)

cell_traffic = df.groupby('square_id').agg({
    'total_traffic': 'sum',
    'total_sms': 'sum',
    'total_calls': 'sum',
    'internet_traffic_mb': 'sum'
}).sort_values('total_traffic', ascending=False)

print("Top 5 busiest grid cells:")
for i, (cell, row) in enumerate(cell_traffic.head(5).iterrows(), 1):
    print(f"  {i}. Cell {cell}: {row['total_traffic']:,.0f} total traffic")

# ============================================
# 3. TRAFFIC DISTRIBUTION
# ============================================
print("\n" + "=" * 40)
print("3. SERVICE DISTRIBUTION (3 Days Total)")
print("=" * 40)

total_sms = df['total_sms'].sum()
total_calls = df['total_calls'].sum()
total_internet_mb = df['internet_traffic_mb'].sum()
total_internet_gb = total_internet_mb / 1024

total_all = total_sms + total_calls + total_internet_mb

print(f"  📱 SMS: {total_sms:,.0f} ({total_sms/total_all*100:.1f}%)")
print(f"  📞 Calls: {total_calls:,.0f} ({total_calls/total_all*100:.1f}%)")
print(f"  🌐 Internet: {total_internet_gb:,.0f} GB ({total_internet_mb/total_all*100:.1f}%)")

# ============================================
# 4. VISUALIZATIONS
# ============================================
print("\n📊 Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Telecom Milan Network Traffic Analysis (Nov 1-3, 2013)', fontsize=14, fontweight='bold')

# Plot 1: Hourly Patterns
ax1 = axes[0, 0]
ax1.plot(hourly['hour'], hourly['total_calls'], 'o-', label='Calls', color='#3498db', linewidth=2)
ax1.plot(hourly['hour'], hourly['total_sms'] / 1000, 's-', label='SMS (ribuan)', color='#e74c3c', linewidth=2)
ax1.set_xlabel('Hour of Day')
ax1.set_ylabel('Volume')
ax1.set_title('Hourly Network Activity')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Internet Traffic by Hour
ax2 = axes[0, 1]
ax2.bar(hourly['hour'], hourly['internet_traffic_mb'] / 1024, color='#2ecc71')
ax2.set_xlabel('Hour of Day')
ax2.set_ylabel('Internet Traffic (GB)')
ax2.set_title('Hourly Internet Traffic')
ax2.grid(True, alpha=0.3)

# Plot 3: Top 10 Busiest Hours
ax3 = axes[1, 0]
top_hours = hourly.nlargest(5, 'total_calls')
ax3.barh([f"{h}:00" for h in top_hours['hour']], top_hours['total_calls'], color='#3498db')
ax3.set_xlabel('Total Calls')
ax3.set_title('Top 5 Busiest Hours')
ax3.invert_yaxis()

# Plot 4: Distribution Pie
ax4 = axes[1, 1]
sizes = [total_sms, total_calls, total_internet_mb]
labels = ['SMS', 'Calls', 'Internet (MB)']
colors = ['#e74c3c', '#3498db', '#2ecc71']
ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax4.set_title('Total Traffic Distribution (3 Days)')

plt.tight_layout()
plt.savefig('telecom_milan_eda.png', dpi=150, bbox_inches='tight')
print("✅ Saved: telecom_milan_eda.png")
plt.show()

print("\n" + "=" * 60)
print("✅ EDA COMPLETE!")
print("=" * 60)