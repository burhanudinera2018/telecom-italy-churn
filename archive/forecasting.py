# forecasting.py (UPDATED)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error

print("=" * 60)
print("🔮 TRAFFIC FORECASTING - TIME SERIES")
print("=" * 60)

# Load hourly aggregated data
df = pd.read_csv('data/telecom_milan_traffic.csv')

# Aggregate per hour (across all grid cells)
hourly_total = df.groupby('hour')['total_traffic'].sum().reset_index()
hourly_total['hour'] = pd.to_datetime(hourly_total['hour'], format='%H').dt.hour

# Create time series (72 hours dari 3 hari)
# Kita punya 3 hari x 24 jam = 72 jam
# Data di-group by hour, perlu di-expand ke 72 jam

# Buat time index untuk 72 jam
dates = pd.date_range('2013-11-01 00:00:00', periods=72, freq='h')
hourly_ts = pd.DataFrame({'datetime': dates})

# Aggregate total traffic per hour dari semua grid cells
hourly_traffic = df.groupby('hour')['total_traffic'].sum().values

# Repeat pattern untuk 72 jam (3 hari)
hourly_traffic_72h = np.tile(hourly_traffic, 3)
hourly_ts['total_traffic'] = hourly_traffic_72h

print(f"📊 Time series: {len(hourly_ts)} hours")

# Train-test split (48 train, 24 test)
train = hourly_ts.iloc[:48]
test = hourly_ts.iloc[48:72]

print(f"📊 Train: {len(train)} jam")
print(f"📊 Test: {len(test)} jam")

# Fit model
model = ExponentialSmoothing(
    train['total_traffic'],
    seasonal_periods=24,
    trend='add',
    seasonal='add'
)
fitted = model.fit()
forecast = fitted.forecast(24)

# Calculate MAPE
mape = mean_absolute_percentage_error(test['total_traffic'], forecast)
print(f"\n📈 MAPE: {mape*100:.1f}%")

# Plot
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(train['datetime'], train['total_traffic'] / 1e6, label='Training', color='blue')
ax.plot(test['datetime'], test['total_traffic'] / 1e6, label='Actual', color='green', linewidth=2)
ax.plot(test['datetime'], forecast / 1e6, 'r--', label=f'Forecast (MAPE={mape*100:.1f}%)', linewidth=2)
ax.set_xlabel('DateTime')
ax.set_ylabel('Total Traffic (Millions)')
ax.set_title('Telecom Milan Traffic Forecasting')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('telecom_milan_forecast.png', dpi=150)
print("✅ Saved: telecom_milan_forecast.png")
plt.show()