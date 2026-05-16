# RUN IN TERMINAL DULU (bukan di streamlit)
# Buat file data_aggregated.py

import pandas as pd

print("Loading data...")
df = pd.read_csv('data/telecom_milan_parsed.csv', parse_dates=['datetime'])

print("Aggregating by hour...")
# Agregasi per hour (dari 14juta jadi 24 baris!)
hourly_agg = df.groupby(df['datetime'].dt.hour).agg({
    'total_calls': 'sum',
    'total_sms': 'sum', 
    'internet_traffic': 'sum'
}).reset_index()
hourly_agg.columns = ['hour', 'total_calls', 'total_sms', 'internet_traffic']
hourly_agg.to_csv('data/hourly_aggregated.csv', index=False)

print("Aggregating by square_id...")
# Agregasi per square (dari 14juta jadi ~10rb baris)
square_agg = df.groupby('square_id').agg({
    'total_calls': 'mean',
    'total_sms': 'mean',
    'internet_traffic': 'mean'
}).reset_index()
square_agg.to_csv('data/square_aggregated.csv', index=False)

print("Done! Data size reduced from 14M to small files")