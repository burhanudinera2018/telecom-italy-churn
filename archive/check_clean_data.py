# check_clean_data.py
import pandas as pd

print("=" * 60)
print("CHECKING CLEAN TELECOM DATA")
print("=" * 60)

# Load file yang baru didownload
df = pd.read_csv('data/clean_telecom_data.csv')

print(f"\n✅ File loaded successfully!")
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")

print(f"\nFirst 5 rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nBasic statistics:")
print(df.describe())

# Check if we have the columns we need
required_cols = ['total_calls', 'total_sms', 'internet_traffic']
available_cols = [col for col in required_cols if col in df.columns]

print(f"\nAvailable required columns: {available_cols}")

if available_cols:
    print("\n✅ Data is valid and ready to use!")
else:
    print("\n⚠️ Column names might be different. Showing all columns:")
    print(df.columns.tolist())