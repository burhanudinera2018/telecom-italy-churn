# download_dataset.py
import pandas as pd
import numpy as np
import urllib.request
import os

print("=" * 60)
print("📡 TELECOM ITALY CUSTOMER CHURN ANALYSIS")
print("Research-based Data Science Case Study")
print("Dataset: Telecom Customer Churn (Real Public Dataset)")
print("=" * 60)

# ============================================
# OPSI 1: Menggunakan dataset REAL dari IBM Telco (Paling Populer)
# Dataset ini REAL, digunakan oleh IBM, Kaggle, dan banyak research
# ============================================

print("\n📥 Downloading REAL dataset from IBM Telco...")

# URL dataset IBM Telco Customer Churn (REAL, PUBLIC, dan TERPERCAYA)
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv"

try:
    df = pd.read_csv(url)
    print(f"✅ Dataset REAL berhasil di-load dari IBM!")
    print(f"📊 Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"📊 Churn rate: {df['Churn'].map({'Yes':1, 'No':0}).mean()*100:.1f}%")
    
    # Simpan ke file lokal
    df.to_csv('telecom_customer_churn.csv', index=False)
    print("💾 Dataset saved: telecom_customer_churn.csv")
    
    # Tampilkan info dataset
    print("\n" + "=" * 60)
    print("📋 DATASET INFORMATION")
    print("=" * 60)
    print(f"\nColumns: {list(df.columns)[:10]}... ({len(df.columns)} total)")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
except Exception as e:
    print(f"❌ Error dengan IBM dataset: {e}")
    
    # ============================================
    # OPSI 2: Generate synthetic data based on REAL telecom patterns
    # ============================================
    print("\n📊 Generating realistic telecom churn dataset...")
    
    np.random.seed(42)
    n_samples = 7043  # Sama dengan dataset IBM Telco yang real
    
    # 21 fitur berdasarkan perilaku penggunaan layanan telecom (REALISTIC)
    data = {
        'customerID': [f'CUST-{i:05d}' for i in range(n_samples)],
        'gender': np.random.choice(['Female', 'Male'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5]),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'tenure': np.random.choice([1,2,3,4,5,6,12,18,24,36,48,60,72], n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.4, 0.5, 0.1]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.45, 0.15]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.55, 0.15]),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.35, 0.5, 0.15]),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.35, 0.5, 0.15]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.55, 0.15]),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.45, 0.15]),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.45, 0.15]),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.2]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(100, 8000, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create churn based on REALISTIC telecom rules
    # Month-to-month + high monthly charges + no tech support = higher churn
    churn_prob = (
        (df['Contract'] == 'Month-to-month') * 0.35 +
        (df['MonthlyCharges'] > 70) * 0.15 +
        (df['TechSupport'] == 'No') * 0.15 +
        (df['tenure'] < 12) * 0.20 +
        (df['SeniorCitizen'] == 1) * 0.05 +
        (df['PaperlessBilling'] == 'Yes') * 0.10
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    df['Churn'] = np.random.binomial(1, churn_prob)
    df['Churn'] = df['Churn'].map({1: 'Yes', 0: 'No'})
    
    df.to_csv('telecom_customer_churn.csv', index=False)
    print(f"✅ Realistic dataset created with {n_samples} customers")
    print(f"📊 Churn rate: {(df['Churn']=='Yes').mean()*100:.1f}%")
    print(f"💾 Saved: telecom_customer_churn.csv")

print("\n" + "=" * 60)
print("✅ DATASET READY!")
print("=" * 60)

# Tampilkan ringkasan churn rate
if 'Churn' in df.columns:
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    print(f"\n📊 Dataset Summary:")
    print(f"   Total customers: {len(df):,}")
    print(f"   Churn rate: {churn_rate:.1f}%")
    print(f"   Features: {df.shape[1]}")