import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# 1. Load Data
print("Loading data...")
df_2023 = pd.read_csv('josaa_2023_all_institutes_full.csv')
df_2024 = pd.read_csv('josaa_2024_reparsed_all_institutes.csv')
df_2025 = pd.read_csv('josaa_2025_opening_closing_ranks_all_institutes.csv')

df_2023['Year'] = 2023
df_2024['Year'] = 2024
df_2025['Year'] = 2025

df = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)

# 2. Preprocessing
print("Preprocessing...")
target = 'Closing Rank'
df[target] = df[target].astype(str).str.replace('P', '')
df[target] = pd.to_numeric(df[target], errors='coerce')
df.dropna(subset=[target], inplace=True)
df[target] = df[target].astype(int)

features = ['Institute', 'Academic Program Name', 'Quota', 'Seat Type', 'Gender', 'Year']
X = df[features].copy()
y = df[target]

# 3. Label Encoding (This is what makes the model small!)
encoders = {}
for col in ['Institute', 'Academic Program Name', 'Quota', 'Seat Type', 'Gender']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

import joblib
import os

# ... (previous loading and encoding steps same as before) ...

# 4. Train Pruned Random Forest (to reduce size)
print("Training Optimized Random Forest...")
model = RandomForestRegressor(
    n_estimators=50,       # Fewer trees
    max_depth=20,          # Limit depth to shrink size
    min_samples_leaf=10,   # Increase leaf size to prune
    random_state=42, 
    n_jobs=-1
)
model.fit(X, y)

# 5. Save Model and Encoders using Compression
print("Saving compressed model with joblib...")
joblib.dump(model, 'college_predictor_compressed.joblib', compress=3)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

size_mb = os.path.getsize('college_predictor_compressed.joblib') / (1024 * 1024)
print(f"Success! New model size: {size_mb:.2f} MB")
