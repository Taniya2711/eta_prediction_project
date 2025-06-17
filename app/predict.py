
import pandas as pd
import numpy as np
import pickle
import os

# --- Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "final_dataframe.csv")
MODEL_10_PATH = os.path.join(BASE_DIR, "models", "model_10.pkl")
MODEL_50_PATH = os.path.join(BASE_DIR, "models", "model_50.pkl")
MODEL_90_PATH = os.path.join(BASE_DIR, "models", "model_90.pkl")
NGB_PATH = os.path.join(BASE_DIR, "models", "ngboost_model.pkl")

# --- Load Data
df = pd.read_csv(DATA_PATH)

# --- LightGBM Quantile Regression Prediction
features_lgb = [
    'urgency_mins', 'late_by_minutes', 'duration_minutes', 'distance_km',
    'route_duration_min', 'num_stops_before'
]

X_lgb = df[features_lgb]

# Load models
with open(MODEL_10_PATH, 'rb') as f: model_10 = pickle.load(f)
with open(MODEL_50_PATH, 'rb') as f: model_50 = pickle.load(f)
with open(MODEL_90_PATH, 'rb') as f: model_90 = pickle.load(f)

# Predict
df["eta_10"] = model_10.predict(X_lgb)
df["eta_50"] = model_50.predict(X_lgb)
df["eta_90"] = model_90.predict(X_lgb)

print("✅ LightGBM ETA prediction completed.")

# --- NGBoost Prediction
features_ngb = [
    'urgency_mins', 'late_by_minutes', 'duration_minutes', 'distance_km',
    'route_duration_min', 'num_stops_before', 'priority_level', 'order_hour',
    'weather_condition_Clear', 'traffic_level_Medium', 'weight_category_encoded_label',
    'vehicle_type_EV', 'location_cluster', 'cluster_avg_duration'
]

X_ngb = df[features_ngb]

with open(NGB_PATH, 'rb') as f:
    ngb_model = pickle.load(f)

pred_dist = ngb_model.pred_dist(X_ngb)
df["eta_mean"] = pred_dist.loc
df["eta_std"] = pred_dist.scale

print("✅ NGBoost ETA prediction completed.")

# --- Save Combined Output
output_path = os.path.join(BASE_DIR, "data", "predicted_eta_df.csv")
df.to_csv(output_path, index=False)
print(f"📦 Saved combined ETA predictions to: {output_path}")
