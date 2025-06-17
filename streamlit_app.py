import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import numpy as np
import folium
import numpy as np
import folium
import geopy
from folium.plugins import AntPath
from geopy.distance import geodesic
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components
import shap




st.set_page_config(layout="wide")
st.title("📦 ETA Prediction & Route Optimization Dashboard")

# --- Load default file from /data
default_file_path = "data/final_dataframe.csv"
uploaded_file = st.file_uploader("📂 Upload a CSV file (optional)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Custom file uploaded and loaded.")
else:
    df = pd.read_csv(default_file_path)
    st.info(f"📌 Using default dataset: `{default_file_path}`")

# --- Show preview
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head(10))

# --- Model selection
model_choice = st.selectbox("🔧 Choose Prediction Model", ["LightGBM Quantile", "NGBoost"])

if model_choice == "LightGBM Quantile":
    st.markdown("**Loaded LightGBM Quantile Regression models...**")
    with open("models/model_10.pkl", "rb") as f: model_10 = pickle.load(f)
    with open("models/model_50.pkl", "rb") as f: model_50 = pickle.load(f)
    with open("models/model_90.pkl", "rb") as f: model_90 = pickle.load(f)

    features = [
        'urgency_mins', 'late_by_minutes', 'duration_minutes', 'distance_km',
        'route_duration_min', 'num_stops_before'
    ]

    X = df[features]
    df["eta_10"] = model_10.predict(X)
    df["eta_50"] = model_50.predict(X)
    df["eta_90"] = model_90.predict(X)

    st.success("✅ ETA predictions (10/50/90th percentiles) generated using LightGBM.")
    st.dataframe(df[['eta_10', 'eta_50', 'eta_90']].head(10))
    # --- Save output
    st.subheader("💾 Save Predictions")
    if st.button("Save as CSV"):
        output_path = "data/predicted_output.csv"
        df.to_csv(output_path, index=False)
        st.success(f"Saved to `{output_path}`")
    st.markdown("### 📈 SHAP Explainability (LightGBM)")

    explainer = shap.Explainer(model_50)
    shap_values = explainer(X)

    fig, ax = plt.subplots(figsize=(10, 6))

# Direct beeswarm plot to the current axes
    shap.plots.beeswarm(shap_values, show=False)

# Show in Streamlit
    st.pyplot(fig)
elif model_choice == "NGBoost":
    st.markdown("**Loaded NGBoost model...**")
    required_ngb_features = [
        'urgency_mins', 'late_by_minutes', 'duration_minutes', 'distance_km',
        'route_duration_min', 'num_stops_before', 'priority_level', 'order_hour',
        'weather_condition_Clear', 'traffic_level_Medium', 'weight_category_encoded_label',
        'vehicle_type_EV', 'location_cluster', 'cluster_avg_duration'
    ]

    missing_cols = [col for col in required_ngb_features if col not in df.columns]

    if missing_cols:
        st.error(f"❌ Cannot use NGBoost — missing columns: {', '.join(missing_cols)}")
        st.stop()
    else:
        with open("models/ngboost_model.pkl", "rb") as f: ngb_model = pickle.load(f)


        X = df[required_ngb_features]
        pred_dist = ngb_model.pred_dist(X)
        df["eta_mean"] = pred_dist.loc
        df["eta_std"] = pred_dist.scale
        df["eta_lower"] = df["eta_mean"] - 1.28 * df["eta_std"]
        df["eta_upper"] = df["eta_mean"] + 1.28 * df["eta_std"]

        st.success("✅ ETA predictions (mean ± std) generated using NGBoost.")
        st.dataframe(df[['eta_mean', 'eta_lower', 'eta_upper']].head(10))
        # --- Save output
        st.subheader("💾 Save Predictions")
        if st.button("Save as CSV"):
            output_path = "data/predicted_output.csv"
            df.to_csv(output_path, index=False)
            st.success(f"Saved to `{output_path}`")
        st.markdown("### 📊 Feature Importance (NGBoost)")
    
        # Average over distribution dimensions (loc, scale) to get a 1D importance score
        importances = ngb_model.feature_importances_
        if len(importances.shape) == 2:
             averaged_importance = importances.mean(axis=0)
        else:
             averaged_importance = importances

        importance_df = pd.DataFrame({
        "feature": required_ngb_features,
        "importance": averaged_importance
        }).sort_values("importance", ascending=True)  # Ascending for barh plot

    # --- Plot using explicit figure to avoid deprecation warning
        fig_ngb, ax_ngb = plt.subplots(figsize=(8, 6))
        ax_ngb.barh(importance_df["feature"], importance_df["importance"], color="darkcyan")
        ax_ngb.set_xlabel("Importance Score")
        ax_ngb.set_title("NGBoost Feature Importance (Avg across loc & scale)")
        st.pyplot(fig_ngb)


st.markdown("### 📊 ETA Interval Width Analysis")

# -- Compatible column inference
if "eta_10" in df.columns and "eta_90" in df.columns:
    df["eta_lower"] = df["eta_10"]
    df["eta_upper"] = df["eta_90"]
    df["eta_mean"] = df.get("eta_50", (df["eta_10"] + df["eta_90"]) / 2)

elif "eta_mean" in df.columns and "eta_std" in df.columns:
    df["eta_lower"] = df["eta_mean"] - 1.28 * df["eta_std"]
    df["eta_upper"] = df["eta_mean"] + 1.28 * df["eta_std"]

# -- Compute interval width and risk level
df["interval_width"] = df["eta_upper"] - df["eta_lower"]
df["uncertainty"] = df["interval_width"] / 2
df["risk_level"] = pd.cut(
    df["uncertainty"],
    bins=[-1, 5, 10, float('inf')],
    labels=["Low", "Medium", "High"]
)

# -- Layout: 2 columns side by side
col1, col2 = st.columns(2)

# -- Histogram of interval widths
with col1:
    st.markdown("#### Histogram: ETA Interval Width (mins)")
    fig1, ax1 = plt.subplots()
    ax1.hist(df["interval_width"], bins=20, color="skyblue", edgecolor="black")
    ax1.set_xlabel("Width (minutes)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("ETA Interval Width")
    st.pyplot(fig1)

# -- Bar chart for risk levels
with col2:
    st.markdown("#### Risk Level Counts")
    fig2, ax2 = plt.subplots()
    df["risk_level"].value_counts().sort_index().plot(kind="bar", color=["green", "orange", "red"], ax=ax2)
    ax2.set_title("Delivery Risk Level Distribution")
    ax2.set_ylabel("Number of Deliveries")
    st.pyplot(fig2)

st.markdown("### 🔟 Top 10 Riskiest Deliveries")
top_risky_df = df.sort_values("uncertainty", ascending=False).head(10)

# Format nicely for readability
display_cols = ["eta_lower", "eta_mean", "eta_upper", "interval_width", "uncertainty", "risk_level"]
if "stop_id" not in top_risky_df.columns:
    top_risky_df["stop_id"] = top_risky_df.index  # fallback if no explicit ID

st.dataframe(top_risky_df[["stop_id"] + display_cols].round(2).reset_index(drop=True))

import folium
from streamlit.components.v1 import html

st.markdown("### 🌍 Map: Medium & High Risk Deliveries")

# Filter only Medium and High risk deliveries
risky_subset = df[df["risk_level"].isin(["Medium", "High"])].copy()

# Pick top 100 most uncertain
top_risky = risky_subset.sort_values(by="uncertainty", ascending=False).head(100)

if not top_risky.empty:
    # Create folium map centered on first risky delivery
    risk_map = folium.Map(
        location=[top_risky.iloc[0]["delivery_lat"], top_risky.iloc[0]["delivery_lon"]],
        zoom_start=11
    )

    # Color-coding based on risk level
    color_map = {"Medium": "orange", "High": "red"}

    # Add markers
    for _, row in top_risky.iterrows():
        folium.CircleMarker(
            location=(row["delivery_lat"], row["delivery_lon"]),
            radius=6,
            color=color_map.get(row["risk_level"], "blue"),
            fill=True,
            fill_opacity=0.7,
            popup=f"ETA ± {row['uncertainty']:.1f} min<br>Risk: {row['risk_level']}",
            tooltip=f"{row['risk_level']} Risk"
        ).add_to(risk_map)

    # Render map in Streamlit
    from streamlit_folium import st_folium
    st_data = st_folium(risk_map, height=600,width=1700)
else:
    st.info("No medium or high risk deliveries found in this dataset.")

# ======================
# 🚚 Route Optimization (Pre-generated Maps)
# ======================
from streamlit.components.v1 import html as st_html
import os

st.subheader("🧭 Pre-Generated Route Visualization")

# Choose which map to load
map_choice = st.selectbox("🗺️ Choose Route Type", ["Geospatial Cluster Map","LightGBM - 5 Stops", "LightGBM - 6 Stops", "NGBoost - 5 Stops", "NGBoost - 6 Stops","LightGBM - 5 stops(risky marked)","LightGBM - 6 stops(risky marked)","NGBoost - 5 stops(risky marked)","NGBoost - 6 stops(risky marked)"])

map_file_dict = {
    "Geospatial Cluster Map":"maps/geospatial_clusters_map.html",
    "LightGBM - 5 Stops": "maps/route_map_lgbm_5 stops.html",
    "LightGBM - 6 Stops": "maps/route_map_lgbm_6 stops.html",
    "NGBoost - 5 Stops": "maps/ngboost_feasible_route_map_5stops.html",
    "NGBoost - 6 Stops": "maps/ngboost_feasible_route_map_6stops.html",
    "LightGBM - 5 stops(risky marked)":"maps/final_risk_route_map_lightgbm_5stops.html",
    "LightGBM - 6 stops(risky marked)":"maps/final_risk_route_map_lightgbm_6stops.html",
    "NGBoost - 5 stops(risky marked)":"maps/ngboost_route_with_risk_map_5stops.html",
    "NGBoost - 6 stops(risky marked)":"maps/ngboost_route_with_risk_map_6stops.html",
}

selected_map_path = map_file_dict[map_choice]

# Check existence and render
if os.path.exists(selected_map_path):
    with open(selected_map_path, "r", encoding="utf-8") as f:
        map_html = f.read()
    st.success(f"Showing: {map_choice}")
    st_html(map_html, height=600, scrolling=False)
else:
    st.error(f"❌ Map file not found: {selected_map_path}")
