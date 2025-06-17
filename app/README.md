📦 ETA Prediction & Route Optimization Dashboard
🚀 Overview
This project simulates a real-world last-mile delivery ETA prediction and routing system, incorporating:

Basic Machine Learnign Models(Linear and Ridge Regression, Random Forest Regressor, XGBoost)

Advanced Machine Learning Models (LightGBM Quantile Regression & NGBoost)

Uncertainty Quantification through confidence intervals / standard deviation

Interactive Streamlit Dashboard for:

ETA prediction

Risk analysis

Route optimization using Google OR-Tools

Visualizations on Folium maps

This project was built from scratch using synthetic data due to the unavailability of suitable real-world datasets and is designed to demonstrate robust Data Science, Optimization, and Software Engineering skills in one unified pipeline.

💡 Features

| Component             | Description                                                    |
| --------------------- | -------------------------------------------------------------- |
| 🔮 ETA Prediction     | LightGBM Quantile Regression & NGBoost with interval estimates |
| 📊 Risk Analysis      | ETA width-based risk classification (Low / Medium / High)      |
| 🧭 Route Optimization | Feasible routes under time window constraints using OR-Tools   |
| 🌍 Visual Mapping     | Delivery risk hotspots + route paths using Folium              |
| 🧠 Explainability     | SHAP plots for model interpretation                            |
| 📦 Deployment Ready   | Streamlit UI + modular structure for scaling to API deployment |


🧰 Technologies Used
Python (Pandas, NumPy, Scikit-learn, LightGBM, NGBoost)

OR-Tools for VRPTW

Folium + Geopy for geospatial routing

SHAP for model explainability

Streamlit for UI

Docker & Git (optional for deployment)

VS Code / Google Colab for development

📦 How to Run Locally
1. Clone the repo:
git clone https://github.com/Taniya2711/eta-prediction-optimizer.git
cd eta-prediction-optimizer

2. Create a virtual environment and activate it:
python -m venv venv
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # macOS/Linux

3. Install dependencies:
pip install -r requirements.txt

4. Run Streamlit app:
streamlit run streamlit_app.py