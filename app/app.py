import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv, find_dotenv

load_dotenv()
HOPSWORK_API_KEY = os.environ.get("HOPSWORK_API_KEY")

# --- Page Config ---
# Set the title and layout of the Streamlit dashboard
st.set_page_config(page_title="AQI Predictor Dashboard", layout="wide")

st.title("🏭 Air Quality Index (AQI) Predictor")
st.write("Visualizing AQI Forecasts using Hopsworks Feature Store & Model Registry.")

# --- Helper Functions ---

@st.cache_resource
def get_hopsworks_project():
    """
    Connect to Hopsworks Project.
    Uses @st.cache_resource to cache the connection object and avoid reconnecting on every rerun.
    """
    project = hopsworks.login(
        project='air_quality_predictor',
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORK_API_KEY
    )
    return project

def get_best_model(project):
    """
    Fetch the best performing model from the Hopsworks Model Registry.
    Criteria: Highest R2 score on training metrics.
    """
    mr = project.get_model_registry()
    
    # List of models to check in the registry
    model_names = ["random_forest_aqi_model", "gradient_boosting_aqi_model", "ridge_aqi_model"]
    
    best_r2 = -float('inf')
    best_model_meta = None
    best_model_name = ""
    models_data = []

    # Iterate through each model name
    for name in model_names:
        try:
            # Get model objects from registry
            models = mr.get_models(name)
            if not models: continue
            
            # Sort by version (descending) to get the latest version
            model = sorted(models, key=lambda m: m.version, reverse=True)[0]
            
            # Extract training metrics (R2, RMSE, MAE)
            metrics = model.training_metrics
            r2 = metrics.get("R2", -1)
            
            # Append metrics to list for display
            models_data.append({
                "Model Name": name,
                "Version": model.version,
                "RMSE": metrics.get("RMSE", 999),
                "MAE": metrics.get("MAE", 999),
                "R2": r2
            })
            
            # Check if this model is the best so far
            if r2 > best_r2:
                best_r2 = r2
                best_model_meta = model
                best_model_name = name
        except Exception as e:
            st.warning(f"Error fetching {name}: {e}")

    # Return the metrics DataFrame and the metadata of the best model
    return pd.DataFrame(models_data), best_model_meta, best_model_name

def fetch_data_from_store(project, days_past=14):
    """
    Fetch historical data from Hopsworks Feature Store.
    We need enough history to calculate rolling means (buffer of 7 days).
    """
    fs = project.get_feature_store()
    try:
        # Get the Feature Group
        fg = fs.get_feature_group(name="aqi_features", version=1)
        print("Feature Group retrieved successfully")
    except:
        st.error("Feature Group not found!")
        return None
        
    # Read data from the Feature Group
    # Try reading from Online Store first (low latency), fallback to Offline
    # Try reading from Online Store first (low latency)
    try:
        df = fg.read(online=True)
        print("data retrieved successfully")
    except Exception as e:
        st.warning(f"Feature Store Online retrieval failed: {e}")
        st.write("Retrying in 2 seconds...")
        time.sleep(2)
        try:
            df = fg.read(online=False)
            print("data retrieved successfully")
        except Exception as e:
            st.warning(f"Feature Store Offline retrieval failed: {e}")
            return None

        
    if df.empty:
        return None

    print(f"Data retrieved: {len(df)} rows.")

    # 5. Preprocess
    # Sort by timestamp to ensure time-based splitting if we wanted time-series split
    df = df.sort_values("timestamp")
    daily_aqi = df.groupby(['year', 'month', 'day'])[['day_of_week','weekend','aqi', 'pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone', 'sulphur_dioxide', 'carbon_monoxide']].mean().reset_index()
    
    # Adding 'date' column for plotting and logic
    daily_aqi['date'] = pd.to_datetime(daily_aqi[['year', 'month', 'day']]).dt.date


    # adding lag features
    cols = [
    'aqi', 'pm2_5', 'pm10', 'nitrogen_dioxide', 
    'ozone', 'sulphur_dioxide', 'carbon_monoxide'
    ]
    for col in cols:
        daily_aqi[f'{col}_lag1'] = daily_aqi[col].shift(1)
        
    df_lagged=daily_aqi.dropna()
    # df_lagged=df_lagged.drop(columns=["pm2_5","pm10","nitrogen_dioxide","ozone","sulphur_dioxide","carbon_monoxide"])

    return df_lagged

def predict_next_3_days(model, df_lagged):
    """
    Predict AQI for next 3 days using recursive approach with persistence for exogenous variables.
    """
        
    # We will simulate the "Next Day" loop
    # We start with existing history
    # Predictions storage
    future_dates = []
    future_preds = []
    
    # We need to predict for 3 days
    for i in range(1, 4): # 1, 2, 3
        # Look at the LAST row of the CURRENT df_lagged (which includes predictions from previous steps)
        last_row = df_lagged.iloc[-1]
        
        # Date for prediction
        next_date = last_row['date'] + timedelta(days=1)
        
        # Prepare Features for Next Day
        # Lags = Values from `last_row`
        features = {
            'year': next_date.year,
            'month': next_date.month,
            'day': next_date.day,
            'day_of_week': next_date.weekday(),
            'weekend': 1 if next_date.weekday() >= 5 else 0,
            
            'aqi_lag1': last_row['aqi'],
            'pm2_5_lag1': last_row['pm2_5'],
            'pm10_lag1': last_row['pm10'],
            'nitrogen_dioxide_lag1': last_row['nitrogen_dioxide'],
            'ozone_lag1': last_row['ozone'],
            'sulphur_dioxide_lag1': last_row['sulphur_dioxide'],
            'carbon_monoxide_lag1': last_row['carbon_monoxide']
        }
        
        # Create DataFrame for Model
        # Ensure order matches training: ['year', 'month', 'day', 'day_of_week', 'weekend', 'aqi_lag1', ... ]
        feature_order = [
            'year', 'month', 'day', 'day_of_week', 'weekend',
            'aqi_lag1', 'pm2_5_lag1', 'pm10_lag1', 'nitrogen_dioxide_lag1',
            'ozone_lag1', 'sulphur_dioxide_lag1', 'carbon_monoxide_lag1'
        ]
        
        X_test = pd.DataFrame([features])[feature_order]
        print(f"Day {i} Prediction Input (X_test):")
        print(X_test)

        # Predict
        pred_aqi = model.predict(X_test)[0]
        
        future_dates.append(next_date)
        future_preds.append(pred_aqi)
        
        # Append to df_lagged so the NEXT iteration can use it as Lag
        new_row = last_row.copy()
        new_row['date'] = next_date
        new_row['aqi'] = pred_aqi # The Predicted AQI
        # Exogenous vars remain same as last_row (Persistence assumption)
        
        df_lagged = pd.concat([df_lagged, pd.DataFrame([new_row])], ignore_index=True)
        
    return pd.DataFrame({'Date': future_dates, 'Predicted AQI': future_preds}), df_lagged
# --- Main App ---

try:
    project = get_hopsworks_project()
    st.sidebar.success("Connection Success")
except:
    st.stop()

# 1. Models
st.header("1. Model Registry")
df_metrics, best_model_meta, best_model_name = get_best_model(project)
st.dataframe(df_metrics.style.highlight_max(color='green', axis=0, subset=['R2']))
st.success(f"Selected: **{best_model_name}**")

# 2. Logic
if best_model_meta:
    with st.spinner("Initializing Forecast..."):
        # Download Load Model
        model_dir = best_model_meta.download()
        filename_map = {
            "random_forest_aqi_model": "random_forest_model.pkl",
            "gradient_boosting_aqi_model": "gradient_boosting_model.pkl",
            "ridge_aqi_model": "ridge_model.pkl"
        }
        fname = filename_map.get(best_model_name, "model.pkl")
        try:
            model = joblib.load(os.path.join(model_dir, fname))
        except:
             model = joblib.load(os.path.join(model_dir, "model.pkl"))

        # Fetch History
        df_hist = fetch_data_from_store(project)
        
        if df_hist is not None:
             # Predict
             st.subheader("2. 3-Day Forecast")
             df_forecast, df_combined = predict_next_3_days(model, df_hist)
             
             st.table(df_forecast)
             
             # Plot
             st.subheader("Trends")
             fig, ax = plt.subplots(figsize=(12,4))
             
             # Combined DF has history + predictions
             # Plot History
             today = datetime.now().date()
             history = df_combined[df_combined['date'] <= today].tail(7)
             forecast = df_combined[df_combined['date'] >= today]
             
             ax.plot(history['date'], history['aqi'], marker='o', label='History')
             ax.plot(forecast['date'], forecast['aqi'], marker='x', linestyle='--', color='red', label='Forecast')
             ax.set_title("AQI Forecast")
             ax.legend()
             ax.grid(True, alpha=0.5)
             st.pyplot(fig)
             
        else:
            st.error("No historical data found in Feature Store.")
