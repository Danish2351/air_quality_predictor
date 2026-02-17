import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
import joblib
import os
import hopsworks
from src import utils

def train_model():
    print("=== Starting Model Training Pipeline ===")
    
    # 1. Connect to Hopsworks
    project = hopsworks.login(
        project='air_quality_predictor',
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=utils.HOPSWORK_API_KEY
    )
    fs = project.get_feature_store()
    
    # Retrieve Feature Group
    print("Retrieving Feature Group...")
    try:
        fg = fs.get_feature_group(name="aqi_features", version=1)
    except Exception as e:
        print(f"Error: Feature Group 'aqi_features' not found. Data ingestion must run first. {e}")
        return

    # Create/Retrieve Feature View
    # We want valid features for prediction
    # Features: year, month, day, hour, day_of_week, weekend, lag features, rolling means.
    # Target: aqi
    # PK: timestamp (we will exclude from training)
    # Time: time (we will exclude from training)

    # 4. create training data
    # Using a simple train_test_split on the retrieved pandas dataframe for simplicity here
    # Alternatively, create_training_data() could materialise a dataset in Hopsworks
    
    # Retrieve all data (offline store)
    # Using read() on feature view or fg
    # For a robust production pipeline, feature_view.train_test_split is better but requires storage connector setup sometimes.
    # We'll use feature_view.get_batch_data() or query.read() and split locally for simplicity in this script.
    
    # Actually, let's use feature_view.training_data() if possible, or just read via query which is simpler for this scale.
    # feature_view.training_data requires setting up a storage connector for caching usually.
    # Let's stick to reading the dataframe directly from the Feature Group query for now to minimize infra setup errors for the user.
    # But ideally we use feature_view to ensure consistency.
    
    
    # Let's read from the Feature Group directly to get everything including label
    # (Feature View is best practice for serving, but for this simple training script direct read is safer if FV setup is complex)
    # Wait, existing `utils.data_cleaning` returns all columns.
    
    # Let's use `feature_view.get_training_data(1)` which splits in-memory if no storage connector specified?
    # Hopsworks `get_training_data` usually requires a storage connector or extensive setup.
    # Simple approach: Read FG, split locally.
    
    print("Reading data from Feature Store...")
    # Use offline=False to get latest if offline job failed? No, offline is better for volume.
    # But given our previous issues with Offline Materialization, we should probably read ONLINE if offline is empty.
    

    df = fg.read(online=True)

    if df.empty:
        print("No data found for training!")
        return

    print(f"Data retrieved: {len(df)} rows.")

    # 5. Preprocess
    # Sort by timestamp to ensure time-based splitting if we wanted time-series split
    df = df.sort_values("timestamp")
    daily_aqi = df.groupby(['year', 'month', 'day'])[['day_of_week','weekend','aqi', 'pm2_5', 'pm10', 'nitrogen_dioxide', 'ozone', 'sulphur_dioxide', 'carbon_monoxide']].mean().reset_index()

    # adding lag features
    cols = [
    'aqi', 'pm2_5', 'pm10', 'nitrogen_dioxide', 
    'ozone', 'sulphur_dioxide', 'carbon_monoxide'
    ]
    for col in cols:
        daily_aqi[f'{col}_lag1'] = daily_aqi[col].shift(1)
        
    df_lagged=daily_aqi.dropna()
    df_lagged=df_lagged.drop(columns=["pm2_5","pm10","nitrogen_dioxide","ozone","sulphur_dioxide","carbon_monoxide"])
    print("df_lagged:  ",df_lagged.columns.tolist())
    # feature scaling
    # features = df_lagged[['aqi','aqi_lag1','pm2_5_lag1','pm10_lag1','nitrogen_dioxide_lag1','ozone_lag1','sulphur_dioxide_lag1','carbon_monoxide_lag1']]
    # scaler = StandardScaler()
    # scaled_features = scaler.fit_transform(features)
    # scaled_df = pd.DataFrame(scaled_features, columns=features.columns)

    # Drop non-feature columns
    # timestamp is PK/EventTime
    # time is readable string/datetime
    # Selection of numeric features only to avoid DTypePromotionError
    # We drop 'time' (datetime) and 'timestamp' (int64/id)
    # The model works best with numeric features only.
    if 'aqi' in df.columns:
        y = df_lagged['aqi']
        X = df_lagged.drop(columns=['aqi'])

    else:
        print("Target column 'aqi' not found!")
        return

    print(f"independent variable: {X.columns.tolist()}")
    print(f"independent variable: {X.info()}")
    print(f"target variable: {y.info()}")


    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False) # Time series split (no shuffle)

    # 6. Train Model
    print("Training Random Forest Regressor...")
    random_forest_model = RandomForestRegressor(    
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=5,
        random_state=42
    )
    random_forest_model.fit(X_train, y_train)
    
    # 7. Evaluate
    y_pred = random_forest_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)*100
    r2 = r2_score(y_test, y_pred)
    
    print(f"Random Forest Regressor Model Evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")
    
    # 8. Register Model
    print("Registering Model in Hopsworks...")
    mr = project.get_model_registry()
    
    # Metrics to save
    metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}
    
    # Create model object
    aqi_model = mr.python.create_model(
        name="random_forest_aqi_model",
        metrics=metrics,
        description="Random Forest Regressor for AQI prediction"
    )
    
    # Save the model artifact
    # create_model saves schema and metadata. To save the actual model object we use `save`.
    # It usually requires saving to a local file/dir first.
    
    model_dir = "model_dir"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        
    joblib.dump(random_forest_model, os.path.join(model_dir, "random_forest_model.pkl"))
    aqi_model.save(model_dir)
    print("Random Forest Model registered successfully!")



    #Gradient Boosting Regressor
    print("Training Gradient Boosting Regressor...")
    gradient_boosting_model = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    gradient_boosting_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = gradient_boosting_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)*100
    r2 = r2_score(y_test, y_pred)
    
    print(f"Gradient Boosting Regressor Model Evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")
    
    # 8. Register Model
    print("Registering Model in Hopsworks...")
    mr = project.get_model_registry()
    
    # Metrics to save
    metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}
    
    # Create model object
    aqi_model = mr.python.create_model(
        name="gradient_boosting_aqi_model",
        metrics=metrics,
        description="Gradient Boosting Regressor for AQI prediction"
    )
    
    model_dir = "model_dir"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        
    joblib.dump(gradient_boosting_model, os.path.join(model_dir, "gradient_boosting_model.pkl"))
    aqi_model.save(model_dir)
    print("Gradient Boosting Model registered successfully!")

    
    #Ridge Regressor
    print("Training Ridge Regressor...")
    ridge_model = Ridge(alpha=1)
    ridge_model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = ridge_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)*100
    r2 = r2_score(y_test, y_pred)
    
    print(f"Ridge Regressor Model Evaluation - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.4f}, R2: {r2:.4f}")
    
    # Register Model
    print("Registering Model in Hopsworks...")
    mr = project.get_model_registry()
    
    # Metrics to save
    metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}
    
    # Create model object
    aqi_model = mr.python.create_model(
        name="ridge_aqi_model",
        metrics=metrics,
        description="Ridge Regressor for AQI prediction"
    )
    
    model_dir = "model_dir"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
        
    joblib.dump(ridge_model, os.path.join(model_dir, "ridge_model.pkl"))
    aqi_model.save(model_dir)
    print("Ridge Model registered successfully!")


if __name__ == "__main__":
    train_model()