import requests
import time
from datetime import datetime
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv, find_dotenv
import hopsworks

load_dotenv(find_dotenv())
HOPSWORK_API_KEY=os.environ.get("HOPSWORK_API_KEY")


def get_aqi_data(lat, lon, start, end):
    
    url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=us_aqi,us_aqi_pm2_5,us_aqi_pm10,us_aqi_nitrogen_dioxide,us_aqi_ozone,us_aqi_sulphur_dioxide,us_aqi_carbon_monoxide&start_hour={start}&end_hour={end}"
    
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None

def data_cleaning(raw_json):
    df = pd.DataFrame(raw_json['hourly'])

    #renaming columns
    df.columns = df.columns.str.replace('us_', '', regex=False)
    df.columns = [col.replace('aqi_', '') if col.startswith('aqi_') else col for col in df.columns]
    
    #creating datetime columns
    df['time'] = pd.to_datetime(df['time'])
    
    # Create timestamp column for Hopsworks compatibility (ms)
    df['timestamp'] = df['time'].astype(np.int64) // 10**6
    
    # Derived features from original (datetime) time column
    df['year'] = df['time'].dt.year.astype(np.int64)
    df['month'] = df['time'].dt.month.astype(np.int64)
    df['day'] = df['time'].dt.day.astype(np.int64)
    df['day_of_week'] = df['time'].dt.dayofweek.astype(np.int64)
    df['weekend'] = (df['day_of_week'] >= 5).astype(np.int64) 

    df['aqi'] = df['aqi'].astype(np.int64)
    df['pm2_5'] = df['pm2_5'].astype(np.int64)
    df['pm10'] = df['pm10'].astype(np.int64)
    df['nitrogen_dioxide'] = df['nitrogen_dioxide'].astype(np.int64)
    df['ozone'] = df['ozone'].astype(np.int64)
    df['sulphur_dioxide'] = df['sulphur_dioxide'].astype(np.int64)
    df['carbon_monoxide'] = df['carbon_monoxide'].astype(np.int64)
    df = df.sort_values('timestamp')
    df = df.dropna()
    return df

def get_feature_group(name="aqi_features", version=1):
    project = hopsworks.login(
        project='sdk_new_project',
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORK_API_KEY
    )
    fs = project.get_feature_store()

    try:
        fg = fs.get_feature_group(name=name, version=version)
    except:
        print("Feature group not found")

    if fg is None:
        fg = fs.create_feature_group(
            name=name,
            version=version,
            primary_key=['timestamp'],
            event_time='timestamp',
            description="Hourly AQI data",
            online_enabled=True
        )
    return fg

def save_data_to_feature_group(df, feature_group, wait_for_job=True):
    """
    Inserts data into the Feature Group.
    Args:
        df (pd.DataFrame): Data to insert.
        feature_group: Hopsworks Feature Group object.
        wait_for_job (bool): Whether to wait for the materialization job to complete.
                       Defaults to True.
    """
    try:
        feature_group.insert(df, write_options={"wait_for_job": wait_for_job})
        print("Data inserted successfully.")
    except Exception as e:
        print(f"Failed to insert data: {e}")
        raise e