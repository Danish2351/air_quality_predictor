import requests
import time
from datetime import datetime
import os
import numpy as np
import pandas as pd
import hopsworks
from src import utils
from dotenv import load_dotenv

load_dotenv()

lat="24.8546842"
lon="67.0207055" 
start_time="2025-07-01T00:00"
time = datetime.now()
end_time = time.strftime("%Y-%m-%dT%H:%M")

print(f"Starting backfill from {start_time} to {end_time}...")

# Fetch data
raw_json = utils.get_aqi_data(lat, lon, start_time, end_time)

if raw_json:
    # Clean data
    df = utils.data_cleaning(raw_json)
    
    if not df.empty:
        print(f"Data fetched and cleaned. Rows: {len(df)}")
        print(df.head())
        
        # Connect to Hopsworks
        try:
            aqi_fg = utils.get_feature_group(name="aqi_features", version=1)
            
            # Insert data using utils helper
            utils.save_data_to_feature_group(df, aqi_fg, wait_for_job=False)
            print("Backfill data saved to Hopsworks Feature Group successfully.")
        except Exception as e:
            print(f"Failed to save data: {e}")
    else:
        print("No data available after cleaning.")
else:
    print("Failed to fetch data from API.")