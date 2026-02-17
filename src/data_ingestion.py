import os
from datetime import datetime, timedelta
from src import utils
from dotenv import load_dotenv

load_dotenv()

lat = "24.8546842"
lon = "67.0207055"

# Calculate time range: last hour to current hour
current_time = datetime.now()
# We want the last completed hour, so we look back 1 hour
start_time = current_time.strftime("%Y-%m-%dT%H:%M")
end_time = current_time.strftime("%Y-%m-%dT%H:%M")

print(f"Fetching data from {start_time} to {end_time}...")
raw_json = utils.get_aqi_data(lat, lon, start_time, end_time)

if raw_json:
    df = utils.data_cleaning(raw_json)
    
    if not df.empty:
        print("Data fetched and cleaned successfully.")
        print(df.head())
        
        # Connect to Hopsworks and get feature group
        try:
            aqi_fg = utils.get_feature_group(name="aqi_features", version=1)
            
            # Insert data using utils helper
            # Hopsworks Feature Groups handle deduplication based on primary key (offline and online stores)
            utils.save_data_to_feature_group(df, aqi_fg)
            print("Data inserted into Hopsworks Feature Group successfully.")
        except Exception as e:
            print(f"Failed to insert data into Hopsworks: {e}")
    else:
        print("No data available after cleaning.")
else:
    print("Failed to fetch data.")