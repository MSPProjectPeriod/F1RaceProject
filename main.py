
import fastf1
import pandas as pd
import os

### LOADING & SAVING SESSION DATA TO CVS
csv_location = 'csv_files/'

#session details
session_year = 2024
session_event = 'Emilia Romagna Grand Prix'
session_type = 'R'
session_name = session_event+'_'+str(session_year)+'_'+session_type
session_driver = 'VER' #VERSTAPPEN

# 1) Enable a cache directory (will be created if it doesn't exist)
cache_dir = './fastf1cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# 2) Load the session with everything
try:
    session = fastf1.get_session(session_year, session_event, session_type)
    session.load(telemetry=True, laps=True, weather=True)
except Exception as e:
    print(f"Failed to load session: {e}")
    exit(1)

'''
# 3) Export laps
laps = session.laps
laps.to_csv(csv_location+session_name+'_laps.csv', index=False)

# 4) Export per-lap telemetry
# (per-lap telemetry is available via Lap.get_car_data())
telemetry_dfs = session.laps.pick_drivers(session_driver).get_telemetry()
telemetry_dfs.to_csv(csv_location+session_name+'_'+session_driver+'_telemetry.csv', index=False)

# 5) Export positional data
pos_data = session.laps.pick_drivers(session_driver).get_pos_data()                  # DataFrame of car position on track  [oai_citation:1‡Welcome to python-forum.io](https://python-forum.io/thread-40191.html?utm_source=chatgpt.com)
pos_data.to_csv(csv_location+session_name+'_'+session_driver+'_posdata.csv', index=False)

# 6) Export weather data
weather = session.laps.pick_drivers(session_driver).get_weather_data()               # WeatherData object as DataFrame  [oai_citation:2‡Welcome to python-forum.io](https://python-forum.io/thread-40191.html?utm_source=chatgpt.com)
weather.to_csv(csv_location+session_name+'_'+session_driver+'_weather.csv', index=False)

print("All session data exported to CSV!")
'''