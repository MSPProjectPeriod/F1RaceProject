
import fastf1
import pandas as pd
import os

### LOADING & SAVING SESSION DATA TO CVS

#session details
year = 2024
event = 'Emilia Romagna Grand Prix'
session_type = 'R'
session_name = event+'_'+str(year)+'_'+session_type

# 1) Enable a cache directory (will be created if it doesn't exist)
cache_dir = './fastf1cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# 2) Load the session with everything
try:
    session = fastf1.get_session(year, event, session_type)
    session.load(telemetry=True, laps=True, weather=True)
except Exception as e:
    print(f"Failed to load session: {e}")
    exit(1)

# 3) Export laps
laps = session.laps
laps.to_csv(session_name+'_laps.csv', index=False)

# 4) Export per-lap telemetry (concatenate all laps)
# (per-lap telemetry is available via Lap.get_car_data())
telemetry_dfs = []
for lap in session.laps.iterlaps():
    df = lap.get_car_data()                       # speed, throttle, brake, gear, etc.  [oai_citation:0‡GitHub](https://github.com/theOehrly/Fast-F1/issues/25?utm_source=chatgpt.com)
    df['Driver'] = lap.driver
    df['LapNumber'] = lap.lap_number
    telemetry_dfs.append(df)
if telemetry_dfs:
    telemetry = pd.concat(telemetry_dfs, ignore_index=True)
    telemetry.to_csv(session_name+'_telemetry.csv', index=False)

# 5) Export positional data
pos_data = session.get_pos_data()                  # DataFrame of car position on track  [oai_citation:1‡Welcome to python-forum.io](https://python-forum.io/thread-40191.html?utm_source=chatgpt.com)
pos_data.to_csv(session_name+'_posdata.csv', index=False)

# 6) Export weather data
weather = session.get_weather_data()               # WeatherData object as DataFrame  [oai_citation:2‡Welcome to python-forum.io](https://python-forum.io/thread-40191.html?utm_source=chatgpt.com)
weather.to_csv(session_name+'_weather.csv', index=False)

print("All session data exported to CSV!")
