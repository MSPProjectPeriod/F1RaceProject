

i = 2

if i == 1:


  import fastf1
  
  fastf1.Cache.enable_cache('./fastf1cache')
  
  session = fastf1.get_session(2024, 'Emilia Romagna Grand Prix', 'R')
  
  session.load(telemetry=True, laps=True, weather=True)
  
  driver_verstappen = session.get_driver('VER')


elif i == 2:


  import fastf1
  import pandas as pd
  import os
  
  
  
  # 1) Enable a cache directory (will be created if it doesn't exist)
  cache_dir = './fastf1cache'
  os.makedirs(cache_dir, exist_ok=True)
  fastf1.Cache.enable_cache(cache_dir)
  
  # 2) Load the session with everything
  session = fastf1.get_session(2024, 'Emilia Romagna Grand Prix', 'R')
  session.load(telemetry=True, laps=True, weather=True)
  
  # 3) Export laps
  laps = session.laps
  laps.to_csv('emilia_romagna_2024_race_laps.csv', index=False)
  
  # 4) Export per-lap telemetry (concatenate all laps)
  #    (per-lap telemetry is available via Lap.get_car_data())
  tel_dfs = []
  for lap in session.laps.iterlaps():
      df = lap.get_car_data()                       # speed, throttle, brake, gear, etc.  [oai_citation:0‡GitHub](https://github.com/theOehrly/Fast-F1/issues/25?utm_source=chatgpt.com)
      df['Driver'] = lap.driver
      df['LapNumber'] = lap.lap_number
      tel_dfs.append(df)
  if tel_dfs:
      telemetry = pd.concat(tel_dfs, ignore_index=True)
      telemetry.to_csv('emilia_romagna_2024_race_telemetry.csv', index=False)
  
  # 5) Export positional data
  pos_data = session.get_pos_data()                  # DataFrame of car position on track  [oai_citation:1‡Welcome to python-forum.io](https://python-forum.io/thread-40191.html?utm_source=chatgpt.com)
  pos_data.to_csv('emilia_romagna_2024_race_posdata.csv', index=False)
  
  # 6) Export weather data
  weather = session.get_weather_data()               # WeatherData object as DataFrame  [oai_citation:2‡Welcome to python-forum.io](https://python-forum.io/thread-40191.html?utm_source=chatgpt.com)
  weather.to_csv('emilia_romagna_2024_race_weather.csv', index=False)
  
  print("All session data exported to CSV!")
