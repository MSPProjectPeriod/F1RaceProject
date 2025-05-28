
import fastf1
import numpy as np
import pandas as pd
import os
import scipy
import matplotlib.pyplot as plt

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

print("Succesfully loaded session! \n")

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

#functions

def get_pit_time_of_driver(session_driver):
    pit_time = pd.Series(dtype='timedelta64[ns]')
    last_pit_time = pd.to_timedelta(0.0, unit='s')
    for index, value in session.laps.pick_drivers(session_driver).Time.items():
        if not pd.isna(session.laps.pick_drivers(session_driver).PitInTime.iloc[index]) or index == (len(session.laps.pick_drivers(session_driver).Time)-1):
            #pit_time_element = session.laps.Time.iloc[index] - last_pit_time
            pit_time_element = session.laps.Time.iloc[index] #trying to get times when pits happen
            pit_time = pd.concat([pit_time, pd.Series(pit_time_element)])
            '''print("Pit time dif: ")
            print (pit_time_element)
            print("Time of Pit: ")
            print(session.laps.Time.iloc[index])
            print("\n")'''
            if not index+1 >= len(session.laps.pick_drivers(session_driver).PitInTime):
                last_pit_time = session.laps.Time.iloc[index+1]
            continue
            
    return pit_time

def get_lap_time_per_pit_time_of_driver(pit_time, session_driver):
    time_per_lap_per_pit_time = pd.Series(dtype='timedelta64[ns]')
    time_per_lap_per_pit_time_per_pit = {}
    pit_index = 0
    for index, value in session.laps.pick_drivers(session_driver).Time.items():
        if not value in pit_time.values:
            time_per_lap = pd.to_timedelta(session.laps.pick_drivers(session_driver).iloc[index].LapTime)
            time_per_lap_per_pit_time = pd.concat([time_per_lap_per_pit_time, pd.Series([time_per_lap])])
        else:
            time_per_lap = pd.to_timedelta(session.laps.pick_drivers(session_driver).iloc[index].LapTime)
            time_per_lap_per_pit_time = (pd.concat([time_per_lap_per_pit_time, pd.Series([time_per_lap])]))
            time_per_lap_per_pit_time_per_pit[f"pit_{pit_index}"] = time_per_lap_per_pit_time.dt.total_seconds()
            time_per_lap_per_pit_time = pd.Series(dtype='timedelta64[ns]')
            pit_index += 1
    return time_per_lap_per_pit_time_per_pit

def get_extra_time_per_lap_per_pit_time_per_pit(time_per_lap_per_pit_time_per_pit):
    extra_time_per_lap_per_pit_time_per_pit = {}
    for pit in range(0,len(time_per_lap_per_pit_time_per_pit)):
        #we want to get extra time from 2nd lap after pitstop therefore [1] since we start at 0 #time measured in seconds
        extra_time_per_lap_per_pit_time_per_pit[f"pit_{pit}"] = (time_per_lap_per_pit_time_per_pit[f"pit_{pit}"].subtract(time_per_lap_per_pit_time_per_pit[f"pit_{pit}"].iloc[1]))
    return extra_time_per_lap_per_pit_time_per_pit 

def plot_times(time_per_lap_per_pit_time_per_pit, extra_time_per_lap_per_pit_time_per_pit):
    '''for pit in range(0,len(time_per_lap_per_pit_time_per_pit)):
        data = time_per_lap_per_pit_time_per_pit[f"pit_{pit}"]

        # Plot using a point plot (scatter style)
        plt.plot(range(len(data)), data.values, 'o')  # 'o' for point markers
        plt.title(f'Point Plot of Lap Times of pit_{pit}')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()'''

    for pit in range(0,len(extra_time_per_lap_per_pit_time_per_pit)):
        data = extra_time_per_lap_per_pit_time_per_pit[f"pit_{pit}"]
        #removing first and last lap due to inconsistencies from starting/pitstopping
        index = range(1,len(data)-1)
        values = data.values[1:(len(data)-1)]

         # Compute linear trend line using pandas (via numpy)
        coeffs = np.polyfit(index,values, 1)
        trend = np.poly1d(coeffs)(index)

        # Compute residuals and standard error
        residuals = values - trend
        std_err = np.std(residuals)

         # Plot uncertainty band (±1 std deviation)
        plt.fill_between(index, trend - std_err, trend + std_err, alpha=0.3, label='±1 Std Error')

        # Plot using a point plot (scatter style)
        plt.plot(index,values, 'o')  # 'o' for point markers
        plt.plot(index, trend, '-', label='Trend Line')    # Trend line
        plt.title(f'Point Plot of Extra- Lap Times of pit_{pit}')
        plt.xlabel('Laps of '+ session_driver + ' in ' + session_name)
        plt.ylabel('Lap Times')
        plt.grid(True)
        plt.show()



###main sequence

#getting pit_time
pit_time = get_pit_time_of_driver(session_driver)
'''print("Pit times: ")
print(pit_time)
print("\n")'''

#getting time for each lap in a pit 
time_per_lap_per_pit_time_per_pit = get_lap_time_per_pit_time_of_driver(pit_time, session_driver)
print("Time per laps per pit (using dictionary for pit): ")
print(time_per_lap_per_pit_time_per_pit)
print("\n")

#getting the difference in time compared to baseline speed (second lap after pit stop)
extra_time_per_lap_per_pit_time_per_pit = get_extra_time_per_lap_per_pit_time_per_pit(time_per_lap_per_pit_time_per_pit)
print("Extra time per laps per pit (using dictionary for pit): ")
print(extra_time_per_lap_per_pit_time_per_pit)
print("\n")

plot_times(time_per_lap_per_pit_time_per_pit,extra_time_per_lap_per_pit_time_per_pit)
