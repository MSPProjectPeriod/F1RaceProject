
import fastf1
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

### LOADING & SAVING SESSION DATA TO CVS
csv_location = 'csv_files/'
os.makedirs(csv_location, exist_ok=True)

# Enable a cache directory (will be created if it doesn't exist)
cache_dir = './fastf1cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

#session details using a dictionary
sessions = {
    "0": [2024, 'Emilia Romagna Grand Prix', 'R', ['VER', 'NOR', 'LEC', 'PIA', 'SAI', 'HAM', 'RUS', 'PER', 'STR', 'TSU', 'HUL', 'MAG', 'RIC', 'OCO', 'ZHO', 'GAS', 'SAR', 'BOT', 'ALO', 'ALB']],
    "1": [2024, 'Monaco Grand Prix', 'R', ['LEC', 'PIA', 'SAI', 'NOR', 'RUS', 'VER', 'HAM', 'TSU', 'ALB', 'GAS', 'ALO', 'RIC', 'BOT', 'STR', 'SAR', 'ZHO']]
}

print("These are the available sessions: \n", sessions['0'],"\n", sessions['1'])
session_selection = input("Enter which session you would like to analyse:  (example: 0, 1, ..)\n")
session_year = sessions.get(session_selection)[0]
session_event = sessions.get(session_selection)[1]
session_type = sessions.get(session_selection)[2]
session_name = session_event+'_'+str(session_year)+'_'+session_type

driver_selection = input("Which driver would you like to analyse? (example: 0 for first driver or VER)\n")
if driver_selection in sessions.get(session_selection)[3]:
    session_driver = driver_selection
elif int(driver_selection) <= len(sessions.get(session_selection)[3]) and int(driver_selection) >= 0:
    session_driver = sessions.get(session_selection)[3][int(driver_selection)]
    print("Picking driver: ", session_driver, "\n")
else:
    print("Driver is not available for this session. \n")
    exit(1)

# Load the session with everything
try:
    session = fastf1.get_session(session_year, session_event, session_type)
    session.load(telemetry=True, laps=True, weather=True)
except Exception as e:
    print(f"Failed to load session: {e}")
    exit(1)

print("Succesfully loaded session! \n")


#removing unnecessary files
for csv_file in os.listdir(csv_location):
    if csv_file != (session_name+'_laps.csv') and csv_file != (session_name+'_'+session_driver+'_posdata.csv') and csv_file != (session_name+'_'+session_driver+'_telemetry.csv') and csv_file != (session_name+'_'+session_driver+'_weather.csv'):
        os.remove(csv_location+csv_file)
print("Removed all prior csv_files that were unnecessary. \n")

session.laps.to_csv(csv_location+session_name+'_laps.csv', index=False)

# Export per-lap telemetry
# (per-lap telemetry is available via Lap.get_car_data())
telemetry_dfs = session.laps.pick_drivers(session_driver).get_telemetry()
telemetry_dfs.to_csv(csv_location+session_name+'_'+session_driver+'_telemetry.csv', index=False)

# Export positional data
pos_data = session.laps.pick_drivers(session_driver).get_pos_data()                  # DataFrame of car position on track  [oai_citation:1‡Welcome to python-forum.io](https://python-forum.io/thread-40191.html?utm_source=chatgpt.com)
pos_data.to_csv(csv_location+session_name+'_'+session_driver+'_posdata.csv', index=False)

# Export weather data
weather = session.laps.pick_drivers(session_driver).get_weather_data()               # WeatherData object as DataFrame  [oai_citation:2‡Welcome to python-forum.io](https://python-forum.io/thread-40191.html?utm_source=chatgpt.com)
weather.to_csv(csv_location+session_name+'_'+session_driver+'_weather.csv', index=False)

#done to check data first hand
print("All session data exported to CSV!")


#functions
#work on this tomorrow
def get_pit_time_of_driver(session_driver):
    pit_time = pd.Series(dtype='timedelta64[ns]')
    for index, value in list(session.laps.pick_drivers(session_driver).Time.items())[1:]:
        if not pd.isna(session.laps.pick_drivers(session_driver).PitInTime.loc[index]) or index == session.laps.pick_drivers(session_driver).index[-1]:
            if pd.isna(session.laps.PitInTime.loc[index]):
                 pit_time_element = session.laps.Time.loc[index] #trying to get times when pits happen
            else:
                pit_time_element = session.laps.PitInTime.loc[index] #trying to get times when pits happen
            pit_time = pd.concat([pit_time, pd.Series(pit_time_element)])
    return pit_time

def get_lap_time_per_pit_time_of_driver(pit_time, session_driver):
    time_per_lap_per_pit_time = pd.Series(dtype='timedelta64[ns]')
    time_per_lap_per_pit_time_per_pit = {}
    pit_index = 0
    for index, value in session.laps.pick_drivers(session_driver).Time.items():
        if not value in pit_time.values and not session.laps.pick_drivers(session_driver).PitInTime.loc[index] in pit_time.values:
            if not pd.isna(session.laps.pick_drivers(session_driver).loc[index].LapTime):
                time_per_lap = pd.to_timedelta(session.laps.pick_drivers(session_driver).loc[index].LapTime)
                time_per_lap_per_pit_time = pd.concat([time_per_lap_per_pit_time, pd.Series([time_per_lap])])
        else:
            time_per_lap = pd.to_timedelta(session.laps.pick_drivers(session_driver).loc[index].LapTime)
            time_per_lap_per_pit_time = (pd.concat([time_per_lap_per_pit_time, pd.Series([time_per_lap])]))
            time_per_lap_per_pit_time_per_pit[f"pit_{pit_index}"] = time_per_lap_per_pit_time.dt.total_seconds()
            time_per_lap_per_pit_time = pd.Series(dtype='timedelta64[ns]')
            pit_index += 1
            continue
    return time_per_lap_per_pit_time_per_pit

def plot_times(time_per_lap_per_pit_time_per_pit):
    width = 8
    height = 6
    plt.figure(figsize=(width, height))

    lap_offset = 1

    for pit in range(0,len(time_per_lap_per_pit_time_per_pit)):
        data = time_per_lap_per_pit_time_per_pit[f"pit_{pit}"]

        index = range(lap_offset,len(data)+lap_offset)
        values = data.values

        # Compute linear trend line ignoring the first data point due to starting lap time delay
        coeffs = np.polyfit(index[1:], values[1:], 1)
        trend = np.poly1d(coeffs)(index)

        # Compute residuals and standard error
        residuals = values - trend
        std_err = np.std(residuals)

        # Plot using a point plot (scatter style)
        plt.plot(index, values, 'o-', label=f'Pit_{pit} Lap Times')  # 'o' for point markers
        plt.plot(index, trend, '-', label=f'Pit_{pit} Trend Line (slope = {coeffs[0]:.4f})')    # Trend line
        # Plot uncertainty band (±1 std deviation)
        plt.fill_between(index, trend - std_err, trend + std_err, alpha=0.3, label=f'Pit_{pit} ±1 Std Error')

        # Increment offset for next pit
        lap_offset += len(data)  # Add 1 to create visual gap:

    plt.title(f'Point Plot of Lap Times for each Pit')
    plt.xlabel('Laps of per Pit of '+ session_driver + ' in ' + session_name)
    print(lap_offset)
    plt.xticks(np.arange(0, lap_offset + 5, 5))
    plt.ylabel('Lap Times per Pit')
    plt.grid(True)
    plt.legend()
    plt.show()



### MAIN SEQUENCE

#getting pit_time
pit_time = get_pit_time_of_driver(session_driver)
print("Pit times: ")
print(pit_time)
print("\n")

#getting time for each lap in a pit 
time_per_lap_per_pit_time_per_pit = get_lap_time_per_pit_time_of_driver(pit_time, session_driver)
print("Time per laps per pit (using dictionary for pit): ")
print(time_per_lap_per_pit_time_per_pit)
print("\n")

plot_times(time_per_lap_per_pit_time_per_pit)
