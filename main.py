
import fastf1
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sympy import symbols, sympify, integrate, lambdify
from scipy.optimize import minimize_scalar
import itertools

### OBJECTS

class driver_Performance:
    def __init__(self, driver=str, pit_intervals=pd.Series(dtype=int), pit_results=None, pit_avr_stoptime=None, original_time=None, lap_count=None):
        self.driver = driver
        self.pit_intervals = pit_intervals #interval in laps
        self.pit_results = pit_results
        self.pit_avr_stoptime = pit_avr_stoptime
        self.original_time = original_time

class pit_strat:
    def __init__(self,pit_intervals=pd.Series(dtype=int), pit_lines=None):
        self.pit_intervals
        self.pit_lines

### FUNCTIONS

def csv_file_manager(session_drivers):
    print(session_drivers)

    # Build a set of all valid filenames
    valid_files = {session_name + '_laps.csv'}
    for driver in session_drivers:
        valid_files.update({
            f"{session_name}_{driver}_posdata.csv",
            f"{session_name}_{driver}_telemetry.csv",
            f"{session_name}_{driver}_weather.csv"
        })

    # Loop once through all files and delete the unnecessary ones
    for csv_file in os.listdir(csv_location):
        full_path = os.path.join(csv_location, csv_file)
        if csv_file not in valid_files and os.path.isfile(full_path):
            os.remove(full_path)

    print("Removed all prior csv_files that were unnecessary. \n")

    #data of the entire session
    session.laps.to_csv(csv_location+session_name+'_laps.csv', index=False)

    #data for each driver

    for driver in session_drivers:
        # Export per-lap telemetry
        # (per-lap telemetry is available via Lap.get_car_data())
        telemetry_dfs = session.laps.pick_drivers(driver).get_telemetry()
        telemetry_dfs.to_csv(csv_location+session_name+'_'+driver+'_telemetry.csv', index=False)

        # Export positional data
        pos_data = session.laps.pick_drivers(driver).get_pos_data()                 
        pos_data.to_csv(csv_location+session_name+'_'+driver+'_posdata.csv', index=False)

        # Export weather data
        weather = session.laps.pick_drivers(driver).get_weather_data()             
        weather.to_csv(csv_location+session_name+'_'+driver+'_weather.csv', index=False)

    #done to check data first hand
    print("All session data exported to CSV!")

def get_pit_time_of_driver(session_driver):
    pit_time = pd.Series(dtype='timedelta64[ns]')
    pit_laps = pd.Series(dtype=int)
    pit_count = 0
    for index, value in list(session.laps.pick_drivers(session_driver).Time.items())[1:]:
        if not pd.isna(session.laps.pick_drivers(session_driver).PitInTime.loc[index]) or index == session.laps.pick_drivers(session_driver).index[-1]:
            if pd.isna(session.laps.PitInTime.loc[index]):
                 pit_time_element = session.laps.Time.loc[index] #trying to get times when pits happen
            else:
                pit_time_element = session.laps.PitInTime.loc[index] #trying to get times when pits happen
            pit_time = pd.concat([pit_time, pd.Series(data=[pit_time_element], index=[pit_count])])

            #get the laps
            pit_lap_element = session.laps.LapNumber.loc[index]
            pit_laps = pd.concat([pit_laps, pd.Series(data=[pit_lap_element], index=[pit_count])])
            pit_count += 1

    return pit_time, pit_laps

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

def get_pit_avr_stoptime(session_driver):

    pit_avr_stoptime = 0.0
    count = 0
    
    pit_in = session.laps.pick_drivers(session_driver).PitInTime
    pit_out = session.laps.pick_drivers(session_driver).PitOutTime

    for index, value in list(pit_in.items())[:-1]:
        in_time = pit_in.loc[index]
        out_time = pit_out.loc[index+1]
        if pd.isna(in_time) and pd.isna(out_time):
            continue  # skip if any value is missing
    
        stop_time = (out_time - in_time).total_seconds()
        pit_avr_stoptime += stop_time
        count += 1
    pit_avr_stoptime = pit_avr_stoptime / count

    return pit_avr_stoptime

def get_pit_trends_coeffs_residuals_data(time_per_lap_per_pit_time_per_pit):
        
    lap_offset = 1
    results = {}

    for pit in range(0,len(time_per_lap_per_pit_time_per_pit)):
        data = time_per_lap_per_pit_time_per_pit[f"pit_{pit}"]

        index = range(lap_offset,len(data)+lap_offset)
        values = data.values

        # Compute linear trend line ignoring the first/last data point due to starting/end lap time delay
        coeffs = np.polyfit(index[1:-1], values[1:-1], 1)
        trends = np.poly1d(coeffs)(index)

        # Compute residuals and standard error
        residuals = values[1:-1] - trends[1:-1]
        std_err = np.std(residuals)
        lap_offset += 1
        
        results[f"pit_{pit}"] = {
        "index": index,
        "values": values,
        "coeffs": coeffs,
        "trends": trends,
        "residuals": residuals,
        "std_err": std_err
        }

    return results

def plt_show_sec(duration):
    plt.show(block=False)
    plt.pause(duration)
    plt.close()

def plot_times(performance, show_seconds):

    #plot dimensions
    width = 8
    height = 6
    plt.figure(figsize=(width, height))

    lap_offset = 1

    for pit in range(0,len(time_per_lap_per_pit_time_per_pit)):

        index = performance.results[f"pit_{pit}"]["index"]
        values = performance.results[f"pit_{pit}"]["values"]
        trends = performance.results[f"pit_{pit}"]["trends"]
        coeffs = performance.results[f"pit_{pit}"]["coeffs"]
        std_err = performance.results[f"pit_{pit}"]["std_err"]
        residuals = performance.results[f"pit_{pit}"]["residuals"]

        # Plot using a point plot (scatter style)
        plt.plot(np.array(index) + lap_offset, values, 'o-', label=f'Pit_{pit} Lap Times')  # 'o' for point markers
        plt.plot(np.array(index) + lap_offset, trends, '-', label=f'Pit_{pit} Trend Line (slope = {coeffs[0]:.4f})')    # Trend line
        # Plot uncertainty band (±1 std deviation)
        plt.fill_between(np.array(index) + lap_offset, trends - std_err,trends + np.std(residuals), alpha=0.3, label=f'Pit_{pit} ±1 Std Error')

        # Increment offset for next pit
        lap_offset += len(values)  # Add 1 to create visual gap:

    plt.title(f'Point Plot of Lap Times for each Pit')
    plt.xlabel('Laps per Pit of '+ performance.driver + ' in ' + session_name)
    plt.xticks(np.arange(0, lap_offset + 5, 5)) #show every 5 lap on x axis
    plt.ylabel('Lap Times (s) per Pit (first and last lap per pit ignored for trend)')
    plt.grid(True)
    plt.legend()
    plt_show_sec(show_seconds)

def optimize_pit(performance, show_seconds=30):

    # Constants
    number_of_laps = int(performance.pit_intervals.iloc[-1])
    average_pit_time = performance.pit_avr_stoptime
    number_of_stints = len(performance.pit_intervals)  # e.g. 3 stints → 2 pit stops


    if number_of_stints < 2:
        print("There are no pit stops for ", performance.driver, " to optimize for.\n")

    else:
        # Get linear coefficients (slope, intercept) for each stint
        stint_funcs = {}
        for stint in range(number_of_stints):
            slope, intercept = performance.results[f'pit_{stint}']["coeffs"]
            stint_penalty =  max(0, -slope) #penalty for being on a stint for long
            stint_funcs[stint] = (slope, intercept, stint_penalty)

        # Generate all valid pit stop combinations
        valid_laps = range(1, number_of_laps - 1)
        possible_pit_combinations = itertools.combinations(valid_laps, number_of_stints - 1)

        # Fast integration of linear function a*x + b from x1 to 
        def fast_integrate_linear(a, b, c, x1, x2):
            
            return (a / 2) * (x2**2 - x1**2) + b * (x2 - x1) + (c / 3) * (x2**3-x1**3)

        def total_race_time(pit_laps):
            lap_points = [1] + list(pit_laps) + [number_of_laps]
            total_time = 0
            for stint in range(number_of_stints):
                a, b, c = stint_funcs[stint]
                start_lap = lap_points[stint]
                end_lap = lap_points[stint + 1]
                total_time += fast_integrate_linear(a, b, c, start_lap, end_lap)
            total_time += average_pit_time * (number_of_stints - 1)
            return total_time

        # Evaluate combinations
        best_combination = None
        best_time = float('inf')
        results = []

        for combo in possible_pit_combinations:
            time = total_race_time(combo)
            results.append((combo, time))
            if time < best_time:
                best_time = time
                best_combination = combo

        # Plot results
        x_vals = [sum(combo)/len(combo) if combo else 0 for combo, _ in results]
        y_vals = [t for _, t in results]

        plt.figure(figsize=(10, 6))
        plt.scatter(x_vals, y_vals, c='blue')
        if best_combination:
            avg_lap = sum(best_combination) / len(best_combination)
            plt.axvline(avg_lap, color='red', linestyle='--',
                        label=f'Optimal Pit(s): {best_combination}')
        plt.title('Total Race Time vs Pit Strategy')
        plt.xlabel(f'Pit stop lap of {performance.driver} in {session_name}')
        plt.ylabel('Total Race Time (s)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt_show_sec(show_seconds) 


### LOADING AND SELECTING SESSION DATA
csv_location = 'csv_files/'
os.makedirs(csv_location, exist_ok=True)

# Enable a cache directory (will be created if it doesn't exist)
cache_dir = './fastf1cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

#session details using a dictionary
sessions = {
    "0": [2024, 'Emilia Romagna Grand Prix', 'R', ['VER', 'NOR', 'LEC', 'PIA', 'SAI', 'HAM', 'RUS', 'PER', 'STR', 'TSU', 'HUL', 'MAG', 'RIC', 'OCO', 'ZHO', 'GAS', 'SAR', 'BOT', 'ALO']], #ALB was excluded due to car faults and Penalty
    "1": [2024, 'Monaco Grand Prix', 'R', ['LEC', 'PIA', 'SAI', 'NOR', 'RUS', 'VER', 'HAM', 'TSU', 'ALB', 'GAS', 'ALO', 'RIC', 'BOT', 'STR', 'SAR', 'ZHO']]
}

#session-selection
print("These are the available sessions: \n", sessions['0'],"\n", sessions['1'])
session_selection = input("Enter which session you would like to analyse:  (example: 0, 1, ..)\n")
if session_selection.isnumeric() and int(session_selection) >= 0 and int(session_selection) <= len(sessions):
    session_year = sessions.get(session_selection)[0]
    session_event = sessions.get(session_selection)[1]
    session_type = sessions.get(session_selection)[2]
    session_name = session_event+'_'+str(session_year)+'_'+session_type
else:
    print("The session you selected is unavailable. Exiting program.")
    exit(1)

#Load the session with everything
try:
    session = fastf1.get_session(session_year, session_event, session_type)
    session.load(telemetry=True, laps=True, weather=True)
except Exception as e:
    print(f"Failed to load session: {e}")
    exit(1)
print("Succesfully loaded session! \n")


driver_selection = input("How many drivers would you like to analyse? (example: VER for just Verstappen, ALL, 2 for two drivers)\n").upper()
if driver_selection in sessions.get(session_selection)[3]:
    session_drivers = [driver_selection]
elif driver_selection == 'ALL':
    session_drivers = sessions.get(session_selection)[3]
elif int(driver_selection) <= len(sessions.get(session_selection)[3]) and int(driver_selection) >= 0:
    session_drivers = input("Which drivers would you like to analyse specifically? (example: VER, NOR)\n").replace(" ", "").split(",")
    print(session_drivers)
    for driver in session_drivers:
        if not driver in sessions.get(session_selection)[3]:
            print("Driver ", driver , " not found in selected session. Will be skipped over.\n")
            session_drivers.remove(driver)
    print("Picking drivers: ", session_drivers, "\n")
else:
    print("Driver is not available for this session. \n")
    exit(1)

csv_file_bool = input("Do you want to process csv files? n/no y/yes\n").upper()
if not csv_file_bool in ['N', 'NO', 'Y', 'YES']:
    print("Invalid answer. Exiting program.")
    exit(1)
elif csv_file_bool in ['Y', 'YES']:
    csv_file_manager(session_drivers)
elif csv_file_bool in ['N', 'NO']:
    print("Skipping over csv file processing.\n")


### ANALYSIS

driver_performances = pd.Series(dtype=object)

#Create Objects with all required information
for driver in session_drivers:

    performance = driver_Performance(driver)

    #getting pit_time
    pit_time, pit_laps = get_pit_time_of_driver(driver)
    performance.pit_avr_stoptime = get_pit_avr_stoptime(driver)
    #print(performance.pit_avr_stoptime)
    """print("Pit times: ", pit_time, "\n")
    print("Pit laps: ", pit_laps, "\n")
    print("\n")"""

    performance.pit_intervals = pit_laps

    performance.original_time = pit_time.iloc[-1]
    
    #getting time for each lap in a pit 
    time_per_lap_per_pit_time_per_pit = get_lap_time_per_pit_time_of_driver(pit_time, driver)
    """print("Time per laps per pit (using dictionary for pit): ")
    print(time_per_lap_per_pit_time_per_pit)
    print("\n")"""

    performance.results = get_pit_trends_coeffs_residuals_data(time_per_lap_per_pit_time_per_pit)

    plot_times(performance, show_seconds=(1))

    #add to series of objects
    driver_performances = pd.concat([pd.Series([performance])])


# Use Objects to do some analysis
for performance in driver_performances:
    optimize_pit(performance, show_seconds=(3))




