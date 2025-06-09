import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import numpy as np
import pandas as pd
import fastf1
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
from sympy import symbols, sympify, integrate, lambdify
from scipy.optimize import minimize_scalar
import itertools

# Load tire info at the beginning
tire_info_df = pd.read_excel('book_tire_information.xlsx')

### OBJECTS

class driver_Performance:
    def __init__(self, driver=str, pit_intervals=pd.Series(dtype=int), results=None, pit_avr_stoptime=None, original_time=None, lap_count=None):
        self.driver = driver
        self.pit_intervals = pit_intervals #interval in laps
        self.results = results
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
            time_per_lap_per_pit_time = pd.to_timedelta(time_per_lap_per_pit_time, errors='coerce')
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
    tiretype = None
    for pit in range(0,len(time_per_lap_per_pit_time_per_pit)):
        data = time_per_lap_per_pit_time_per_pit[f"pit_{pit}"]
        
        index = range(lap_offset,len(data)+lap_offset)
        values = data.values
        
        # Compute linear trend line ignoring the first/last data point due to starting/end lap time delay
        coeffs = np.polyfit(index[1:-1], values[1:-1], 1)
        trends = np.poly1d(coeffs)(index)

        coeffs_quad = np.polyfit(index[1:-1], values[1:-1], 2)
        coeffs_cube = np.polyfit(index[1:-1], values[1:-1], 3)
        coeffs_exp = np.polyfit(index[1:-1], np.log(values[1:-1]), 1)

        # Compute residuals and standard error
        residuals = values[1:-1] - trends[1:-1]
        std_err = np.std(residuals)
        lap_offset += 1
        
        results[f"pit_{pit}"] = {
        "index": index,
        "values": values,
        "coeffs": coeffs,
        "trends": trends,
        "coeffs_quad": coeffs_quad,
        "coeffs_cube": coeffs_cube,
        "coeffs_exp": coeffs_exp,
        "residuals": residuals,
        "std_err": std_err,
        "tiretype": tiretype
        }

    return results

def get_pit_tiretype(performance,driver):
    pit_count = 0

    for index, value in list(session.laps.pick_drivers(driver).Time.items())[1:]:
        if not pd.isna(session.laps.pick_drivers(driver).PitInTime.loc[index]) or index == session.laps.pick_drivers(driver).index[-1]:
            if pd.isna(session.laps.PitInTime.loc[index]):
                tiretype = session.laps.pick_drivers(driver).Compound.loc[index] #trying to get compounds when pits happen
                performance.results[f"pit_{pit_count}"]["tiretype"] = str(tiretype).strip().upper()
                pit_count += 1
            else:
                tiretype =  session.laps.pick_drivers(driver).Compound.loc[index] #trying to get compounds when pits happen
                performance.results[f"pit_{pit_count}"]["tiretype"] = str(tiretype).strip().upper()
                pit_count += 1
        if pit_count == len(performance.pit_intervals):
            break

    return performance.results

def plt_show_sec(duration=None):
    if duration is None:
        plt.show()
    else:
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


# New function: plot_combined_equations
def plot_combined_equations(driver_performances, compound, compound_groups):
    import matplotlib.pyplot as plt
    # Build compound_map from tire_info_df, like in export_driver_performances_to_csv
    race_index = int(session_selection) + 1
    tire_row = tire_info_df[tire_info_df['Race #'] == race_index]
    if not tire_row.empty:
        tire_row = tire_row.iloc[0]
        compound_map = {
            'HARD': tire_row['Hard'],
            'MEDIUM': tire_row['Medium'],
            'SOFT': tire_row['Soft']
        }
    else:
        compound_map = {}

    # Reverse map: compound to tiretype (e.g. "C3" -> "SOFT")
    compound_to_tiretype = {v: k for k, v in compound_map.items()}

    # Restrict x to the hard max lap count for the given compound
    race_index = int(session_selection) + 1
    hard_max_raw = tire_info_df.loc[tire_info_df['Race #'] == race_index, f"{compound} Max Laps"].values
    hard_max = int(np.ceil(hard_max_raw[0])) if len(hard_max_raw) > 0 and pd.notna(hard_max_raw[0]) else 20
    x_full = np.linspace(1, hard_max + 15, 120)
    x_r2 = np.linspace(1, hard_max, 100)
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()
    titles = ['Linear', 'Quadratic', 'Cubic', 'Exponential']
    coeff_keys = ['lin', 'quad', 'cube', 'exp']

    # --- Load R² values from the final CSV file for each driver and fit type ---
    # Build a dictionary: r2_dict[compound][fit_type] = list of R² values
    corrected_event_name = race_name_corrections.get(session_event, session_event)
    season_year = session.event.year
    csv_filename = csv_location + f"csv_results_{session_selection}_{season_year}_{corrected_event_name.replace(' ', '_')}_Grand_Prix.csv"
    import os
    # Add check for missing/empty CSV file
    if not os.path.exists(csv_filename) or os.path.getsize(csv_filename) == 0:
        print(f"⚠️ Skipping plot: CSV file '{csv_filename}' is empty or missing.")
        return
    r2_dict = {}
    df_csv = pd.read_csv(csv_filename)
    # Only keep rows that are not the average equation (i.e., not "AVERAGE_EQUATION_<compound>")
    for _, row in df_csv.iterrows():
        cmpd = str(row.get("compound", "")).strip()
        driver = str(row.get("driver", ""))
        if driver.startswith("AVERAGE_EQUATION"):
            continue
        if cmpd == "":
            continue
        if cmpd not in r2_dict:
            r2_dict[cmpd] = {"lin": [], "quad": [], "cube": [], "exp": []}
        # Try to parse each R² value, skip if missing or nan
        for key in ["lin", "quad", "cube", "exp"]:
            col = f"r2_{key}"
            val = row.get(col, "")
            try:
                valf = float(val)
                if not np.isnan(valf):
                    r2_dict[cmpd][key].append(valf)
            except Exception:
                pass
    # If no CSV or missing compound, fallback to empty lists
    for cmpd in compound_groups:
        if cmpd not in r2_dict:
            r2_dict[cmpd] = {"lin": [], "quad": [], "cube": [], "exp": []}
        for key in ["lin", "quad", "cube", "exp"]:
            if key not in r2_dict[cmpd]:
                r2_dict[cmpd][key] = []

    for i, (title, key) in enumerate(zip(titles, coeff_keys)):
        ax = axs[i]
        ax.set_title(f"{title} Fits for Compound {compound}")
        ax.set_xlabel("Lap Index")
        ax.set_ylabel("Lap Time (s)")

        # Plot each driver's stint equation for this compound and fit type
        for perf in driver_performances:
            for data in getattr(perf, "results", {}).values():
                cpd = compound_map.get(str(data.get("tiretype", "")).strip().upper())
                if cpd != compound:
                    continue
                if key == 'exp' and "coeffs_exp" in data:
                    a = np.exp(data["coeffs_exp"][1])
                    b = data["coeffs_exp"][0]
                    ax.plot(x_full, a * np.exp(b * x_full), alpha=0.3)
                elif key == 'lin' and "coeffs" in data:
                    coeffs = data["coeffs"]
                    y = coeffs[0] * x_full + coeffs[1]
                    ax.plot(x_full, y, alpha=0.3)
                elif f"coeffs_{key}" in data and key != 'lin':
                    coeffs = data[f"coeffs_{key}"]
                    if key == 'quad':
                        y = coeffs[0] * x_full**2 + coeffs[1] * x_full + coeffs[2]
                    elif key == 'cube':
                        y = coeffs[0] * x_full**3 + coeffs[1] * x_full**2 + coeffs[2] * x_full + coeffs[3]
                    ax.plot(x_full, y, alpha=0.3)

        # Plot average for this compound and fit type, and include R² in the label
        r2_val = None
        nan_contributors = []
        nan_present = False
        if key in compound_groups[compound] and compound_groups[compound][key]:
            avg = np.mean(compound_groups[compound][key], axis=0)
            # --- Use R² values from CSV for this compound and fit type ---
            r2_values_list = r2_dict.get(compound, {}).get(key, [])
            # Exclude NaNs or empty
            filtered_r2s = [v for v in r2_values_list if not np.isnan(v)]
            if filtered_r2s:
                r2_val = np.mean(filtered_r2s)
            # Build the label for the average line, including R² if available
            label = f'Average Fit (R²={r2_val:.3f})' if r2_val is not None else 'Average Fit'

            # Now plot the average curve
            if key == 'exp':
                a = np.exp(np.mean([c[1] for c in compound_groups[compound][key]]))
                b = np.mean([c[0] for c in compound_groups[compound][key]])
                ax.plot(x_full, a * np.exp(b * x_full), 'k--', label=label, linewidth=2)
            else:
                if key == 'lin':
                    y = avg[0] * x_full + avg[1]
                elif key == 'quad':
                    y = avg[0] * x_full**2 + avg[1] * x_full + avg[2]
                elif key == 'cube':
                    y = avg[0] * x_full**3 + avg[1] * x_full**2 + avg[2] * x_full + avg[3]
                else:
                    y = None
                ax.plot(x_full, y, 'k--', label=label, linewidth=2)

            # After plotting the average line, add axvline for compound_avg and compound_max
            # Find the correct tire_row for this compound (from tire_info_df)
            race_index = int(session_selection) + 1
            tire_row = tire_info_df[tire_info_df['Race #'] == race_index]
            if not tire_row.empty:
                tire_row = tire_row.iloc[0]
                # Find the column name for this compound
                compound_avg = tire_row.get(f"{compound} Avg Laps", None)
                compound_max = tire_row.get(f"{compound} Max Laps", None)
                # Only plot label on first subplot to avoid duplicates
                if pd.notna(compound_avg):
                    ax.axvline(compound_avg, color='orange', linestyle='--', label='Soft max (avg)' if i == 0 else None)
                if pd.notna(compound_max):
                    ax.axvline(compound_max, color='red', linestyle='--', label='Hard Max' if i == 0 else None)
            # If not found, skip axvline

        # --- Add shaded region for extrapolated data (beyond hard_max to x_full[-1]) ---
        ax.axvspan(hard_max, x_full[-1], color='gray', alpha=0.2, label='Extrapolated Region' if i == 0 else None)

        ax.legend()

    plt.tight_layout()
    plt.show()

def export_driver_performances_to_csv(driver_performances, csv_location, show_individual_fit_plots=False):
    corrected_event_name = race_name_corrections.get(session_event, session_event)
    # Get tire info for this event using Race # (session_selection + 1)
    race_index = int(session_selection) + 1
    tire_row = tire_info_df[tire_info_df['Race #'] == race_index]
    if not tire_row.empty:
        tire_row = tire_row.iloc[0]
        compound_map = {
            'HARD': tire_row['Hard'],
            'MEDIUM': tire_row['Medium'],
            'SOFT': tire_row['Soft']
        }
    else:
        compound_map = {}
    headers = [
        'driver',
        'stint', 'tiretype', 'compound',
        'compound_avg', 'compound_max',
        'pit_interval',
        'function_lin',
        'function_quad',
        'function_cube',
        'function_exp',
        'r2_lin', 'r2_quad', 'r2_cube', 'r2_exp',
        'std_err',
        'value_mean', 'residual_std',
        'besttrendline',
        'besttrendline_r2'
    ]
    # Add comments column if not present
    if "comments" not in headers:
        headers.append("comments")

    # Add season/year to filename
    season_year = session.event.year
    session_number = session_selection
    csv_filename = f"csv_results_{session_number}_{season_year}_{corrected_event_name.replace(' ', '_')}_Grand_Prix.csv"
    filename = csv_location + csv_filename

    # Erase file content if it already exists and is not empty
    if os.path.isfile(filename) and os.path.getsize(filename) > 0:
        open(filename, 'w').close()  # truncate the file

    # Compute average equations for each compound used
    avg_eq_rows = []
    compound_groups = {}
    for perf in driver_performances:
        for stint_name, data in perf.results.items():
            tiretype_raw = str(data.get("tiretype", "")).strip().upper()
            compound = compound_map.get(tiretype_raw)
            # print(f"Processing stint: {stint_name} | Driver: {perf.driver} | Compound: {compound} | Tiretype: {tiretype_raw}")
            if not compound:
                continue  # Skip if mapping fails
            if compound not in compound_groups:
                compound_groups[compound] = {
                    "lin": [], "quad": [], "cube": [], "exp": [],
                    "std_err": [], "value_mean": [], "residual_std": []
                }
            # Always collect coefficient arrays if present
            if "coeffs" in data:
                compound_groups[compound]["lin"].append(data["coeffs"])
            if "coeffs_quad" in data:
                compound_groups[compound]["quad"].append(data["coeffs_quad"])
            if "coeffs_cube" in data:
                compound_groups[compound]["cube"].append(data["coeffs_cube"])
            if "coeffs_exp" in data:
                compound_groups[compound]["exp"].append(data["coeffs_exp"])
            # Collect statistical metrics for this stint, per compound
            if "std_err" in data:
                compound_groups[compound]["std_err"].append(data["std_err"])
            # REPLACED BLOCK: value_mean collection with NaN/empty check and logging
            if "values" in data:
                values_array = np.array(data["values"])
                if values_array.size > 0 and not np.isnan(values_array).all():
                    compound_groups[compound]["value_mean"].append(np.mean(values_array))
                else:
                    # print(f"Skipping empty or NaN-only values for compound {compound}")
                    pass
            if "residuals" in data and len(data["residuals"]) > 0:
                compound_groups[compound]["residual_std"].append(np.std(data["residuals"]))

    for compound, coeff_sets in compound_groups.items():
        row = {key: "" for key in headers}
        row["driver"] = f"AVERAGE_EQUATION_{compound}"
        tiretype_text = [k for k, v in compound_map.items() if v == compound]
        row["tiretype"] = tiretype_text[0].capitalize() if tiretype_text else "Unknown"
        row["compound"] = compound

        soft_avg_raw = tire_row.get(f"{compound} Avg Laps", None)
        hard_max_raw = tire_row.get(f"{compound} Max Laps", None)
        row["compound_avg"] = int(np.ceil(soft_avg_raw)) if pd.notna(soft_avg_raw) else ""
        row["compound_max"] = int(np.ceil(hard_max_raw)) if pd.notna(hard_max_raw) else ""

        # Prepare for R2 calculation
        try:
            hard_max_laps = int(row["compound_max"])
        except Exception:
            hard_max_laps = None
        x = np.arange(1, hard_max_laps + 1) if hard_max_laps else np.arange(1, 21)
        r2_lin = r2_quad = r2_cube = r2_exp = ""

        nan_contributors_r2 = {"lin": [], "quad": [], "cube": [], "exp": []}
        nan_contributors_valmean = []

        # Linear
        if coeff_sets["lin"]:
            avg = np.mean(coeff_sets["lin"], axis=0)
            row["function_lin"] = f"{avg[0]:.5f} * x + {avg[1]:.5f}"
            y_true = []
            y_true_contributors = []
            for perf in driver_performances:
                for stint_name, data in perf.results.items():
                    cpd = compound_map.get(str(data.get("tiretype", "")).strip().upper())
                    if cpd == compound and "values" in data:
                        vals = np.array(data["values"])
                        if vals.size > 0:
                            if np.isnan(vals).any():
                                nan_contributors_r2["lin"].append(f"{perf.driver}-{stint_name}")
                            if not np.isnan(vals).all():
                                y_true.extend(vals[~np.isnan(vals)].tolist())
                                y_true_contributors.append(f"{perf.driver}-{stint_name}")
            y_true = np.array(y_true)
            x_r2 = np.arange(1, hard_max_laps + 1) if hard_max_laps else np.arange(1, 21)
            if y_true.size < x_r2.size:
                if y_true.size > 0:
                    pad_val = y_true[-1]
                    y_true = np.concatenate([y_true, np.full(x_r2.size - y_true.size, pad_val)])
                else:
                    y_true = np.zeros(x_r2.size)
            elif y_true.size > x_r2.size:
                y_true = y_true[:x_r2.size]
            y_pred = avg[0] * x_r2 + avg[1]
            if y_true.size > 0 and not np.isnan(y_true).all():
                y_true_mean = np.mean(y_true)
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - y_true_mean) ** 2)
                r2_lin = 1 - ss_res / ss_tot if ss_tot != 0 else ""
                row["r2_lin"] = round(r2_lin, 5)
                if nan_contributors_r2["lin"]:
                    row["comments"] += f"NaNs ignored in average R² for {compound} (linear); contributors: {', '.join(nan_contributors_r2['lin'])}. "
            else:
                row["r2_lin"] = ""
        else:
            row["r2_lin"] = ""

        # Quadratic
        if coeff_sets["quad"]:
            avg = np.mean(coeff_sets["quad"], axis=0)
            row["function_quad"] = f"{avg[0]:.5f} * x**2 + {avg[1]:.5f} * x + {avg[2]:.5f}"
            y_true = []
            for perf in driver_performances:
                for stint_name, data in perf.results.items():
                    cpd = compound_map.get(str(data.get("tiretype", "")).strip().upper())
                    if cpd == compound and "values" in data:
                        vals = np.array(data["values"])
                        if vals.size > 0:
                            if np.isnan(vals).any():
                                nan_contributors_r2["quad"].append(f"{perf.driver}-{stint_name}")
                            if not np.isnan(vals).all():
                                y_true.extend(vals[~np.isnan(vals)].tolist())
            y_true = np.array(y_true)
            x_r2 = np.arange(1, hard_max_laps + 1) if hard_max_laps else np.arange(1, 21)
            if y_true.size < x_r2.size:
                if y_true.size > 0:
                    pad_val = y_true[-1]
                    y_true = np.concatenate([y_true, np.full(x_r2.size - y_true.size, pad_val)])
                else:
                    y_true = np.zeros(x_r2.size)
            elif y_true.size > x_r2.size:
                y_true = y_true[:x_r2.size]
            y_pred = avg[0] * (x_r2 ** 2) + avg[1] * x_r2 + avg[2]
            if y_true.size > 0 and not np.isnan(y_true).all():
                y_true_mean = np.mean(y_true)
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - y_true_mean) ** 2)
                r2_quad = 1 - ss_res / ss_tot if ss_tot != 0 else ""
                row["r2_quad"] = round(r2_quad, 5)
                if nan_contributors_r2["quad"]:
                    row["comments"] += f"NaNs ignored in average R² for {compound} (quadratic); contributors: {', '.join(nan_contributors_r2['quad'])}. "
            else:
                row["r2_quad"] = ""
        else:
            row["r2_quad"] = ""

        # Cubic
        if coeff_sets["cube"]:
            avg = np.mean(coeff_sets["cube"], axis=0)
            row["function_cube"] = f"{avg[0]:.5f} * x**3 + {avg[1]:.5f} * x**2 + {avg[2]:.5f} * x + {avg[3]:.5f}"
            y_true = []
            for perf in driver_performances:
                for stint_name, data in perf.results.items():
                    cpd = compound_map.get(str(data.get("tiretype", "")).strip().upper())
                    if cpd == compound and "values" in data:
                        vals = np.array(data["values"])
                        if vals.size > 0:
                            if np.isnan(vals).any():
                                nan_contributors_r2["cube"].append(f"{perf.driver}-{stint_name}")
                            if not np.isnan(vals).all():
                                y_true.extend(vals[~np.isnan(vals)].tolist())
            y_true = np.array(y_true)
            x_r2 = np.arange(1, hard_max_laps + 1) if hard_max_laps else np.arange(1, 21)
            if y_true.size < x_r2.size:
                if y_true.size > 0:
                    pad_val = y_true[-1]
                    y_true = np.concatenate([y_true, np.full(x_r2.size - y_true.size, pad_val)])
                else:
                    y_true = np.zeros(x_r2.size)
            elif y_true.size > x_r2.size:
                y_true = y_true[:x_r2.size]
            y_pred = avg[0] * (x_r2 ** 3) + avg[1] * (x_r2 ** 2) + avg[2] * x_r2 + avg[3]
            if y_true.size > 0 and not np.isnan(y_true).all():
                y_true_mean = np.mean(y_true)
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - y_true_mean) ** 2)
                r2_cube = 1 - ss_res / ss_tot if ss_tot != 0 else ""
                row["r2_cube"] = round(r2_cube, 5)
                if nan_contributors_r2["cube"]:
                    row["comments"] += f"NaNs ignored in average R² for {compound} (cubic); contributors: {', '.join(nan_contributors_r2['cube'])}. "
            else:
                row["r2_cube"] = ""
        else:
            row["r2_cube"] = ""

        # Exponential
        if coeff_sets["exp"]:
            log_a = np.log([c[1] for c in coeff_sets["exp"]])
            b = [c[0] for c in coeff_sets["exp"]]
            row["function_exp"] = f"{np.exp(np.mean(log_a)):.5f} * e**({np.mean(b):.5f} * x)"
            y_true = []
            for perf in driver_performances:
                for stint_name, data in perf.results.items():
                    cpd = compound_map.get(str(data.get("tiretype", "")).strip().upper())
                    if cpd == compound and "values" in data:
                        vals = np.array(data["values"])
                        if vals.size > 0:
                            if np.isnan(vals).any():
                                nan_contributors_r2["exp"].append(f"{perf.driver}-{stint_name}")
                            if not np.isnan(vals).all():
                                y_true.extend(vals[~np.isnan(vals)].tolist())
            y_true = np.array(y_true)
            x_r2 = np.arange(1, hard_max_laps + 1) if hard_max_laps else np.arange(1, 21)
            if y_true.size < x_r2.size:
                if y_true.size > 0:
                    pad_val = y_true[-1]
                    y_true = np.concatenate([y_true, np.full(x_r2.size - y_true.size, pad_val)])
                else:
                    y_true = np.zeros(x_r2.size)
            elif y_true.size > x_r2.size:
                y_true = y_true[:x_r2.size]
            a = np.exp(np.mean(log_a))
            b_mean = np.mean(b)
            y_pred = a * np.exp(b_mean * x_r2)
            if y_true.size > 0 and not np.isnan(y_true).all():
                y_true_mean = np.mean(y_true)
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - y_true_mean) ** 2)
                r2_exp = 1 - ss_res / ss_tot if ss_tot != 0 else ""
                row["r2_exp"] = round(r2_exp, 5)
                if nan_contributors_r2["exp"]:
                    row["comments"] += f"NaNs ignored in average R² for {compound} (exponential); contributors: {', '.join(nan_contributors_r2['exp'])}. "
            else:
                row["r2_exp"] = ""
        else:
            row["r2_exp"] = ""

        # Add statistical metrics to the row if available (refactored for empty sets)
        row["std_err"] = round(np.mean(coeff_sets["std_err"]), 5) if coeff_sets["std_err"] else ""
        # New block: filter out NaNs from value_mean before averaging
        filtered_values = [v for v in coeff_sets["value_mean"] if not np.isnan(v)]
        row["value_mean"] = round(np.mean(filtered_values), 5) if filtered_values else ""
        if len(filtered_values) < len(coeff_sets["value_mean"]):
            nan_contributors_valmean = []
            for perf in driver_performances:
                for stint_name, data in perf.results.items():
                    cpd = compound_map.get(data.get("tiretype", "").strip().upper())
                    if cpd == compound:
                        values_array = np.array(data.get("values", []))
                        if values_array.size > 0 and np.isnan(values_array).any():
                            nan_contributors_valmean.append(f"{perf.driver}-{stint_name}")
            row["comments"] += f"NaNs ignored in value_mean for {compound}; contributors: {', '.join(nan_contributors_valmean)}. "
        row["residual_std"] = round(np.mean(coeff_sets["residual_std"]), 5) if coeff_sets["residual_std"] else ""

        # --- Begin: Determine best trendline for average row ---
        # Gather R2 and function strings, skipping NaNs/missing
        r2_dict = {
            'lin': row.get("r2_lin", ""),
            'quad': row.get("r2_quad", ""),
            'cube': row.get("r2_cube", ""),
            'exp': row.get("r2_exp", "")
        }
        eqn_dict = {
            'lin': row.get("function_lin", ""),
            'quad': row.get("function_quad", ""),
            'cube': row.get("function_cube", ""),
            'exp': row.get("function_exp", "")
        }
        # Only consider R2 values that are numbers and not NaN
        valid_r2 = {k: v for k, v in r2_dict.items() if isinstance(v, (float, int)) and not np.isnan(v)}
        # If some R2s are string representations of numbers, try to convert
        for k in list(r2_dict.keys()):
            if k not in valid_r2:
                try:
                    v = float(r2_dict[k])
                    if not np.isnan(v):
                        valid_r2[k] = v
                except Exception:
                    continue
        best_fit = max(valid_r2.items(), key=lambda x: x[1])[0] if valid_r2 else ""
        best_eq_string = eqn_dict[best_fit] if best_fit and eqn_dict[best_fit] else ""
        best_eq_type = best_fit
        # Store only the raw equation string in besttrendline, without R²
        row["besttrendline"] = str(best_eq_string) if best_eq_string else ""
        # Add besttrendline_r2 column
        row["besttrendline_r2"] = r2_dict[best_eq_type] if best_eq_type and r2_dict[best_eq_type] != "" else ""
        # --- End: Determine best trendline for average row ---

        avg_eq_rows.append(row)
    season_year = session.event.year
    # Determine total laps in the session
    # Try session.total_laps, otherwise fallback to session.laps['LapNumber'].max()
    total_laps = getattr(session, "total_laps", None)
    if total_laps is None or (isinstance(total_laps, float) and np.isnan(total_laps)):
        if hasattr(session, "laps") and "LapNumber" in session.laps.columns:
            total_laps = int(session.laps["LapNumber"].max())
        else:
            total_laps = ""
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # Write the season/year as a metadata row at the top
        # writer.writerow([f"Season: {session.event.year}"])  # (removed, now in column)
        # Insert "Total Laps" after "Season" in the header row
        writer.writerow(["Season", "Total Laps"] + headers)  # write header with "Season" and "Total Laps" as first columns

        # Safeguard: Ensure 'driver' key exists in each average row
        for row in avg_eq_rows:
            if "driver" not in row:
                row["driver"] = f"AVERAGE_EQUATION_UNKNOWN"
        # Write average equation rows for each compound, including comments, with Total Laps as second column
        for row in avg_eq_rows:
            writer.writerow([season_year, total_laps] + [row.get(col, "") for col in headers])

        # --- Begin: Setup for combined subplot plotting if enabled ---
        if show_individual_fit_plots:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(20, 25))
            axes = axes.flatten()
            plot_idx = 0
        # --- End: Setup for subplot plotting ---

        for performance in driver_performances:
            driver = getattr(performance, "driver", "UNKNOWN")
            for stint_name, data in performance.results.items():
                tiretype = data.get('tiretype', 'UNKNOWN')
                # Map compound
                compound = compound_map.get(str(tiretype).strip().upper(), 'UNKNOWN')
                # Add compound_avg and compound_max columns after compound
                soft_avg_raw = tire_row.get(f"{compound} Avg Laps", None)
                hard_max_raw = tire_row.get(f"{compound} Max Laps", None)
                soft_avg_laps = int(np.ceil(soft_avg_raw)) if pd.notna(soft_avg_raw) else None
                hard_max_laps = int(np.ceil(hard_max_raw)) if pd.notna(hard_max_raw) else None
                coeffs = data.get('coeffs', None)
                coeffs_quad = data.get('coeffs_quad', None)
                coeffs_cube = data.get('coeffs_cube', None)
                coeffs_exp = data.get('coeffs_exp', None)

                if coeffs is not None and len(coeffs) >= 2:
                    slope = float(coeffs[0])
                    intercept = float(coeffs[1])
                else:
                    slope = intercept = None

                std_err = round(float(data.get('std_err', 0.0)), 5)
                index = data.get('index', range(0))
                pit_interval = len(index) if isinstance(index, range) else None
                function_lin = f"{slope:.5f} * x + {intercept:.5f}"
                function_quad = f"{coeffs_quad[0]:.5f} * x**2 + {coeffs_quad[1]:.5f} * x + {coeffs_quad[2]:.5f}"
                function_cube = f"{coeffs_cube[0]:.5f} * x**3 + {coeffs_cube[1]:.5f} * x**2 + {coeffs_cube[2]:.5f} * x + {coeffs_cube[3]:.5f}"
                function_exp = f"{np.exp(coeffs_exp[1]):.5f} * e**({coeffs_exp[0]:.5f} * x)"
                values = np.array(data.get('values', []))
                # Filter out NaNs for 
                #value_mean and log if any were present
                filtered_values = values[~np.isnan(values)] if values.size else np.array([])
                value_mean = round(float(np.mean(filtered_values)), 5) if filtered_values.size else None
                comments = ""
                if filtered_values.size < values.size:
                    nan_count = np.isnan(values).sum()
                    print(f"⚠️ {nan_count} NaN values found in values for driver {driver}, stint {stint_name}, compound {compound}, tiretype {tiretype}, ignoring them in average.")
                    comments = f"{nan_count} NaNs ignored for {driver}, {stint_name}, {tiretype} ({compound})"
                residuals = np.array(data.get('residuals', []))
                residual_std = round(float(np.std(residuals)), 5) if residuals.size else None

                # --- Begin R2 calculations for this stint ---
                r2_lin = ""
                r2_quad = ""
                r2_cube = ""
                r2_exp = ""
                # Before computing R², check if values.size < 5
                if values.size < 5:
                    msg = f"R² not computed for {driver}, {stint_name}, {tiretype} due to insufficient data (<5 laps). "
                    print(f"⚠️ {msg}")
                    comments = (comments + " " if comments else "") + msg
                    # All r2_* remain ""
                else:
                    # Linear
                    if coeffs is not None and values.size:
                        x_vals = np.arange(1, values.size + 1)
                        y_pred = coeffs[0] * x_vals + coeffs[1]
                        y_true = values
                        y_mean = np.mean(y_true)
                        ss_res = np.sum((y_true - y_pred) ** 2)
                        ss_tot = np.sum((y_true - y_mean) ** 2)
                        r2_lin = round(1 - ss_res / ss_tot, 5) if ss_tot else ""
                    # Quadratic
                    if coeffs_quad is not None and values.size:
                        x_vals = np.arange(1, values.size + 1)
                        y_pred = coeffs_quad[0] * x_vals**2 + coeffs_quad[1] * x_vals + coeffs_quad[2]
                        y_true = values
                        y_mean = np.mean(y_true)
                        ss_res = np.sum((y_true - y_pred) ** 2)
                        ss_tot = np.sum((y_true - y_mean) ** 2)
                        r2_quad = round(1 - ss_res / ss_tot, 5) if ss_tot else ""
                    # Cubic
                    if coeffs_cube is not None and values.size:
                        x_vals = np.arange(1, values.size + 1)
                        y_pred = coeffs_cube[0] * x_vals**3 + coeffs_cube[1] * x_vals**2 + coeffs_cube[2] * x_vals + coeffs_cube[3]
                        y_true = values
                        y_mean = np.mean(y_true)
                        ss_res = np.sum((y_true - y_pred) ** 2)
                        ss_tot = np.sum((y_true - y_mean) ** 2)
                        r2_cube = round(1 - ss_res / ss_tot, 5) if ss_tot else ""
                    # Exponential
                    if coeffs_exp is not None and values.size:
                        x_vals = np.arange(1, values.size + 1)
                        a = np.exp(coeffs_exp[1])
                        b = coeffs_exp[0]
                        y_pred = a * np.exp(b * x_vals)
                        y_true = values
                        y_mean = np.mean(y_true)
                        ss_res = np.sum((y_true - y_pred) ** 2)
                        ss_tot = np.sum((y_true - y_mean) ** 2)
                        r2_exp = round(1 - ss_res / ss_tot, 5) if ss_tot else ""
                # --- End R2 calculations for this stint ---

                # --- Begin: Determine best trendline ---
                # numpy already imported at top
                r2_dict = {
                    'lin': r2_lin,
                    'quad': r2_quad,
                    'cube': r2_cube,
                    'exp': r2_exp
                }
                valid_r2 = {k: v for k, v in r2_dict.items() if isinstance(v, (float, int)) and not np.isnan(v)}
                best_fit = max(valid_r2.items(), key=lambda x: x[1])[0] if valid_r2 else ""
                # Instead of best_equation as a dict, we want the equation string for the best fit
                besttrendline = ""
                if best_fit == "lin":
                    besttrendline = function_lin
                elif best_fit == "quad":
                    besttrendline = function_quad
                elif best_fit == "cube":
                    besttrendline = function_cube
                elif best_fit == "exp":
                    besttrendline = function_exp
                else:
                    besttrendline = ""
                # Store only the raw equation string in besttrendline, without R²
                besttrendline = str(besttrendline) if besttrendline else ""
                # Add besttrendline_r2 column value for this row
                besttrendline_r2 = r2_dict[best_fit] if best_fit and r2_dict[best_fit] != "" else ""
                # --- End: Determine best trendline ---

                # --- Begin: Optionally plot individual fits for debugging ---
                if show_individual_fit_plots:
                    x_vals = np.arange(1, len(values) + 1)
                    ax = axes[plot_idx]
                    ax.plot(x_vals, values, label='Actual', marker='o')
                    # Collect fit coefficients for each model
                    coeff_dict = {
                        'lin': coeffs,
                        'quad': coeffs_quad,
                        'cube': coeffs_cube,
                        'exp': coeffs_exp
                    }
                    for model_key, coeffs_model in coeff_dict.items():
                        if coeffs_model is not None:
                            if model_key == 'exp':
                                y_pred = coeffs_model[0] * np.exp(coeffs_model[1] * x_vals)
                            else:
                                y_pred = np.polyval(coeffs_model[::-1], x_vals)
                            ax.plot(x_vals, y_pred, label=f'{model_key} fit')
                    ax.set_title(f'{driver} - {stint_name} - {compound} ({tiretype})')
                    ax.set_xlabel('Lap')
                    ax.set_ylabel('Value Mean')
                    ax.legend()
                    plot_idx += 1
                # --- End: Optionally plot individual fits for debugging ---

                # Write row with Total Laps after Season
                writer.writerow([
                    season_year,
                    total_laps,
                    driver or "",
                    stint_name or "",
                    tiretype or "",
                    compound or "",
                    soft_avg_laps if soft_avg_laps is not None else "",
                    hard_max_laps if hard_max_laps is not None else "",
                    pit_interval if pit_interval is not None else "",
                    function_lin if function_lin is not None else "",
                    function_quad if function_quad is not None else "",
                    function_cube if function_cube is not None else "",
                    function_exp if function_exp is not None else "",
                    r2_lin,
                    r2_quad,
                    r2_cube,
                    r2_exp,
                    std_err if std_err is not None else "",
                    value_mean if value_mean is not None else "",
                    residual_std if residual_std is not None else "",
                    besttrendline,
                    besttrendline_r2,
                    comments
                ])

        # --- Show all subplots at the end if enabled ---
        if show_individual_fit_plots:
            plt.tight_layout()
            plt.show()
        # --- End subplot show ---

    print(f"Exported results to {filename}")
    

### LOADING AND SELECTING SESSION DATA
csv_location = 'csv_files/'
os.makedirs(csv_location, exist_ok=True)

# Enable a cache directory (will be created if it doesn't exist)
cache_dir = './fastf1cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

#session details using a dictionary
sessions = {
    "0": [2024, 'Bahrain Grand Prix', 'R', []],
    "1": [2024, 'Saudi Arabian Grand Prix', 'R', []],
    "2": [2024, 'Australian Grand Prix', 'R', []],
    "3": [2024, 'Japanese Grand Prix', 'R', []],
    "4": [2024, 'Chinese Grand Prix', 'R', []],
    "5": [2024, 'Miami Grand Prix', 'R', []],
    "6": [2024, 'Emilia Romagna Grand Prix', 'R', []],
    "7": [2024, 'Monaco Grand Prix', 'R', []],
    "8": [2024, 'Canadian Grand Prix', 'R', []],
    "9": [2024, 'Spanish Grand Prix', 'R', []],
    "10": [2024, 'Austrian Grand Prix', 'R', []],
    "11": [2024, 'British Grand Prix', 'R', []],
    "12": [2024, 'Hungarian Grand Prix', 'R', []],
    "13": [2024, 'Belgian Grand Prix', 'R', []],
    "14": [2024, 'Dutch Grand Prix', 'R', []],
    "15": [2024, 'Italian Grand Prix', 'R', []],
    "16": [2024, 'Azerbaijan Grand Prix', 'R', []],
    "17": [2024, 'Singapore Grand Prix', 'R', []],
    "18": [2024, 'United States Grand Prix', 'R', []],
    "19": [2024, 'Mexico City Grand Prix', 'R', []],
    "20": [2024, 'São Paulo Grand Prix', 'R', []],
    "21": [2024, 'Las Vegas Grand Prix', 'R', []],
    "22": [2024, 'Qatar Grand Prix', 'R', []],
    "23": [2024, 'Abu Dhabi Grand Prix', 'R', []]
}

race_name_corrections = {
    "Emilia Romagna Grand Prix": "Italy (Emilia-Romagna)",
    "Monaco Grand Prix": "Monaco"
}

#session-selection
# print("These are the available sessions: \n")
# for key in sorted(sessions.keys(), key=int):
#     print(f"{key}: {sessions[key]}")

session_selection = input("Enter which session you would like to analyse (default is 0):\n").strip() or "0"
if session_selection.isnumeric() and session_selection in sessions:
    # Use event name, type, and year for session loading
    session_year = sessions[session_selection][0]
    session_event = sessions[session_selection][1]
    session_type = sessions[session_selection][2]
    session_name = session_event + '_' + str(session_year) + '_' + session_type
else:
    print("The session you selected is unavailable. Exiting program.")
    exit(1)

 # Load the session with everything
try:
    session = fastf1.get_session(session_year, session_event, session_type)
    session.load(telemetry=True, laps=True, weather=True)
    # Patch: Use event date for season_year instead of event['Year']
    # season_year = session.event['Year']
    season_year = pd.to_datetime(session.event['EventDate']).year
except Exception as e:
    print(f"Failed to load session: {e}")
    exit(1)
print("Succesfully loaded session! \n")


print("Defaulting to analysis of all drivers in selected session.\n")
session_drivers = session.laps['Driver'].unique().tolist()

csv_file_bool = input("Do you want to process csv files? (press Enter to skip) y/yes to confirm\n").upper()
if csv_file_bool in ['Y', 'YES']:
    csv_file_manager(session_drivers)
else:
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
    performance.results = get_pit_tiretype(performance,driver) 

    # plot_times(performance, show_seconds=None)

    #add to series of objects
    driver_performances = pd.concat([driver_performances,pd.Series([performance])])

# Use Objects to do some analysis
print(len(driver_performances))

# After export_driver_performances_to_csv's compound_groups are generated, call plot_combined_equations for each compound.
# To do this, we need to duplicate the compound_groups generation logic here, since export_driver_performances_to_csv does not return it.
# So, reconstruct compound_groups as in export_driver_performances_to_csv:
corrected_event_name = race_name_corrections.get(session_event, session_event)
race_index = int(session_selection) + 1
tire_row = tire_info_df[tire_info_df['Race #'] == race_index]
if not tire_row.empty:
    tire_row = tire_row.iloc[0]
    compound_map = {
        'HARD': tire_row['Hard'],
        'MEDIUM': tire_row['Medium'],
        'SOFT': tire_row['Soft']
    }
else:
    compound_map = {}
compound_groups = {}
for perf in driver_performances:
    for stint_name, data in perf.results.items():
        tiretype_raw = str(data.get("tiretype", "")).strip().upper()
        compound = compound_map.get(tiretype_raw)
        if not compound:
            continue
        if compound not in compound_groups:
            compound_groups[compound] = {
                "lin": [], "quad": [], "cube": [], "exp": [],
                "std_err": [], "value_mean": [], "residual_std": []
            }
        if "coeffs" in data:
            compound_groups[compound]["lin"].append(data["coeffs"])
        if "coeffs_quad" in data:
            compound_groups[compound]["quad"].append(data["coeffs_quad"])
        if "coeffs_cube" in data:
            compound_groups[compound]["cube"].append(data["coeffs_cube"])
        if "coeffs_exp" in data:
            compound_groups[compound]["exp"].append(data["coeffs_exp"])
        if "std_err" in data:
            compound_groups[compound]["std_err"].append(data["std_err"])
        if "values" in data:
            values_array = np.array(data["values"])
            if values_array.size > 0 and not np.isnan(values_array).all():
                compound_groups[compound]["value_mean"].append(np.mean(values_array))
        if "residuals" in data and len(data["residuals"]) > 0:
            compound_groups[compound]["residual_std"].append(np.std(data["residuals"]))

for compound in compound_groups:
    plot_combined_equations(driver_performances, compound, compound_groups)

export_driver_performances_to_csv(driver_performances, csv_location)
