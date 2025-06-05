#!/usr/bin/env python3

import fastf1
import pandas as pd
import os

try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not installed; proceeding without progress bars.")
    tqdm = lambda x, **kwargs: x

# Enable cache
cache_dir = '/Users/foml/coding/MSP/period 6/f1cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)


year = 2024
schedule = fastf1.get_event_schedule(year)

print("Races that will be processed:")
for _, event in schedule.iterrows():
    print(f"- {event['EventName']}")
print()

sessions = ['R']
base_dir = 'csv_exports'
os.makedirs(base_dir, exist_ok=True)

errors = []

for race_num, (_, event) in enumerate(tqdm(schedule.iterrows(), total=len(schedule), desc="Races"), start=1):
    gp = event['EventName']
    gp_slug = gp.replace(' ', '_').lower()
    race_dir = os.path.join(base_dir, f"{race_num:02d}_{gp_slug}")
    os.makedirs(race_dir, exist_ok=True)

    for ses in tqdm(sessions, desc=f"Sessions for {gp_slug}", leave=False):
        laps_file = os.path.join(race_dir, f"{year}_{race_num:02d}_{gp_slug}_{ses}_laps.csv")
        telemetry_file = os.path.join(race_dir, f"{year}_{race_num:02d}_{gp_slug}_{ses}_telemetry.csv")

        if os.path.exists(laps_file) and os.path.exists(telemetry_file):
            print(f"All files already exist for {gp} {ses}, skipping session load.")
            continue

        try:
            session = fastf1.get_session(year, gp, ses)
            session.load(telemetry=True, laps=True)
        except Exception as e:
            errors.append({'race_num': race_num, 'gp': gp, 'session': ses, 'error': str(e)})
            continue

        # Always export both laps and telemetry data
        laps = session.laps
        laps.to_csv(laps_file, index=False)

        tel_dfs = []
        for _, lap in session.laps.iterlaps():
            df = lap.get_car_data()
            df['Driver'] = lap.Driver
            df['LapNumber'] = lap.LapNumber
            tel_dfs.append(df)
        if tel_dfs:
            telemetry = pd.concat(tel_dfs, ignore_index=True)
            telemetry.to_csv(telemetry_file, index=False)

if errors:
    df_errors = pd.DataFrame(errors)
    report_file = os.path.join(base_dir, f"{year}_error_report.csv")
    df_errors.to_csv(report_file, index=False)
    print(f"\nError report written to {report_file}")
    print("Errors details:")
    for err in errors:
        print(f"- Race {err['race_num']} - {err['gp']} {err['session']}: {err['error']}")