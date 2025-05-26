import fastf1

fastf1.Cache.enable_cache('./fastf1cache')

session = fastf1.get_session(2024, 'EMI', 'R')

session.load(telemetry=True, laps=True, weather=True)

driver_verstappen = session.get_driver('VER')
