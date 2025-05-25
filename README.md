# F1RaceProject
The Repository for the Project Period June 2025.

Possible Ideas:
0. Train a model to find the ideal stats to win for new tracks using the dataset given from https://github.com/toUpperCase78/formula1-datasets

1. Aerodynamic Drag and Speed Analysis
Goal: Estimate aerodynamic drag coefficient for different cars or circuits.
Use telemetry to plot speed vs time curves during straights with and without DRS.
Fit drag models (e.g., quadratic air resistance) to observed deceleration/speed loss in sectors.
Compare drag performance between cars/teams.

2. Tire Degradation Modeling
Goal: Model how tire wear affects lap time over stints, factoring in temperature and compound.
Fit regression or time series models to lap time degradation.
Compare soft vs medium vs hard tires.
Incorporate ambient track temperature to model frictional efficiency.

3. Energy Expenditure and Efficiency
Goal: Estimate total mechanical energy used over a lap and analyze how efficiently it's used.
Integrate speed × force (or estimate from acceleration) to get energy.
Compare energy per lap to lap time to define a performance/efficiency metric.
Study how braking zones and cornering affect energy loss.

4. Cornering and Downforce Dynamics
Goal: Estimate downforce or coefficient of friction in cornering from lateral acceleration.
Use ac=v2rac​=rv2​ to estimate effective friction coefficient.
Compare performance in high-speed vs low-speed corners for different teams or drivers.
Study weather impact (rain vs dry) on cornering performance.

5. Weather Impact on Race Dynamics
Goal: Quantify how rain or temperature changes affect lap times and race strategy.
Analyze how wet races change pit stop frequency, tire choice, and lap consistency.
Model rain's effect on traction (possibly via increased lap variability).
Study effect of track temperature on tire compound performance.

6. Crash and Safety Analysis
Goal: Analyze dynamics of known crashes or safety car periods.
Use telemetry before crash to estimate impact velocity and energy.
Model impulse and force assuming deceleration over known time/distance.
Analyze how crashes cluster (laps, weather, track type).

7. Optimal Pit Stop Strategy Simulation
Goal: Simulate and optimize pit stop timing for minimal race time.
Build a simulation model of a full race with variable stints.
Include penalties for tire wear (lap time increases).
Test how weather or safety car affects the ideal strategy.

