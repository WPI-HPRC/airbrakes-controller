hi

all the code used in the pdr for the mc is in old stuff directory all the plotting code is in data_processing
tc2 is the 2% change and weather is the only weather sim
lowk just ignore that besides the 2% part cause ill reimplement that into mc_sim_weather_only at some point

data processing has all the plot generation and the file to get the weather data

rocket data is just the csvs of the rocket drag and thrust stuff and weather data

the important files are the weather.py and MC_sim_weather_only 
weather.py is the weather class generated from a month in midland texas that gets used by the mc
the old code just put error on a predetermined weather enviornment so this is a lot better 
MC_sim_weather_only is the actual mc sim it is missing the new rocket data im working on the serializer stuff seperatly to make implementation easier
An important difference between this and the pdr code is I forgot to account for the change in coefficient of drag at higher speeds so that was added here

code was written with claude

UPDATING ROCKET DATA FROM OPENROCKET
run:  python update_from_openrocket.py
it pulls the latest designs from https://github.com/WPI-HPRC/openrockets and writes into rocket_data/:
  power_on_drag.csv, power_off_drag.csv  (Mach,Cd from the saved OpenRocket sim)
  thrust_curve.csv                        (time,thrust)
  rocket_params.json                      (radius, mass, inertia, CG, nose/fins/tail/motor positions, parachutes)
options: --list (show designs and sims), --ork "file.ork", --sim "sim name", --no-pull
the design must have a simulation that was run AND saved in OpenRocket before it was pushed,
otherwise there is no drag data in the .ork file.

MC_sim_openrocket.py = same MC as MC_sim_weather_only but the rocket comes from OpenRocket:
it runs update_from_openrocket.py at start-up and builds the rocket from rocket_data/rocket_params.json.
results go to mc_openrocket_out/ (with rocket_params_used.json recording which design was flown).
