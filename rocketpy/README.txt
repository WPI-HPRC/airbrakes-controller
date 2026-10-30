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