so like only look at airbrake_montecarlo_tc2.py and airbrake_montecarlo_weather_tc.py

the weather MC will only run a MC varying the weather
The tc2 will have a 2% error on rocket measurements with the option of varying weather

MC_better is an updated version of weather MC to have more realistic weather conditions pulled from real data
Uses the weather_stuff file has the sounding profile class that is used in MC_better
run fetchWeather at least once to get the csv thatll be used

all other files are either for data processing
or like actual data

claude was used for most of the code