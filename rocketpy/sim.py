from rocketpy import Environment, GenericMotor, SolidMotor, Rocket, Flight
import rocketpy
from datetime import datetime, timedelta

# Environment
env = Environment(latitude=39.389700, longitude=-8.288964, elevation=113)
# tomorrow = datetime.date.today() + datetime.timedelta(days=1)
# env.set_date((tomorrow.year, tomorrow.month, tomorrow.day, 12))
# env.set_atmospheric_model(type="Ensemble", file="GEFS")
env.set_atmospheric_model(type="standard_atmosphere")

# Motor
# motor = SolidMotor(
#     thrust_source="C:/Users/Nicho/Documents/airbrakes-controller/rocketpy/AeroTech_O5500X-PS.eng",
#     dry_mass=16.78,
#     dry_inertia=(1.33, 1.33, 0.00140),
#     nozzle_radius=0.0381,
#     grain_number=6,
#     grain_density=1815,
#     grain_outer_radius=0.042799,
#     grain_initial_inner_radius=0.01905,
#     grain_initial_height=0.18415,
#     grain_separation=0.0015875,
#     grains_center_of_mass_position=0.397,
#     center_of_dry_mass_position=0.317,
#     nozzle_position=0,
#     burn_time=3.9,
#     throat_radius=11 / 1000,
#     coordinate_system_orientation="nozzle_to_combustion_chamber",
# )

motor = GenericMotor(
    thrust_source="AeroTech_O5500X-PS.eng",  # RASP file, see note below
    burn_time=4.03,                 # or the last time value in your file
    propellant_initial_mass=9.779,  # kg, published
    dry_mass=7.004,                 # kg, 16.783 total - 9.779 propellant

    # --- estimates (not published), all positions measured from the nozzle ---
    chamber_radius=0.045,           # m, approx. propellant/casing bore
    chamber_height=1.10,            # m, approx. length of the propellant stack
    chamber_position=0.70,          # m, propellant center from nozzle
    nozzle_radius=0.035,            # m, approx.
    nozzle_position=0,
    center_of_dry_mass_position=0.65,   # m, slightly aft of mid-length (nozzle is heavy)
    dry_inertia=(1.16, 1.16, 0.0155),   # kg*m^2 about dry CoM: m*L^2/12 with L=1.408 m; axial ~ m*r^2
    coordinate_system_orientation="nozzle_to_combustion_chamber",
    # coordinate_system_orientation="combustion_chamber_to_nozzle",
)
# print(f"Total Impulse of the Solid Motor: {motor.total_impulse} Ns")

# Rocket
rocket = Rocket(
    radius=0.131 / 2,
    mass=19.262,
    inertia=(26.7, 26.7, 0.0919),
    power_off_drag="power_off_drag.csv",
    power_on_drag="power_on_drag.csv",
    center_of_mass_without_motor=2.06,
    coordinate_system_orientation="nose_to_tail",
)

# rail_buttons = rocket.set_rail_buttons(
#     upper_button_position=0.0818,
#     lower_button_position=-0.618,
#     angular_position=45,
# )

rocket.add_motor(motor, position=4.1)

nose_cone = rocket.add_nose(length=0.432, kind="Ogive", position=0)

fin_set = rocket.add_trapezoidal_fins(
    n=3,
    root_chord=0.457,
    tip_chord=0.0762,
    span=0.165,
    position=3.5,
    cant_angle=0,
    # airfoil=("../../../data/airfoils/NACA0012-radians.txt", "radians"),
)

tail = rocket.add_tail(
    top_radius=0.131 / 2, bottom_radius=0.0889 / 2, length=0.131, position=4.0
)
Main = rocket.add_parachute(
    "Main",
    cd_s=10.0,
    trigger=800,
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
)

Drogue = rocket.add_parachute(
    "Drogue",
    cd_s=1.0,
    trigger="apogee",
    sampling_rate=105,
    lag=1.5,
    noise=(0, 8.3, 0.5),
)

rocket.draw()

# Flight
test_flight = Flight(
    rocket=rocket,
    environment=env,
    rail_length=5,
    inclination=84,
    heading=133,
)
# test_flight.all_info()