from rocketpy import Environment, SolidMotor, Rocket, Flight
import rocketpy
from datetime import datetime, timedelta

#change
flight_day = datetime.now() + timedelta(days=1)
flight_location = {"latitude" : 42.27369444253446,
                   "longitude" : -71.8054886493408,
                   "elavation" : 155
}


#set to WPI, MUST CHANGE
env = Environment(latitude=flight_location["latitude"], longitude=flight_location["longitude"], elevation=flight_location["elavation"])

env.set_date(flight_day)

#not working idk why default is international standard atmosphere
# env.set_atmospheric_model(type="Forecast", file="GFS")

# env.info()

#this needs to be fixed
import numpy as np
import xml.etree.ElementTree as ET

def load_rse_thrust(path):
    root = ET.parse(path).getroot()
    # t in seconds, f in newtons
    return np.array([(float(e.get("t")), float(e.get("f")))
                     for e in root.iter("eng-data")])

thrust = load_rse_thrust("../Models/Motor/EngineData/AeroTech_O5500X-PS.rse")

Motor = SolidMotor(
    thrust_source=thrust,
    burn_time=thrust[-1, 0],
    #the rest is pasted from the website must change
    dry_mass=1.815,
    dry_inertia=(0.125, 0.125, 0.002),
    nozzle_radius=33 / 1000,
    grain_number=5,
    grain_density=1815,
    grain_outer_radius=33 / 1000,
    grain_initial_inner_radius=15 / 1000,
    grain_initial_height=120 / 1000,
    grain_separation=5 / 1000,
    grains_center_of_mass_position=0.397,
    center_of_dry_mass_position=0.317,
    nozzle_position=0,
    # burn_time=3.9,
    throat_radius=11 / 1000,
    coordinate_system_orientation="nozzle_to_combustion_chamber",
)

rocket = Rocket(
    radius=127 / 2000,
    mass=14.426,
    inertia=(6.321, 6.321, 0.034),
    # power_off_drag="../data/rockets/calisto/powerOffDragCurve.csv",
    # power_on_drag="../data/rockets/calisto/powerOnDragCurve.csv",
    center_of_mass_without_motor=0,
    coordinate_system_orientation="tail_to_nose",
)

rocket.add_motor(Motor, position=-1.255)