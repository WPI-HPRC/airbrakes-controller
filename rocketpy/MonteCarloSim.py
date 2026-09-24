from Stociastic_rocket import(
    stochastic_env,
    stochastic_flight,
    stochastic_rocket,
)
from sim import rocket
from rocketpy import AirBrakes
import shutil
from pathlib import Path

#altitude to deploy airbrakes
deployment_level = 20000

def controller_function(
    time, sampling_rate, state, state_history, observed_variables, air_brakes, sensors, environment
):
    # state = [x, y, z, vx, vy, vz, e0, e1, e2, e3, wx, wy, wz]
    altitude_ASL = state[2]
    altitude_AGL = altitude_ASL - environment.elevation
    vx, vy, vz = state[3], state[4], state[5]

    # Get winds in x and y directions
    wind_x, wind_y = environment.wind_velocity_x(altitude_ASL), environment.wind_velocity_y(altitude_ASL)

    # Calculate Mach number
    free_stream_speed = (
        (wind_x - vx) ** 2 + (wind_y - vy) ** 2 + (vz) ** 2
    ) ** 0.5
    mach_number = free_stream_speed / environment.speed_of_sound(altitude_ASL)

    # Get previous state from state_history
    previous_state = state_history[-1]
    previous_vz = previous_state[5]

    # If we wanted to we could get the returned values from observed_variables:
    # returned_time, deployment_level, drag_coefficient = observed_variables[-1]

    # Check if the rocket has reached burnout
    if time < rocket.motor.burn_out_time:
        return None

    # If below 1500 meters above ground level, air_brakes are not deployed
    if altitude_AGL < deployment_level:
        air_brakes.deployment_level = 0

    # Full deployment if above certain altitude
    else:
        air_brakes.deployment_level = 1

    # Return variables of interest to be saved in the observed_variables list
    return (
        time,
        air_brakes.deployment_level,
        air_brakes.drag_coefficient(air_brakes.deployment_level, mach_number),
    )

def create_airbrakes(path):
    rocket.air_brakes.remove()
    airbrakes = rocket.add_airbrakes(
        drag_coefficient_curve=path,
        controller_function=controller_function,
        sampling_rate=10,
        reference_area=None,
        clamp=True,
        initial_observed_variables=[0, 0, 0],
        override_rocket_drag=False,
        name="Air Brakes",
    )
    
def create_csv():
    shutil.rmtree("./csv")
    path = Path("./csv")
    path.mkdir(parents=True, exists_ok=True)
    #probably linearly interpolated but should def find equation to verify
    
    