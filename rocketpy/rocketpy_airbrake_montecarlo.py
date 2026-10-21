"""
RocketPy Monte Carlo study of airbrake drag coefficient.

This script is based on the supplied sim.py rocket model.

For each nominal airbrake Cd, RocketPy's built-in StochasticAirBrakes and
MonteCarlo classes are used to run NUM_SIMULATIONS independent flights.

Airbrake behavior:
    - Starts fully retracted.
    - The controller first waits until the rocket has reached Mach 0.8.
    - On the subsequent ascent, when Mach drops below 0.8, the airbrakes
      immediately command 100% deployment.
    - Once deployed, they remain fully deployed.

The airbrake Cd is sampled as a normal stochastic parameter:
    Cd ~ Normal(nominal Cd, CD_RELATIVE_STD * nominal Cd)

The existing rocket power-on/power-off drag curves are retained because
override_rocket_drag=False.

Tested against the current RocketPy 1.13 documentation/API.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from rocketpy import Environment, GenericMotor, Rocket, Flight, MonteCarlo
from rocketpy.stochastic import (
    StochasticEnvironment,
    StochasticFlight,
    StochasticRocket,
    StochasticAirBrakes,
)


# ============================================================================
# USER SETTINGS
# ============================================================================

CD_VALUES = np.arange(0.1, 1.01, 0.1)

# Monte Carlo flights for each nominal Cd.
NUM_SIMULATIONS = 100

# Relative standard deviation of Cd.
# 0.05 means sigma = 5% of the nominal Cd.
CD_RELATIVE_STD = 0.05

# Deployment threshold.
DEPLOYMENT_MACH = 0.8

# Use False first while validating the model. True can be used later to
# parallelize the Monte Carlo simulations.
PARALLEL = False
N_WORKERS = None

OUTPUT_DIR = Path("airbrake_montecarlo_results")


# ============================================================================
# ENVIRONMENT
# ============================================================================

def create_environment():
    env = Environment(
        latitude=39.389700,
        longitude=-8.288964,
        elevation=113,
    )

    env.set_atmospheric_model(type="standard_atmosphere")
    return env


# ============================================================================
# MOTOR
# ============================================================================

def create_motor():
    return GenericMotor(
        thrust_source="AeroTech_O5500X-PS.eng",
        burn_time=4.03,
        propellant_initial_mass=9.779,
        dry_mass=7.004,

        chamber_radius=0.045,
        chamber_height=1.10,
        chamber_position=0.70,
        nozzle_radius=0.035,
        nozzle_position=0,

        center_of_dry_mass_position=0.65,
        dry_inertia=(1.16, 1.16, 0.0155),

        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )


# ============================================================================
# ROCKET
# ============================================================================

def create_rocket():
    motor = create_motor()

    rocket = Rocket(
        radius=0.131 / 2,
        mass=19.262,
        inertia=(26.7, 26.7, 0.0919),
        power_off_drag="power_off_drag.csv",
        power_on_drag="power_on_drag.csv",
        center_of_mass_without_motor=2.06,
        coordinate_system_orientation="nose_to_tail",
    )

    rocket.add_motor(motor, position=4.1)

    nose_cone = rocket.add_nose(
        length=0.432,
        kind="Ogive",
        position=0,
    )

    fin_set = rocket.add_trapezoidal_fins(
        n=3,
        root_chord=0.457,
        tip_chord=0.0762,
        span=0.165,
        position=3.5,
        cant_angle=0,
    )

    tail = rocket.add_tail(
        top_radius=0.131 / 2,
        bottom_radius=0.0889 / 2,
        length=0.131,
        position=4.0,
    )

    main = rocket.add_parachute(
        "Main",
        cd_s=10.0,
        trigger=800,
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    drogue = rocket.add_parachute(
        "Drogue",
        cd_s=1.0,
        trigger="apogee",
        sampling_rate=105,
        lag=1.5,
        noise=(0, 8.3, 0.5),
    )

    return rocket, nose_cone, fin_set, tail, main, drogue


# ============================================================================
# AIRBRAKE CONTROLLER
# ============================================================================

def make_controller(environment):
    """
    Create the airbrake controller.

    We use the 6-argument controller form supported by RocketPy:
        time, sampling_rate, state, state_history,
        observed_variables, air_brakes

    Mach is calculated from the current inertial velocity and the standard
    atmosphere.  Since this simulation uses standard_atmosphere with no wind,
    freestream speed is simply the magnitude of the velocity vector.

    observed_variables stores:
        [armed, last_mach]

    'armed' prevents the brakes from deploying immediately at launch, since
    the rocket necessarily starts below Mach 0.8.  The brakes only deploy
    after the rocket has first reached Mach 0.8 and subsequently falls below
    Mach 0.8 while still ascending.
    """

    def controller_function(
        time,
        sampling_rate,
        state,
        state_history,
        observed_variables,
        air_brakes,
    ):
        altitude_asl = state[2]
        vx, vy, vz = state[3], state[4], state[5]

        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        speed_of_sound = environment.speed_of_sound(altitude_asl)

        mach = speed / speed_of_sound

        # Initial observed variables are [armed, last_mach].
        if observed_variables:
            armed = bool(observed_variables[-1][0])
        else:
            armed = False

        # First pass above Mach 0.8 arms the controller.
        if not armed and mach >= DEPLOYMENT_MACH:
            armed = True

        # Once armed, deploy when the rocket comes back below Mach 0.8
        # during ascent.  Deployment is latched at 1.0.
        if (
            armed
            and air_brakes.deployment_level < 1.0
            and vz > 0
            and mach < DEPLOYMENT_MACH
        ):
            air_brakes.deployment_level = 1.0

        # Keep the brakes deployed after the trigger.
        # Otherwise leave them at their current deployment level.
        return armed, mach

    return controller_function


# ============================================================================
# ADD AIRBRAKES
# ============================================================================

def add_airbrakes(rocket, environment):
    """
    Add the deterministic airbrake/controller pair.

    The normalized curve is:
        deployment 0 -> Cd 0
        deployment 1 -> Cd 1

    StochasticAirBrakes later multiplies this curve by the sampled
    drag_coefficient_curve_factor. Therefore a nominal factor of 0.5
    produces Cd = 0.5 when fully deployed.
    """

    drag_coefficient_curve = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 5.0, 1.0],
    ])

    controller_function = make_controller(environment)

    air_brakes, controller = rocket.add_air_brakes(
        drag_coefficient_curve=drag_coefficient_curve,
        controller_function=controller_function,
        sampling_rate=20,
        clamp=True,
        reference_area=None,
        initial_observed_variables=[False, 0.0],
        override_rocket_drag=False,
        return_controller=True,
        name="Air Brakes",
        controller_name="Air Brakes Controller",
    )

    return air_brakes, controller


# ============================================================================
# STOCHASTIC OBJECTS
# ============================================================================

def create_stochastic_objects(nominal_cd):
    """
    Build the deterministic model first, then its RocketPy stochastic
    counterparts.

    RocketPy's MonteCarlo class requires StochasticEnvironment,
    StochasticRocket, and StochasticFlight objects rather than the
    deterministic Environment/Rocket/Flight objects.
    """

    environment = create_environment()

    (
        rocket,
        nose_cone,
        fin_set,
        tail,
        main,
        drogue,
    ) = create_rocket()

    air_brakes, controller = add_airbrakes(
        rocket,
        environment,
    )

    # The standard Flight is the base object for StochasticFlight.
    # RocketPy's airbrake documentation specifically recommends
    # time_overshoot=False for discrete airbrake controllers.
    flight = Flight(
        rocket=rocket,
        environment=environment,
        rail_length=5,
        inclination=84,
        heading=133,
        time_overshoot=False,
    )

    # No environmental/flight/rocket uncertainty is introduced other than
    # the airbrake Cd.  Passing deterministic values to the stochastic
    # classes makes those inputs have zero standard deviation.
    stochastic_environment = StochasticEnvironment(
        environment=environment,
    )

    stochastic_rocket = StochasticRocket(
        rocket=rocket,
    )

    # Add the deterministic components to the stochastic rocket.  RocketPy
    # automatically wraps deterministic components in their corresponding
    # stochastic models with zero uncertainty when no uncertainty is supplied.
    stochastic_rocket.add_motor(
        rocket.motor,
        position=4.1,
    )

    stochastic_rocket.add_nose(
        nose_cone,
        position=0,
    )

    stochastic_rocket.add_trapezoidal_fins(
        fin_set,
        position=3.5,
    )

    stochastic_rocket.add_tail(
        tail,
        position=4.0,
    )

    stochastic_rocket.add_parachute(main)
    stochastic_rocket.add_parachute(drogue)

    # This is the important part:
    # RocketPy's StochasticAirBrakes multiplies the airbrake Cd curve by
    # the sampled drag_coefficient_curve_factor.
    stochastic_air_brakes = StochasticAirBrakes(
        air_brakes=air_brakes,
        drag_coefficient_curve_factor=(
            float(nominal_cd),
            float(nominal_cd * CD_RELATIVE_STD),
        ),
    )

    stochastic_rocket.add_air_brakes(
        stochastic_air_brakes,
        controller,
    )

    stochastic_flight = StochasticFlight(
        flight=flight,
        time_overshoot=False,
    )

    return (
        stochastic_environment,
        stochastic_rocket,
        stochastic_flight,
    )


# ============================================================================
# CUSTOM MONTE CARLO OUTPUTS
# ============================================================================

def make_data_collector(nominal_cd):
    """
    Return callbacks for values that are not part of MonteCarlo's default
    output list.

    The actual airbrake Cd is read from the stochastic AirBrakes object in
    the generated Flight. At deployment=1 and Mach=0.8, the normalized
    curve is exactly 1, so this value is the sampled fully-deployed Cd.
    """

    def get_nominal_cd(flight):
        return float(nominal_cd)

    def get_actual_airbrake_cd(flight):
        air_brakes = flight.rocket.air_brakes[0]
        return float(air_brakes.drag_coefficient(1.0, DEPLOYMENT_MACH))

    return {
        "nominal_airbrake_cd": get_nominal_cd,
        "airbrake_cd": get_actual_airbrake_cd,
    }


# ============================================================================
# RUN ONE MONTE CARLO GROUP
# ============================================================================

def run_one_cd_monte_carlo(nominal_cd, output_dir):
    print(f"Running Cd = {nominal_cd:.3f}")

    (
        stochastic_environment,
        stochastic_rocket,
        stochastic_flight,
    ) = create_stochastic_objects(nominal_cd)

    filename = output_dir / f"cd_{nominal_cd:.3f}"

    monte_carlo = MonteCarlo(
        filename=str(filename),
        environment=stochastic_environment,
        rocket=stochastic_rocket,
        flight=stochastic_flight,
        export_list=[
            "apogee",
            "apogee_time",
            "max_mach_number",
        ],
        data_collector=make_data_collector(nominal_cd),
    )

    monte_carlo.simulate(
        number_of_simulations=NUM_SIMULATIONS,
        append=False,
        parallel=PARALLEL,
        n_workers=N_WORKERS,
        include_function_data=False,
    )

    return monte_carlo


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Starting RocketPy airbrake Cd Monte Carlo sweep.")
    print(f"Cd values: {[float(x) for x in CD_VALUES]}")
    print(f"Simulations per Cd: {NUM_SIMULATIONS}")
    print(f"Airbrake deployment Mach: {DEPLOYMENT_MACH}")
    print(
        f"Cd relative standard deviation: "
        f"{CD_RELATIVE_STD * 100:.1f}%"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for index, nominal_cd in enumerate(CD_VALUES, start=1):
        print("\n" + "=" * 72)
        print(
            f"Running Cd = {float(nominal_cd):.3f} "
            f"({index}/{len(CD_VALUES)})"
        )

        mc = run_one_cd_monte_carlo(
            float(nominal_cd),
            OUTPUT_DIR,
        )

        # MonteCarlo.results is a dictionary of lists, one list per
        # exported parameter.
        result_df = pd.DataFrame(mc.results)

        all_results.append(result_df)

        # Also save a simple CSV for this Cd group.
        group_csv = (
            OUTPUT_DIR / f"cd_{float(nominal_cd):.3f}_results.csv"
        )
        mc.export_outputs_to_csv(str(group_csv))

        print(f"Saved: {group_csv}")

    # Combine all Cd groups into one CSV.
    combined = pd.concat(
        all_results,
        ignore_index=True,
    )

    # Put the most relevant columns first.
    preferred = [
        "nominal_airbrake_cd",
        "airbrake_cd",
        "apogee",
        "apogee_time",
        "max_mach_number",
    ]

    remaining = [
        column for column in combined.columns
        if column not in preferred
    ]

    combined = combined[
        [column for column in preferred if column in combined.columns]
        + remaining
    ]

    combined_csv = OUTPUT_DIR / "airbrake_cd_montecarlo_results.csv"
    combined.to_csv(combined_csv, index=False)

    print("\n" + "=" * 72)
    print("Monte Carlo sweep complete.")
    print(f"Combined results: {combined_csv.resolve()}")
    print(f"Total simulations: {len(combined)}")


if __name__ == "__main__":
    main()
