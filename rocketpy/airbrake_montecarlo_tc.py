"""
Monte Carlo sweep of airbrake drag area (CdA) with RocketPy.

For each airbrake CdA in AIRBRAKE_CDA_M2 a full MonteCarlo run (N_SIMS flights) is
performed. In every flight the airbrakes stay retracted until the rocket has
burned out AND its Mach number drops below MACH_DEPLOY, then they go to 100 %
deployment.

Outputs (in OUT_DIR):
  airbrake_mc_samples.csv  - one row per flight: airbrake_cda_m2, sample, apogee ASL/AGL
  airbrake_mc_summary.csv  - mean/std/min/max apogee per CdA
  mc_cda_<cm2>.*.txt      - raw RocketPy MonteCarlo files (inputs/outputs/errors)

Written against RocketPy >= 1.6 (controller functions take a `sensors` argument
and StochasticAirBrakes exists). If a keyword name differs in your version,
check the class docstrings: help(StochasticRocket), help(StochasticAirBrakes).
"""

import os
from math import sqrt

import numpy as np
import pandas as pd

from rocketpy import Environment, Flight, GenericMotor, Motor, Parachute, Rocket
from rocketpy import MonteCarlo
from rocketpy.stochastic import (
    StochasticEnvironment,
    StochasticFlight,
    StochasticGenericMotor,
    StochasticNoseCone,
    StochasticParachute,
    StochasticRocket,
    StochasticTail,
    StochasticTrapezoidalFins,
    StochasticAirBrakes,
)

# ----------------------------------------------------------------------------
# User settings
# ----------------------------------------------------------------------------
# Airbrake drag force is 0.5*rho*v^2*Cd*A, so only the product CdA (m^2) matters.
# Because the brake area is not known yet, sweep CdA directly: the reference area is
# fixed at 1 m^2 so the "Cd" handed to RocketPy is numerically equal to CdA in m^2.
# Afterwards, required area = CdA / Cd_of_your_panel_design.
# AIRBRAKE_CDA_M2 = [0.0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.02]  # CdA values to sweep (m^2)
AIRBRAKE_CDA_M2 = [0.015]
AIRBRAKE_REF_AREA = 1.0    # m^2, keep at 1.0 so Cd == CdA
MACH_DEPLOY = 0.8          # fully deploy below this Mach (after burnout)
ENG_FILE = "AeroTech_O5500X-PS.eng"
# End of the thrust curve (last time in the .eng); brakes cannot deploy before this
BURN_TIME = Motor.import_eng(ENG_FILE)[2][-1][0]
N_SIMS = 200             # flights per CdA value
PARALLEL = False           # True needs the __main__ guard below (already there)
N_WORKERS = 4
OUT_DIR = "airbrake_mc_out"

# Nominal environment (module level so the controller can read speed of sound)
ENV = Environment(latitude=39.389700, longitude=-8.288964, elevation=113)
# Small constant-ish wind so the wind factors in StochasticEnvironment do something.
# pressure/temperature None -> standard atmosphere, same as your original sim.
# Set wind_u/wind_v to 0 if you want a no-wind baseline.
ENV.set_atmospheric_model(
    type="custom_atmosphere",
    pressure=None,
    temperature=None,
    wind_u=[(0, 2.0), (3000, 6.0)],
    wind_v=[(0, 1.0), (3000, 3.0)],
)


# ----------------------------------------------------------------------------
# Airbrake controller: fully deploy below MACH_DEPLOY after burnout
# ----------------------------------------------------------------------------
def controller_function(
    time, sampling_rate, state, state_history, observed_variables, air_brakes, sensors
):
    z = state[2]  # altitude ASL
    vx, vy, vz = state[3], state[4], state[5]
    speed = sqrt(vx**2 + vy**2 + vz**2)  # ground-relative speed (ignores wind)
    mach = speed / ENV.speed_of_sound(z)

    if time > BURN_TIME and mach < MACH_DEPLOY:
        air_brakes.deployment_level = 1.0
    else:
        air_brakes.deployment_level = 0.0

    return (time, air_brakes.deployment_level, mach)


# ----------------------------------------------------------------------------
# Build deterministic + stochastic objects for one CdA value
# ----------------------------------------------------------------------------
def build_montecarlo(cda, filename):
    # ---- Deterministic template (same as your sim.py) ----
    # NOTE: a .eng file must be loaded with load_from_eng_file. Passing the path to
    # GenericMotor(thrust_source=...) reads it as a plain CSV (header row -> NaN thrust).
    motor = GenericMotor.load_from_eng_file(
        ENG_FILE,
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
    nose = rocket.add_nose(length=0.432, kind="Ogive", position=0)
    fins = rocket.add_trapezoidal_fins(
        n=3, root_chord=0.457, tip_chord=0.0762, span=0.165, position=3.5, cant_angle=0
    )
    tail = rocket.add_tail(
        top_radius=0.131 / 2, bottom_radius=0.0889 / 2, length=0.131, position=4.0
    )
    main = rocket.add_parachute(
        "Main", cd_s=10.0, trigger=800, sampling_rate=105, lag=1.5, noise=(0, 8.3, 0.5)
    )
    drogue = rocket.add_parachute(
        "Drogue", cd_s=1.0, trigger="apogee", sampling_rate=105, lag=1.5,
        noise=(0, 8.3, 0.5),
    )
    # Cd curve must be a function of (deployment_level, mach). A bare float gives a
    # 1-D Function and fails, so use a callable. Cd scales with deployment level so
    # closed brakes (level 0) add no drag and fully open (level 1) gives exactly `cda`.
    def cd_curve(deployment_level, mach):
        return cda * deployment_level

    air_brakes, controller = rocket.add_air_brakes(
        drag_coefficient_curve=cd_curve,
        controller_function=controller_function,
        sampling_rate=10,
        reference_area=AIRBRAKE_REF_AREA,
        clamp=True,
        initial_observed_variables=[0, 0, 0],
        override_rocket_drag=False,
        name="AirBrakes",
        return_controller=True,
    )

    flight = Flight(
        rocket=rocket, environment=ENV, rail_length=5, inclination=84, heading=133
    )

    # ---- Stochastic wrappers (1-sigma values are placeholders: tune them) ----
    s_env = StochasticEnvironment(
        environment=ENV,
        elevation=(113, 5),
        wind_velocity_x_factor=(1.0, 0.25),
        wind_velocity_y_factor=(1.0, 0.25),
    )

    s_motor = StochasticGenericMotor(
        motor,
        total_impulse=(motor.total_impulse, 0.03 * motor.total_impulse),
        propellant_initial_mass=(9.779, 0.10),
        dry_mass=(7.004, 0.05),
    )

    s_rocket = StochasticRocket(
        rocket=rocket,
        radius=0.0,  # 0 std -> not varied
        mass=(19.262, 0.3),
        inertia_11=(26.7, 0.5),
        inertia_22=(26.7, 0.5),
        inertia_33=0.0,
        center_of_mass_without_motor=(2.06, 0.01),
        power_off_drag_factor=(1.0, 0.05),
        power_on_drag_factor=(1.0, 0.05),
    )
    s_rocket.add_motor(s_motor, position=(4.1, 0.005))
    s_rocket.add_nose(StochasticNoseCone(nose, length=0.001), position=(0.0, 0.001))
    s_rocket.add_trapezoidal_fins(
        StochasticTrapezoidalFins(fins, root_chord=0.001, tip_chord=0.001, span=0.001),
        position=(3.5, 0.001),
    )
    s_rocket.add_tail(StochasticTail(tail), position=(4.0, 0.001))
    s_rocket.add_parachute(StochasticParachute(main, cd_s=0.3, lag=0.1))
    s_rocket.add_parachute(StochasticParachute(drogue, cd_s=0.1, lag=0.1))
    # Airbrake CdA is held at the swept value (no scatter) so each MC batch is one CdA
    s_rocket.add_air_brakes(
        StochasticAirBrakes(air_brakes), controller
    )

    s_flight = StochasticFlight(
        flight=flight, inclination=(84, 1), heading=(133, 2)
    )

    return MonteCarlo(
        filename=filename,
        environment=s_env,
        rocket=s_rocket,
        flight=s_flight,
    )


# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    for cda in AIRBRAKE_CDA_M2:
        print(f"\n=== Airbrake CdA = {cda:.4f} m^2 ===")
        # No "." in the base name: MonteCarlo treats everything after a dot as an extension
        tag = f"mc_cda_{int(round(cda * 1e4)):04d}"  # CdA in cm^2
        mc = build_montecarlo(cda, os.path.join(OUT_DIR, tag))
        mc.simulate(
            number_of_simulations=N_SIMS,
            append=False,
            parallel=PARALLEL,
            n_workers=N_WORKERS if PARALLEL else None,
        )
        
        mc.import_results()

        apogees = np.asarray(mc.results["apogee"], dtype=float)  # m ASL
        for i, ap in enumerate(apogees):
            rows.append(
                {
                    "airbrake_cda_m2": cda,
                    "sample": i,
                    "apogee_asl_m": ap,
                    "apogee_agl_m": ap - ENV.elevation,
                }
            )

    samples = pd.DataFrame(rows)
    samples.to_csv(os.path.join(OUT_DIR, "airbrake_mc_samples.csv"), index=False)

    summary = (
        samples.groupby("airbrake_cda_m2")["apogee_agl_m"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={c: f"apogee_agl_{c}" for c in ["mean", "std", "min", "max"]})
    )
    summary.to_csv(os.path.join(OUT_DIR, "airbrake_mc_summary.csv"), index=False)
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":  # required on Windows, especially with PARALLEL=True
    main()