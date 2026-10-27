"""
Component Monte Carlo: every rocket / motor / launch parameter gets a NORMAL
distribution with a standard deviation of PCT (default 2 %) of its nominal value,
and the airbrake CdA is swept. Weather can be varied too (VARY_WEATHER).

Uses the rocket, weather model and helpers from airbrake_montecarlo.py, so keep both
files in the same folder. Tested against RocketPy 1.13.

Outputs (in OUT_DIR):
  airbrake_mc_components_samples.csv - one row per flight: CdA, apogee, max speed,
                                       every sampled input (columns starting "in_")
  airbrake_mc_components_summary.csv - mean/std/min/max apogee per CdA
  mc_comp_cda_<cm2>.*.txt            - raw RocketPy MonteCarlo files
"""

import json
import os
from math import sqrt

import numpy as np
import pandas as pd

from rocketpy import Flight, GenericMotor, Parachute, Rocket, MonteCarlo
from rocketpy.stochastic import (
    StochasticAirBrakes, StochasticFlight, StochasticGenericMotor, StochasticNoseCone,
    StochasticParachute, StochasticRocket, StochasticTail, StochasticTrapezoidalFins,
)

from airbrake_montecarlo_weather_tc import (
    AIRBRAKE_REF_AREA, BURN_TIME, ENG_FILE, ENV, MACH_DEPLOY, WEATHER,
    WeatherStochasticEnvironment, apply_weather,
)

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
PCT = 0.02                 # 1-sigma as a fraction of each nominal value (2 %)
VARY_WEATHER = True        # also vary weather (uses the stds in WEATHER); False = nominal weather
AIRBRAKE_CDA_M2 = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
N_SIMS = 100              # flights per CdA value
PARALLEL = False
N_WORKERS = 4
OUT_DIR = "airbrake_mc_components_out_tc2"
# ----------------------------------------------------------------------------


def n2(x):
    """(mean, std, 'normal') with std = PCT * |mean|. Zero-valued parameters stay fixed."""
    return (x, PCT * abs(x), "normal")


def controller_function(
    time, sampling_rate, state, state_history, observed_variables, air_brakes,
    sensors, environment,
):
    vx, vy, vz = state[3], state[4], state[5]
    mach = sqrt(vx**2 + vy**2 + vz**2) / environment.speed_of_sound(state[2])
    # burn time is randomised too, so keep a 10 % margin on the "after burnout" guard
    air_brakes.deployment_level = (
        1.0 if (time > 1.1 * BURN_TIME and mach < MACH_DEPLOY) else 0.0
    )
    return (time, air_brakes.deployment_level, mach)


def build_montecarlo(cda, filename):
    apply_weather(ENV)  # nominal weather for the template flight

    # ---- Deterministic template (nominal values) ----
    motor = GenericMotor.load_from_eng_file(
        ENG_FILE, propellant_initial_mass=9.779, dry_mass=7.004, chamber_radius=0.045,
        chamber_height=1.10, chamber_position=0.70, nozzle_radius=0.035, nozzle_position=0,
        center_of_dry_mass_position=0.65, dry_inertia=(1.16, 1.16, 0.0155),
        coordinate_system_orientation="nozzle_to_combustion_chamber",
    )
    rocket = Rocket(
        radius=0.131 / 2, mass=19.262, inertia=(26.7, 26.7, 0.0919),
        power_off_drag="power_off_drag.csv", power_on_drag="power_on_drag.csv",
        center_of_mass_without_motor=2.06, coordinate_system_orientation="nose_to_tail",
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
        "Drogue", cd_s=1.0, trigger="apogee", sampling_rate=105, lag=1.5, noise=(0, 8.3, 0.5)
    )

    def cd_curve(deployment_level, mach):  # Cd*A (reference area is 1 m^2)
        return cda * deployment_level

    air_brakes, controller = rocket.add_air_brakes(
        drag_coefficient_curve=cd_curve, controller_function=controller_function,
        sampling_rate=10, reference_area=AIRBRAKE_REF_AREA, clamp=True,
        initial_observed_variables=[0, 0, 0], override_rocket_drag=False,
        name="AirBrakes", return_controller=True,
    )
    flight = Flight(rocket=rocket, environment=ENV, rail_length=5, inclination=84, heading=133)

    # ---- Stochastic wrappers: normal, sigma = PCT of nominal ----
    weather = WEATHER if VARY_WEATHER else {
        **WEATHER, "wind_speed_factor_std": 0.0, "wind_dir_std_deg": 0.0,
        "temp_offset_std_k": 0.0, "pressure_factor_std": 0.0,
    }
    s_env = WeatherStochasticEnvironment(ENV, weather)

    s_motor = StochasticGenericMotor(
        motor,
        total_impulse=n2(motor.total_impulse),
        burn_out_time=n2(motor.burn_out_time),
        propellant_initial_mass=n2(9.779),
        dry_mass=n2(7.004),
        dry_inertia_11=n2(1.16), dry_inertia_22=n2(1.16), dry_inertia_33=n2(0.0155),
        chamber_radius=n2(0.045), chamber_height=n2(1.10), chamber_position=n2(0.70),
        nozzle_radius=n2(0.035),
        center_of_dry_mass_position=n2(0.65),
    )
    s_rocket = StochasticRocket(
        rocket=rocket,
        radius=n2(0.131 / 2), mass=n2(19.262),
        inertia_11=n2(26.7), inertia_22=n2(26.7), inertia_33=n2(0.0919),
        power_off_drag_factor=(1, PCT, "normal"), power_on_drag_factor=(1, PCT, "normal"),
        center_of_mass_without_motor=n2(2.06),
    )
    s_rocket.add_motor(s_motor, position=n2(4.1))
    s_rocket.add_nose(StochasticNoseCone(nose, length=n2(0.432)), position=n2(0.0))
    s_rocket.add_trapezoidal_fins(
        StochasticTrapezoidalFins(
            fins, root_chord=n2(0.457), tip_chord=n2(0.0762), span=n2(0.165)
        ),
        position=n2(3.5),
    )
    s_rocket.add_tail(
        StochasticTail(tail, top_radius=n2(0.131 / 2), bottom_radius=n2(0.0889 / 2),
                       length=n2(0.131)),
        position=n2(4.0),
    )
    s_rocket.add_parachute(StochasticParachute(main, cd_s=n2(10.0), lag=n2(1.5)))
    s_rocket.add_parachute(StochasticParachute(drogue, cd_s=n2(1.0), lag=n2(1.5)))
    # the swept airbrake CdA itself also carries the same 2 % (manufacturing) scatter
    s_rocket.add_air_brakes(
        StochasticAirBrakes(air_brakes, drag_coefficient_curve_factor=(1, PCT, "normal")),
        controller,
    )
    s_flight = StochasticFlight(
        flight=flight, rail_length=n2(5), inclination=n2(84), heading=n2(133),
        terminate_on_apogee=True,   # only apogee is needed, so stop there
    )
    return MonteCarlo(
        filename=filename, environment=s_env, rocket=s_rocket, flight=s_flight,
        export_list=["apogee", "apogee_time", "apogee_x", "apogee_y", "max_speed",
                     "max_mach_number"],
    )


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten(v, f"{prefix}{k}."))
        elif isinstance(v, list):
            for i, x in enumerate(v):
                if isinstance(x, dict):
                    out.update(_flatten(x, f"{prefix}{k}[{i}]."))
                elif isinstance(x, (int, float)) and not isinstance(x, bool):
                    out[f"{prefix}{k}[{i}]"] = x
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[prefix + k] = v
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for cda in AIRBRAKE_CDA_M2:
        print(f"\n=== Airbrake CdA = {cda:.4f} m^2 ===")
        tag = f"mc_comp_cda_{int(round(cda * 1e4)):04d}"  # CdA in cm^2, no "." in name
        mc = build_montecarlo(cda, os.path.join(OUT_DIR, tag))
        mc.simulate(number_of_simulations=N_SIMS, append=False, parallel=PARALLEL,
                    n_workers=N_WORKERS if PARALLEL else None)
        mc.import_results()

        idx = [int(i) for i in mc.results["index"]]
        with open(os.path.join(OUT_DIR, tag + ".inputs.txt"), encoding="utf-8") as f:
            inputs = {r["index"]: _flatten(r) for r in map(json.loads, f) if "index" in r}
        d = {"airbrake_cda_m2": cda, "sample": idx}
        d["apogee_asl_m"] = np.asarray(mc.results["apogee"], dtype=float)
        d["apogee_agl_m"] = d["apogee_asl_m"] - ENV.elevation
        d["max_speed_ms"] = np.asarray(mc.results["max_speed"], dtype=float)
        d["max_mach"] = np.asarray(mc.results["max_mach_number"], dtype=float)
        df = pd.DataFrame(d)
        inp = pd.DataFrame([inputs.get(i, {}) for i in idx]).add_prefix("in_")
        frames.append(pd.concat([df, inp], axis=1))

    samples = pd.concat(frames, ignore_index=True)
    samples.to_csv(os.path.join(OUT_DIR, "airbrake_mc_components_samples.csv"), index=False)
    summary = (
        samples.groupby("airbrake_cda_m2")["apogee_agl_m"]
        .agg(["count", "mean", "std", "min", "max"]).reset_index()
        .rename(columns={c: f"apogee_agl_{c}" for c in ["mean", "std", "min", "max"]})
    )
    summary.to_csv(os.path.join(OUT_DIR, "airbrake_mc_components_summary.csv"), index=False)
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()