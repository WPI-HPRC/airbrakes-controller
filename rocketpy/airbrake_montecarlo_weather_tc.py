"""
Weather-only Monte Carlo sweep of airbrake drag area (CdA) with RocketPy.

For each airbrake CdA in AIRBRAKE_CDA_M2 a full MonteCarlo run (N_SIMS flights) is
performed. The ONLY thing that varies from flight to flight is the weather:
  * wind speed      (whole profile scaled by a random factor)
  * wind direction  (whole profile rotated by a random offset)
  * air temperature (random offset added to the standard-atmosphere profile)
  * air pressure    (standard-atmosphere profile scaled by a random factor)
Motor, rocket mass/inertia, drag, parachutes, launch angles etc. stay at nominal.
The airbrakes stay retracted until burnout AND Mach < MACH_DEPLOY, then go to 100 %.

Outputs (in OUT_DIR):
  airbrake_mc_samples.csv  - one row per flight: CdA, sampled weather, apogee ASL/AGL
  airbrake_mc_summary.csv  - mean/std/min/max apogee per CdA
  mc_cda_<cm2>.*.txt       - raw RocketPy MonteCarlo files (inputs/outputs/errors)

Tested against RocketPy 1.13.
"""

import json
import os
from math import cos, radians, sin, sqrt

import numpy as np
import pandas as pd

from rocketpy import Environment, Flight, GenericMotor, Motor, MonteCarlo, Rocket
from rocketpy.stochastic import (
    StochasticAirBrakes,
    StochasticEnvironment,
    StochasticFlight,
    StochasticGenericMotor,
    StochasticNoseCone,
    StochasticParachute,
    StochasticRocket,
    StochasticTail,
    StochasticTrapezoidalFins,
)

# ----------------------------------------------------------------------------
# User settings
# ----------------------------------------------------------------------------
# Airbrake drag force is 0.5*rho*v^2*Cd*A, so only the product CdA (m^2) matters.
# Brake area is not known yet, so sweep CdA directly: the reference area is fixed at
# 1 m^2 so the "Cd" handed to RocketPy is numerically equal to CdA in m^2.
# Afterwards: required area = CdA / Cd_of_your_panel_design.
# AIRBRAKE_CDA_M2 = [0.0, 0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016, 0.018, 0.02, 0.022, 0.024, 0.026, 0.028, 0.03]
# AIRBRAKE_CDA_M2 = [0.032, 0.034, 0.036, 0.038, 0.04, 0.042, 0.044, 0.048, 0.05]
# AIRBRAKE_CDA_M2 = [0.052, 0.054, 0.056, 0.058, 0.06, 0.062, 0.064, 0.068, 0.07]
# AIRBRAKE_CDA_M2 = [0.072, 0.074, 0.076, 0.078, 0.8, 0.082, 0.084, 0.086, 0.088, 0.09]
# AIRBRAKE_CDA_M2 = [0.08]
AIRBRAKE_CDA_M2 = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
AIRBRAKE_REF_AREA = 1.0    # m^2, keep at 1.0 so Cd == CdA
MACH_DEPLOY = 0.8          # fully deploy below this Mach (after burnout)
ENG_FILE = "AeroTech_O5500X-PS.eng"
# End of the thrust curve (last time in the .eng); brakes cannot deploy before this
BURN_TIME = Motor.import_eng(ENG_FILE)[2][-1][0]
N_SIMS = 20               # flights per CdA value
PARALLEL = False
N_WORKERS = 4
OUT_DIR = "airbrake_mc_out_weather_t3"

# ---- Weather model (EDIT THESE to match your launch site / season) ----
# Nominal wind profile. Altitudes are metres ABOVE GROUND LEVEL, speeds in m/s,
# direction is the compass direction the wind blows FROM (270 = from the west).
LAUNCH_LAT, LAUNCH_LON, LAUNCH_ELEV = 31.9974, -102.0780, 847
WEATHER = {
    # "wind_alt_agl_m": [0, 500, 1500, 3000, 5000, 8000, 10000],
    # "wind_speed_ms":  [3,   4,    6,    9,   13,   18,    20],
    "wind_alt_agl_m": [0, 900, 1829, 2743, 3658, 5486, 7315, 9144],
    "wind_speed_ms": [15/2, 19/2, 12/2, 14/2, 12/2, 10/2, 9/2, 7/2],
    "wind_from_deg": [150, 150, 200, 190, 230, 220, 200, 30],
    # 1-sigma of the random perturbations applied to each flight:
    "wind_speed_factor_std": 0.6,   # 0.35 -> speeds typically 65 %..135 % of nominal
    "wind_dir_std_deg": 30.0,        # wind direction offset
    "temp_offset_std_k": 5.0,        # +/- K added to standard-atmosphere temperature
    "pressure_factor_std": 0.01,     # 0.01 -> +/-1 % (~10 hPa) pressure
}

'''
TODO fix weather to be based on real conditions
'''

# Pristine ISA profiles used as the baseline for temperature / pressure
_ISA = Environment(latitude=LAUNCH_LAT, longitude=LAUNCH_LON, elevation=LAUNCH_ELEV)
ISA_P, ISA_T = _ISA.pressure_ISA, _ISA.temperature_ISA


def apply_weather(env, temp_offset_k=0.0, pressure_factor=1.0,
                  wind_speed_factor=1.0, wind_dir_offset_deg=0.0):
    """(Re)build the atmosphere of `env` in place for one weather sample."""
    w = WEATHER
    n = len(w["wind_alt_agl_m"])
    if not (len(w["wind_speed_ms"]) == len(w["wind_from_deg"]) == n):
        raise ValueError("wind_alt_agl_m, wind_speed_ms and wind_from_deg must be the same length")
    alts = [a + LAUNCH_ELEV for a in w["wind_alt_agl_m"]]  # RocketPy wants ASL
    speeds = [max(0.0, s * wind_speed_factor) for s in w["wind_speed_ms"]]
    # the random offset rotates the whole direction profile by the same angle
    dirs = [d + wind_dir_offset_deg for d in w["wind_from_deg"]]
    # wind blows FROM direction d: u = east component, v = north component.
    # Components (not angles) are what RocketPy interpolates between altitudes.
    wind_u = [(a, -s * sin(radians(d))) for a, s, d in zip(alts, speeds, dirs)]
    wind_v = [(a, -s * cos(radians(d))) for a, s, d in zip(alts, speeds, dirs)]
    env.set_atmospheric_model(
        type="custom_atmosphere",
        pressure=lambda z: ISA_P(z) * pressure_factor,
        temperature=lambda z: ISA_T(z) + temp_offset_k,
        wind_u=wind_u,
        wind_v=wind_v,
    )


class WeatherStochasticEnvironment(StochasticEnvironment):
    """StochasticEnvironment whose only randomness is the weather model above."""

    def __init__(self, environment, weather):
        self.weather = dict(weather)
        super().__init__(environment)  # no elevation/gravity/wind-factor variation

    def _set_stochastic(self, seed=None):  # also called per worker with its own seed
        super()._set_stochastic(seed)
        self._weather_rng = np.random.default_rng(seed)

    def create_object(self):
        w, rng = self.weather, self._weather_rng
        sample = {
            "wind_speed_factor": max(0.0, rng.normal(1.0, w["wind_speed_factor_std"])),
            "wind_dir_offset_deg": rng.normal(0.0, w["wind_dir_std_deg"]),
            "temp_offset_k": rng.normal(0.0, w["temp_offset_std_k"]),
            "pressure_factor": rng.normal(1.0, w["pressure_factor_std"]),
        }
        apply_weather(
            self.obj,
            temp_offset_k=sample["temp_offset_k"],
            pressure_factor=sample["pressure_factor"],
            wind_speed_factor=sample["wind_speed_factor"],
            wind_dir_offset_deg=sample["wind_dir_offset_deg"],
        )
        # MonteCarlo logs last_rnd_dict to <name>.inputs.txt
        self.last_rnd_dict = {k: float(v) for k, v in sample.items()}
        return self.obj


# Nominal environment (no perturbation) for the deterministic template flight
ENV = Environment(latitude=LAUNCH_LAT, longitude=LAUNCH_LON, elevation=LAUNCH_ELEV)
apply_weather(ENV)


# ----------------------------------------------------------------------------
# Airbrake controller: fully deploy below MACH_DEPLOY after burnout
# ----------------------------------------------------------------------------
def controller_function(
    time, sampling_rate, state, state_history, observed_variables, air_brakes,
    sensors, environment,
):
    z = state[2]  # altitude ASL
    vx, vy, vz = state[3], state[4], state[5]
    speed = sqrt(vx**2 + vy**2 + vz**2)  # ground-relative speed
    # speed of sound of THIS flight's (perturbed) atmosphere
    mach = speed / environment.speed_of_sound(z)

    if time > BURN_TIME and mach < MACH_DEPLOY:
        air_brakes.deployment_level = 1.0
    else:
        air_brakes.deployment_level = 0.0

    return (time, air_brakes.deployment_level, mach)


# ----------------------------------------------------------------------------
# Build deterministic + stochastic objects for one CdA value
# ----------------------------------------------------------------------------
def build_montecarlo(cda, filename):
    apply_weather(ENV)  # reset to nominal weather for the template flight

    # ---- Deterministic template (same as your sim.py) ----
    # A .eng file must be loaded with load_from_eng_file (a bare path passed to
    # GenericMotor(thrust_source=...) is read as CSV and the header row becomes NaN).
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

    # Cd curve must be a function of (deployment_level, mach); RocketPy treats level 0
    # as zero drag, and level 1 gives exactly `cda` (reference area is 1 m^2).
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

    # ---- Stochastic wrappers: EVERYTHING nominal except the weather ----
    # NOTE: in RocketPy a bare number in a Stochastic* argument means "std deviation",
    # so positions are given as (nominal, 0.0) tuples and other args are left unset.
    s_env = WeatherStochasticEnvironment(ENV, WEATHER)
    s_motor = StochasticGenericMotor(motor)
    s_rocket = StochasticRocket(rocket=rocket)
    s_rocket.add_motor(s_motor, position=(4.1, 0.0))
    s_rocket.add_nose(StochasticNoseCone(nose), position=(0.0, 0.0))
    s_rocket.add_trapezoidal_fins(StochasticTrapezoidalFins(fins), position=(3.5, 0.0))
    s_rocket.add_tail(StochasticTail(tail), position=(4.0, 0.0))
    s_rocket.add_parachute(StochasticParachute(main))
    s_rocket.add_parachute(StochasticParachute(drogue))
    s_rocket.add_air_brakes(StochasticAirBrakes(air_brakes), controller)

    # Only apogee is needed, so stop each flight there (much faster)
    s_flight = StochasticFlight(flight=flight, terminate_on_apogee=True)

    return MonteCarlo(
        filename=filename,
        environment=s_env,
        rocket=s_rocket,
        flight=s_flight,
        export_list=["apogee", "apogee_time", "apogee_x", "apogee_y", "max_mach_number"],
    )


# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------
WEATHER_COLS = ["wind_speed_factor", "wind_dir_offset_deg", "temp_offset_k", "pressure_factor"]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []

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

        # RocketPy's results hold outputs only; read the sampled weather from the
        # inputs log and match it to each flight by its index.
        with open(os.path.join(OUT_DIR, tag + ".inputs.txt"), encoding="utf-8") as f:
            inputs = {r["index"]: r for r in map(json.loads, f) if "index" in r}
        idx = [int(i) for i in mc.results["index"]]
        d = {"airbrake_cda_m2": cda, "sample": idx}
        for c in WEATHER_COLS:
            d[c] = [inputs.get(i, {}).get(c, np.nan) for i in idx]
        d["apogee_asl_m"] = np.asarray(mc.results["apogee"], dtype=float)
        d["apogee_agl_m"] = d["apogee_asl_m"] - ENV.elevation
        frames.append(pd.DataFrame(d))

    samples = pd.concat(frames, ignore_index=True)
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