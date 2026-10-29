"""
Weather-only Monte Carlo sweep of airbrake drag area (CdA) with RocketPy.

For each airbrake CdA in AIRBRAKE_CDA_M2 a full MonteCarlo run (N_SIMS flights) is
performed. The ONLY thing that varies from flight to flight is the weather: every
flight is flown through one complete, real radiosonde sounding (wind speed + direction,
temperature and pressure at every altitude) drawn from the launch-season climatology
loaded by weather_stuff.py. Motor, rocket mass/inertia, drag, parachutes, launch
angles etc. stay at nominal.

Airbrake model
  * CdA (m^2) is the drag area of the fully deployed brakes at LOW Mach (<= 0.3).
  * The drag area is then scaled by a Mach-dependent compressibility factor
    (AIRBRAKE_MACH_FACTOR) and by the deployment level.
  * The airbrakes stay retracted until burnout AND Mach < MACH_DEPLOY, then go to 100 %.

Outputs (in OUT_DIR):
  airbrake_mc_samples.csv  - one row per flight: CdA, sounding used, apogee ASL/AGL
  airbrake_mc_summary.csv  - mean/std/min/max apogee per CdA
  mc_cda_<cm2>.*.txt       - raw RocketPy MonteCarlo files (inputs/outputs/errors)

Needs weather_stuff.py next to this file and a sounding CSV (see SOUNDINGS_CSV).
Written for RocketPy 1.13.
"""

import json
import os
from math import sqrt

import numpy as np
import pandas as pd

from rocketpy import GenericMotor, Environment, Motor, MonteCarlo, Rocket, Flight
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

import weather_stuff as ws

# ----------------------------------------------------------------------------
# User settings
# ----------------------------------------------------------------------------
# Airbrake drag force is 0.5*rho*v^2*Cd*A, so only the product CdA (m^2) matters.
# The reference area is fixed at 1 m^2 so the "Cd" handed to RocketPy equals CdA in m^2.
# Afterwards: required area = CdA / Cd_of_your_panel_design (Cd at low Mach).
AIRBRAKE_CDA_M2 = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
AIRBRAKE_REF_AREA = 1.0    # m^2, keep at 1.0 so Cd == CdA
MACH_DEPLOY = 0.8          # fully deploy below this Mach (after burnout)
ENG_FILE = "AeroTech_O5500X-PS.eng"
# End of the thrust curve (last time in the .eng); brakes cannot deploy before this
BURN_TIME = Motor.import_eng(ENG_FILE)[2][-1][0]
N_SIMS = 5               # flights per CdA value
PARALLEL = False
N_WORKERS = 4
OUT_DIR = "t4"

# ---- Airbrake drag vs Mach and deployment ----
# Multiplier applied to the low-Mach CdA, as (Mach, factor) points, linearly
# interpolated and held constant outside the range. Shape: nearly flat in low subsonic,
# compressibility rise through the transonic range, peak just above M=1, slow decline
# supersonic. >>> PLACEHOLDER typical of a bluff plate normal to the flow. <<<
# Replace with CFD / wind-tunnel data for YOUR panel (divide Cd(M) by Cd at M<=0.3).
AIRBRAKE_MACH_FACTOR = [
    (0.0, 1.00), (0.3, 1.00), (0.5, 1.04), (0.6, 1.08), (0.7, 1.13), (0.8, 1.20),
    (0.9, 1.32), (1.0, 1.45), (1.1, 1.52), (1.3, 1.50), (1.6, 1.42), (2.0, 1.32),
    (3.0, 1.25),
]
# Effective area vs deployment level = level ** DEPLOY_EXPONENT. RocketPy defines the
# level as the deployed AREA fraction, so 1.0 (linear) is the default. Raise above 1 if
# small extensions are less effective (e.g. panels still inside the boundary layer).
DEPLOY_EXPONENT = 1.0

# ---- Weather (real radiosondes, see weather_stuff.py) ----
LAUNCH_LAT, LAUNCH_LON, LAUNCH_ELEV = 31.9974, -102.0780, 847
SOUNDINGS_CSV = "soundings_KMAF.csv"  # written by fetch_soundings.py (Midland/Odessa, TX)
LAUNCH_MMDD = "06-15"            # centre of launch season, "MM-DD" (match fetch_soundings.py)
SEASON_WINDOW_DAYS = 14          # use soundings within +/- this many days (any year)
SOUNDING_HOURS_UTC = (0,)        # EDIT: 0 = 00Z (evening local), 12 = 12Z (dawn local)
MIN_SOUNDING_TOP_AGL_M = 12000.0 # sounding must reach this far above the pad (> apogee)
MAX_SURFACE_WIND_MS = 9.0        # drop soundings you would scrub for (~20 mph); None = keep all
# Fixed seed => in serial mode every CdA is flown through the SAME soundings in the SAME
# order (paired comparison: apogee differences come from the brakes, not the weather luck).
# Set None for an independent random draw per flight. Ignored when PARALLEL=True.
WEATHER_SEED = 12345

WEATHER_COLS = [
    "sounding_label", "surface_wind_ms", "surface_wind_from_deg",
    "wind_mean_0_5km_ms", "surface_temp_k", "surface_pressure_pa",
]


# ----------------------------------------------------------------------------
# Airbrake drag model
# ----------------------------------------------------------------------------
_MACH_PTS = np.array([p[0] for p in AIRBRAKE_MACH_FACTOR], dtype=float)
_FACT_PTS = np.array([p[1] for p in AIRBRAKE_MACH_FACTOR], dtype=float)


def airbrake_cd_factor(deployment_level, mach):
    """Drag-area multiplier at (deployment level, Mach) relative to full deployment, low Mach."""
    level = min(max(float(deployment_level), 0.0), 1.0)
    return (level ** DEPLOY_EXPONENT) * float(np.interp(mach, _MACH_PTS, _FACT_PTS))


# ----------------------------------------------------------------------------
# Weather: one real sounding per flight
# ----------------------------------------------------------------------------
class SoundingStochasticEnvironment(StochasticEnvironment):
    """StochasticEnvironment whose only randomness is which real sounding is flown."""

    def __init__(self, environment, profiles, launch_elev_m, order_seed=None):
        self.profiles = list(profiles)
        self.launch_elev_m = launch_elev_m
        self._call_count = 0
        # fixed shuffled order (no repeats until every sounding was used) or None = random
        self._order = (
            None if order_seed is None
            else np.random.default_rng(order_seed).permutation(len(self.profiles))
        )
        super().__init__(environment)  # no elevation/gravity/wind-factor variation

    def _set_stochastic(self, seed=None):  # also called per worker with its own seed
        super()._set_stochastic(seed)
        self._rng = np.random.default_rng(seed)

    def create_object(self):
        if self._order is not None:
            idx = int(self._order[self._call_count % len(self._order)])
            self._call_count += 1
        else:
            idx = int(self._rng.integers(len(self.profiles)))
        prof = self.profiles[idx]
        ws.apply_profile(self.obj, prof, self.launch_elev_m)
        # MonteCarlo logs last_rnd_dict to <name>.inputs.txt
        self.last_rnd_dict = {
            "sounding_index": idx, "sounding_label": prof.label,
            **ws.describe(prof, self.launch_elev_m),
        }
        return self.obj


# Nominal environment for the deterministic template flight (weather set in build_montecarlo)
ENV = Environment(latitude=LAUNCH_LAT, longitude=LAUNCH_LON, elevation=LAUNCH_ELEV)


# ----------------------------------------------------------------------------
# Airbrake controller: fully deploy below MACH_DEPLOY after burnout
# ----------------------------------------------------------------------------
def controller_function(
    time, sampling_rate, state, state_history, observed_variables, air_brakes,
    sensors, environment,
):
    z = state[2]  # altitude ASL
    # Air-relative speed (what RocketPy uses for drag and for the Cd(Mach) lookup):
    # ground velocity minus the wind of THIS flight's sounding at the current altitude.
    vx = state[3] - environment.wind_velocity_x(z)
    vy = state[4] - environment.wind_velocity_y(z)
    vz = state[5]
    mach = sqrt(vx**2 + vy**2 + vz**2) / environment.speed_of_sound(z)

    if time > BURN_TIME and mach < MACH_DEPLOY:
        air_brakes.deployment_level = 1.0
    else:
        air_brakes.deployment_level = 0.0

    return (time, air_brakes.deployment_level, mach)


# ----------------------------------------------------------------------------
# Build deterministic + stochastic objects for one CdA value
# ----------------------------------------------------------------------------
def build_montecarlo(cda, filename, profiles):
    ws.apply_profile(ENV, profiles[0], LAUNCH_ELEV)  # nominal weather for the template flight

    # ---- Deterministic template ----
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

    # RocketPy calls this with (deployment_level, free-stream Mach). Level 0 is always
    # treated as zero drag. Reference area is 1 m^2, so the returned value is the drag
    # AREA in m^2: low-Mach CdA * Mach factor * deployment factor.
    def cd_curve(deployment_level, mach):
        return cda * airbrake_cd_factor(deployment_level, mach)

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
    s_env = SoundingStochasticEnvironment(
        ENV, profiles, LAUNCH_ELEV,
        order_seed=None if PARALLEL else WEATHER_SEED,
    )
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
def load_profiles():
    profiles, stats = ws.load_soundings(
        SOUNDINGS_CSV,
        launch_elev_m=LAUNCH_ELEV,
        launch_mmdd=LAUNCH_MMDD,
        window_days=SEASON_WINDOW_DAYS,
        hours_utc=SOUNDING_HOURS_UTC,
        min_top_agl_m=MIN_SOUNDING_TOP_AGL_M,
        max_surface_wind_ms=MAX_SURFACE_WIND_MS,
    )
    if not profiles:
        raise RuntimeError(f"No usable soundings in {SOUNDINGS_CSV}: {stats}")
    if len(profiles) < N_SIMS:
        print(f"WARNING: only {len(profiles)} soundings for {N_SIMS} flights; "
              "soundings will be reused.")
    return profiles


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    profiles = load_profiles()
    frames = []

    for cda in AIRBRAKE_CDA_M2:
        print(f"\n=== Airbrake CdA = {cda:.4f} m^2 ===")
        # No "." in the base name: MonteCarlo treats everything after a dot as an extension
        tag = f"mc_cda_{int(round(cda * 1e4)):04d}"  # CdA in cm^2
        mc = build_montecarlo(cda, os.path.join(OUT_DIR, tag), profiles)
        mc.simulate(
            number_of_simulations=N_SIMS,
            append=False,
            parallel=PARALLEL,
            n_workers=N_WORKERS if PARALLEL else None,
        )
        mc.import_results()

        # RocketPy's results hold outputs only; read the sounding info from the
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