"""
Realistic launch-site weather for the airbrake Monte Carlo, built from REAL
radiosonde (weather-balloon) soundings instead of a hand-typed profile that is
scaled / rotated / offset.

Why this is more realistic than the old random factors
------------------------------------------------------
Each Monte Carlo flight gets one complete, historical sounding (wind speed AND
direction at every altitude, temperature and pressure) measured by the balloon
released at the nearest NWS upper-air station on a day inside your launch season.
That keeps everything physically self-consistent:
  * wind speed and direction change with height the way they really do (shear,
    veering, low-level jets) instead of the whole profile being scaled/rotated;
  * temperature and pressure come from the same air mass as the wind;
  * the spread of the weather is the real climatological spread for the season.

This module deliberately does NOT import rocketpy so it can be tested on its own.
Soundings are downloaded once with fetch_soundings.py and cached as a CSV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

KT_TO_MS = 0.514444
G0 = 9.80665       # m/s^2
R_AIR = 287.053    # J/(kg K)
LAPSE = 0.0065     # K/m, only used to extend temperature below the lowest level


def _scalar_if_0d(a):
    return float(a) if np.ndim(a) == 0 else a


@dataclass(eq=False)
class SoundingProfile:
    """One measured sounding. All heights are metres ABOVE SEA LEVEL (as RocketPy wants)."""

    label: str
    h_t: np.ndarray       # heights of the temperature samples
    temp_k: np.ndarray
    h_p: np.ndarray       # heights of the pressure samples
    ln_p: np.ndarray      # ln(pressure in Pa), strictly decreasing with height
    h_w: np.ndarray       # heights of the wind samples
    wind_u: np.ndarray    # east component, m/s
    wind_v: np.ndarray    # north component, m/s

    # --- callables handed to RocketPy: single argument z (m ASL), valid at ANY z ---
    def temperature(self, z):
        z = np.asarray(z, dtype=float)
        t = np.interp(z, self.h_t, self.temp_k)  # constant above the top
        t = np.where(z < self.h_t[0], self.temp_k[0] + LAPSE * (self.h_t[0] - z), t)
        return _scalar_if_0d(t)

    def pressure(self, z):
        z = np.asarray(z, dtype=float)
        z0, z1 = self.h_p[0], self.h_p[-1]
        lnp = np.interp(z, self.h_p, self.ln_p)
        # outside the measured range extend hydrostatically with the local temperature
        t0, t1 = float(self.temperature(z0)), float(self.temperature(z1))
        lnp = np.where(z < z0, self.ln_p[0] + G0 * (z0 - z) / (R_AIR * t0), lnp)
        lnp = np.where(z > z1, self.ln_p[-1] - G0 * (z - z1) / (R_AIR * t1), lnp)
        return _scalar_if_0d(np.exp(lnp))

    def wind(self, z):
        z = np.asarray(z, dtype=float)
        return np.interp(z, self.h_w, self.wind_u), np.interp(z, self.h_w, self.wind_v)

    @property
    def wind_top(self):
        return float(self.h_w[-1])


def wind_from_components(u, v):
    """(speed m/s, compass direction the wind blows FROM in deg) from u (east), v (north)."""
    return float(np.hypot(u, v)), float(np.degrees(np.arctan2(-u, -v)) % 360.0)


def describe(profile: SoundingProfile, launch_elev_m: float) -> dict:
    """A few numbers that summarise a profile; logged next to every flight."""
    z = np.arange(launch_elev_m, launch_elev_m + 5000.0 + 1.0, 100.0)
    u, v = profile.wind(z)
    spd, frm = wind_from_components(u[0], v[0])
    return {
        "surface_wind_ms": spd,
        "surface_wind_from_deg": frm,
        "wind_mean_0_5km_ms": float(np.mean(np.hypot(u, v))),
        "surface_temp_k": float(profile.temperature(launch_elev_m)),
        "surface_pressure_pa": float(profile.pressure(launch_elev_m)),
    }


def apply_profile(env, profile: SoundingProfile, launch_elev_m: float, step_m: float = 100.0):
    """(Re)build the atmosphere of a RocketPy Environment, in place, from one sounding."""
    zs = np.arange(launch_elev_m, profile.wind_top, step_m)
    zs = np.append(zs, profile.wind_top)
    u, v = profile.wind(zs)
    env.set_atmospheric_model(
        type="custom_atmosphere",
        # single-argument lambdas, exactly like the original script (RocketPy inspects the arity)
        pressure=lambda z: profile.pressure(z),
        temperature=lambda z: profile.temperature(z),
        wind_u=list(zip(zs.tolist(), u.tolist())),
        wind_v=list(zip(zs.tolist(), v.tolist())),
    )


# ----------------------------------------------------------------------------
# Loading / quality control
# ----------------------------------------------------------------------------
def _covers(h, lo, hi, max_gap):
    """True if the sorted heights h span [lo, hi] without a data gap larger than max_gap."""
    if len(h) < 4:
        return False, "too_few_levels"
    if h[0] > lo + 300.0:
        return False, "no_surface_level"
    if h[-1] < hi:
        return False, "low_top"
    gaps = np.diff(h)
    relevant = (h[1:] > lo) & (h[:-1] < hi)
    if relevant.any() and gaps[relevant].max() > max_gap:
        return False, "data_gap"
    return True, ""


def _build_profile(g, label, launch_elev_m, min_top_agl_m, max_gap_m):
    lo, hi = launch_elev_m, launch_elev_m + min_top_agl_m
    g = g.copy()
    # calm wind is often reported with a missing direction
    g.loc[(g["speed_kts"] == 0) & g["drct"].isna(), "drct"] = 0.0

    t = g[["height_m", "tmpc"]].dropna().drop_duplicates("height_m").sort_values("height_m")
    p = g[["height_m", "pressure_mb"]].dropna().drop_duplicates("height_m").sort_values("height_m")
    w = g[["height_m", "drct", "speed_kts"]].dropna().drop_duplicates("height_m").sort_values("height_m")

    # keep only pressure samples that decrease with height (RocketPy inverts p(z))
    keep, last = [], np.inf
    for lnp in np.log(p["pressure_mb"].to_numpy() * 100.0):
        if lnp < last:
            keep.append(True)
            last = lnp
        else:
            keep.append(False)
    p = p[np.array(keep, dtype=bool)] if len(p) else p

    for df in (t, p, w):
        ok, why = _covers(df["height_m"].to_numpy(), lo, hi, max_gap_m)
        if not ok:
            return None, why

    speed = w["speed_kts"].to_numpy() * KT_TO_MS
    d = np.radians(w["drct"].to_numpy())
    return SoundingProfile(
        label=label,
        h_t=t["height_m"].to_numpy(float),
        temp_k=t["tmpc"].to_numpy(float) + 273.15,
        h_p=p["height_m"].to_numpy(float),
        ln_p=np.log(p["pressure_mb"].to_numpy(float) * 100.0),
        h_w=w["height_m"].to_numpy(float),
        wind_u=-speed * np.sin(d),   # wind blows FROM d: east component
        wind_v=-speed * np.cos(d),   # north component
    ), ""


def load_soundings(
    csv_path,
    *,
    launch_elev_m,
    launch_mmdd,
    window_days=14,
    hours_utc=(0,),
    min_top_agl_m=12000.0,
    max_gap_m=3000.0,
    max_surface_wind_ms=None,
    verbose=True,
):
    """
    Read the cached sounding CSV (see fetch_soundings.py) and return a list of
    SoundingProfile objects that pass quality control.

    launch_mmdd          "MM-DD" centre of your launch season
    window_days          keep soundings within +/- this many days of launch_mmdd (any year)
    hours_utc            nominal balloon times to keep: 0 (=00Z, evening local) and/or 12
                         (=12Z, dawn local). Pick the one closest to your launch time.
    min_top_agl_m        sounding must reach this height above the pad (must exceed your apogee)
    max_gap_m            reject soundings with a larger data gap below min_top_agl_m
    max_surface_wind_ms  drop soundings whose surface wind exceeds this (flights would be
                         scrubbed in such wind). None = keep everything.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"validutc", "pressure_mb", "height_m", "tmpc", "drct", "speed_kts"}
    if need - set(df.columns):
        raise ValueError(f"{csv_path} is missing columns {sorted(need - set(df.columns))}")

    df["valid"] = pd.to_datetime(df["validutc"], utc=True, errors="coerce")
    for c in ["pressure_mb", "height_m", "tmpc", "drct", "speed_kts"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # sentinel / garbage values -> NaN
    df.loc[~df["pressure_mb"].between(1, 1100), "pressure_mb"] = np.nan
    df.loc[~df["tmpc"].between(-100, 60), "tmpc"] = np.nan
    df.loc[~df["speed_kts"].between(0, 250), "speed_kts"] = np.nan
    df.loc[~df["drct"].between(0, 360), "drct"] = np.nan
    df = df.dropna(subset=["valid", "height_m"])

    month, day = (int(x) for x in launch_mmdd.split("-"))
    target = pd.Timestamp(year=2001, month=month, day=day).dayofyear
    diff = (df["valid"].dt.dayofyear - target).abs()
    diff = np.minimum(diff, 365 - diff)
    nominal_hour = (df["valid"] + pd.Timedelta(minutes=30)).dt.hour
    df = df[(diff <= window_days) & nominal_hour.isin(list(hours_utc))]

    keys = ["station", "valid"] if "station" in df.columns else ["valid"]
    stats = {"candidates": 0, "accepted": 0}
    profiles = []
    for key, g in df.groupby(keys):
        stats["candidates"] += 1
        valid = key[-1]
        prof, why = _build_profile(
            g, f"{valid:%Y-%m-%d %H}Z", launch_elev_m, min_top_agl_m, max_gap_m
        )
        if prof is None:
            stats[why] = stats.get(why, 0) + 1
            continue
        if max_surface_wind_ms is not None:
            u, v = prof.wind(launch_elev_m)
            if float(np.hypot(u, v)) > max_surface_wind_ms:
                stats["surface_wind_too_high"] = stats.get("surface_wind_too_high", 0) + 1
                continue
        profiles.append(prof)
        stats["accepted"] += 1

    if verbose:
        print(f"Soundings: {stats['accepted']} usable out of {stats['candidates']} "
              f"in the season window  (details: {stats})")
    return profiles, stats