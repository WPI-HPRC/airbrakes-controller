"""
Plot Monte Carlo apogee vs airbrake drag area (CdA) with +/-1 standard deviation
error bars, from airbrake_mc_samples.csv written by airbrake_montecarlo.py.

Bottom x-axis: CdA in m^2 (what the simulation actually swept).
Top x-axis:    equivalent brake area in cm^2 for an ASSUMED panel Cd, so you can
               read off "how much plate area do I need" for your design Cd.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
SAMPLES_CSV = "t4/airbrake_mc_samples.csv"
# SAMPLES_CSV = "airbrake_mc_out_weather_t3/airbrake_mc_samples.csv"
OUT_PNG = "t4/apogee_vs_cda.png"
OUT_TABLE = "t4/apogee_vs_cda_table.csv"
APOGEE_COL = "apogee_agl_m"   # or "apogee_asl_m"
USE_FEET = False              # plot apogee in feet
ASSUMED_PANEL_CD = 1.0        # for the top axis only; set to your panel's expected Cd
TARGET_APOGEE_FT = 30000      # target apogee in feet (converted to metres below); None = off
TARGET_BAND = 0.05            # +/- fraction around the target (0.05 = +/-5 %)
# ----------------------------------------------------------------------------

df = pd.read_csv(SAMPLES_CSV)
scale = 3.28084 if USE_FEET else 1.0
unit = "ft" if USE_FEET else "m"
df["apogee"] = df[APOGEE_COL] * scale

g = df.groupby("airbrake_cda_m2")["apogee"]
tab = g.agg(n="count", mean="mean", std=lambda x: x.std(ddof=1), min="min", max="max")
tab = tab.reset_index()
tab["equiv_area_cm2_at_assumed_cd"] = tab["airbrake_cda_m2"] / ASSUMED_PANEL_CD * 1e4
tab.to_csv(OUT_TABLE, index=False)
print(tab.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(
    tab["airbrake_cda_m2"], tab["mean"], yerr=tab["std"],
    fmt="o-", capsize=4, lw=1.5, label=r"Mean apogee $\pm 1\sigma$",
)
if TARGET_APOGEE_FT is not None:
    target_m = TARGET_APOGEE_FT * 0.3048                 # 25,000 ft = 7,620 m
    lo_m, hi_m = target_m * (1 - TARGET_BAND), target_m * (1 + TARGET_BAND)
    print(f"target {target_m:.0f} m, band {lo_m:.0f} - {hi_m:.0f} m")
    ax.axhline(target_m * scale, color="orange", ls=":", lw=2,
               label=f"Target {TARGET_APOGEE_FT:,} ft ({target_m:,.0f} m)")
    ax.axhline(hi_m * scale, color="red", ls=":", lw=1.8,
               label=f"+{TARGET_BAND:.0%} ({hi_m:,.0f} m)")
    ax.axhline(lo_m * scale, color="red", ls=":", lw=1.8,
               label=f"-{TARGET_BAND:.0%} ({lo_m:,.0f} m)")

ax.set_xlabel(r"Airbrake drag area $C_d A$ (m$^2$)")
ax.set_ylabel(f"Apogee ({unit} {'AGL' if 'agl' in APOGEE_COL else 'ASL'})")
n = int(tab["n"].min())
ax.set_title(f"Monte Carlo apogee vs airbrake CdA ({n}+ flights per point)")
ax.grid(alpha=0.3)
ax.legend()

# Secondary axis: area for an assumed panel Cd (area = CdA / Cd)
secax = ax.secondary_xaxis(
    "top",
    functions=(lambda x: x / ASSUMED_PANEL_CD * 1e4, lambda a: a * ASSUMED_PANEL_CD / 1e4),
)
secax.set_xlabel(f"Equivalent brake area (cm$^2$) if panel $C_d$ = {ASSUMED_PANEL_CD:g}")

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=200)
print("saved", OUT_PNG)
plt.show()