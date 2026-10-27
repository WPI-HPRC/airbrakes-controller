import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ============================================================
# Settings
# ============================================================

# FILE_NAME = "./airbrake_mc_out/mc_cda_0150.outputs.txt"
# FILE_NAME = "./airbrake_mc_components_out_1-5_t3/mc_comp_cda_0200.outputs.txt"
FILE_NAME = "./airbrake_mc_out_weather_t3/mc_cda_0000.outputs.txt"

TARGET_APOGEE_FT = 25_000
PERCENT_RANGE = 0.05

NUM_BINS = 50


# ============================================================
# Target apogee and ±5% range
# ============================================================

TARGET_APOGEE_M = TARGET_APOGEE_FT * 0.3048

LOWER_LIMIT = TARGET_APOGEE_M * (1 - PERCENT_RANGE)
UPPER_LIMIT = TARGET_APOGEE_M * (1 + PERCENT_RANGE)


# ============================================================
# Read Monte Carlo results
# ============================================================

apogees = []

with open(FILE_NAME, "r") as file:
    for line in file:
        if line.strip():
            result = json.loads(line)
            apogees.append(result["apogee"] - 847)#847 is from the launch elevation

apogees = np.array(apogees)


# ============================================================
# Determine how many simulations hit the target range
# ============================================================

in_target = (
    (apogees >= LOWER_LIMIT) &
    (apogees <= UPPER_LIMIT)
)

num_in_target = np.sum(in_target)
total_simulations = len(apogees)

percentage_in_target = (
    num_in_target / total_simulations * 100
)


# ============================================================
# Print results
# ============================================================

print(f"Target apogee:       {TARGET_APOGEE_FT:,} ft")
print(f"Target apogee:       {TARGET_APOGEE_M:.2f} m")
print(f"-5% limit:           {LOWER_LIMIT:.2f} m")
print(f"+5% limit:           {UPPER_LIMIT:.2f} m")
print()
print(f"Total simulations:   {total_simulations}")
print(f"Within ±5%:          {num_in_target}")
print(f"Percentage:          {percentage_in_target:.2f}%")


# ============================================================
# Histogram
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(
    apogees,
    bins=NUM_BINS,
    edgecolor="black"
)


# ============================================================
# Target boundaries
# ============================================================

ax.axvline(
    LOWER_LIMIT,
    linestyle="--",
    linewidth=2,
    label="-5% target"
)

ax.axvline(
    TARGET_APOGEE_M,
    linestyle="-",
    linewidth=2,
    label="25,000 ft target"
)

ax.axvline(
    UPPER_LIMIT,
    linestyle="--",
    linewidth=2,
    label="+5% target"
)


# ============================================================
# X-axis resolution
# ============================================================

# Major ticks every 10 meters
ax.xaxis.set_major_locator(
    ticker.MultipleLocator(100)
)

# Minor ticks every 2 meters
ax.xaxis.set_minor_locator(
    ticker.MultipleLocator(25)
)

# Make minor ticks visible
ax.tick_params(
    axis="x",
    which="minor",
    length=3
)


# ============================================================
# Labels and formatting
# ============================================================

ax.set_xlabel("Apogee (m)")
ax.set_ylabel("Number of Occurrences")
ax.set_title("Monte Carlo Apogee Distribution")

ax.grid(
    axis="y",
    alpha=0.3
)

ax.legend()

plt.tight_layout()
plt.show()