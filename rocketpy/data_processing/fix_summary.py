"""
Rebuild airbrake_mc_samples.csv and airbrake_mc_summary.csv from the raw RocketPy
MonteCarlo files left in the output directory by airbrake_montecarlo_weather_tc.py.

The main script overwrites the two CSVs on every run, but the per-CdA raw files
(mc_cda_<cm2>.inputs.txt / .outputs.txt) stay on disk, so they can be used to
recreate the CSVs covering EVERY CdA value that has ever been simulated.

Usage:
    python rebuild_airbrake_summary.py                      # uses ./airbrake_mc_out_weather
    python rebuild_airbrake_summary.py --dir my_out_dir
    python rebuild_airbrake_summary.py --elevation 847      # launch site elevation ASL (m)

No RocketPy install is needed; only numpy and pandas.
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

DEFAULT_DIR = "../airbrake_mc_out_weather"
DEFAULT_ELEVATION_M = 847  # LAUNCH_ELEV in the main script
WEATHER_COLS = ["wind_speed_factor", "wind_dir_offset_deg", "temp_offset_k", "pressure_factor"]

# The main script names files mc_cda_<CdA in cm^2, zero padded>.<kind>.txt
TAG_RE = re.compile(r"^(mc_cda_(\d+))\.outputs\.txt$")


def read_jsonl(path):
    """Read a file of one-JSON-object-per-line, skipping blank/corrupt lines."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                print(f"  warning: {os.path.basename(path)} line {n} is not valid JSON, skipped")
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_run(out_dir, tag, cda):
    """Return a DataFrame with one row per flight for a single CdA run."""
    outputs = read_jsonl(os.path.join(out_dir, tag + ".outputs.txt"))
    inputs = {r["index"]: r for r in read_jsonl(os.path.join(out_dir, tag + ".inputs.txt"))
              if "index" in r}

    records = []
    for pos, o in enumerate(outputs):
        if "apogee" not in o:
            continue
        idx = o.get("index", pos)  # fall back to line order if no index is stored
        rec = {"airbrake_cda_m2": cda, "sample": int(idx)}
        for c in WEATHER_COLS:
            rec[c] = inputs.get(idx, {}).get(c, np.nan)
        rec["apogee_asl_m"] = float(o["apogee"])
        records.append(rec)
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dir", default=DEFAULT_DIR, help="directory holding the raw MonteCarlo files")
    ap.add_argument("--elevation", type=float, default=DEFAULT_ELEVATION_M,
                    help="launch elevation ASL in metres (to convert apogee ASL -> AGL)")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        raise SystemExit(f"Directory not found: {args.dir}")

    frames = []
    for name in sorted(os.listdir(args.dir)):
        m = TAG_RE.match(name)
        if not m:
            continue
        tag, cm2 = m.group(1), int(m.group(2))
        cda = cm2 / 1e4  # cm^2 -> m^2
        df = load_run(args.dir, tag, cda)
        print(f"{tag}: CdA = {cda:.4f} m^2, {len(df)} flights")
        if len(df):
            frames.append(df)

    if not frames:
        raise SystemExit("No mc_cda_*.outputs.txt files with results were found.")

    samples = pd.concat(frames, ignore_index=True)
    samples["apogee_agl_m"] = samples["apogee_asl_m"] - args.elevation
    samples = samples.sort_values(["airbrake_cda_m2", "sample"]).reset_index(drop=True)
    samples.to_csv(os.path.join(args.dir, "airbrake_mc_samples.csv"), index=False)

    summary = (
        samples.groupby("airbrake_cda_m2")["apogee_agl_m"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={c: f"apogee_agl_{c}" for c in ["mean", "std", "min", "max"]})
    )
    summary.to_csv(os.path.join(args.dir, "airbrake_mc_summary.csv"), index=False)
    print("\n", summary.to_string(index=False))
    print(f"\nWrote airbrake_mc_samples.csv and airbrake_mc_summary.csv to {args.dir}")


if __name__ == "__main__":
    main()