"""
Download historical weather-balloon soundings for the launch season and cache them
as ONE csv file (run this once, it needs internet; the Monte Carlo itself is offline).

Source: Iowa Environmental Mesonet RAOB archive
        https://mesonet.agron.iastate.edu/archive/raob/
        CSV service: /cgi-bin/request/raob.py   (args: station, sts, ets)
Columns returned: station, validUTC, levelcode, pressure_mb, height_m, tmpc, dwpc,
                  drct, speed_kts, bearing, range_sm

Nearest NWS upper-air station to the launch site near Midland, TX
(31.9974 N, 102.0780 W) is Midland/Odessa, TX (MAF, ~872 m ASL, ~15 km from the
site). Edit the settings below if you launch somewhere else.

Usage:  python fetch_soundings.py
"""

import datetime as dt
import io
import time
import urllib.parse
import urllib.request

import pandas as pd

# ---------------- settings ----------------
STATIONS_TO_TRY = ["KMAF", "MAF"]   # Midland/Odessa; first id that returns data wins
LAUNCH_MMDD = "06-15"               # centre of the launch season (keep in sync with the Monte Carlo script)
WINDOW_DAYS = 14                    # +/- days around it
YEARS = range(2005, 2027)           # years to pull (years with no data are skipped)
OUT_CSV = "../rocket_data/soundings_KMAF.csv"      # must match SOUNDINGS_CSV in the Monte Carlo script
BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/raob.py"
# ------------------------------------------


def fetch_csv(url, retries=3, pause=5.0):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "airbrake-mc-sounding-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as exc:  # network hiccup, 5xx, rate limit ...
            last = exc
            time.sleep(pause * (attempt + 1))
    raise RuntimeError(f"giving up on {url}: {last}")


def fetch_year(station, year):
    m, d = (int(x) for x in LAUNCH_MMDD.split("-"))
    centre = dt.datetime(year, m, d)
    start = centre - dt.timedelta(days=WINDOW_DAYS + 1)
    end = centre + dt.timedelta(days=WINDOW_DAYS + 2)   # end time is exclusive
    q = urllib.parse.urlencode({
        "station": station,
        "sts": start.strftime("%Y-%m-%dT00:00:00Z"),
        "ets": end.strftime("%Y-%m-%dT00:00:00Z"),
    })
    try:
        text = fetch_csv(f"{BASE_URL}?{q}")
    except RuntimeError as exc:   # one bad year should not kill the whole download
        print(f"  warning: {exc}")
        return None
    if not text.strip() or "validUTC" not in text.split("\n", 1)[0]:
        return None   # empty answer / error page
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    return df if len(df) else None


def main():
    for station in STATIONS_TO_TRY:
        frames = []
        for year in YEARS:
            df = fetch_year(station, year)
            n = 0 if df is None else df["validUTC"].nunique()
            print(f"{station} {year}: {n} soundings")
            if df is not None:
                frames.append(df)
            time.sleep(1.0)  # be polite to the free service
        if frames:
            out = pd.concat(frames, ignore_index=True)
            out.to_csv(OUT_CSV, index=False)
            print(f"\nSaved {out['validUTC'].nunique()} soundings ({len(out)} rows) to {OUT_CSV}")
            return
        print(f"No data for station id {station}, trying next id ...")
    raise SystemExit("No soundings downloaded. Check STATIONS_TO_TRY / YEARS / your connection, "
                     "or download the csv by hand from the IEM RAOB archive page.")


if __name__ == "__main__":
    main()