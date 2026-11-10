""" how to use  - python update_from_openrocket.py  
Pull the latest OpenRocket designs from GitHub and convert one into RocketPy inputs.

    python update_from_openrocket.py                      # default design + first sim with data
    python update_from_openrocket.py --list               # show designs and their simulations
    python update_from_openrocket.py --ork "2026-2027 30k.ork" --sim "O5500 in Monahans"
    python update_from_openrocket.py --no-pull            # use the copy already downloaded

What it does
  1. git clone / git pull https://github.com/WPI-HPRC/openrockets into ./openrockets_source
     (ignored by git, it is just a local cache).
  2. Opens the .ork file. An .ork is a zip holding one XML file ("rocket.ork") with the
     rocket's parts AND the data of every simulation that was run and saved in OpenRocket.
  3. Writes into ./rocket_data:
       power_on_drag.csv   Mach,Cd while the motor burns   (RocketPy power_on_drag)
       power_off_drag.csv  Mach,Cd from burnout to apogee  (RocketPy power_off_drag)
       thrust_curve.csv    time,thrust from the OpenRocket sim
       rocket_params.json  radius, masses, inertias, CG, nose/fins/tail/motor positions,
                           parachutes - all in SI units, positions in metres from the
                           nose tip (RocketPy coordinate_system_orientation="nose_to_tail")

Important: open the design in OpenRocket, run the simulation and SAVE before pushing it to
GitHub. Only saved simulation results end up in the .ork file; a design whose sims were never
run has no drag data and is refused.

Only the Python standard library is used, so no installs are needed beyond git.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

REPO_URL = "https://github.com/WPI-HPRC/openrockets.git"
BRANCH = "main"
DEFAULT_ORK = "2026-2027 30k.ork"

HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE / "openrockets_source"
OUT_DIR = HERE / "rocket_data"

# OpenRocket nose-cone shape -> RocketPy add_nose(kind=...)
NOSE_KINDS = {
    "ogive": "ogive",
    "conical": "conical",
    "ellipsoid": "elliptical",
    "power": "powerseries",
    "parabolic": "powerseries",
}


# ----------------------------------------------------------------------------
# 1. Get the latest designs
# ----------------------------------------------------------------------------
def pull_repo():
    if (SOURCE_DIR / ".git").exists():
        print(f"[git] Updating {SOURCE_DIR.name} ...")
        cmd = ["git", "-C", str(SOURCE_DIR), "pull", "--ff-only", "origin", BRANCH]
    else:
        print(f"[git] Downloading {REPO_URL} ...")
        cmd = ["git", "clone", "--branch", BRANCH, REPO_URL, str(SOURCE_DIR)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        sys.exit(f"git failed:\n{result.stderr.strip()}")
    print("      " + (result.stdout.strip().splitlines() or ["done"])[-1])


def load_ork(path):
    with zipfile.ZipFile(path) as z:
        name = "rocket.ork" if "rocket.ork" in z.namelist() else z.namelist()[0]
        return ET.fromstring(z.read(name))


# ----------------------------------------------------------------------------
# 2. Read the simulation data
# ----------------------------------------------------------------------------
def list_sims(root):
    """[(name, branch_element_or_None)] for every simulation in the file."""
    sims = []
    for sim in root.iter("simulation"):
        branch = sim.find("flightdata/databranch")
        if branch is not None and branch.find("datapoint") is None:
            branch = None
        sims.append((sim.findtext("name"), branch, sim))
    return sims


def read_branch(branch):
    """Return (columns dict name -> list of floats, events dict type -> first time)."""
    names = branch.get("types").split(",")
    rows = [[float(v) for v in dp.text.split(",")] for dp in branch.findall("datapoint")]
    cols = {n: [r[i] for r in rows] for i, n in enumerate(names)}
    events = {}
    for ev in branch.findall("event"):
        events.setdefault(ev.get("type"), float(ev.get("time")))
    return cols, events


def index_at(times, t):
    """Index of the first sample at or after time t."""
    for i, ti in enumerate(times):
        if ti >= t:
            return i
    return len(times) - 1


def drag_curve(mach, cd, i0, i1):
    """Mach/Cd pairs between two indices, sorted by Mach, duplicate Machs averaged."""
    buckets = {}
    for m, c in zip(mach[i0:i1 + 1], cd[i0:i1 + 1]):
        if math.isfinite(m) and math.isfinite(c) and m > 0:
            buckets.setdefault(round(m, 3), []).append(c)
    return [(m, sum(v) / len(v)) for m, v in sorted(buckets.items())]


# ----------------------------------------------------------------------------
# 3. Read the rocket's parts and work out where they sit
# ----------------------------------------------------------------------------
def num(elem, tag, default=None):
    """Read a number; OpenRocket writes 'auto 0.065' when a value is inherited."""
    text = elem.findtext(tag)
    if text is None:
        return default
    try:
        return float(text.split()[-1])
    except ValueError:
        return default


def walk_positions(stage):
    """
    Yield (element, top_position_m) for every part, top measured from the nose tip.
    Body parts (nose, tubes, transitions) are stacked one after another; inner parts
    are placed relative to their parent with OpenRocket's top/middle/bottom/absolute rules.
    """
    def place(children, parent_top, parent_len):
        prev_end = parent_top
        for c in children:
            length = num(c, "length", 0.0)
            if c.tag == "trapezoidfinset":
                length = num(c, "rootchord", 0.0)
            ao = c.find("axialoffset")
            method = ao.get("method") if ao is not None else "after"
            off = float(ao.text) if ao is not None else 0.0
            if method == "top":
                top = parent_top + off
            elif method == "bottom":
                top = parent_top + parent_len - length + off
            elif method == "middle":
                top = parent_top + (parent_len - length) / 2 + off
            elif method == "absolute":
                top = off
            else:  # "after": stacked below the previous body part
                top = prev_end + off
                prev_end = top + length
            yield c, top, length
            yield from place(c.findall("subcomponents/*"), top, length)

    yield from place(stage.findall("subcomponents/*"), 0.0, 0.0)


def read_geometry(root):
    stage = root.find("rocket/subcomponents/stage")
    if len(root.findall("rocket/subcomponents/stage")) > 1:
        print("  ! More than one stage found; only the first (top) stage is converted.")

    geo = {"fins": [], "parachutes": [], "rail_buttons": []}
    body_end = 0.0
    for c, top, length in walk_positions(stage):
        name = c.findtext("name")
        if c.tag in ("nosecone", "bodytube", "transition"):
            body_end = max(body_end, top + length)

        if c.tag == "nosecone" and "nose" not in geo:
            shape = c.findtext("shape")
            param = num(c, "shapeparameter", 0.0)
            kind = NOSE_KINDS.get(shape, shape)
            if shape == "haack":
                kind = "lvhaack" if param > 0.1 else "von karman"
            geo["nose"] = {
                "name": name, "kind": kind, "openrocket_shape": shape,
                "shape_parameter": param, "length": length,
                "base_radius": num(c, "aftradius"), "position": top,
            }
        elif c.tag == "transition" and top > 0:
            geo["tail"] = {
                "name": name, "length": length, "position": top,
                "top_radius": num(c, "foreradius"), "bottom_radius": num(c, "aftradius"),
            }
        elif c.tag == "trapezoidfinset":
            geo["fins"].append({
                "name": name, "n": int(num(c, "fincount", 3)),
                "root_chord": num(c, "rootchord"), "tip_chord": num(c, "tipchord"),
                "span": num(c, "height"), "sweep_length": num(c, "sweeplength"),
                "cant_angle_deg": math.degrees(num(c, "cant", 0.0)),
                "position": top,  # leading edge of the root chord
            })
        elif c.tag == "parachute":
            cd, d = num(c, "cd"), num(c, "diameter")
            event = c.findtext("deployevent")
            geo["parachutes"].append({
                "name": name, "cd": cd, "diameter": d,
                "cd_s": cd * math.pi * (d / 2) ** 2,
                "trigger": "apogee" if event == "apogee"
                           else num(c, "deployaltitude") if event == "altitude" else event,
                "deploy_event": event, "deploy_delay_s": num(c, "deploydelay", 0.0),
                "position": top,
            })
        elif c.tag == "railbutton":
            geo["rail_buttons"].append({"name": name, "position": top + length / 2})

        mount = c.find("motormount")
        if mount is not None and mount.find("motor") is not None:
            m = mount.find("motor")
            geo["motor_mount"] = {
                "name": name, "mount_bottom": top + length,
                "overhang": num(mount, "overhang", 0.0),
                "designation": m.findtext("designation"),
                "manufacturer": m.findtext("manufacturer"),
                "diameter": num(m, "diameter"), "length": num(m, "length"),
            }
    geo["rocket_length"] = body_end
    return geo


# ----------------------------------------------------------------------------
# 4. Put it together
# ----------------------------------------------------------------------------
def convert(ork_path, sim_name=None):
    root = load_ork(ork_path)
    sims = list_sims(root)
    with_data = [s for s in sims if s[1] is not None]
    if not with_data:
        sys.exit(f"'{ork_path.name}' has no saved simulation results. Open it in OpenRocket, "
                 "run a simulation, save, push, and try again.")
    if sim_name:
        match = [s for s in with_data if s[0] == sim_name]
        if not match:
            sys.exit(f"No simulation with data called '{sim_name}'. "
                     f"Choices: {[s[0] for s in with_data]}")
        name, branch, sim_el = match[0]
    else:
        name, branch, sim_el = with_data[0]
    status = sim_el.get("status")
    print(f"[ork] {ork_path.name}  ->  simulation '{name}' (status: {status})")
    if status != "uptodate":
        print("  ! OpenRocket marks this simulation as OUTDATED: the design changed after it "
              "was run. Re-run it in OpenRocket and save for accurate data.")

    cols, events = read_branch(branch)
    t = cols["Time"]
    i_lift = index_at(t, events.get("liftoff", 0.0))
    i_burn = index_at(t, events["burnout"])
    i_apo = index_at(t, events["apogee"])

    # --- drag ---
    mach, cd = cols["Mach number"], cols["Drag coefficient"]
    power_on = drag_curve(mach, cd, i_lift, i_burn)
    power_off = drag_curve(mach, cd, i_burn, i_apo)

    # --- thrust ---
    thrust = [(ti, f) for ti, f in zip(t, cols["Thrust"]) if ti <= t[i_burn] + 1e-9]
    thrust.append((t[i_burn] + 0.001, 0.0))

    # --- mass properties (OpenRocket's values include the motor) ---
    M0, Mb = cols["Mass"][0], cols["Mass"][i_burn]
    m_motor0, m_motor_b = cols["Motor mass"][0], cols["Motor mass"][i_burn]
    cg0 = cols["CG location"][0]
    I_long0, I_rot0 = cols["Longitudinal moment of inertia"][0], cols["Rotational moment of inertia"][0]
    ref_len = cols["Reference length"][0]

    geo = read_geometry(root)
    mm = geo.get("motor_mount")

    rocket = {
        "radius": ref_len / 2,
        "length": geo["rocket_length"],
        "mass_with_motor_liftoff": M0,
        "cg_with_motor_liftoff": cg0,
        "inertia_with_motor_liftoff": [I_long0, I_long0, I_rot0],
        "inertia_with_motor_burnout": [cols["Longitudinal moment of inertia"][i_burn]] * 2
                                      + [cols["Rotational moment of inertia"][i_burn]],
        "mass_without_motor": M0 - m_motor0,
    }

    motor = None
    if mm:
        nozzle = mm["mount_bottom"] + mm["overhang"]          # aft end of the motor
        L, r = mm["length"], mm["diameter"] / 2
        z_m = nozzle - L / 2                                  # motor CG ~ its middle
        m_r = M0 - m_motor0
        cg_r = (M0 * cg0 - m_motor0 * z_m) / m_r
        # Remove the motor (treated as a solid cylinder) with the parallel-axis theorem.
        I_m_long = m_motor0 * (3 * r ** 2 + L ** 2) / 12
        I_r_long = I_long0 - m_r * (cg_r - cg0) ** 2 - I_m_long - m_motor0 * (z_m - cg0) ** 2
        I_r_rot = I_rot0 - m_motor0 * r ** 2 / 2
        rocket["center_of_mass_without_motor"] = cg_r
        rocket["inertia_without_motor_approx"] = [I_r_long, I_r_long, I_r_rot]
        motor = {
            "designation": mm["designation"], "manufacturer": mm["manufacturer"],
            "diameter": mm["diameter"], "length": L,
            "position_nozzle": nozzle,  # use as rocket.add_motor(motor, position=...)
            "total_mass": m_motor0, "dry_mass": m_motor_b,
            "propellant_mass": m_motor0 - m_motor_b,
            "burn_time": t[i_burn],
            "total_impulse": sum((thrust[k + 1][0] - thrust[k][0]) *
                                 (thrust[k + 1][1] + thrust[k][1]) / 2
                                 for k in range(len(thrust) - 1)),
        }

    fd = sim_el.find("flightdata")
    params = {
        "_source": {
            "repo": REPO_URL, "file": ork_path.name, "simulation": name,
            "simulation_status": status,
            "openrocket_commit": git_head(),
            "converted": datetime.now().isoformat(timespec="seconds"),
            "units": "SI (m, kg, kg*m^2, s, N); positions from nose tip, nose_to_tail",
            "notes": [
                "inertia_without_motor_approx removes the motor as a solid cylinder; "
                "rocket inertia/CG without motor are estimates.",
                "Rocket inertia order is [I_11, I_22, I_33] = [pitch, yaw, roll] like RocketPy.",
                "parachute trigger is metres above ground or 'apogee'.",
            ],
        },
        "openrocket_results": {
            "apogee_m": float(fd.get("maxaltitude")),
            "max_velocity_ms": float(fd.get("maxvelocity")),
            "max_mach": float(fd.get("maxmach")),
            "time_to_apogee_s": float(fd.get("timetoapogee")),
        },
        "rocket": rocket,
        "motor": motor,
        "nose": geo.get("nose"),
        "fins": geo["fins"],
        "tail": geo.get("tail"),
        "parachutes": geo["parachutes"],
        "rail_buttons": geo["rail_buttons"],
    }
    return params, power_on, power_off, thrust


def git_head():
    r = subprocess.run(["git", "-C", str(SOURCE_DIR), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows([(f"{a:.4g}", f"{b:.4g}") for a, b in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ork", default=DEFAULT_ORK, help=f"design file (default: {DEFAULT_ORK})")
    ap.add_argument("--sim", help="simulation name inside the design (default: first with data)")
    ap.add_argument("--list", action="store_true", help="list designs and simulations, then stop")
    ap.add_argument("--no-pull", action="store_true", help="skip git, use the local copy")
    args = ap.parse_args()

    if not args.no_pull or not SOURCE_DIR.exists():
        pull_repo()

    if args.list:
        for ork in sorted(SOURCE_DIR.glob("*.ork")):
            print(f"\n{ork.name}")
            for name, branch, sim in list_sims(load_ork(ork)):
                tag = "has data" if branch is not None else "no data (not simulated)"
                print(f"   - {name}  [{sim.get('status')}, {tag}]")
        return

    ork_path = SOURCE_DIR / args.ork
    if not ork_path.exists():
        sys.exit(f"'{args.ork}' not found. Available: {[p.name for p in SOURCE_DIR.glob('*.ork')]}")

    params, power_on, power_off, thrust = convert(ork_path, args.sim)

    OUT_DIR.mkdir(exist_ok=True)
    write_csv(OUT_DIR / "power_on_drag.csv", power_on)
    write_csv(OUT_DIR / "power_off_drag.csv", power_off)
    write_csv(OUT_DIR / "thrust_curve.csv", thrust)
    with open(OUT_DIR / "rocket_params.json", "w") as f:
        json.dump(params, f, indent=2)

    r, m = params["rocket"], params["motor"]
    print(f"[out] {OUT_DIR}")
    print(f"      power_on_drag.csv  {len(power_on)} points, Mach {power_on[0][0]}-{power_on[-1][0]}")
    print(f"      power_off_drag.csv {len(power_off)} points, Mach {power_off[0][0]}-{power_off[-1][0]}")
    print(f"      thrust_curve.csv   {len(thrust)} points")
    print("      rocket_params.json")
    print(f"\n  radius {r['radius']:.4f} m | length {r['length']:.3f} m | "
          f"mass w/o motor {r['mass_without_motor']:.3f} kg")
    if m:
        print(f"  motor {m['manufacturer']} {m['designation']} at {m['position_nozzle']:.3f} m | "
              f"prop {m['propellant_mass']:.3f} kg | dry {m['dry_mass']:.3f} kg")
    print(f"  OpenRocket apogee {params['openrocket_results']['apogee_m']:.0f} m AGL")


if __name__ == "__main__":
    main()
