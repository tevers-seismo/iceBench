"""
iceBench — associate a continuous run (chunk-by-chunk)
=======================================================
Associates the picks from a run_continuous() run into events with PyOcto, using
the Associator settings from associator_config

This reads the run one chunk-parquet at a time and associates each chunk separately,
to keep memory bounded on long continuous runs.
Events/assignments are written per chunk, and optionally exported to
NonLinLoc .obs files.

Input:  a run directory from run_continuous(), plus a station inventory.
Output: per-chunk events + assignments in the run directory, and NonLinLoc .obs files.
"""

import os, sys
import pandas as pd
import iceBench
from obspy import read_inventory, UTCDateTime
from associator_config import associator


print(f"Association started.")

# 1. Specify the inventory
inv = read_inventory("/path/to/inventory.xml")

# 2. Convert Stations into correct PyOcto Format
stations = iceBench.associate.stations_df(associator,inv) # do coordinate transformation


# 3. Specify the run directory that contains the picks to be associated
run_dir = "/path/to/run_dir"

# 4. Specify the directory where the NonLinLoc .obs files will be written
obs_nlloc_dir = "/path/to/NLLoc/obs/"


# 5. Specify grid id and threshold : only picks of this picking setup will be associated)
gid = "phasenet_volpick_ov090_max_trans"
P_th = 0.198
S_th = 0.043

t0_wall = UTCDateTime() # wall clock, timing only

# 6. Run the association in chunks 
# XXX This would read all picks in with full information, Not recommended for large continuous runs -> Memory concerns!!!
# picks = iceBench.store.read_run(run_dir, kind="picks")
# Alternative implemented below: read in chunk for chunk and associate each chunk separately

folder = os.path.join(run_dir, "picks")
files = sorted(f for f in os.listdir(folder) if f.endswith(".parquet"))
if not files:
    sys.exit("No files in run_dir/picks!")

for i, f in enumerate(files):
    key = os.path.splitext(f)[0]                # chunk timestamp, reused as the output key
    print("Associating: ", key)

    picks = pd.read_parquet(os.path.join(folder, f))
    picks = picks.sort_values("peak_time").reset_index(drop=True)
    if picks.empty:
        continue

    events, assignments = iceBench.associate.associate_picks(picks, associator, stations)

    if len(events):
        associator.transform_events(events)
    for df in (events, assignments):
        df["grid_id"], df["P_th"], df["S_th"], df["chunk"] = gid, P_th, S_th, key

    # 7. Write the results to the run directory
    iceBench.store.write_segment(assignments, run_dir, kind="assignments", key=key)
    iceBench.store.write_segment(events, run_dir, kind="events", key=key)   # events last -> resume-safe

    if assignments.empty:
        continue

    # 8. Save to NonLinLoc .obs file
    assignments["station"] = assignments["station"].str.split(".").str[1]
    obs_path = os.path.join(obs_nlloc_dir, f"{key}.obs")
    associator.to_nonlinloc(assignments, obs_path, pick_std=0.05)

    print(f"[{i+1}/{len(files)}] {key}: {len(picks)} picks -> {len(events)} events")

    # 6. Summary
    print(f"{len(events)} events, {len(assignments)}/{len(picks)} picks assigned")


print(f"Done in {UTCDateTime() - t0_wall:.0f} s. Run dir: {run_dir}")
