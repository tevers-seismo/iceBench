"""
iceBench — association example 2: associate the automatic picks
================================================================
Association groups individual phase picks into events. This script applies the
PyOcto settings tuned in association example 1 (imported from associator_config)
to the AUTOMATIC picks produced by run_benchmark().

It associates one picking setup at a time: picks are sliced to a single grid_id
and its chosen P/S thresholds, then PyOcto builds events -> events, assignments
which are written back into the run directory (kind="events" and
kind="assignments").

RUN ORDER — this must be run BEFORE evaluation example 2:
  run_benchmark_example.py   -> creates the picks
  THIS script                -> creates the assignments (associated auto picks)
  run_benchmark_eval2.py     -> reads kind="assignments" and scores residuals
Evaluation example 2 reads the assignments this script writes, so it will find
nothing (or stale results) if association has not been run for that grid_id.

Prerequisite: PyOcto parameters fixed via association example 1.
Input:  a run directory from run_benchmark(), plus a station inventory.
Output: events and assignments parquet files in the run directory, per grid_id.
"""

import iceBench
from obspy import read_inventory, UTCDateTime
from associator_config import associator, params, velocity

t_start = UTCDateTime()  # wall clock, timing only

# 1. Specify the inventory
inv = read_inventory("/path/to/inventory.xml")


# 2. Convert Stations into correct PyOcto Format
stations = iceBench.associate.stations_df(associator,inv) # do coordinate transformation
# XXX: Careful: the station ids in the inventory must match the station names in the analyst catalog.
# -> Strip the location code if necessary !
# stations["id"] = stations["id"].str.rstrip(".SW")   # remove location code to enable matching


# 3. Step: Load the automated picks from the Run
print("Reading picks from run directory...")
run_dir = "/path/to/run_dir"  # Specify the path to the run directory that that was created by run_benchmark() in run_benchmark_example.py
picks_auto = iceBench.store.read_run(run_dir=run_dir, kind="picks")


# 4. Specify grid id and threshold : only picks of this picking setup will be associated)
gid = "phasenet_volpick_ov090_max_trans"
P_th = 0.198
S_th = 0.043


# 5. Filter the picks to only those of the specified grid_id and thresholds
# pick-level slice: P picks at P_th, S picks at S_th, one config
picks_cut = picks_auto[(picks_auto.grid_id == gid) &
               (((picks_auto.phase == "P") & (picks_auto.threshold == P_th)) |
                ((picks_auto.phase == "S") & (picks_auto.threshold == S_th)))]
picks_cut = picks_cut.reset_index(drop=True)                     
print(f"slice: {len(picks_cut)} picks "
      f"({(picks_cut.phase=='P').sum()} P @ {P_th:.3g}, {(picks_cut.phase=='S').sum()} S @ {S_th:.3g})")


# 6. : Associate the picks
print("Associating picks...")
events, assignments = iceBench.associate.associate_picks(picks_cut, associator, stations)
associator.transform_events(events)


# 7. : Write the results to the run directory
print("Writing results to run directory...")
for df in (events, assignments):  # add gid and thresholds to df
    df["grid_id"], df["P_th"], df["S_th"] = gid, P_th, S_th
iceBench.store.write_segment(events, run_dir, kind="events", key=gid)
iceBench.store.write_segment(assignments, run_dir, kind="assignments", key=gid)

print(f"assoc events: {len(events)}  (ceiling: INSERT NUMBER OF ANALYST EVENTS HERE)")
print(f"picks assigned: {len(assignments)} / {len(picks_cut)}")
print(events[["idx", "time", "picks", "x", "y", "z"]].to_string(index=False))

t_end = UTCDateTime()

print(f"Association finished. Saved in run directory: {run_dir}")
print(f"Association Start: {t_start} | End: {t_end} | Duration: {t_end - t_start:.1f} s")

# 8 . Example for how to read in the results
# events = iceBench.read_association_results(run_dir, kind="events", gid=gid)
# assignments = iceBench.read_association_results(run_dir, kind="assignments", gid=gid)

# 9. Save to NonLinLoc .obs file 
obs_path = f"/path/to/NLLoc/obs/{gid}.obs"
assignments["station"] = assignments["station"].str.split(".").str[1]
#XXX: Specify pick_std (pick uncertainty) in seconds, e.g. 0.05s
associator.to_nonlinloc(assignments, obs_path, pick_std=0.05) 