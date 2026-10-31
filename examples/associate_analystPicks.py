"""
iceBench — association example 1: tune PyOcto on analyst picks
===============================================================
Association groups individual phase picks into events. PyOcto's association
parameters must be chosen before associating the automatic picks.

This is association example 1. It runs PyOcto on the MANUAL analyst picks, whose
event assignement is already known, so you can check that PyOcto recovers the
expected events and iterate on its settings. Its only purpose is to FIND the
correct PyOcto parameters.

Association example 2 (separate file) then applies those tuned settings to the
automatic picks produced by run_benchmark().

Input: a directory of manual analyst picks (read with load_manual_picks).
Output: PyOcto associations for the analyst picks, for inspecting parameter choices.
"""

import yaml
import pandas as pd
import iceBench
from obspy import read_inventory
from associator_config import associator, params, velocity

# 0. Specify the inventory
inv = read_inventory("/path/to/inventory.xml")


# 1. Get the reference catalog
analyst_cat = "/path/to/analyst_catalog.xml"


# 2. Convert Stations into correct PyOcto Format
stations = iceBench.stations_df(associator,inv) # do coordinate transformation
# XXX: Careful: the station ids in the inventory must match the station names in the analyst catalog.
# -> Strip the location code if necessary !
# stations["id"] = stations["id"].str.rstrip(".SW")   # remove location code to enable matching


# 3. Get Manual picks
manualPicks_df,man_cat = iceBench.load_manual_picks(analyst_cat)
print(man_cat)


# 4. Rename columns in the analyst picks table to match the output schema of automatic picks, and input schema of PyOcto
picks = manualPicks_df.rename(columns={"station": "trace_id", "t_manual": "peak_time"})
picks["peak_value"] = 1.0  # analyst pick = full confidence


# 5. Do Association
events, assignments = iceBench.associate.associate_picks(picks,associator,stations)


# 6. Retransform events into correct coordinate system
associator.transform_events(events)


# 7. Print out Results
n_manual = manualPicks_df["event_id"].nunique()
print(f"MP events: {len(events)}  (expect {n_manual})")
print(f"Manual picks assigned: {len(assignments)} / {len(picks)}")
print(events[["idx", "time", "picks", "x", "y", "z"]].to_string(index=False))

print("\nAssignments:")
print(assignments[["event_idx", "pick_idx", "residual"]].to_string(index=False))


# 8. Save the association parameters and sumamrised results to a yaml file
stamp = pd.Timestamp.now()
with open(f"association_params_{stamp}.yaml", "w") as f:
    yaml.safe_dump(
        {
            "params": params,
            "velocity": velocity,
            "tuned_on": str(analyst_cat),
            "result": {
                "n_events": int(len(events)),
                "n_manual_events": int(n_manual),
                "picks_assigned": int(len(assignments)),
                "picks_total": int(len(picks)),
            },
        },
        f,
        sort_keys=False,
    )

# 9. Save to NonLinLoc .obs file 
obs_path = f"/path/to/NLLoc/obs/Analyst.obs"
assignments["station"] = assignments["station"].str.split(".").str[1]
#XXX: Specify pick_std (pick uncertainty) in seconds, e.g. 0.05s
associator.to_nonlinloc(assignments, obs_path, pick_std=0.05) 