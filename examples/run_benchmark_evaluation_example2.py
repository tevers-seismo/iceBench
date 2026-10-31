"""
iceBench — evaluation example 2: pick residuals against manual analyst picks
=============================================================================
In iceBench, creating picks, associating them and evaluating them are separate
steps:
  run_benchmark() / run_continuous()  - create picks (see run_benchmark_example.py).
  association (associate example 2)   - group picks into events (writes assignments).
  evaluation (this script)            - score those picks.

This is evaluation example 2. It compares the ASSOCIATED auto picks against the
manual analyst picks: each manual pick is matched to the closest auto pick on the
same station within a tolerance, and pick residuals (t_manual - t_auto) are
summarised per grid point, threshold and phase.

RUN ORDER — association must be run first:
  run_benchmark_example.py   -> creates the picks
  associate_example2.py      -> creates the assignments (associated auto picks)
  THIS script                -> reads kind="assignments" and scores residuals

Evaluation example 1 (separate file) instead compares auto picks against the
STA/LTA reference events and reports detection/recall only;

Input: a run directory produced by run_benchmark() with association applied
       (associate example 2), plus the manual analyst picks.
"""

import os
import numpy as np
import pandas as pd

import iceBench


print("Start matching...")

# 1. Step: Load the manual picks
df_manual, _ = iceBench.load_manual_picks("path/to/manual_picks_dir") # reads all *.xml files in the specified directory together

# 2. Step: Load the automated picks from the residuals run
run_dir = "/path/to/run_dir"     # Example : .../iceBench_runs/20260821T095633
picks_assigned = iceBench.store.read_run(run_dir=run_dir, kind ="assignments")

# 3. Step: Clean station code to enable correct matching
picks_assigned["station"] = picks_assigned["station"].str.split(".").str[:2].str.join(".")

# 4. Step: Match the manual picks to the automated picks; closest will be picked
match_tolerance = 0.5 # specify match tolerance [s] --> [-0.5, +0.5] around the manual pick time
matched = iceBench.match_picks(df_manual, picks_assigned, tolerance=match_tolerance) # For each grid_id & threshold, find the closest auto pick to each manual pick within the tolerance (in seconds). Returns a dataframe of matched picks.

# 5. Step: Write out the match results for further analysis
matched.to_parquet(
    os.path.join(run_dir, f"matched_tol{match_tolerance:g}_POST_association.parquet"),
    index=False)

# 6. Step: Calculate residual statistics for the matched picks
df_stats = iceBench.residual_stats(matched = matched,by=("grid_id", "threshold", "phase"))
# XXX: Do what you wish with the df_stats, e.g. save to parquet or csv,...

print("Done.")
