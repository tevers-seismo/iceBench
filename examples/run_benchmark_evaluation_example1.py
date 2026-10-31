"""
iceBench — evaluation example 1: detection against STA/LTA reference picks
============================================================================
In iceBench, creating picks and evaluating them are separate steps:
  run_benchmark() / run_continuous()  - create picks (see run_benchmark_example.py).
  evaluation (this script)            - score those picks.

This is evaluation example 1. It compares the auto picks against the STA/LTA
reference events (the reference_author picks that defined the benchmark windows)
and counts, per grid point and threshold, how many reference events were matched
-> matches and recall per phase ("(P)", "(S)").
-> Detection only; no association and no pick-residual accuracy.

Evaluation example 2 (separate file) instead compares ASSOCIATED auto picks
against manual analyst picks and reports pick residuals; it requires
association to be run first.

Input: the run directory produced by run_benchmark().
"""

import iceBench

# Step 1: Get all picks of the run
run_dir = "/path/to/run_dir"  # Specify the path to the run directory that that was created by run_benchmark() in run_benchmark_example.py

# Step 2: Get the reference catalog
cpath = "path/to/run_config_file.yaml"
reference_cat = iceBench.load_config(cpath).catalog

# Step 3: Read the automatic picks from the run directory
print("Reading picks from run directory...")
picks = iceBench.store.read_run(run_dir=run_dir, kind="picks")

# Step 4: Load the Reference picks from reference_author (e.g. STA/LTA picks)
print("Read reference picks...")
reference_picks, catalog = iceBench.load_manual_picks(path=reference_cat, skip_author=None)

# Step 5: Evaluate the picks against the reference picks by comparing peak_time vs t_ref grouped by grid_id and threshold, output: pandas DataFrame with n_events, n_matched/recall/n_missed per phase ("(P)", "(S)")
print("Calculate detections...")
detections = iceBench.detection_stats(
    reference_picks,
    picks,
    p_window=(0.3, 0.1),
    s_window=(0.1, 0.5),
    by=("grid_id", "threshold"),
    out_dir=run_dir,
)

print(f"Detections saved to parquet in run directory: {run_dir}/detection_stats_*.parquet")
