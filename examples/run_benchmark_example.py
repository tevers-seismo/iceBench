"""=============================================================================
iceBench run configuration

Two run modes read this file:
  run_benchmark()   - cuts a window around each event in a catalog and runs the
                      pickers on it. Requires a catalog of reference picks (used to
                      define the windows). Scoring is evaluated in a second step.
  run_continuous()  - runs the pickers over a continuous time range in chunks and
                      writes the picks; no reference picks, no scoring.

This script demonstrates run_benchmark(). The run is driven by a YAML config
(see run_config_example.yaml); keys marked "run_benchmark() only" there apply here.

"""

from obspy.clients.fdsn import Client
import iceBench
from obspy import UTCDateTime

# Specify your Data Client
client = Client("...")

# config file that defines network, grid, thresholds and reference catalog settings
cpath = "path/to/run_config_file.yaml"

t_start = UTCDateTime()  # wall-clock start, for timing only

run_dir, manual = iceBench.run_benchmark(
    config_path=cpath,
    client=client,
    inv=None,
    pre_filt=None,
    water_level=None,
    associate=False,
    verbose=True,
)

t_end = UTCDateTime()  # wall-clock end, for timing only


print(f"Run finished. Run directory: {run_dir}")
print(f"Run Start: {t_start} | End: {t_end} | Duration: {t_end - t_start:.1f} s")
