"""
iceBench — run_continuous() example
====================================
iceBench has two run modes that produce picks:
  run_benchmark()   - cuts a window around each event in a catalog and runs the
                      pickers on it (see run_benchmark_example.py). Needs a catalog.
  run_continuous()  - runs the pickers over a continuous time range in chunks.
                      No catalog or reference picks; produces picks only.

This script demonstrates run_continuous(). It processes [t_start, t_end] in
non-overlapping chunks of chunk_length seconds (e.g. 12h chunks), running the selected grid points
over every chunk and writing the picks to the run directory. 
Association is separate steps (see assoicate_autoPick_continous)

Config: driven by the same YAML as run_benchmark() (run_config_example.yaml), but
the "run_benchmark() only" keys there are ignored here. 

"""
import iceBench
from obspy.clients.fdsn import Client
from obspy import UTCDateTime


# 1. Specify your Data Client
client = Client("...")

# 2. Specify config file that defines network, grid, thresholds 
cpath = "path/to/run_config_file.yaml"
print(f"Config: {cpath}")

# 4. Specify the time frame and chunk length
start = UTCDateTime("2025-03-20T00:00:00") 
end = UTCDateTime("2025-04-02T00:00:00")  # time frame ends on 2025-04-02T00:00:00
chunk_len = 12 * 3600 # chunk length [s]

gids = None  # None takes all grid_ids from the config file, otherwise select here

t0_wall = UTCDateTime() # wall clock, timing only
print("Continous Run started.")

run_dir = iceBench.run_continuous(
    config_path=cpath,
    client=client,
    t_start=start,
    t_end=end,
    chunk_length=chunk_len,
    grid_ids=gids,
    write_quakeml=False,
    inv=None,
    pre_filt=None,
    water_level=None,
    verbose=True,
)


t1_wall = UTCDateTime() # wall clock, timing only


print(f"Continous Run finished. Run directory: {run_dir}")
print(f"Run Start: {t0_wall} | End: {t1_wall} | Duration: {t1_wall - t0_wall:.1f} s")