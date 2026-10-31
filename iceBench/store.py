import os
import re
from datetime import datetime, timezone

import pandas as pd
import yaml

SUBDIRS = ("picks", "events", "assignments") # subdirectories of a run directory, one per kind of output


def make_run_id():
    """
    Timestamp identifying one invocation of the pipeline.

    OUTPUT:
        str, UTC timestamp, e.g. "20260804T141230"
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def create_run_dir(output_root, run_id, config_raw=None):
    """
    Create the directory tree of a run and store the configuration next to the
    results, so the grid_ids stay decodable later.

    INPUT:
        output_root: str, directory the run directory is created in
        run_id: str, from make_run_id
        config_raw: dict or None, the raw YAML content, written as config.yaml

    OUTPUT:
        str, path of the run directory
    """
    run_dir = os.path.join(output_root, run_id)
    for sub in SUBDIRS:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)

    if config_raw is not None:
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(config_raw, f, sort_keys=False)

    return run_dir


def event_key(event):
    """
    Cuts event.resource_id to a file name safe key.
    Kept as close to its resource_id as possible.
    Takes the last path component of the resource_id and strips anything that is
    not safe in a file name.

    EXAMPLE : resource_id="smi:local/20260730111536" -> key="20260730111536" 

    INPUT:
        event: obspy Event

    OUTPUT:
        str, e.g. "20260730111536"
    """
    tail = str(event.resource_id).rstrip("/").split("/")[-1] 
    key = re.sub(r"[^A-Za-z0-9._-]", "_", tail)

    return key or "event"


def event_keys(catalog):
    """
    Keys of all events of a catalog. If there are duplicate events they get a numeric
    suffix so that no event silently overwrites another one.

    INPUT:
        catalog: obspy Catalog

    OUTPUT:
        list of str, same order as the catalog
    """
    keys, seen = [], {}
    for event in catalog:
        key = event_key(event)
        if key in seen:
            seen[key] += 1
            print(f"NOTE: duplicate event key '{key}' "
              f"(same resource_id tail); stored as '{key}_{seen[key]}'. "
              f"Event {event.resource_id} may be a duplicate in the catalog and will be "
              f"processed and counted twice.")
            key = f"{key}_{seen[key]}" #event key gets altered to avoid overwriting another event with the same key
        else:
            seen[key] = 0
        keys.append(key)

    return keys


def segment_path(run_dir, kind, key):
    """
    Path of the output file of one segment.

    INPUT:
        run_dir: str, from create_run_dir
        kind: str, one of "picks", "events", "assignments"
        key: str, event key or continuous chunk key

    OUTPUT:
        str, file path
    """
    if kind not in SUBDIRS:
        raise ValueError(f"kind must be one of {SUBDIRS}, got {kind}.")

    return os.path.join(run_dir, kind, f"{key}.parquet")


def write_segment(df, run_dir, kind, key):
    """
    Write the results of one segment. Parquet files are immutable. One file
    per segment. Contains all picks of all grid points and thresholds.

    INPUT:
        df: pandas DataFrame
        run_dir: str, from create_run_dir
        kind: str, one of "picks", "events", "assignments"
        key: str, event key or chunk key

    OUTPUT:
        str, path of the written file
    """
    path = segment_path(run_dir, kind, key)
    df.to_parquet(path, index=False)

    return path


def read_run(run_dir, kind="picks",filters=None): #XXX: Memory concerns! 
    """
    Read all segments of a run into one DataFrame.

    INPUT:
        run_dir: str, from create_run_dir
        kind: str, one of "picks", "events", "assignments"
        filters: list or None, pyarrow predicate pushdown applied per file,
                 e.g. [("grid_id", "in", gids)] or
                 [("grid_id", "==", gid), ("threshold", "in", ths)];
                 filtering happens during the read, so only matching rows
                 ever occupy memory

    OUTPUT:
        pandas DataFrame, empty when nothing has been written yet
    """
    folder = os.path.join(run_dir, kind)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".parquet"))
    if not files:
        return pd.DataFrame()
    
    return pd.concat([pd.read_parquet(os.path.join(folder, f), filters=filters)
                      for f in files],ignore_index=True)


def read_assignments_events(run_dir:str,kind:str,gid:str):
    return pd.read_parquet(str(run_dir+"/"+kind+"/"+gid+".parquet"))


def write_quakeml_per_config(picks, run_dir, key):
    """
    Write the picks of one segment as QuakeML, one file per (grid_id, threshold)
    so every catalog holds exactly one configuration and downstream tools never
    see the same arrival at several thresholds.

    INPUT:
        picks: pandas DataFrame from predict_segment
        run_dir: str, from create_run_dir
        key: str, segment key

    OUTPUT:
        list of str, paths of the written files
    """
    from obspy import UTCDateTime
    from obspy.core.event import Catalog, Event, Pick, WaveformStreamID, Comment

    paths = []
    for (grid_id, th), group in picks.groupby(["grid_id", "threshold"]):
        event = Event()
        event.comments = [Comment(text=f"grid_id={grid_id} threshold={th}",
                                  force_resource_id=False)]
        for _, r in group.iterrows():
            net, sta = r["station"].split(".")
            p = Pick(time=UTCDateTime(r["peak_time"]),
                     waveform_id=WaveformStreamID(network_code=net,
                                                  station_code=sta),
                     phase_hint=r["phase"], evaluation_mode="automatic")
            # trigger window stored as asymmetric uncertainties around the peak
            p.time_errors.lower_uncertainty = r["peak_time"] - r["start_time"]
            p.time_errors.upper_uncertainty = r["end_time"] - r["peak_time"]
            p.comments = [Comment(text=f"peak_value={r['peak_value']:.4f}",
                                  force_resource_id=False)]
            event.picks.append(p)

        folder = os.path.join(run_dir, "quakeml", grid_id)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{key}_th{th:g}.xml")
        Catalog(events=[event]).write(path, format="QUAKEML")
        paths.append(path)

    return paths
