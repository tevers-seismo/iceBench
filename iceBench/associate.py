import pandas as pd
import pyocto

_STATIONS = {}


def stations_df(associator, inventory):
    """
    Converts obspy inventory into Station table in PyOcto format, cached because the projection is the same
    for every segment of a run.

    INPUT:
        associator: pyocto OctoAssociator
        inventory: obspy Inventory

    OUTPUT:
        pandas DataFrame
    """
    key = id(inventory)
    if key not in _STATIONS:                                    # string identifier of station, needs to match identifier in pick_df
        _STATIONS[key] = associator.inventory_to_df(inventory)  # x, y, z coordinates of stations in km in local coordinate projection,
                                                                # z-axis is oriented downwards so stations above ground have negative sign
                                                                # optional: station delays like in NonLinLoc
    return _STATIONS[key]   # stations_df in PyOcto format


def associate_picks(picks, associator, stations):
    """
    Associate a single pick set. The pick table already holds everything PyOcto needs,
    so no seisbench objects are rebuilt.

    INPUT:
        picks: pandas DataFrame with trace_id, peak_time, peak_value, phase
        associator: pyocto OctoAssociator
        stations: pandas DataFrame from stations_df

    OUTPUT:
        events, assignments: pandas DataFrames as returned by pyocto
    """
    pick_df = picks.reset_index(drop=True).assign(      # pick_idx = row position
        station=picks["trace_id"].values,               # must match id in stations_df
        time=picks["peak_time"].values,                 # unix timestamp
        probability=picks["peak_value"].values,         # PyOcto's optional confidence
    )
    # all other pick columns (grid_id, threshold, start/end_time, t_ref, ...)
    # pass through pyocto into the assignments unchanged

    return associator.associate(pick_df, stations)


def associate_segment(picks, associator, stations, thresholds=None):
    """
    Associate every (grid_id, threshold) combination of one segment.

    INPUT:
        picks: pandas DataFrame from predict_segment
        associator: pyocto OctoAssociator
        stations: pandas DataFrame from stations_df
        thresholds: iterable or None, restrict to this subset of thresholds

    OUTPUT:
        events, assignments: pandas DataFrames with grid_id and threshold columns.
        assignments keeps pick_idx, which indexes into the picks of that group only
    """
    ev_all, as_all = [], []
    for (grid_id, th), group in picks.groupby(["grid_id", "threshold"]):
        if thresholds is not None and th not in thresholds:
            continue

        group = group.reset_index(drop=True)
        events, assignments = associate_picks(group, associator, stations)
        # associator.transform_events(events)
        for df in (events, assignments):
            df["grid_id"], df["threshold"] = grid_id, th

        ev_all.append(events)
        as_all.append(assignments)

    events = pd.concat(ev_all, ignore_index=True) if ev_all else pd.DataFrame()
    assignments = pd.concat(as_all, ignore_index=True) if as_all else pd.DataFrame()

    return events, assignments


def build_associator(params, velocity):
    """
    Build a PyOcto associator from plain dicts, so the parameters can live in
    the run configuration. We use a homogenous Velocity Model, as PyOcto states
    that is is sufficient for regions with shallow seismicity and of limited size.

    INPUT:
        params: dict, keyword arguments of OctoAssociator.from_area (e.g. association_cutoff_distance)
        velocity: dict, keyword arguments of VelocityModel0D

    OUTPUT:
        pyocto OctoAssociator
    """
    return pyocto.OctoAssociator.from_area(
        velocity_model=pyocto.VelocityModel0D(**velocity), **params
    )

def read_association_results(run_dir: str, kind: str, gid: str):
    """Read one grid point's parquet table from a run directory.

    INPUT:
        run_dir: str, path to the run directory created by run_benchmark()
        kind:    str, subfolder / table type, e.g. "picks" or "assignments"
        gid:     str, grid_id whose file to read (filename without .parquet),
                 e.g. "phasenet_original_ov075_avg_trans"
    OUTPUT:
        pd.DataFrame, the contents of <run_dir>/<kind>/<gid>.parquet
    """
    return pd.read_parquet(str(run_dir + "/" + kind + "/" + gid + ".parquet"))  # <run_dir>/<kind>/<gid>.parquet