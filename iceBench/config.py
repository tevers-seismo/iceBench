from dataclasses import dataclass, field

import numpy as np
import yaml


@dataclass(frozen=True)
class GridPoint:
    """
    One point of the model grid. Frozen so it can be used as a dict key.

    model: str, seisbench class name, e.g. "PhaseNet" or "EQTransformer"
    weights: str, pretrained weight set, e.g. "original"
    overlap: float, fraction of the window when < 1, sample count when >= 1
    stacking: str, "avg" or "max"
    time_domain: str, "real" or "transformed"
    """
    model: str
    weights: str
    overlap: float
    stacking: str
    time_domain: str

    @property
    def grid_id(self):
        """
        Readable identifier of this grid point, used in file names and as a
        column in every output table.

        Example: phasenet_original_ov090_avg_real

        OUTPUT:
            str, e.g. "phasenet_original_ov090_avg_real"
        """
        if self.overlap < 1:
            ov = f"ov{round(self.overlap * 100):03d}"
        else:
            ov = f"ov{int(self.overlap)}smp"
        domain = "trans" if self.time_domain == "transformed" else "real"

        return f"{self.model.lower()}_{self.weights}_{ov}_{self.stacking}_{domain}" 

    @property
    def annotate_kwargs(self):
        """
        Arguments handed to picker.annotate for this grid point.

        OUTPUT:
            dict
        """
        return {"overlap": self.overlap, "stacking": self.stacking}


@dataclass
class RunConfig:
    """
    Everything a run needs that is not part of the model grid. Preprocessing is
    fixed for all runs, only the response removal can be switched on.

    catalog: str, path to the manual pick event files or a single file
    station_list: list of str, "NET.STA.LOC.CHA"
    output_root: str, directory the run directory is created in
    grid: list of GridPoint
    p_thresholds, s_thresholds: np.ndarray, trigger thresholds per phase
    window_length: float, length of the fetched window in seconds
    window_start_before_eventtime: float, seconds of data before the anchor pick
    reference_author: str or None, author of the anchor pick, None uses the first pick
    batch_size: int, seisbench batch size
    remove_response: bool, response removal, off for the icequake benchmark
    sampling_rate: float or None, LOWEST raw rate among the stations. Only used
                   to validate the window length against the transformed domain,
                   the per station rates are always read from the data itself.
                   None fetches the first segment to determine it
    match_tolerance: float, symmetric matching window of match_picks in seconds
    detection_window: tuple (lower, upper), asymmetric window of detection_stats,
                      lower seconds early to upper seconds late
    """
    catalog: str
    station_list: list
    output_root: str
    grid: list
    p_thresholds: np.ndarray
    s_thresholds: np.ndarray
    window_length: float = 180.0
    window_start_before_eventtime: float = 60.0
    reference_author: str = "nrichels"
    batch_size: int = 256
    remove_response: bool = False
    sampling_rate: float = None
    match_tolerance: float = 0.5
    detection_window: tuple = (0.5, 0.2)
    raw: dict = field(default_factory=dict)


def build_thresholds(spec):
    """
    Build a threshold list from a short specification.

    INPUT:
        spec: dict with keys min, max, n and optionally spacing ("log" or "linear"),
              or a plain list of values

    OUTPUT:
        np.ndarray of thresholds, ascending
    """
    if isinstance(spec, (list, tuple)):
        return np.sort(np.asarray(spec, dtype=float))

    spacing = spec.get("spacing", "log")
    if spacing == "log":
        return np.logspace(np.log10(spec["min"]), np.log10(spec["max"]), spec["n"])
    if spacing == "linear":
        return np.linspace(spec["min"], spec["max"], spec["n"])

    raise ValueError(f"spacing must be 'log' or 'linear', got {spacing}.")


def expand_grid(spec):
    """
    Expand the grid specification into the individual grid points.
    Stacking is collapsed to a single value when overlap is 0, because with
    non overlapping windows avg and max give identical curves.

    INPUT:
        spec: dict with keys models (mapping model name to a list of weight sets),
              overlap, stacking, time_domain

    OUTPUT:
        list of GridPoint, in a deterministic order
    """
    grid = []
    for model, weight_sets in spec["models"].items():
        for weights in weight_sets:
            for overlap in spec["overlap"]:
                stackings = spec["stacking"] if overlap else spec["stacking"][:1]
                for stacking in stackings:
                    for domain in spec["time_domain"]:
                        grid.append(GridPoint(model, weights, float(overlap),
                                              stacking, domain))

    ids = [g.grid_id for g in grid]
    if len(set(ids)) != len(ids):
        raise ValueError("Grid contains duplicate grid_ids, check the specification.")

    return grid


def load_config(path):
    """
    Read a run configuration from YAML.

    INPUT:
        path: str, path to the YAML file

    OUTPUT:
        RunConfig
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    run = raw["run"]
    th = raw["thresholds"]

    return RunConfig(
        catalog=run["catalog"],
        station_list=run["station_list"],
        output_root=run["output_root"],
        grid=expand_grid(raw["grid"]),
        p_thresholds=build_thresholds(th["P"]),
        s_thresholds=build_thresholds(th["S"]),
        window_length=run.get("window_length", 180.0),
        window_start_before_eventtime=run.get("window_start_before_eventtime", 60.0),
        reference_author=run.get("reference_author", "nrichels"),
        batch_size=run.get("batch_size", 256),
        remove_response=run.get("remove_response", False),
        sampling_rate=run.get("sampling_rate"),
        match_tolerance=raw.get("evaluation", {}).get("match_tolerance", 0.5),
        detection_window=tuple(raw.get("evaluation", {})
                               .get("detection_window", (0.5, 0.2))),
        raw=raw,
    )
