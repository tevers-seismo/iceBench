import os

from . import store
from .config import load_config
from .evaluate import load_manual_picks
from .models import check_window_length
from .predict import predict_segment
from .sources import CatalogWindowSource, ContinuousSource


def run_source(source, config, run_dir, verbose=True, write_quakeml=False):
    """
    Run the grid over every segment of a source and write one pick file per
    segment. Segments are either catalog windows or just chuncks (continous), depending on the source type.

    INPUT:
        source: CatalogWindowSource or ContinuousSource
        config: RunConfig
        run_dir: str, from store.create_run_dir
        verbose: bool
        write_quakeml: bool, additionally write the picks of each segment as QuakeML

    OUTPUT:
        str, the run directory
    """
    thresholds = {"P": config.p_thresholds, "S": config.s_thresholds}
    n = len(source) # number of segments to process, either catalog events or continuous chunks

    for i, (key, event_id, t_ref, st) in enumerate(source, start=1): # This is where the streams are downloaded and processed one by one for each segment
        if len(st) == 0: # No data available for this segment
            if verbose:
                print(f"[{i}/{n}] {key} no data")
            continue

        rate = min(tr.stats.sampling_rate for tr in st)
        if config.sampling_rate and rate < config.sampling_rate: # Safety Check
            print(f"WARNING {key}: data at {rate} Hz below configured "
                  f"sampling_rate {config.sampling_rate} Hz, windows may be too "
                  f"short for seisbench models. ")

        picks = predict_segment(st, config.grid, thresholds, config.batch_size)
        picks["segment_key"] = key
        picks["event_id"] = event_id
        picks["t_ref"] = t_ref.timestamp if t_ref is not None else float("nan")

        store.write_segment(picks, run_dir, "picks", key)
        if write_quakeml:
            store.write_quakeml_per_config(picks, run_dir, key)
        if verbose:
            print(f"[{i}/{n}] {key} {len(st)} traces, {len(picks)} total picks for current segment")
            # i: current segment number, n: total number of segments, 
            # key: segment key, 
            # len(st): number of traces in this segments stream
            # len(picks): number of total picks produced for this segment across the entire sweep

    return run_dir


def run_benchmark(config_path, client, inv=None, pre_filt=None, water_level=None,
                  associate=False, verbose=True):
    """
    Run the grid over the manually picked catalog.

    INPUT:
        config_path: str, path to the run YAML
        client: obspy client
        inv, pre_filt, water_level: response removal, skipped when inv is None
        associate: bool, run the associator inside the loop, not implemented yet
        verbose: bool

    OUTPUT:
        run_dir: str
        manual: pandas DataFrame of manual picks
    """
    if associate:
        raise NotImplementedError(
            "In-loop association is not implemented. For now only association as postprocessing due do runtime concerns.")

    config = load_config(config_path) #parses the config file
    manual, catalog = load_manual_picks(config.catalog) # loads input catalog e.g. with manual picked events or other reference times marked with a pick
    if manual.empty:
        print("WARNING: No picks in the catalog that are not from reference author; picks will be produced but residual statistics"
              " will be empty. If you want to include all picks, simply set reference_author=None in the config file.")

    run_dir = store.create_run_dir(config.output_root, store.make_run_id(),
                                   config.raw)
    manual.to_parquet(os.path.join(run_dir, "manual_picks.parquet"), index=False)

    source = CatalogWindowSource(catalog, client, config, inv, pre_filt, water_level)

    # Setting a sampling rate in the config file  avoids fetching the first segment twice
    raw_rate = config.sampling_rate or _lowest_rate(source)
    check_window_length(config.grid, config.window_length, raw_rate)

    run_source(source, config, run_dir, verbose)

    return run_dir, manual


def run_continuous(config_path, client, t_start, t_end, chunk_length,
                   grid_ids=None, write_quakeml=False, inv=None, pre_filt=None,
                   water_level=None, verbose=True):
    """
    Run one or more grid points over a continuous time range, chunks do not overlap.

    INPUT:
        config_path: str, path to the run YAML
        client: obspy client
        t_start, t_end: obspy UTCDateTime
        chunk_length: float, seconds
        grid_ids: iterable of str or None, iterable: restrict to these grid points
                  None :runs the whole grid of the config
        write_quakeml: bool, additionally write one QuakeML file per chunk and per (grid_id, threshold) combination
        inv, pre_filt, water_level: response removal, skipped when inv is None
        verbose: bool

    OUTPUT:
        str, the run directory
    """
    config = load_config(config_path)
    if grid_ids is not None:
        wanted = set(grid_ids)
        config.grid = [g for g in config.grid if g.grid_id in wanted]
        missing = wanted - {g.grid_id for g in config.grid}
        if missing:
            raise ValueError(f"grid_ids not contained in the config grid: {sorted(missing)}")

    run_dir = store.create_run_dir(config.output_root, store.make_run_id(),
                                   config.raw)

    source = ContinuousSource(t_start, t_end, chunk_length, client, config, inv,
                              pre_filt, water_level)

    raw_rate = config.sampling_rate or _lowest_rate(source)
    check_window_length(config.grid, chunk_length, raw_rate)

    return run_source(source, config, run_dir, verbose,
                      write_quakeml=write_quakeml)


def _lowest_rate(source):
    """
    Finds minimum sampling rate of the first segment. Only necessary if no sampling_rate is set in the config file, otherwise the config file value is used.

    INPUT:
        source: CatalogWindowSource or ContinuousSource

    OUTPUT:
        float, lowest sampling rate found in the first segment with data
    """
    for _, _, _, st in source:
        if len(st):
            return min(tr.stats.sampling_rate for tr in st)

    raise ValueError("No segment with data, cannot determine the sampling rate.")
