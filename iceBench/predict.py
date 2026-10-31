import pandas as pd

from .models import get_picker
from .transform import delate_stream, backtransform_picks

PHASES = ("P", "S")


def annotate_stream(st, grid_point, batch_size=256):
    """
    Probability curves of one stream for one grid point. In the transformed
    domain the stream is relabelled to the model rate first, so seisbench does
    not resample.

    INPUT:
        st: obspy Stream in real time
        grid_point: GridPoint
        batch_size: int

    OUTPUT:
        preds: obspy Stream of probability traces, in the transformed frame when
               grid_point.time_domain is "transformed"
        picker: seisbench WaveformModel used
    """
    picker = get_picker(grid_point.model, grid_point.weights)
    st_in = st if grid_point.time_domain == "real" else \
        delate_stream(st, picker.sampling_rate) # This is where the timetransformation happens

    preds = picker.annotate(st_in, batch_size=batch_size, # This is where the seisbench model is applied to the stream: see seisbench.models.base.annotate() for details
                            **grid_point.annotate_kwargs) #annotate_kwarg: overlap, stacking

    return preds, picker


def picks_over_thresholds(preds, picker, st, grid_point, thresholds):
    """
    Turn one set of probability curves into picks for every stated threshold. The phase
    channel is selected once, then only the trigger runs per threshold, and the
    two phases are swept independently.

    INPUT:
        preds: obspy Stream from annotate_stream
        picker: seisbench WaveformModel that produced preds
        st: obspy Stream in real time, needed to back transform
        grid_point: GridPoint
        thresholds: dict {"P": array, "S": array}

    OUTPUT:
        list of dict, one per pick
    """
    rows = []
    for phase in PHASES:
        curve = preds.select(channel=f"*_{phase}")  # preds is obspy stream of annotations
        if len(curve) == 0:
            continue

        for th in thresholds[phase]:
            picks = picker.picks_from_annotations(curve, float(th), phase) # see seisbench.models.base.picks_from_annotations() for details
            if grid_point.time_domain == "transformed":
                picks = backtransform_picks(st, picks, picker.sampling_rate) # st is original stream in real time, picks are in transformed time

            for p in picks:
                rows.append({
                    "grid_id": grid_point.grid_id,
                    "threshold": float(th),
                    "phase": phase,
                    "station": ".".join(str(p.trace_id).split(".")[:2]),
                    "trace_id": str(p.trace_id),
                    "peak_time": p.peak_time.timestamp,
                    "peak_value": float(p.peak_value),
                    "start_time": p.start_time.timestamp,
                    "end_time": p.end_time.timestamp,
                })

    return rows


def predict_segment(st, grid, thresholds, batch_size=256):
    """
    Run the whole grid on one waveform window.

    INPUT:
        st: obspy Stream in real time, already preprocessed
        grid: list of GridPoint
        thresholds: dict {"P": array, "S": array}
        batch_size: int

    OUTPUT:
        pandas DataFrame of picks, times as unix timestamps
    """
    rows = [] #XXX :  Memory concerns? Carful this can get very large if the grid is large and the thresholds are low, 
    for grid_point in grid:
        preds, picker = annotate_stream(st, grid_point, batch_size)
        rows += picks_over_thresholds(preds, picker, st, grid_point, thresholds)

    df = pd.DataFrame(rows)
    if df.empty:  # keep the schema stable for empty segments
        df = pd.DataFrame(columns=["grid_id", "threshold", "phase", "station",
                                   "trace_id", "peak_time", "peak_value",
                                   "start_time", "end_time"])

    return df
