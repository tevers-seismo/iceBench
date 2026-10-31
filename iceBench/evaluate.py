import os
import glob

import numpy as np
import pandas as pd
from obspy import read_events, Catalog

PHASES = ("P", "S")


def load_manual_picks(path, pattern="*.xml", skip_author="nrichels"):
    """
    Read the manual picks. The files are expected to have been through
    set_pick_author_and_mode, so low quality picks carry onset "questionable" and author.

    INPUT:
        path: str, event file, directory of event files, or glob pattern
        pattern: str, file pattern used when path is a directory
        skip_author: str or None, picks by this author are reference
                     picks (e.g. STA/LTA) and are kept in the catalog but not in the table

    OUTPUT:
        df: pandas DataFrame, one row per manual pick, times as unix timestamps
        catalog: obspy Catalog with all events unchanged
    """
    files = sorted(glob.glob(os.path.join(path, pattern))) if os.path.isdir(path) \
        else sorted(glob.glob(path))
    if not files:
        raise ValueError(f"No event files found for {path}")

    rows, catalog = [], Catalog()
    for f in files:
        for event in read_events(f):
            catalog.append(event)
            for pick in event.picks:
                ci = pick.creation_info
                author = ci.author if ci else None
                if pick.phase_hint is None or author == skip_author: # We don't want the picks from skip_authoer as manual picks
                    continue

                # "impulsive" means the rewrite from "impulsive" to "questionable" with set_pick_author_and_mode() was not applied to this file and
                # low quality picks would silently be counted as normal ones
                if pick.onset == "impulsive":
                    raise ValueError(f"{os.path.basename(f)} still has onset "
                                     f"'impulsive', run set_pick_author_and_mode() first"
                                      " if you meant 'impulsive' just as obspyck-set flag for 'questionable'."
                                      " If ,in your catalog, 'impulsive' means really 'impulsive phase onset' ignore this error.")

                wid = pick.waveform_id
                rows.append({
                    "event_id": str(event.resource_id),
                    "file": os.path.basename(f),
                    "station": f"{wid.network_code}.{wid.station_code}",
                    "channel": wid.channel_code,
                    "phase": str(pick.phase_hint).upper(),
                    "quality": "questionable" if pick.onset == "questionable"
                               else "normal",
                    "onset": pick.onset,
                    "evaluation_mode": pick.evaluation_mode,
                    "polarity": pick.polarity,
                    "t_manual": pick.time.timestamp,
                    "unc_lower": pick.time_errors.lower_uncertainty,
                    "unc_upper": pick.time_errors.upper_uncertainty,
                    "author": author,
                })

    return pd.DataFrame(rows), catalog


def match_picks(manual, picks, tolerance=0.5):
    """
    Match every manual pick to the nearest automatic pick of the same station
    and phase, separately for each grid point and threshold. Matching runs per
    phase, so P and S may use different threshold lists without producing
    spurious missed rows for the other phase.
    Residual convention: t_manual - t_auto, positive means the automatic pick
    is early. Convention copied from Shi P.(2024): From labquakes to megathrusts:.., Figure 4

    INPUT:
        manual: DataFrame containing manual picks from load_manual_picks; left dataframe of the merge_asof
        picks: DataFrame from predict_segment or store.read_run; right dataframe of the merge_asof
        tolerance: float, largest accepted |residual| in seconds

    OUTPUT:
        pandas DataFrame, one row per manual pick per (grid_id, threshold) of
        its own phase, residual is NaN when nothing was found inside tolerance
    """
    out = []
    for phase in manual["phase"].unique():
        man_ph = manual[manual["phase"] == phase].sort_values("t_manual")   # Filter out other phase
        pick_ph = picks[picks["phase"] == phase]                            # Filter out other phase

        for (grid_id, th), group in pick_ph.groupby(["grid_id", "threshold"],
                                                    sort=False):
            matched = pd.merge_asof(
                man_ph,
                group.sort_values("peak_time")[["station", "peak_time",
                                                "peak_value", "start_time",
                                                "end_time"]],
                left_on="t_manual", right_on="peak_time",
                by="station", direction="nearest", tolerance=tolerance,
            )
            matched["grid_id"], matched["threshold"] = grid_id, th
            matched["residual"] = matched["t_manual"] - matched["peak_time"] # positive: automatic pick early, negative: automatic pick late
            out.append(matched)

    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def residual_stats(matched, by=("grid_id", "threshold", "phase")):
    """
    Mean, standard deviation, mean absolute error and counts of the residuals.

    INPUT:
        matched: DataFrame from match_picks
        by: tuple of column names to group by, add "quality" to split 'normal' and 'questionable' picks

    OUTPUT:
        pandas DataFrame with columns n, mean, std, mae, n_manual, n_missed
    """
    def agg(g):
        """
        Summary statistics of the residuals of one group.

        INPUT:
            g: pandas DataFrame, rows of one (grid_id, threshold, phase, quality, station, event_id / file, author, polarity, ...) group
               from matched; residual is NaN for unmatched manual picks

        OUTPUT:
            pandas Series with
                by (grid_id, threshold, phase, ...) : group keys as columns
                n        : number of matched manual picks, CAREFUL : one automatic pick may serve several
                mean     : mean residual [s], bias, positive = automatic pick early, negative = automatic pick late
                std      : sample std of the residuals [s], NaN if fewer than 2 matched
                mae      : mean absolute residual [s]
                n_manual : number of total manual picks in the group, matched + missed
                n_missed : number of manual picks without a match inside the tolerance = missed picks of this group
        """
        r = g["residual"].dropna().to_numpy(dtype=float) # numpy array of residuals, NaN means no match within tolerance, so drop them for the statistics
        return pd.Series({
            "n": r.size,
            "mean": np.mean(r) if r.size else np.nan,
            "std": np.std(r, ddof=1) if r.size > 1 else np.nan,     # ddof=1 -> sample std (N-1 divisor) and not population std (N divisor): consistent with plot_residual_hist
            "mae": np.mean(np.abs(r)) if r.size else np.nan,
            "n_manual": len(g),
            "n_missed": len(g) - r.size,
        })

    return matched.groupby(list(by)).apply(agg, include_groups=False).reset_index() # include_groups=False : agg doesn't need the group keys; # reset_index() : group keys become columns


def noise_stats(noise_manual, picks, window=(0.5, 0.5), by=("grid_id", "threshold"),
                out_dir=None):
    """
    False detections on the noise catalog. Counts automatic P and S picks inside
    [t_manual - window[0], t_manual + window[1]] around each noise reference pick.
    A window with at least one pick of a phase is a false detection of that phase.
    CAREFUL: picks must already be filtered to the station of the noise picks.

    INPUT:
        noise_manual: DataFrame of noise reference picks, one per noise event
        picks: DataFrame from store.read_run, filtered to the reference station
        window: tuple (lower, upper), seconds before and after the noise pick
        by: tuple of column names to group by
        out_dir: str or None, write the result as
                 noise_stats_w<lower>_<upper>.parquet into this directory

    OUTPUT:
        pandas DataFrame, one row per group, columns
        n_windows, n_false/false_rate/n_picks per phase ("(P)", "(S)"),
        window_lower, window_upper
    """
    lower, upper = window
    by = list(by)

    m_ref = np.sort(noise_manual["t_manual"].to_numpy())                    # automatic pick times as array, sorted
    n_windows = len(m_ref)
    pk = picks[["phase", "peak_time", *by]].sort_values("peak_time")        # drop unnecessary columns; sorted; * unpacks tuple

    rows = []
    for keys, group in pk.groupby(by, sort=False):                           # splits the picks into groups of by
        keys = keys if isinstance(keys, tuple) else (keys,)                  # isintance is needed because group_key is a tuple when len(by) > 1, otherwise it is a single value
        row = dict(zip(by, keys), n_windows=n_windows)

        for phase in ("P", "S"):
            t_auto = group.loc[group["phase"] == phase, "peak_time"].to_numpy()     # automatic pick times as array, filtered by phase
            lo = np.searchsorted(t_auto, m_ref - lower, side="left")                # index of the first automated pick with time >= m − lower (window start, inclusive)
            hi = np.searchsorted(t_auto, m_ref + upper, side="right")               # index just past the last pick with time <= m + upper (window end, inclusive)
            counts = hi - lo                                                        # counts number of automatic picks inside the window 
            hit_ph = counts > 0                                                  
            row[f"n_false ({phase})"] = int(hit_ph.sum())                       # FP : counts windows that have at least one pick of current phase
            row[f"FPR ({phase})"] =hit_ph.sum() / n_windows if n_windows else np.nan    # calculates False Positive Rate as FP/n_windows = Fall-out
            row[f"n_picks ({phase})"] = int(counts.sum())                           # counts total number of automatic picks of that phase that fall inside the noise windows 

        row["window_lower"], row["window_upper"] = lower, upper
        rows.append(row)

    out = pd.DataFrame(rows)
    if out_dir is not None:
        out.to_parquet(os.path.join(out_dir, f"noise_stats_w{lower:g}_{upper:g}.parquet"),
                       index=False)

    return out


# def noise_stats_old(noise_manual, picks, window=(0.5, 0.5), by=("grid_id", "threshold"),
#                    out_dir=None):
#     """
#     False detections on the noise catalog, per phase. For each noise reference
#     pick and each grouping, counts automatic picks of the same station inside
#     [t_manual - window[0], t_manual + window[1]]. A noise window with at least
#     one pick is a false detection.

#     INPUT:
#         noise_manual: DataFrame of noise reference picks, one per noise event,
#                       restrict to the reference station before calling
#         picks: DataFrame from predict_segment or store.read_run
#         window: tuple (lower, upper), seconds before and after the noise pick
#         by: tuple of column names to group by
#         out_dir: str or None, write the result as
#                  noise_stats_w<lower>_<upper>.parquet into this directory

#     OUTPUT:
#         pandas DataFrame, one row per group and phase, columns
#         phase, n_windows, n_false (windows with >= 1 pick), false_rate,
#         n_picks (all picks inside the windows), window_lower, window_upper
#     """
#     lower, upper = window
#     by = list(by)

#     man = noise_manual[["station", "t_manual"]].sort_values("t_manual")             # drop unnecessary columns; sorting
#     n_windows = len(man)
#     pk = picks[["station", "phase", "peak_time", *by]].sort_values("peak_time")     # drop unnecessary columns; sorting

#     rows = []
#     for (*keys, phase), group in pk.groupby([*by, "phase"], sort=False):            # splits the picks into groups of by,phase
#         n_false, n_picks = 0, 0
#         for sta, g in group.groupby("station", sort=False):                         # loops over station within that group
#             t_auto = g["peak_time"].to_numpy()                                      # automatic pick times as array
#             m_ref = man.loc[man["station"] == sta, "t_manual"].to_numpy()           # noise reference time as array
#             lo = np.searchsorted(t_auto, m_ref - lower, side="left")                # index of the first automated pick with time >= m − lower (window start, inclusive)
#             hi = np.searchsorted(t_auto, m_ref + upper, side="right")               # index just past the last pick with time <= m + upper (window end, inclusive)
#             counts = hi - lo                                                        # counts number of automatic picks inside the window 
#             n_false += int((counts > 0).sum())  # counts # of FP windows overall across all stations in this group
#             n_picks += int(counts.sum())        # counts tota number of automatic picks that fall inside the noise windows
#         rows.append(dict(zip(by, keys), phase=phase, n_windows=n_windows,
#                          n_false=n_false,
#                          false_rate=n_false / n_windows if n_windows else np.nan,
#                          n_picks=n_picks, window_lower=lower, window_upper=upper))

#     out = pd.DataFrame(rows)
#     if out_dir is not None:
#         name = f"noise_stats_w{lower:g}_{upper:g}.parquet"
#         out.to_parquet(os.path.join(out_dir, name), index=False)

#     return out


def detection_stats(manual, picks, p_window=(0.5, 0.2), s_window=None,
                    by=("grid_id", "threshold"), out_dir=None):
    """
    Event level detection, interval based. Counts reference events with at least
    one automatic P pick inside [t_manual - p_window[0], t_manual + p_window[1]].
    The same is computed for S picks (window s_window):
    CAREFUL: picks must already be filtered to the station of the reference picks.

    INPUT:
        manual: DataFrame of reference picks (P onsets), one per event,
                restricted to the reference station
        picks: DataFrame from store.read_run, filtered to the reference station
        p_window: tuple (lower, upper), seconds the P pick may be early or late
        s_window: tuple or None, window for the S presence count,
                  None uses p_window
        by: tuple of column names to group by
        out_dir: str or None, write the result as
                 detection_stats_wp<lower>_<upper>_ws<lower>_<upper>.parquet into this directory

    OUTPUT:
        pandas DataFrame, one row per group, columns n_events,
        n_matched/recall/n_missed per phase ("(P)", "(S)"),
        window_lower/upper (P) and (S)
    """
    windows = {"P": p_window, "S": s_window if s_window is not None else p_window}
    by = list(by)

    m_ref = np.sort(manual["t_manual"].to_numpy())      # reference times (P onsets), sorted by time
    n_events = len(m_ref)
    pk = picks[["phase", "peak_time", *by]].sort_values("peak_time") # drop unnecessary columns, sorted by peak time

    rows = []
    for keys, group in pk.groupby(by, sort=False): # iteration through the groups of by, e.g. (grid_id, threshold) 
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(by, keys), n_events=n_events)

        for phase in ("P", "S"):
            lower, upper = windows[phase]
            t_auto = group.loc[group["phase"] == phase, "peak_time"].to_numpy()     # automatic pick times as array, filtered by phase
            lo = np.searchsorted(t_auto, m_ref - lower, side="left")                # index of the first automated pick with time >= m − lower (window start, inclusive)
            hi = np.searchsorted(t_auto, m_ref + upper, side="right")               # index just past the last pick with time <= m + upper (window end, inclusive)
            hit_ph = (hi - lo) > 0                                                  # True if event has >= 1 automated pick of this phase
            row[f"n_matched ({phase})"] = int(hit_ph.sum())                         # TP counts events that have at least one pick of current phase
            row[f"n_missed ({phase})"] = int(n_events - hit_ph.sum())               # FN counts events that have no pick of current phase
            row[f"Rec ({phase})"] = hit_ph.sum() / n_events if n_events else np.nan # Recall
            row[f"window_lower ({phase})"], row[f"window_upper ({phase})"] = lower, upper

        rows.append(row)

    out = pd.DataFrame(rows)
    if out_dir is not None:
        lo_p, up_p = windows["P"]
        lo_s, up_s = windows["S"]
        out.to_parquet(os.path.join(out_dir, f"detection_stats_wp{lo_p:g}_{up_p:g}_ws{lo_s:g}_{up_s:g}.parquet"),
                       index=False)
    return out

# def detection_stats_old(manual, picks, window=(0.5, 0.2), by=("grid_id", "threshold"),
#                     out_dir=None):
#     """
#     Event level detection, interval based: a manual pick counts as matched when
#     ANY automatic pick of the same station and phase lies inside
#     [t_manual - window[0], t_manual + window[1]].
#     Independent of the nearest neighbour matching of match_picks.

#     INPUT:
#         manual: DataFrame from load_manual_picks, CAREFUL: restrict to the reference
#                 station and phase before calling
#         picks: DataFrame from predict_segment or store.read_run
#         window: tuple (lower, upper), seconds the automatic pick may be early
#                 or late relative to the manual pick
#         by: tuple of column names to group by
#         out_dir: str or None, write the result as
#                  detection_stats_w<lower>_<upper>_<phase>.parquet into this directory

#     OUTPUT:
#         pandas DataFrame with columns n_events, n_matched, n_missed,
#         recall (= n_matched / n_events), phase (single phase of manual, or "mixed"),
#         window_lower, window_upper
#     """
#     lower, upper = window
#     by = list(by)

#     phases = manual["phase"].unique()
#     if len(phases) > 1:
#         print(f"WARNING: manual contains several phases {sorted(phases)}; restrict "
#               f"to one station and phase before calling. n_events counts picks, "
#               f"not events, and the output is labeled phase='mixed'.")
#     phase_label = phases[0] if len(phases) == 1 else "mixed"

#     man = (manual[["station", "phase", "t_manual"]]                                 # drop unnecessary columns; sorting, reset index after sorting
#            .sort_values("t_manual").reset_index(drop=True))
#     pk = picks[["station", "phase", "peak_time", *by]].sort_values("peak_time")     # drop unnecessary columns; sorting ; *by unpacks by tuple into separate columns

#     rows = []
#     for group_key, group in pk.groupby(by, sort=False):
#         # any pick in [t-lower, t+upper]  <=>  nearest earlier pick within lower
#         # OR nearest later pick within upper (merge_asof tolerances are inclusive)
#         g = group[["station", "phase", "peak_time"]]
#         before = pd.merge_asof(man, g, left_on="t_manual", right_on="peak_time",    # for each reference pick (left), find nearest automatic pick (right) in the given direction; match only if that nearest-before pick is within lower seconds;
#                                by=["station", "phase"], direction="backward",       # only match picks of the same station and phase; 
#                                tolerance=lower)["peak_time"].notna()                # peak_time gets turned into boolean: True if a match was found, False if no match within tolerance
#         after = pd.merge_asof(man, g, left_on="t_manual", right_on="peak_time",     
#                               by=["station", "phase"], direction="forward",         # same but in other direction
#                               tolerance=upper)["peak_time"].notna()
#         hits = int((before | after).sum())                                          # before and after are boolean arrays, | is elementwise OR, sum counts the number of True values

#         keys = group_key if isinstance(group_key, tuple) else (group_key,)
#         rows.append(dict(zip(by, keys),
#                          n_events=len(man), n_matched=hits,                        # if manual contains 'P' and 'S' picks, n_events is the total number of picks, not the number of events
#                          n_missed=len(man) - hits,
#                          recall=hits / len(man) if len(man) else np.nan,
#                          phase=phase_label,
#                          window_lower=lower, window_upper=upper))
#         # isintance is needed because group_key is a tuple when len(by) > 1, otherwise it is a single value
#         # zip(by, ...) → [("grid_id", "phasenet_original_ov050_avg_real"), ("threshold", 0.1)]
#         # dict(zip(by, ...),n_events=172, n_matched=158, n_missed=14)   → {"grid_id": "phasenet_original_ov050_avg_real", "threshold": 0.1, "n_events": 172, "n_matched": 158, "n_missed": 14}        

#     out = pd.DataFrame(rows)
#     if out_dir is not None:
#         name = f"detection_stats_w{lower:g}_{upper:g}_{phase_label}.parquet"
#         out.to_parquet(os.path.join(out_dir, name), index=False)  # parquet keeps dtypes

#     return out