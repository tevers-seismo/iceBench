import numpy as np
import matplotlib.pyplot as plt


def plot_residual_hist(matched, phases=("P", "S"), split_quality=False, bins=50,
                       tolerance=None, title=None, figsize=(12, 5)):
    """
    Histogram of the pick time residuals, one panel per phase, annotated with
    mean, std, mae and the number of compared picks. Expects a single grid point
    and threshold, filter beforehand.

    INPUT:
        matched: DataFrame from match_picks, CAREFUL: Filter beforehand to a single grid point and threshold
        phases: iterable of phase names
        split_quality: bool, overlay normal and questionable manual picks
        bins: int, number of histogram bins
        tolerance: float or None, symmetric x limit in seconds
                    -> float: Useful as histograms become comparative when the same range is used
                    -> None: uses the full range of the residuals
        title: str or None
        figsize: tuple

    OUTPUT:
        matplotlib figure
    """
    fig, ax = plt.subplots(1, len(phases), figsize=figsize, squeeze=False)
    ax = ax[0]
    rng = (-tolerance, tolerance) if tolerance else None

    for a, ph in zip(ax, phases):
        sub = matched[matched["phase"] == ph]
        groups = [("all", sub)] if not split_quality else \
            [(q, sub[sub["quality"] == q]) for q in ("normal", "questionable")]

        lines = []
        for label, g in groups:       # "all" or "normal" and "questionable", plotted on same axis
            r = g["residual"].dropna().to_numpy(dtype=float)    # numpy array of residuals, drop naN for statistics because they are unmatched manual picks
            if r.size == 0:
                print("WARNING: No matched picks for phase", ph, "and quality", label, "in this grid point and threshold.")
                continue
            a.hist(r, bins=bins, range=rng, histtype="stepfilled", alpha=0.6,       # 'stepfilled': filled histogram with a line around it at alpha opacity -> overlapping histograms are visible
                   label=f"{label} (N={r.size})")               # N = number of matched manual picks
            lines.append(f"{label}: mean {np.mean(r):.4f} s, "
                         f"std {np.std(r, ddof=1):.4f} s, "     # ddof=1 -> sample std (N-1 divisor) and not population (N divisor): consistent with residual_stats
                         f"mae {np.mean(np.abs(r)):.4f} s")

        a.axvline(0, color="k", lw=1.0)
        a.set_title(f"{ph} phase")
        a.set_xlabel("residual (manual - seisbench) [s]")
        if tolerance:
            a.set_xlim(-tolerance, tolerance)
            a.set_xticks(np.arange(-tolerance, tolerance, 0.1), minor = True)
            #a.set_xticks(np.arange(-tolerance, tolerance, 0.01), minor = True)
        if lines:
            a.text(0.02, 0.98, "\n".join(lines), transform=a.transAxes, va="top",   # print statistics box
                   fontsize=8, family="monospace",
                   bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85))
        a.legend(loc="upper right", fontsize=8)

    ax[0].set_ylabel("counts")
    if title:
        fig.suptitle(title)
    fig.tight_layout()

    return fig


def plot_metric_vs_threshold(stats, metric="std", phase="P", grid_ids=None,
                             figsize=(10, 6),cmap="viridis",logx=True,legend=True):
    """
    One metric against the trigger threshold, one line per grid point.

    INPUT:
        stats: DataFrame from residual_stats or detection_stats
        metric: str, column to plot, e.g. "std", "mae", "n_missed"
        phase: str, phase to show
        grid_ids: iterable or None, restrict to these grid points
        figsize: tuple
        cmap: str, matplotlib colormap
        logx: bool, logarithmic threshold axis

    OUTPUT:
        matplotlib figure
    """
    sub = stats[stats["phase"] == phase]
    if grid_ids is not None:
        sub = sub[sub["grid_id"].isin(grid_ids)]

    ids = sorted(sub["grid_id"].unique())
    colors = plt.get_cmap(cmap)(np.linspace(0, 0.9, len(ids)))

    fig, ax = plt.subplots(figsize=figsize)
    for gid, c in zip(ids, colors):
        g = sub[sub["grid_id"] == gid].sort_values("threshold")
        ax.plot(g["threshold"], g[metric], "-o", ms=3, color=c, label=gid)

    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("trigger threshold")
    ax.set_ylabel(metric)
    ax.set_title(f"{phase} phase")
    if legend:
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()

    return fig
