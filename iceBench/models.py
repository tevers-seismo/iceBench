import seisbench.models as sbm

_CACHE = {}


def get_picker(model, weights):
    """
    Load a pretrained seisbench model, cached so each one is built only once
    per session and stays on the preferred device.

    INPUT:
        model: str, seisbench class name, e.g. "PhaseNet"
        weights: str, pretrained weight set

    OUTPUT:
        seisbench WaveformModel
    """
    key = (model, weights)
    if key not in _CACHE:
        picker = getattr(sbm, model).from_pretrained(weights) # uses the model name stated in config file to load the model and weights from the seisbench library
        picker.to_preferred_device() # From the seisbench docs: Move the model to an accelerator if available: checks for CUDA, MPS and XPU accelerators (in this order).
        _CACHE[key] = picker

    return _CACHE[key]


def window_seconds(picker, time_domain, raw_rate):
    """
    Seisbench resamples input stream to picker.sampling_rate and then processes it in windows of picker.in_samples.
    In the 'real' time_domain seisbench resamples, in the 'transformed' time_domain resampling is avoided. Therefore shorter windows are possible in the 'transformed' setting.

    INPUT:
        picker: seisbench WaveformModel
        time_domain: str, "real" or "transformed"
        raw_rate: float, true sampling rate of the data

    OUTPUT:
        float, seconds
    """
    rate = picker.sampling_rate if time_domain == "real" else raw_rate

    return picker.in_samples / rate # window length in real seconds


def check_window_length(grid, window_length, raw_rate):
    """
    Verify that the fetched window is long enough for every grid point. A cut
    shorter than the model window makes seisbench return an empty stream with
    only a log warning.

    INPUT:
        grid: list of GridPoint
        window_length: float, seconds
        raw_rate: float, lowest sampling rate in the network

    OUTPUT:
        None, raises ValueError on the first grid point that does not fit
    """
    for g in grid:
        need = window_seconds(get_picker(g.model, g.weights), g.time_domain, raw_rate)
        if window_length < need:
            raise ValueError(f"{g.grid_id} needs {need:.2f}s, "
                             f"window_length is {window_length}s.")
