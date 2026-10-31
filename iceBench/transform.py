import copy


def delate_stream(st, target_rate):
    """
    Stretch every trace in time by relabelling its sampling rate to target_rate.
    The samples and start_time are untouched. Obspy automatically recalculates end_time from the new rate and the number of samples. 

    INPUT:
        st: obspy Stream in real time
        target_rate: float, picker.sampling_rate

    OUTPUT:
        obspy Stream, copy with relabelled rates
    """
    st_t = st.copy() # makes a deepcopy of the stream : see obspy.core.stream.Stream.copy() for details
    for tr in st_t:
        tr.stats.sampling_rate = target_rate

    return st_t


def backtransform_picks(st, picks_t, target_rate):
    """
    Map picks from the transformed frame back to real time, each against the
    start time and sampling rate of its own station.

    INPUT:
        st: obspy Stream in real time, the input of delate_stream
        picks_t: seisbench PickList in transformed time
        target_rate: float, the target_rate used in delate_stream

    OUTPUT:
        seisbench PickList in real time
    """
    info = {f"{tr.stats.network}.{tr.stats.station}":
            (tr.stats.starttime, tr.stats.sampling_rate / target_rate) for tr in st} # needed to infer rescaling rate -> dil 

    picks = copy.deepcopy(picks_t)
    for p in picks:
        t0, dil = info[".".join(str(p.trace_id).split(".")[:2])] #  seisbench trace_id: NET.STA.LOC -> we drop location to match obspy trace_id: NET.STA -> looks up in dictionary
        p.start_time = t0 + (p.start_time - t0) / dil
        p.peak_time = t0 + (p.peak_time - t0) / dil
        p.end_time = t0 + (p.end_time - t0) / dil

    return picks
