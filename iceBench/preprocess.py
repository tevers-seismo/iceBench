def preprocess(st, correct_for_instrument=False, inv=None, pre_filt=None,
               water_level=None):
    """
    Fixed preprocessing recipe for one station. No decimation: the model rate is
    reached either by seisbench itself (real domain) or by the time transformation.

    INPUT:
        st: obspy Stream of one station, as fetched
        correct_for_instrument: bool, remove the instrument response
        inv: obspy Inventory, required when correct_for_instrument is set
        pre_filt: tuple or None, passed to remove_response
        water_level: float or None, passed to remove_response

    OUTPUT:
        obspy Stream
    """
    st.merge()  # duplicated waveforms collapse without masking
    st.detrend("linear") #other options: see obpsy.trace documentations

    if correct_for_instrument:
        st.remove_response(inv, pre_filt=pre_filt, water_level=water_level)

    return st
