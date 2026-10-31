from math import ceil

from obspy import Stream

from .preprocess import preprocess
from .store import event_keys


def reference_pick_time(event, author=None):
    """
    Anchor time of an event. With an refernce author given the matching pick is used,
    otherwise the first pick.

    INPUT:
        event: obspy Event
        author: str or None, creation_info.author of the anchor pick ; in the config file : reference_author

    OUTPUT:
        obspy UTCDateTime
    """
    #XXX : is this correct? -> Yes as long as the reference pick is the first pick of the event
    if author is None:
        return event.picks[0].time

    for p in event.picks:
        if p.creation_info is not None and p.creation_info.author == author:
            return p.time

    raise ValueError(f"No pick by '{author}' in event {event.resource_id}.")


def fetch_window(t0, t1, client, station_list, correct_for_instrument=False,
                 inv=None, pre_filt=None, water_level=None):
    """
    Fetch and preprocess one time window, station by station. A station that
    fails anywhere in fetch or preprocessing is skipped, the others continue.

    INPUT:
        t0, t1: obspy UTCDateTime, window bounds
        client: obspy client
        station_list: list of str, "NET.STA.LOC.CHA"
        correct_for_instrument: bool, remove the instrument response
        inv, pre_filt, water_level: response removal settings

    OUTPUT:
        obspy Stream, stations without data are simply absent
    """
    st = Stream()
    for nsc in station_list:
        net, stat, _, chan = nsc.split(".")
        try:
            st_i = client.get_waveforms(network=net, station=stat, location="*",
                                        channel=chan, starttime=t0, endtime=t1)
            st_i = preprocess(st_i, correct_for_instrument, inv, pre_filt,
                              water_level)
            st += st_i
        except Exception:  # not KeyboardInterrupt, so ctrl-c stops the run
            print(nsc, ".... no data")

    return st


class CatalogWindowSource:
    """
    Yields one waveform window per event of a catalog. (icequakes or noise or whatever you consider as an event )
    """

    def __init__(self, catalog, client, config, inv=None, pre_filt=None,
                 water_level=None):
        """
        INPUT:
            catalog: obspy Catalog
            client: obspy client
            config: RunConfig, correct_for_instrument comes from remove_response
            inv, pre_filt, water_level: response removal settings
        """
        self.catalog = catalog
        self.client = client
        self.config = config
        self.inv, self.pre_filt, self.water_level = inv, pre_filt, water_level
        self.keys = event_keys(catalog)

    def __len__(self):
        return len(self.catalog)

    def __iter__(self): # careful with calling this function, it will stepwise download the data for each event in the catalog and yield a stream for each event 
        """
        OUTPUT:
            yields (key, event_id, t_ref, stream) per event
        """
        c = self.config
        for key, event in zip(self.keys, self.catalog):
            t_ref = reference_pick_time(event, c.reference_author) # for whatever event is marked with a pick in the catalog
            t0 = t_ref - c.window_start_before_eventtime
            st = fetch_window(t0, t0 + c.window_length, self.client, 
                              c.station_list, c.remove_response, self.inv,
                              self.pre_filt, self.water_level)
            yield key, str(event.resource_id), t_ref, st


class ContinuousSource:
    """
    Yields consecutive chunks of a continuous time range. Chunks do not overlap.
    """

    def __init__(self, t_start, t_end, chunk_length, client, config, inv=None,
                 pre_filt=None, water_level=None):
        """
        INPUT:
            t_start, t_end: obspy UTCDateTime, range to process
            chunk_length: float, chunk length in seconds
            client: obspy client
            config: RunConfig, station_list and remove_response are used
            inv, pre_filt, water_level: response removal settings
        """
        self.t_start, self.t_end = t_start, t_end
        self.chunk_length = chunk_length
        self.client = client
        self.config = config
        self.inv, self.pre_filt, self.water_level = inv, pre_filt, water_level

    def __len__(self):
        return ceil((self.t_end - self.t_start) / self.chunk_length) # Number of chunks 

    def __iter__(self):
        """
        OUTPUT:
            yields (key, None, None, stream) per chunk, key is the chunk start time
        """
        t0 = self.t_start
        while t0 < self.t_end:
            t1 = min(t0 + self.chunk_length, self.t_end)

            if (t1 - t0) < self.chunk_length:
                print(f"NOTE: last chunk {t0.strftime('%Y%m%dT%H%M%S')} is only "
                      f"{t1 - t0:.0f}s long ({self.chunk_length:.0f}s configured); "
                      f"it may be shorter than the model window and yield no picks.")

            key = t0.strftime("%Y%m%dT%H%M%S") #UTCDateTime("2025-03-21T12:00:00") ->  "20250321T120000"
            st = fetch_window(t0, t1, self.client, self.config.station_list,
                              self.config.remove_response, self.inv,
                              self.pre_filt, self.water_level)
            yield key, None, None, st
            t0 = t1
