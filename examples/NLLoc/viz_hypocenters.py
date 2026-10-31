"""
iceBench — NLLoc <-> geographic coordinate helpers and QGIS exporters
======================================================================
Utilities for turning NonLinLoc output into QGIS-format,
all in the frozen SDC frame used for the location grids (TRANS SDC <lat> <lon> 0).

Provides:
  - grid geometry:  nll_grid(...) turns VGGRID/LOCGRID numbers into node axes,
                    cell edges and imshow/pcolormesh extents (map, EW, NS views).
  - coordinate transforms (local azimuthal-equidistant about LAT0/LON0, in km):
      sdc_converter(x, y, z)       NLLoc SDC km -> (lon, lat, depth_km)
      sdc_forward(lat, lon, depth) WGS84 deg    -> SDC (x east, y north, z=depth) km
  - QGIS exporters (write CSV point layers):
      exportCoords2QGIS(cat)   catalog origins -> one row per event
      exportPDF2QGIS(scat)     one NLLoc scatter cloud -> one row per sample
  - read_nlloc_scatter(...)    read a NonLinLoc .scat file into a structured array
                               (adapted from obspy; see note below).

Convention: stations and the datum are at 0 m; depths are km below the datum,
and elev_m = ELST_MEAN - depth places points relative to the mean station
elevation for 3D / vertical-exaggeration views in QGIS.

NOTE: the origin (LAT0, LON0), grid size (XNUM/YNUM/ZNUM, DX/DY/DZ) and
ELST_MEAN are study-specific (Grenzgletscher).
The origin must match the TRANS line used to build the NLLoc grids ( in nlloc_manager script ).
"""

import numpy as np
import pyproj
import pandas as pd


# frozen SDC origin for all NLLoc grids: TRANS SDC <lat> <lon> 0
# Station, lat, lon
#QGISgrid
LAT0 = 45.9243
LON0 = 7.8341
EL0 = 0.0       # we set everything to 0 m elevation

XNUM = 180
YNUM = 136
ZNUM = 48

DX = 0.0125
DY = 0.0125
DZ = 0.0125

ELST_MEAN = 3.724


def nll_grid(nx, ny, nz, x0, y0, z0, dx, dy=None, dz=None):
    # NLLoc VGGRID/LOCGRID numbers -> plottable node axes, edges, extents.
    # INPUT:
    #   nx..dz: the VGGRID values (gridType omitted); dy/dz default to dx
    # OUTPUT:
    #   dict: node axes x/y/z [km], cell edges xe/ye/ze,
    #         extent_map/extent_ew/extent_ns for imshow, box = outline limits
    dy = dx if dy is None else dy
    dz = dx if dz is None else dz
    x = x0 + dx * np.arange(nx)
    y = y0 + dy * np.arange(ny)
    z = z0 + dz * np.arange(nz)
    xe = x0 - dx/2 + dx * np.arange(nx + 1)     # cell edges for pcolormesh
    ye = y0 - dy/2 + dy * np.arange(ny + 1)
    ze = z0 - dz/2 + dz * np.arange(nz + 1)
    return dict(x=x, y=y, z=z, xe=xe, ye=ye, ze=ze,
                extent_map=(xe[0], xe[-1], ye[0], ye[-1]),
                extent_ew=(xe[0], xe[-1], ze[-1], ze[0]),   # depth downward
                extent_ns=(ye[0], ye[-1], ze[-1], ze[0]),
                box=dict(x=(x[0], x[-1]), y=(y[0], y[-1]), z=(z[0], z[-1])))

GRID_C = nll_grid(XNUM, YNUM, ZNUM, 0.0, 0.0, 0.0, DX,DY,DZ)   # your VGGRID/LOCGRID


_to_geo = pyproj.Transformer.from_crs(
    f"+proj=aeqd +lat_0={LAT0} +lon_0={LON0} +datum=WGS84 +units=km",
    "EPSG:4326", always_xy=True)

_to_xy = pyproj.Transformer.from_crs(
    "EPSG:4326",
    f"+proj=aeqd +lat_0={LAT0} +lon_0={LON0} +datum=WGS84 +units=km",
    always_xy=True)

def sdc_converter(x, y, z):
    # NLLoc SDC km -> (lon, lat, depth_km) via local azimuthal-equidistant (<0.1 m vs SDC)
    lon, lat = _to_geo.transform(x, y)
    return lon, lat, z

def sdc_forward(lat, lon, depth_km=0.0):
    # lat/lon [deg] -> NLLoc SDC frame (x east, y north [km], z = depth_km)
    # INPUT:
    #   lat, lon: float or arrays, WGS84 degrees
    #   depth_km: float or array, km below the NLLoc datum (passes through)
    # OUTPUT:
    #   x, y, z: km in the NLLoc frame
    x, y = _to_xy.transform(lon, lat)      # always_xy: lon first!
    return x, y, depth_km


def exportCoords2QGIS(cat, outfile=None):
    """
    Write catalog origins to a CSV that QGIS loads as a point layer.
    INPUT:
      cat:     obspy Catalog, each event with >=1 origin
      outfile: str, path of the CSV to write
    OUTPUT:
      df:      pandas DataFrame, the table that was written
    """
    rows = []
    for ev in cat:
        o = ev.preferred_origin() or ev.origins[0]
        m = ev.preferred_magnitude()                      # may be None
        d = (o.depth if o.depth is not None else np.nan)/1000    # obspy: metres, positive down
        rows.append({
            "id":      str(ev.resource_id).split("/")[-1],
            "time":    o.time.isoformat(),                # ISO -> QGIS parses as datetime
            "lon":     o.longitude,                       # X
            "lat":     o.latitude,                        # Y
            "depth_m": d,                            # depth in km
            "elev_m":  ELST_MEAN -d,                                # Z for 3D / vertical exaggeration
            "time_key":  o.time.strftime("%Y-%m-%d %H:%M"),           # primary key: date+hour+minute
            "time_unix": float(o.time.timestamp),                     # raw, for merge_asof
            "time_unixRound": round(float(o.time.timestamp), -1),          # fallback key: 10 s bucket
        })
    df = pd.DataFrame(rows)
    if outfile:
      df.to_csv(outfile, index=False)
    return df

def exportPDF2QGIS(scat, elev0=ELST_MEAN, outfile=None):
    # Write one scatter cloud (already lat/lon-converted) to a QGIS CSV.
    # INPUT:
    #   scat: structured array from read_nlloc_scatter(..., coordinate_converter=
    #         sdc_converter) - fields x=lon, y=lat, z=depth_km, pdf
    #   event_id: str, tag column (e.g. from the filename)
    #   elev0: float, datum elevation [m a.s.l.]
    #   outfile: str or None, CSV path
    # OUTPUT:
    #   df: DataFrame (event_id, lon, lat, depth_km, elev_m, pdf)
    df = pd.DataFrame({
        "lon":      scat["x"],                    # converter: x -> lon
        "lat":      scat["y"],                    # converter: y -> lat
        "depth_m": scat["z"],
        "elev_m":   ELST_MEAN -scat["z"]/1000 ,
        "pdf":      scat["pdf"],
    })
    if outfile:
        df.to_csv(outfile, index=False)
    return df


# From obspy 
def read_nlloc_scatter(filename, coordinate_converter=None):
    """
    Read a NonLinLoc Scatter file into a Numpy array.

    .. note::

        Coordinate conversion from coordinate frame of NonLinLoc model files /
        location run to WGS84 has to be specified explicitly by the user if
        necessary.

    :type filename: str
    :param filename: Filename with NonLinLoc scatter.
    :type coordinate_converter: callable
    :param coordinate_converter: Function to convert (x, y, z)
        coordinates of NonLinLoc output to geographical coordinates and depth
        in meters (longitude, latitude, depth in kilometers).
        If left ``None``, NonLinLoc (x, y, z) output is left unchanged (e.g. if
        it is in geographical coordinates already like for NonLinLoc in
        global mode).
        The function should accept three arguments x, y, z (each of type
        :class:`numpy.ndarray`) and return a tuple of three
        :class:`numpy.ndarray` (lon, lat, depth in kilometers).
    :returns: NonLinLoc scatter information as structured numpy array (fields:
        "x", "y", "z", "pdf").
    """
    # omit the first 4 values (header information) and reshape
    dtype = np.dtype([
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("pdf", "<f4")])
    data = np.fromfile(filename, dtype=dtype)[4:]
    if coordinate_converter:
        data["x"], data["y"], data["z"] = coordinate_converter(
            data["x"], data["y"], data["z"])
    return data

