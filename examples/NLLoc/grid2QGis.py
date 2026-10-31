"""
iceBench — export the NLLoc grid extent to QGIS
================================================
Writes the location grid geometry as QGIS-loadable CSVs, converted from the SDC
frame (km) to WGS84 lon/lat with viz_hypocenters.sdc_converter. Use it to overlay
the NLLoc search volume on a map.

Produces three files:
  grid_corners.csv  - the four grid-box corners (LL, LR, UR, UL) as points.
  grid_full.csv     - every grid node (lon/lat + original x/y in km).
  nll_box_poly.csv  - the grid outline as a single WKT POLYGON (load as a polygon
                      layer; the ring is closed back to the first corner).

Grid geometry and the SDC origin come from viz_hypocenters.GRID_C and must match
the TRANS/VGGRID used to build the NLLoc grids.
"""

import pandas as pd
import numpy as np

import viz_hypocenters 

bx, by = viz_hypocenters.GRID_C["box"]["x"], viz_hypocenters.GRID_C["box"]["y"]
cx = np.array([bx[0], bx[1], bx[1], bx[0]])   # LL, LR, UR, UL
cy = np.array([by[0], by[0], by[1], by[1]])
lon_c, lat_c, _ = viz_hypocenters.sdc_converter(cx, cy, np.zeros(4))

pd.DataFrame({"corner": ["LL", "LR", "UR", "UL"],
              "lon": lon_c, "lat": lat_c}).to_csv("/path/to/QGis/grid_corners.csv", index=False)

X, Y = np.meshgrid(viz_hypocenters.GRID_C["x"], viz_hypocenters.GRID_C["y"])          # node axes [km], from nll_grid
lon_g, lat_g, _ = viz_hypocenters.sdc_converter(
    X.ravel(), Y.ravel(), np.zeros(X.size))

pd.DataFrame({"lon": lon_g, "lat": lat_g,
              "x_km": X.ravel(), "y_km": Y.ravel()}
             ).to_csv("/path/to/QGis/grid_full.csv", index=False)


ring = list(zip(lon_c, lat_c)) + [(lon_c[0], lat_c[0])]   # close the ring
wkt = "POLYGON((" + ", ".join(f"{lo} {la}" for lo, la in ring) + "))"
pd.DataFrame({"wkt": [wkt], "name": ["NLLoc grid box"]}).to_csv("/path/to/QGis/nll_box_poly.csv", index=False)