import pandas as pd
import numpy as np

import sys
sys.path.append("/home/tevers/MTH/MTH2026/")

import viz_hypocenters 

bx, by = viz_hypocenters.GRID_C["box"]["x"], viz_hypocenters.GRID_C["box"]["y"]
cx = np.array([bx[0], bx[1], bx[1], bx[0]])   # LL, LR, UR, UL
cy = np.array([by[0], by[0], by[1], by[1]])
lon_c, lat_c, _ = viz_hypocenters.sdc_converter(cx, cy, np.zeros(4))

pd.DataFrame({"corner": ["LL", "LR", "UR", "UL"],
              "lon": lon_c, "lat": lat_c}).to_csv("/home/tevers/MTH/MTH2026/MTH_Data/thesis_plotting_storage/events/QGis/grid_corners.csv", index=False)

X, Y = np.meshgrid(viz_hypocenters.GRID_C["x"], viz_hypocenters.GRID_C["y"])          # node axes [km], from nll_grid
lon_g, lat_g, _ = viz_hypocenters.sdc_converter(
    X.ravel(), Y.ravel(), np.zeros(X.size))

pd.DataFrame({"lon": lon_g, "lat": lat_g,
              "x_km": X.ravel(), "y_km": Y.ravel()}
             ).to_csv("/home/tevers/MTH/MTH2026/MTH_Data/thesis_plots/events/QGis/grid_full.csv", index=False)


ring = list(zip(lon_c, lat_c)) + [(lon_c[0], lat_c[0])]   # close the ring
wkt = "POLYGON((" + ", ".join(f"{lo} {la}" for lo, la in ring) + "))"
pd.DataFrame({"wkt": [wkt], "name": ["NLLoc grid box"]}).to_csv("/home/tevers/MTH/MTH2026/MTH_Data/thesis_plotting_storage/events/QGis/nll_box_poly.csv", index=False)