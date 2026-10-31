"""
iceBench — build the NonLinLoc station file (GTSRCE) from an inventory
=======================================================================
Setup step for the NonLinLoc location chain. Reads a station inventory
(StationXML) and writes a GTSRCE station file that NLLoc/Grid2Time use to place
the stations: one line per station, 'GTSRCE <code> LATLON <lat> <lon> <z> <elev>'.

Stations are written at the 0 m datum (z_srce = elev = 0.0), matching the z=0
station convention used throughout iceBench; edit those fields if you locate with
true station elevations.

Run this once before nlloc_manager.py — the file it writes is the STATION_FILE
that nlloc_manager.py expects. The station codes here (bare STA, e.g. S01) must
match the station labels in the .obs phase files produced by the associator.

Input:  a StationXML inventory.
Output: a GTSRCE text file at station_file.
"""

from obspy import read_inventory

# 0. Specify the inventory
inv = read_inventory("/path/to/inventory.xml")

station_file = "/path/to/NLLoc/obs/station_coordinates.txt"

with open(station_file, "w") as out:
    out.write("#GTSRCE  label  type  lat  lon  z_srce  elev\n")
    for net in inv:
        for sta in net:
            # STA lat lon elevation_m
            out.write(f"GTSRCE {sta.code} LATLON {sta.latitude} "
              f"{sta.longitude} 0.0 0.0\n")

print(f"{sum(len(n) for n in inv)} stations written to {station_file}")



