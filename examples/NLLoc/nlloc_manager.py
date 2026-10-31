#!/usr/bin/env python3
"""
nlloc_manager.py — NonLinLoc manager for iceBench locations
===========================================================
Generates NonLinLoc control files and runs the full location chain
(Vel2Grid -> Grid2Time P/S -> NLLoc) for every (velocity model x observation
set) combination, from a single set of definitions at the top of this file.

It is the downstream step after association: the .obs phase files produced by
the associator examples (associate_analystPicks.py / associate_autoPicks.py and
the continuous associator) are located here against one or more velocity models.

How it works:
  - MODELS     : name -> (grid geometry, LAYER lines)   — the velocity models.
  - OBS_SETS   : name -> (.obs glob, output subdir)      — the phase picks to locate.
  - TEMPLATE   : nlloc_template.in, an annotated control file with __PLACEHOLDERS__
                 that make_control() fills per model / wave type / observation set.
  - build_grids(vmod)      : Vel2Grid + Grid2Time (P and S) -> travel-time grids.
  - locate(vmod, obs_set)  : NLLoc for one model x obs set; archives the control
                             file next to its results.

Usage:
  1. Ensure the NonLinLoc binaries (Vel2Grid, Grid2Time, NLLoc) are reachable,
     and station_coordinates.txt and nlloc_template.in are in place.
  2. Edit the RUN section at the bottom (which models, which obs sets, whether to
     rebuild grids), then you can run this script
  Travel-time grids only need rebuilding when a velocity model or the grid
  geometry changes (compute_grids=True); locating reuses existing grids.

Coordinate frame: TRANS SDC with a frozen origin (see TRANS) — changing it
invalidates existing grids and requires a full regrid.

Requires: a working NonLinLoc installation, this script only controls the application.
"""

import os
import subprocess
import sys
from datetime import datetime

# ----------------------------------------------------------------- fixed setup
# XXX SET YOUR DIRECTORIES HERE
ROOT = "/path/to/installation/NLLoc"  # base dir: holds obs/, model/, time/, loc/, controls/, binaries
BIN = ROOT                                  # dir containing NLLoc/Vel2Grid/Grid2Time links
STATION_FILE = f"{ROOT}/station_coordinates.txt"
TRANS = "TRANS SDC 45.9243 7.8341 0"        # frozen origin - never change without regridding

# grid geometry (nx ny nz  x0 y0 z0  d) - cubic only (PLFD); one geometry for all models
GRID_FINE = (180, 136, 48, 0.0, 0.0, 0.0, 0.0125)     # 12.5 m

# velocity models: name -> (grid, [LAYER lines])  - single source of truth
MODELS = {
    "vModFIB": (GRID_FINE, [ #vModF_1 : velocity model key
        "LAYER    0.000  0.223 244.286 0.117 128.571 615     0",
        "LAYER    0.014  3.643   0    1.917    0    917     0",
        "LAYER    0.400  5.000   0    2.550    0    2750    0"]),
    "vModIB": (GRID_FINE, [                    # no-firn twin of C_0: same ice/bedrock
        "LAYER    0.000  3.643   0    1.917    0    917     0",
        "LAYER    0.400  5.000   0    2.550    0    2750    0"]),
}

# XXX: SET YOUR OBS DIRECTORIES HERE
# observation sets: name -> (obs glob, output subdir under loc/)
OBS_SETS = {

    # automatic picks from one picker setup, all events in one obs file
    "gid_key":        (f"{ROOT}/obs/key.obs",                  "name/for/loc/dir"),

    # a continuous run: many .obs files (e.g. one per time chunk) matched by a wildcard
    "gid_key_cont": (f"{ROOT}/obs/*.obs",  "name/for/loc/dir_cont"),
}


# ----------------------------------------------------------------- control file
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nlloc_template.in")


def make_control(vmod, wavetype="P", obs_set="MP_manual"):
    # Fill the full annotated template (user's original .in, all comments kept).
    # INPUT:
    #   vmod: str, model name (key of MODELS)
    #   wavetype: "P" or "S", the Grid2Time pass this file drives
    #   obs_set: str, key of OBS_SETS (fills LOCFILES; harmless for grid stages)
    # OUTPUT:
    #   str, complete control file content
    grid, layers = MODELS[vmod]
    nx, ny, nz, x0, y0, z0, d = grid
    obs_glob, loc_sub = OBS_SETS[obs_set]
    out = open(TEMPLATE).read()
    for k, v in {
        "__MODEL_ROOT__": f"{ROOT}/model/{vmod}/layer",
        "__TIME_ROOT__": f"{ROOT}/time/{vmod}/layer",
        "__GRID__": f"{nx} {ny} {nz}  {x0} {y0} {z0}  {d} {d} {d}",
        "__GRID_LOC__": f"{nx} {ny} {nz} {x0} {y0} {z0} {d} {d} {d}",
        "__LAYERS__": "\n".join(layers),
        "__WAVETYPE__": wavetype,
        "__OBS_GLOB__": obs_glob,
        "__LOC_OUT__": f"{ROOT}/loc/{loc_sub}/{vmod}/grenz",
    }.items():
        out = out.replace(k, v)
    return out


# ----------------------------------------------------------------- run helpers
def run(prog, control_path, log_path):
    # Run one NLLoc program on a control file, tee output to a log.
    # INPUT: prog: str, program name; control_path/log_path: str
    # OUTPUT: int, return code (raises on nonzero)
    with open(log_path, "a") as log:
        log.write(f"\n===== {prog} {control_path} {datetime.now()}\n")
        r = subprocess.run([f"{BIN}/{prog}", control_path],
                           stdout=log, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"{prog} failed (rc {r.returncode}), see {log_path}")
    return r.returncode


def build_grids(vmod):
    # Vel2Grid + Grid2Time P + Grid2Time S for one model.
    cdir = f"{ROOT}/controls"
    os.makedirs(cdir, exist_ok=True)
    for sub in ("model", "time"):
        os.makedirs(f"{ROOT}/{sub}/{vmod}", exist_ok=True)
    log = f"{cdir}/{vmod}_grids.log"
    for w in ("P", "S"):
        cp = f"{cdir}/{vmod}_grids_{w}.in"
        open(cp, "w").write(make_control(vmod, wavetype=w))
        if w == "P":
            print(f"[{vmod}] Vel2Grid ...")
            run("Vel2Grid", cp, log)
        print(f"[{vmod}] Grid2Time {w} ...")
        run("Grid2Time", cp, log)
    print(f"[{vmod}] grids done -> time/{vmod}/")


def locate(vmod, obs_set):
    # NLLoc for one (model, observation set) combination; archives the control file.
    _, loc_sub = OBS_SETS[obs_set]
    outdir = f"{ROOT}/loc/{loc_sub}/{vmod}"
    os.makedirs(outdir, exist_ok=True)
    cp = f"{outdir}/nlloc_{vmod}__{obs_set}.in"        # archived next to the results
    open(cp, "w").write(make_control(vmod, obs_set=obs_set))
    print(f"[{vmod} x {obs_set}] NLLoc ...")
    run("nlloc", cp, f"{outdir}/nlloc.log")
    print(f"[{vmod} x {obs_set}] done -> {outdir}/")


# ----------------------------------------------------------------- RUN section
# Edit this block, then you can run this script to build grids and/or locate
if __name__ == "__main__":
    vmods = [
        "vModIB",
        "vModFIB",
    ]  # velocity model(s) to use; names must be keys of MODELS
    obs_sets = ["gid_key", "gid_key_cont"]  # specify observation sets that must be located; names must be keys of OBS_SETS
    compute_grids = True  # True: (re)build the model + travel-time grids for each vmod ; (needed once per model, or after changing a model / the grid)
    locate_list = True  # True: run NLLoc for every (vmod x obs_set) combination below

    # build travel-time grids first
    if compute_grids:
        for vm in vmods:
            build_grids(vm)

    # locate each observation set against each velocity model
    if locate_list:
        for vm in vmods:
            for ob in obs_sets:
                locate(vm, ob)
