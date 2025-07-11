#!/usr/bin/env python

"""
First script to build the grid
It will store a *npz file named: `heim_grid.npz` containing the following:
 - BOUNDARIES: grid extension
 - SPACING: in km of the grid in 3 dimensions
 - REFERENCE_POINT: Center point of my grid

"""

import sys
import numpy as np
import heimdall
from heimdall import io as gio
from heimdall import locator as glctr
from heimdall import custom_logger as CL
#
from pathlib import Path

logger = CL.init_logger(Path(sys.argv[0]).name, lvl="INFO",
                        log_file="BuildGrid.log")

# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


def build_heimdall_grid(confs):
    #   0       1       2       3        4         5
    # lonmin, lonmax, latmin, latmax, depminKM, depmaxKM  --> bound.
    boundaries = confs.BOUNDARIES
    logger.info("Building grid --> LON: %s / LAT: %s / DEP: %s" % (
                boundaries[0:2], boundaries[2:4], boundaries[4:6]))

    if confs.CENTRE:
        logger.info("Reference Point SPECIFIED --> LON: %f / LAT: %f" % (
                    confs.CENTRE[0], confs.CENTRE[1]))
        assert confs.CENTRE[0] >= boundaries[0]
        assert confs.CENTRE[0] <= boundaries[1]
        assert confs.CENTRE[1] >= boundaries[2]
        assert confs.CENTRE[1] <= boundaries[3]
        reference_point = confs.CENTRE
    else:
        # Take the lowest left corner
        reference_point = [boundaries[0], boundaries[2]]
    #
    logger.info("Reference point:  LON: %f / LAT: %f /" % (
                    reference_point[0], reference_point[1]))
    logger.info("Storing")
    np.savez('heim_grid.npz',
             boundaries=boundaries,
             spacing_km=confs.SPACING_XYZ,
             reference_point=reference_point,
             tag=confs.GRID_TAG,
             version=heimdall.__version__)

    # ---------- 1. Initialize  LOCATOR
    HG = glctr.HeimdallLocator(boundaries,
                               spacing_x=confs.SPACING_XYZ[0],
                               spacing_y=confs.SPACING_XYZ[1],
                               spacing_z=confs.SPACING_XYZ[2],
                               reference_point_lonlat=reference_point)
    (xgr, ygr, zgr) = HG.get_grid()
    (reflon, reflat) = HG.get_grid_reference()
    SHAPES = [(len(xgr), len(ygr)), (len(xgr), len(zgr)), (len(ygr), len(zgr))]
    logger.state("The 2D planes sizes (XY, XZ, YZ):  %r" % SHAPES)


if __name__ == "__main__":
    try:
        assert Path(sys.argv[1]).exists()
    except:
        logger.error("Configuration file-path non existent!")
        sys.exit()
    CONFIG = gio.read_configuration_file(sys.argv[1], check_version=True)
    build_heimdall_grid(CONFIG.BUILD_GRID)
