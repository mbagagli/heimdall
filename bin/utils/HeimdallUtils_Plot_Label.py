#!/usr/bin/env python

import sys
import numpy as np
from pathlib import Path
import argparse

from heimdall import io as gio
from heimdall import plot as gplt
from heimdall import coordinates as gcr
from heimdall import utils as gut
from heimdall import locator as glctr
from heimdall import custom_logger as CL

logger = CL.init_logger(Path(sys.argv[0]).name, lvl="INFO")


# ====================================================================

def parse_cli():
    """Simple parsing utility for the script."""
    p = argparse.ArgumentParser(
        description="Plot TimeSeries and Grid images slices contained in NPZ plot files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("-g", "--graph", metavar="GRAPH", type=str,
                   required=True, dest="gnn",
                   help="Heimdall GNN graph file (.npz)")
    p.add_argument("-gr", "--grid", metavar="GRID", type=str,
                   required=True, dest="grd",
                   help="Heimdall spatial grid file (.npz)")
    p.add_argument("-f", "--file", metavar="NPZ_FILE", type=str,
                   required=True, dest="npz",
                   help="NPZ file path to plot")
    p.add_argument(
        "-r", "--reference",
        metavar=("LON", "LAT", "DEPm"),        # shows up nicely in --help
        type=float,                    # convert each token to float
        nargs=3,                       # expect exactly two numbers
        dest="origin",
        help="Reference epicentre [lon lat] – stations are sorted by distance",
    )
    p.add_argument(
        "-c", "--configuration",
        dest="confs",
        type=str, required=True,
        metavar="CONFIGURATION_FILE",
        help="Configurations file used for the data-preparation."
    )
    # p.add_argument(
    #     "-w", "--window",
    #     dest="overlap",
    #     type=float,
    #     default=4.5,
    #     metavar="SECONDS",
    #     help="Sliding window overlap in seconds (default: 4.5)"
    # )
    # p.add_argument(
    #     "-df", "--samplingrate",
    #     dest="df",
    #     type=float,
    #     default=100.0,
    #     metavar="HZ",
    #     help="Sampling frequency in Hz (default: 100.0)"
    # )
    p.add_argument("--plot-stations", dest="plot_stations",
                   action="store_true",
                   help="Plot only windows containing at least one pick")
    p.add_argument("--only-windows-with-picks", dest="only_picks",
                   action="store_true",
                   help="Plot only windows containing at least one pick")
    return p.parse_args()


def main(gnnpath, gridpath, filepath, config_file,
         reference_point=[], plot_stations=False,
         only_windows_with_picks=False):
    """Render all requested windows in *npz_file*."""
    heim_grid = np.load(gridpath, allow_pickle=True)
    heim_gnn = np.load(gnnpath, allow_pickle=True)
    confs = gio.read_configuration_file(config_file, check_version=True)
    statdict = heim_gnn["stations_order"].item()
    npz = np.load(filepath, allow_pickle=True)

    logger.info("GNN keys:  %r" % heim_gnn.files)
    logger.info("NPZ keys:  %r" % npz.files)
    logger.info("Plotting:  %s" % Path(filepath).name)
    total_batches = npz['Xdet'].shape[0]
    how_many_channels = npz['Xdet'].shape[2]
    logger.info(" ...found  %d  channels" % how_many_channels)

    df = confs.PREPARE_DATA.DOWNSAMPLE.new_df
    overlap = int((confs.PREPARE_DATA.SLICING.wlen_seconds -
                   confs.PREPARE_DATA.SLICING.slide_seconds)*df)+1
    slide = int(confs.PREPARE_DATA.SLICING.slide_seconds*df)
    wlen = int(confs.PREPARE_DATA.SLICING.wlen_seconds*df)+1

    # if only_windows_with_picks:
    #     idx_to_plot = [xx for xx, yy in enumerate(npz['Ydet']) if np.max(yy) >= 0.99]
    # else:
    #     idx_to_plot = [xx for xx, yy in enumerate(npz['Ydet'])]
    if only_windows_with_picks:
        logger.warning("Option 'only_windows_with_picks' currently unavailable")


    # ===============================================================
    # ===============================================================
    if reference_point:
        logger.warning("Sorting by REFERENCE:  %.5f  %.5f" % (
                       reference_point[0], reference_point[1]))
        dict_epicenter = gcr.sort_stations_by_distance(
                            reference_point[0], reference_point[1],
                            heim_gnn['stations_coordinate'].item())
    else:
        dict_epicenter = None
    # ===============================================================
    # ===============================================================

    r_ts = gut.__merge_windows__(npz["R"], overlap, method="shift")
    y_ts = gut.__merge_windows__(npz["Ydet"], overlap, method="shift")

    # opst = build_mseed(r_ts, det_ts, heim_gnn, npz["heim_eq_ot"].item(), 1/df)
    # opst.write("%s.mseed" % Path(filepath).stem, format="MSEED")

    # -------- locator
    HG = glctr.HeimdallLocator(
        heim_grid['boundaries'],
        spacing_x=heim_grid['spacing_km'][0],
        spacing_y=heim_grid['spacing_km'][1],
        spacing_z=heim_grid['spacing_km'][2],
        reference_point_lonlat=heim_grid['reference_point']
    )
    # (xgr, ygr, zgr) = HG.get_grid()
    _ = gplt.plot_heimdall_flow_continuous(
            npz["Xdet"], npz["Ydet"], npz["R"], npz["Ydet"],
            [_win[0] for _win in npz["Yloc"]],
            [_win[1] for _win in npz["Yloc"]],
            [_win[2] for _win in npz["Yloc"]],
            HG.get_grid(),
            [0.0 for xx in range(npz["Xdet"].shape[0])],
            heim_gnn,
            HG, (r_ts, y_ts),  # merged windows
            plot_stations=plot_stations,
            order=dict_epicenter,
            reference_locations_1=np.tile(HG.map_on_grid(reference_point),
                                          (npz["Xdet"].shape[0], 1)),
            store_dir="PLOT__%s__.figs" % (Path(filepath).name),
            window_length=wlen,
            sliding=slide,
            suptitle=str(Path(filepath).name),
            show_cfs=False)


if __name__ == "__main__":      # pragma: no cover
    args = parse_cli()
    main(
        gnnpath=args.gnn,
        gridpath=args.grd,
        filepath=args.npz,
        config_file=args.confs,
        reference_point=args.origin,
        # overlap=args.overlap,
        # df=args.df,
        plot_stations=args.plot_stations,
        only_windows_with_picks=args.only_picks,
    )
