#!/usr/bin/env python

import sys
import numpy as np
import obspy
from pathlib import Path
import argparse

from heimdall import plot as gplt
from heimdall import utils as gut
from heimdall import custom_logger as CL

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm

logger = CL.init_logger(Path(sys.argv[0]).name, lvl="INFO")

plt.style.use('seaborn-v0_8-white')  # ggplot / seaborn-v0_8-white
plt.rcParams.update({
    'font.size': 10,
    'axes.labelweight': 'bold',
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'axes.edgecolor': 'gray',
    'axes.linewidth': 0.8,
})


# ====================================================================
# ====================================================================
# ====================================================================

def parse_cli():
    parser = argparse.ArgumentParser(
        description="Plot Heimdall NPZ and export MSEED."
    )

    parser.add_argument(
        "-g", "--graph",
        dest="gnn",
        type=str,
        required=True,
        metavar="GNN_FILE",
        help="Path to the GNN metadata NPZ file."
    )

    parser.add_argument(
        "-e", "--event",
        dest="event",
        type=str,
        required=True,
        metavar="EVENT_FILE",
        help="Path to the Heimdall event NPZ result file."
    )

    parser.add_argument(
        "-o", "--overlap",
        dest="overlap",
        type=float,
        default=4.5,
        metavar="SECONDS",
        help="Sliding window overlap in seconds (default: 4.5)"
    )

    parser.add_argument(
        "-s", "--sampling-rate",
        dest="df",
        type=float,
        default=100.0,
        metavar="HZ",
        help="Sampling frequency in Hz (default: 100.0)"
    )

    return parser.parse_args()


def plot_additional_stations(xmat, ymat, rmat, dets, gnn,
                             store_dir="BatchResults"):
    """
    Plot predictions and real signals for each station in a batch of events.

    Args:
        xmat (np.ndarray): Input features for the batch, shape (B, N, C, T).
        ymat (np.ndarray): Ground truth signals, same shape as xmat.
        rmat (np.ndarray): Raw waveform matrix for plotting, same shape as xmat.
        dets (np.ndarray): Detection/prediction outputs, shape (B, N, C, T).
        gnn (dict): Dictionary with station order mapping and other metadata.
        store_dir (str): Directory to store generated figures.

    Returns:
        List[matplotlib.figure.Figure]: List of generated matplotlib figures.
    """
    logger.warning("Plotting all stations")
    fig_list = []
    for ii in tqdm(range(xmat.shape[0])):
        # Create the figure
        fig = plt.figure(figsize=(19, 7))
        gs = GridSpec(6, 12, figure=fig)

        _, _y, _r, _p = xmat[ii], ymat[ii], rmat[ii], dets[ii]
        condition = np.any(_y >= 0.95, axis=-1)
        _tot_picks = np.sum(np.any(condition, axis=1))

        # Plot the timeseries on the left (rectangular panels) in columns 1-3
        for cc in range(4):
            # 4 columns
            for rr in range(6):
                xx = (cc*6)+rr
                ax = fig.add_subplot(gs[rr, (0+cc*3):(3+cc*3)])
                #
                ax.plot(_r[xx][1] + _r[xx][2], color="darkgray", alpha=0.6, label="real_NE")
                ax.plot(_r[xx][0], color="black", alpha=0.6, label="real_Z")
                ax.set_ylabel([kk for kk, vv in gnn["stations_order"].item().items() if vv==xx][0])
                lf_ax2 = ax.twinx()
                # for _cfs in range(_x[xx].shape[0]):
                #     lf_ax2.plot(_x[xx][_cfs], alpha=0.7, color=f"C{_cfs}", label=f"CF_{_cfs}")
                lf_ax2.plot(_y[xx][0], alpha=0.7, label="Y", color="darkred", ls="--")

                # PREDICTIONS
                lf_ax2.plot(_p[xx][0], alpha=0.7, label="prediction", color="purple")
                if _p[xx].shape[0] > 1:
                    # picker_type
                    lf_ax2.plot(_p[xx][1], alpha=0.7, label="prediction_P", color="darkblue")
                    lf_ax2.plot(_p[xx][2], alpha=0.7, label="prediction_S", color="darkred")

                lf_ax2.set_ylim([-0.2, 1.2])
                # # ax.set_ylabel(sorted_indices_ylabel[idx])  # station name
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                lf_ax2.spines['top'].set_visible(False)
                lf_ax2.spines['right'].set_visible(False)
                if xx != 5:
                    ax.spines['bottom'].set_visible(False)
                    lf_ax2.spines['right'].set_visible(False)

                if cc == 0 and xx == 0:
                    ax.legend(loc='upper left')
                    # lf_ax2.legend(loc='upper right')
                    ax.set_title("Total Stations with signal: %d" % _tot_picks)

        # Adjust layout for better spacing
        plt.tight_layout()

        if store_dir:
            Path(store_dir).mkdir(parents=True, exist_ok=True)
            fig.savefig(str(Path(store_dir) / ("Results_%03d_allStation.png" % ii)), dpi=310)
            fig.savefig(str(Path(store_dir) / ("Results_%03d_allStation.pdf" % ii)), dpi=310)
        #
        fig_list.append(fig)
        plt.close()
    #
    return fig_list


def main(gnn, filepath, overlap=4.5, df=100.0):
    """
    Generate visualizations and MSEED output from a Heimdall event NPZ file.

    Args:
        gnn (str or Path): Path to the Heimdall GNN metadata (.npz).
        filepath (str or Path): Path to the Heimdall event NPZ result file.
        overlap (float): Overlap in seconds between sliding windows.
        df (float): Sampling frequency in Hz.

    Returns:
        None
    """
    npz = np.load(filepath, allow_pickle=True)
    heim_gnn = np.load(gnn, allow_pickle=True)
    logger.info("Plotting:  %s" % (Path(filepath).name))

    r_ts = gut.__merge_windows__(npz["ev_R"], int(overlap*df)+1, method="shift")
    y_ts = gut.__merge_windows__(npz["ev_Y"], int(overlap*df)+1, method="shift")
    det_ts = gut.__merge_windows__(npz["ev_det"], int(overlap*df)+1, method="avg")  # method="max")

    opst = build_mseed(r_ts, det_ts, heim_gnn, npz["heim_eq_ot"].item(), 1/df)
    opst.write("%s.mseed" % Path(filepath).stem, format="MSEED")

    _ = gplt.plot_heimdall_flow_continuous(
            npz["ev_X"], npz["ev_Y"], npz["ev_R"], npz["ev_det"],
            npz["ev_xy_img"],  npz["ev_xz_img"], npz["ev_yz_img"],
            [npz["grid_x"], npz["grid_y"], npz["grid_z"]],
            npz["ev_verdict"],
            heim_gnn,
            npz["HG"].item(), (r_ts, y_ts),  # merged windows
            plot_stations=True,
            order=npz["order"].item(),
            reference_locations_1=np.tile(npz["eq_on_grid"],
                                          (npz["ev_X"].shape[0], 1)),
            store_dir="Event__%s__.figs" % (Path(filepath).name),
            window_length=npz["window_length"].item(),
            sliding=npz["sliding"].item(),
            suptitle="OT: %s  Ml: %.2f\nLON: %.4f  LAT: %.4f  DEP_KM: %.2f  PDF: %.2f (%.2f)" % (
                     npz["heim_eq_ot"], npz["heim_eq_mag"] if npz["heim_eq_mag"] else -9999.9,
                     npz["heim_eq_lon"], npz["heim_eq_lat"], npz["heim_eq_z_on_grid"]*10**-3,
                     npz["median_pdf3d"],  npz["std_pdf3d"]),
            show_cfs=False)

    _ = plot_additional_stations(
            npz["ev_X"], npz["ev_Y"], npz["ev_R"], npz["ev_det"], heim_gnn,
            store_dir="Event__%s__.figs" % (Path(filepath).name))


def build_mseed(inmat_real, inmat_det, gnn, starttime, deltat, tag="DL"):
    """
    Construct an ObsPy Stream from real and predicted signals and export it as MSEED.

    Args:
        inmat_real (np.ndarray): Real waveform signals, shape (N, 3, T) where N = num stations.
        inmat_det (np.ndarray): Predicted detections, shape (N, 3, T).
        gnn (dict): Heimdall GNN metadata including station ordering.
        starttime (UTCDateTime): Event origin time.
        deltat (float): Time increment between samples.
        tag (str): Channel tag prefix (e.g., "DL").

    Returns:
        obspy.core.stream.Stream: Combined ObsPy stream containing real and predicted traces.
    """
    def _convert_to_obspy(data, station, channel, network,
                          location, startt, deltat):
        header = {
            'station': station,
            'network': network,
            'location': location,
            'channel': channel,
            'starttime': startt,
            'delta': deltat,
            'sampling_rate': 1/deltat,
        }
        return obspy.core.trace.Trace(
                    data=data,
                    header=header)

    logger.info("Creating MSEED file")
    assert inmat_real.shape == inmat_det.shape
    st = obspy.core.stream.Stream()
    for mat in range(inmat_real.shape[0]):
        (net, stat) = [kk for kk, vv in gnn["stations_order"].item().items()
                       if vv == mat][0].split(".")

        # ===========  REAL
        # data, station, channel, network, location, startt, deltat
        for _c, chan in enumerate((tag+"Z", tag+"N", tag+"E")):
            obspy_trace = _convert_to_obspy(inmat_real[mat][_c], stat,
                                            chan, net, "", starttime, deltat)
            st.append(obspy_trace)

        # ===========  PRED
        # data, station, channel, network, location, startt, deltat
        for _c, chan in enumerate((tag+"Q", tag+"P", tag+"S")):
            obspy_trace = _convert_to_obspy(inmat_det[mat][_c], stat,
                                            chan, net, "", starttime, deltat)
            st.append(obspy_trace)

    return st


if __name__ == "__main__":
    args = parse_cli()
    main(args.gnn, args.event, overlap=args.overlap, df=args.df)
    logger.info("DONE")
