#!/usr/bin/env python

"""
Heimdall Earthquake Detection Pipeline.

This script loads GNN graph/grid structures, initializes Heimdall models,
processes MSEED streams, predicts events, extracts detections and locations,
and optionally generates plots/statistics.

Usage:
    python script.py -graph path/to/graph.npz -grid path/to/grid.npz -conf config.yml ...

Functions:
    _parse_cli: Parse command line arguments.
    apply_cli_parameters: Set global parameters from parsed args.
    AnalyticClassifier: Classify using analytic spatial consistency & variance.
    __init_model_heimdall__: Load Heimdall model with weights.
    __prepare_batches__: Create DataLoader batches from feature matrices.
    go_with_the_flow: Perform inference using Heimdall model.
    predict: Run prediction pipeline on processed data.
    locate_triggered_events: Calculate event statistics and location.
    find_max_value_with_coordinates: Find max value + coordinates in matrix.
    do_plane_statistics: Filter outliers and calculate stats on plane data.
    triangulate_3d_point: Combine 2D plane results into 3D estimate.
    __extract_folder__: Read .mseed files into Obspy stream.
    __prepare_data_buffer__: Process Obspy stream into Heimdall input.
    main: Main entry point.

"""

import sys
import io
from tqdm import tqdm
import time
from pathlib import Path
import argparse

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
#
import obspy
from obspy import UTCDateTime as UTC

import heimdall
from heimdall import utils as gutl
from heimdall import io as gio
from heimdall import models as gmdl
from heimdall import plot as gplt
from heimdall import locator as glctr
from heimdall import magnitude as gmag
from heimdall import custom_logger as CL

from scipy.ndimage import center_of_mass

__log_name__ = "HEIMDALL_prediction.log"
logger = CL.init_logger(__log_name__, lvl="INFO", log_file=__log_name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def _parse_cli():
    """
    Parse command-line arguments for the Heimdall pipeline.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    p = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        description="Run Heimdall on MSEED files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core files / dirs -------------------------------------------------------
    p.add_argument("-graph", "-g", metavar="GRAPH",
                   help="Heimdall GNN graph file (.npz)")
    p.add_argument("-grid", "-grd", metavar="GRID",
                   help="Heimdall spatial grid file (.npz)")
    p.add_argument("-conf", "-c", dest="config", metavar="YAML",
                   help="Configuration YAML file")
    p.add_argument("-weigths", "-w", dest="modweigths", metavar="MODEL",
                   help="path to *.pt HEIMDALL model weights")
    p.add_argument("-folders", "-f", dest="outfolder", metavar="DIR", nargs='+',
                   help="One or more folder paths containing mseed files to process")
    #
    p.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    p.add_argument("--threshold-prob", type=float, default=0.05,
                   help="Probability threshold for image-location (all planes max. value must exceed this value)")
    p.add_argument("--threshold-coherence", type=float, default=4000.0,
                   help="Coherence threshold for x,y,z dimensions (meters)")
    p.add_argument("--threshold-sta-obs-mag", type=float, default=0.2,
                   help="STA observation magnitude threshold")
    p.add_argument("--buffer-signal", type=int, default=5,
                   help="Number of signal buffer frames to trigger event recording")
    p.add_argument("--buffer-noise", type=int, default=0,
                   help="Number of noise buffer frames allowed")
    p.add_argument("--stream-win-slide", type=int, default=3600*24,
                   help="Stream window slide in seconds")
    p.add_argument("--store-event-plot-data", action="store_true",
                   help="Enable plot generation (default: False)")
    return p


def apply_cli_parameters(args):
    """
    Set global pipeline parameters from parsed command-line arguments.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        bool: True if parameters successfully applied.
    """
    global BATCH_SIZE, HEIMDALL_WEIGHTS, MAKE_PLOTS
    global THRESHOLD_PROB, THRESHOLD_COHERENCE, THRESHOLD_STA_OBS_MAG
    global BUFF_SIG, BUFF_NOISE, STREAM_WIN_SLIDE

    MAKE_PLOTS = args.store_event_plot_data
    BATCH_SIZE = args.batch_size
    THRESHOLD_PROB = args.threshold_prob
    THRESHOLD_COHERENCE = args.threshold_coherence
    THRESHOLD_STA_OBS_MAG = args.threshold_sta_obs_mag
    BUFF_SIG = args.buffer_signal
    BUFF_NOISE = args.buffer_noise
    STREAM_WIN_SLIDE = args.stream_win_slide
    HEIMDALL_WEIGHTS = args.modweigths
    return True


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def AnalyticClassifier(pdf_xy, pdf_xz, pdf_yz,
                       grids_x, grids_y, grids_z,
                       pdf_threshold=0.1,
                       var_threshold=3.0,
                       dist_threshold=3000,
                       extract="max"):  # com (center of mass)
    """
    Perform analytic classification based on spatial consistency, compactness,
    and plane PDF maxima or center-of-mass.

    Args:
        pdf_xy (ndarray): XY plane PDF matrix.
        pdf_xz (ndarray): XZ plane PDF matrix.
        pdf_yz (ndarray): YZ plane PDF matrix.
        grids_x (ndarray): Grid vector X.
        grids_y (ndarray): Grid vector Y.
        grids_z (ndarray): Grid vector Z.
        pdf_threshold (float): Minimum PDF threshold.
        var_threshold (float): Maximum variance threshold.
        dist_threshold (float): Maximum 3D distance threshold.
        extract (str): "max" or "com" to use maximum or center-of-mass.

    Returns:
        tuple: (is_event (bool), diagnostics (dict), coordinates (ndarray))
    """

    # -----------------------------------------------------------------
    def __compute_spatial_variance__(pdf, grid_x, grid_y):
        prob_flat = pdf.flatten()
        prob_flat /= prob_flat.sum()
        mean_x = (grid_x.flatten() * prob_flat).sum()
        mean_y = (grid_y.flatten() * prob_flat).sum()
        var_x = ((grid_x.flatten() - mean_x)**2 * prob_flat).sum()
        mean_y = (grid_y.flatten() * prob_flat).sum()
        var_y = (grid_y.flatten() * prob_flat).sum()
        return mean_x, mean_y, var_x, var_y

    def __compute_3D_spatial_delta__(inarr):
        (xdelta, ydelta, zdelta) = (np.abs(inarr[0][0] - inarr[1][0]),
                                    np.abs(inarr[0][1] - inarr[2][0]),
                                    np.abs(inarr[1][1] - inarr[2][1]))
        lendelta = np.sqrt(xdelta**2 + ydelta**2 + zdelta**2)
        return lendelta
    # -----------------------------------------------------------------

    if extract.lower() in ("com", "center_of_mass"):
        com_xy = center_of_mass(pdf_xy)
        com_xz = center_of_mass(pdf_xz)
        com_yz = center_of_mass(pdf_yz)
    elif extract.lower() in ("max", "maximum"):
        com_xy = np.unravel_index(np.argmax(pdf_xy), pdf_xy.shape)
        com_xz = np.unravel_index(np.argmax(pdf_xz), pdf_xz.shape)
        com_yz = np.unravel_index(np.argmax(pdf_yz), pdf_yz.shape)
    else:
        raise ValueError("Extract par must be either MAX or COM!")

    xy_coords_ongrid = (grids_x[int(np.round(com_xy[0]+1e-8))],
                        grids_y[int(np.round(com_xy[1]+1e-8))])
    xz_coords_ongrid = (grids_x[int(np.round(com_xz[0]+1e-8))],
                        grids_z[int(np.round(com_xz[1]+1e-8))])
    yz_coords_ongrid = (grids_y[int(np.round(com_yz[0]+1e-8))],
                        grids_z[int(np.round(com_yz[1]+1e-8))])

    delta_dist = __compute_3D_spatial_delta__(
            np.array([xy_coords_ongrid, xz_coords_ongrid, yz_coords_ongrid]))
    spatial_consistency = delta_dist < dist_threshold

    # Evaluate MaximumValue of PDFs
    mxy = np.max(pdf_xy)
    mxz = np.max(pdf_xz)
    myz = np.max(pdf_yz)
    marr = np.array([mxy, mxz, myz])
    pdf_sustain = np.all(marr >= pdf_threshold)

    # Evaluate spatial compactness using variance
    var_xy = np.var(pdf_xy)
    var_xz = np.var(pdf_xz)
    var_yz = np.var(pdf_yz)
    avg_variance = np.mean([var_xy, var_xz, var_yz])
    spatial_compactness = avg_variance < var_threshold

    # Final decision
    is_event = spatial_compactness and spatial_consistency and pdf_sustain

    return (is_event,
            {'compactness': spatial_compactness,
             'consistency': spatial_consistency,
             'pdf_sustain': pdf_sustain,
             'avg_variance': avg_variance,
             'max_dist': delta_dist,
             'max_pdf_planes': marr},
            np.array([xy_coords_ongrid, xz_coords_ongrid, yz_coords_ongrid]))


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@  MODEL  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def __init_model_heimdall__(heim_weights, shapes, stations):
    """
    Initialize Heimdall model, load pretrained weights, and move to device.

    Args:
        heim_weights (str): Path to model weights (.pt).
        shapes (list): List of tuples with output plane shapes.
        stations (ndarray): Normalized station coordinates.

    Returns:
        HEIMDALL: Initialized Heimdall model instance.
    """
    stations = torch.as_tensor(stations, dtype=torch.float32)   # on CPU
    model = gmdl.HEIMDALL(stations_coords=stations,
                          location_output_sizes=shapes)
    logger.info("Loading weights: %s" % heim_weights)
    model.load_state_dict(torch.load(heim_weights,
                          map_location=torch.device(device)))
    model.to(device)
    return model


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def __prepare_batches__(xin, edges, weights, batch_size=BATCH_SIZE):
    """
    Prepare PyTorch Geometric DataLoader batches from raw matrices.

    Args:
        xin (ndarray): Feature matrices [B, N, C, T].
        edges (ndarray): Edge indices.
        weights (ndarray): Edge weights.
        batch_size (int): Batch size.

    Returns:
        DataLoader: Loader with Data objects ready for model input.
    """
    def __scale_zero_one(mat, dimension=-1):
        min_vals = torch.min(mat, dim=dimension, keepdim=True).values
        max_vals = torch.max(mat, dim=dimension, keepdim=True).values
        mat = (mat - min_vals) / (max_vals - min_vals + 1e-8)
        return mat

    def __scale_minusone_one(mat, dimension=-1):
        min_vals = torch.min(mat, dim=dimension, keepdim=True).values
        max_vals = torch.max(mat, dim=dimension, keepdim=True).values
        mat = 2 * (mat - min_vals) / (max_vals - min_vals + 1e-8) - 1
        return mat

    production_data_list = [
            # shape: [N, C, T]
            Data(
                x=__scale_minusone_one(
                    torch.tensor(xin[idx], dtype=torch.float32)),
                edge_index=torch.tensor(edges, dtype=torch.long),
                edge_attr=torch.tensor(weights, dtype=torch.float32),
                num_nodes=xin[idx].shape[0]    # N = num.stations
            )
            for idx in range(xin.shape[0])     # E = num events
        ]
    production_loader = DataLoader(
                    production_data_list, batch_size=batch_size, shuffle=False)
    return production_loader


def go_with_the_flow(X, edges, weights,
                     mod_heim,
                     batch_size=BATCH_SIZE):
    """
    Run detection and location inference pipeline on batch data.

    Args:
        X (ndarray): Input data [B, N, C, T].
        edges (ndarray): Graph edge indices.
        weights (ndarray): Graph edge weights.
        mod_heim (HEIMDALL): Heimdall model.
        batch_size (int): Batch size.

    Returns:
        tuple: (detections, (locations_xy, locations_xz, locations_yz), verdicts, proc_time)
    """
    def __unflatten_BN(tensor, B, N):
        # tensor: [B*N, C, F] → [B, N, C, F]
        return tensor.view(B, N, *tensor.shape[1:])

    _startt = time.time()
    mod_heim.eval()

    # --- Prepare the batches
    B, N, _, _ = X.shape
    predict_loader = __prepare_batches__(
        X, edges, weights, batch_size=batch_size)

    detections, locations_xy, locations_xz, locations_yz, verdicts = [], [], [], [], []
    with torch.no_grad():

        for (batch_idx, batch) in enumerate(tqdm(predict_loader)):
            # _inputs = batch.x.to(device)
            # _edges = batch.edge_index.to(device)
            # _weights = batch.edge_attr.to(device)
            _inputs = batch.x.to(device)  # , non_blocking=True)
            _edges = batch.edge_index.to(device)  # , non_blocking=True)
            _weights = batch.edge_attr.to(device)  # , non_blocking=True)
            _batch_vector = batch.batch.to(device)  # , non_blocking=True)

            # 1. Run DETECTOR
            (_detections, (_location_xy, _location_xz, _location_yz), _) = (
                mod_heim(_inputs, _edges, _weights, _batch_vector))

            # 2. Run CLASSIFIER
            _verdict = torch.zeros(_location_xy.shape[0]).to(device)

            # APPEND to CPU to free memory
            detections.append(_detections.cpu())
            locations_xy.append(_location_xy.cpu())
            locations_xz.append(_location_xz.cpu())
            locations_yz.append(_location_yz.cpu())
            verdicts.append(_verdict.cpu())

            # Cleanup
            del _detections, _location_xy, _location_xz, _location_yz, _verdict
            if (batch_idx + 1) % 300 == 0:
                torch.cuda.empty_cache()

    # MERGE
    detections = torch.cat(detections, dim=0)
    detections = __unflatten_BN(detections, B, N)
    locations_xy = torch.cat(locations_xy, dim=0)
    locations_xz = torch.cat(locations_xz, dim=0)
    locations_yz = torch.cat(locations_yz, dim=0)
    verdicts = torch.cat(verdicts, dim=0)

    _endt = time.time()
    proc_time = _endt - _startt
    return (detections, (locations_xy, locations_xz, locations_yz), verdicts,
            proc_time)


def predict(heim_gnn, heim_grid, npz,
            Heimdall, store_dir,
            current_event=0, chunk_idx=0):
    """
    Process one stream chunk: run prediction, extract events, locate events, store outputs.

    Args:
        heim_gnn (dict): Loaded Heimdall graph.
        heim_grid (dict): Loaded Heimdall grid.
        npz (dict): Prepared data in NumPy NPZ format.
        Heimdall (HEIMDALL): Initialized Heimdall model.
        store_dir (str): Directory to store results.
        current_event (int): Starting index for events.
        chunk_idx (int): Index of the current chunk.

    Returns:
        int: Number of detected events.
    """
    STOREDIR = Path(store_dir)
    STOREDIR_STATISTICS = STOREDIR / "HeimdallResults.STATISTICS"
    STOREDIR_FIGURES = STOREDIR / "HeimdallResults.FIGURES"
    STOREDIR_STATISTICS.mkdir(exist_ok=True, parents=True)
    STOREDIR_FIGURES.mkdir(exist_ok=True, parents=True)

    logger.info("HEIMDALL predicting...")

    # ---------- 1. Initialize  LOCATOR + GNN
    EDGES = heim_gnn["edges"]
    WEIGHTS = heim_gnn["weights"]
    HG = glctr.HeimdallLocator(heim_grid['boundaries'],
                               spacing_x=heim_grid['spacing_km'][0],
                               spacing_y=heim_grid['spacing_km'][1],
                               spacing_z=heim_grid['spacing_km'][2],
                               reference_point_lonlat=heim_grid['reference_point'])
    (xgr, ygr, zgr) = HG.get_grid()
    (reflon, reflat) = HG.get_grid_reference()
    HM = gmag.HeimdallMagnitude("/data/m.bagagli/MSEED.FOLDER/NetworksResponse_COSEISMIQ.xml",
                                ref_amps_dict="/data/m.bagagli/MSEED.FOLDER/M1.99_REFERENCE.npz")

    (detections, (locations_xy, locations_xz, locations_yz), verdicts, proc_time) = (
        go_with_the_flow(npz["X"], EDGES, WEIGHTS, Heimdall))

    # Bring everything back to host machine:
    Xplot = npz["X"]
    Yplot = npz["Y"]
    Rplot = npz["R"]
    #
    detections = detections.cpu().numpy()
    locations_xy = locations_xy.cpu().numpy()
    locations_xz = locations_xz.cpu().numpy()
    locations_yz = locations_yz.cpu().numpy()
    verdicts = verdicts.cpu().numpy()

    # -------------------------------------------- 4. Collect Events
    # ------->     FINAL LOCATION / FINALERROR
    # For each window in verdict:
    #   - scan for VERDICT >= 0.5 (signal),
    #       - then extract max-value per plane
    #       - append all of them into matrices
    #       - do statistics on each plane to return the best records among thos
    #           and remove outliers
    #       - collect all triplets per plane (valid) and calculate the std deviation
    assert (detections.shape[0] == locations_xy.shape[0] ==
            locations_yz.shape[0] == verdicts.shape[0])
    logger.info("Extracting EVENTS")

    startt = time.time()

    # -------------------------------------------------
    def clean_memory(thr):
        if len(memory_verdicts) > thr:
            memory_x.pop(0)
            memory_y.pop(0)
            memory_r.pop(0)
            memory_xy.pop(0)
            memory_xz.pop(0)
            memory_yz.pop(0)
            memory_xy_images.pop(0)
            memory_xz_images.pop(0)
            memory_yz_images.pop(0)
            memory_detections.pop(0)
            memory_windows_idx.pop(0)
            memory_windows_start_time.pop(0)
            memory_above_threshold.pop(0)
            memory_verdicts.pop(0)
            memory_verdicts_new.pop(0)  # only if using judger
            memory_verdicts_new_allresults.pop(0)  # --> only if using judger

    # -------------------------------------------------

    # Extract info for main ts  --> DO NOT CHANGE
    overlap = int((npz["slicing"].item()['wlen_seconds'] -
                   npz["slicing"].item()['slide_seconds']) *
                  npz["downsample"].item()['new_df']) + 1
    window_length = int(npz["slicing"].item()['wlen_seconds'] *
                        npz["downsample"].item()['new_df'] + 1)
    increment = int(npz["slicing"].item()['slide_seconds'] *
                    npz["downsample"].item()['new_df'])

    event_detections = []
    buffer_signal_threshold, buffer_noise_threshold = BUFF_SIG, BUFF_NOISE

    memory_windows_idx = []
    memory_windows_start_time = []
    memory_verdicts = []
    memory_verdicts_new = []
    memory_verdicts_new_allresults = []
    memory_detections = []
    #
    memory_x = []
    memory_y = []
    memory_r = []
    #
    memory_xy = []
    memory_yz = []
    memory_xz = []
    #
    memory_xy_images = []
    memory_xz_images = []
    memory_yz_images = []
    #
    memory_above_threshold = []
    all_classifier_time = []

    start_date = obspy.UTCDateTime(str(npz["start_date"]))
    _ev_start_idx, noise_count, signal_count, on_event = 0, 0, 0, False

    for _win in tqdm(range(verdicts.shape[0])):
        # # --> Calculate everything
        _max_xy = find_max_value_with_coordinates(  # is (x,y,val)
                        locations_xy[_win], xgr, ygr)
        _max_xz = find_max_value_with_coordinates(
                        locations_xz[_win], xgr, zgr)
        _max_yz = find_max_value_with_coordinates(
                        locations_yz[_win], ygr, zgr)
        _windows_start_time = start_date + (
                _win * npz["slicing"].item()['slide_seconds'])

        _all_above_threshold = all(x >= THRESHOLD_PROB for x in
                                   (_max_xy[-1], _max_xz[-1], _max_yz[-1]))
        # --> Store in Memory
        memory_x.append(Xplot[_win])
        memory_y.append(Yplot[_win])
        memory_r.append(Rplot[_win])
        memory_xy.append(_max_xy)
        memory_xz.append(_max_xz)
        memory_yz.append(_max_yz)
        memory_xy_images.append(locations_xy[_win])
        memory_xz_images.append(locations_xz[_win])
        memory_yz_images.append(locations_yz[_win])
        memory_detections.append(detections[_win])
        memory_windows_idx.append(_win)
        memory_windows_start_time.append(_windows_start_time)
        memory_above_threshold.append(True if _all_above_threshold else False)
        memory_verdicts.append(verdicts[_win])

        # ==========================  MB - NewClassifier
        (judger, _all_results, _) = AnalyticClassifier(
                                            locations_xy[_win],
                                            locations_xz[_win],
                                            locations_yz[_win],
                                            xgr, ygr, zgr,
                                            pdf_threshold=THRESHOLD_PROB,
                                            dist_threshold=THRESHOLD_COHERENCE)

        memory_verdicts_new.append(judger)
        memory_verdicts_new_allresults.append(_all_results)

        # ============================================================
        # ============================================================
        # ==================   TRIGGER DECLARATION EVENT
        if judger:
            # ---> SIGNAL
            noise_count = 0  # reset noise count
            signal_count += 1
            if (not on_event) and (signal_count >= buffer_signal_threshold):
                # INITIALIZE the EVENT
                on_event = True
                global_idx_event = buffer_signal_threshold
            elif on_event:
                # Keep recording the window-count
                global_idx_event += 1

        else:
            # ---> NOISE
            if on_event:
                global_idx_event += 1  # still keeping track
                noise_count += 1
                if noise_count > buffer_noise_threshold:
                    # Max noise window consecutive found, close event
                    _start_relative_cut = -global_idx_event
                    _end_relative_cut = -noise_count  # or -buffer_noise_threshold
                    _new_event = {
                        "total_windows_event": signal_count,
                        "windows_start_time": memory_windows_start_time[_start_relative_cut:_end_relative_cut],
                        "verdict": np.array(memory_verdicts[_start_relative_cut:_end_relative_cut]),
                        "verdict_new": np.array(memory_verdicts_new[_start_relative_cut:_end_relative_cut]),
                        "verdict_new_allresults": np.array(memory_verdicts_new_allresults[_start_relative_cut:_end_relative_cut]),
                        "X": np.array(memory_x[_start_relative_cut:_end_relative_cut]),
                        "Y": np.array(memory_y[_start_relative_cut:_end_relative_cut]),
                        "R": np.array(memory_r[_start_relative_cut:_end_relative_cut]),
                        "xy": np.array(memory_xy[_start_relative_cut:_end_relative_cut]),
                        "xz": np.array(memory_xz[_start_relative_cut:_end_relative_cut]),
                        "yz": np.array(memory_yz[_start_relative_cut:_end_relative_cut]),
                        "xy_images": np.array(memory_xy_images[_start_relative_cut:_end_relative_cut]),
                        "xz_images": np.array(memory_xz_images[_start_relative_cut:_end_relative_cut]),
                        "yz_images": np.array(memory_yz_images[_start_relative_cut:_end_relative_cut]),
                        "det": np.array(memory_detections[_start_relative_cut:_end_relative_cut])}
                    event_detections.append(_new_event)
                    global_idx_event, noise_count, signal_count, on_event = 0, 0, 0, False
            else:
                # Not on_event yet => any single noise window means
                # we haven't triggered anything, so reset signal count
                signal_count, global_idx_event = 0, 0

        # Clean memory periodically (after every 60 windows)
        clean_memory(60)

    # Check if there’s an ongoing event at the end of the loop
    if on_event:
        global_idx_event += 1  # still keeping track
        _new_event = {
            "total_windows_event": signal_count,
            "windows_start_time": memory_windows_start_time[-global_idx_event:],
            "verdict": np.array(memory_verdicts[-global_idx_event:]),
            "verdict_new": np.array(memory_verdicts_new[-global_idx_event:]),
            "verdict_new_allresults": np.array(memory_verdicts_new_allresults[-global_idx_event:]),
            "X": np.array(memory_x[-global_idx_event:]),
            "Y": np.array(memory_y[-global_idx_event:]),
            "R": np.array(memory_r[-global_idx_event:]),
            "xy": np.array(memory_xy[-global_idx_event:]),
            "xz": np.array(memory_xz[-global_idx_event:]),
            "yz": np.array(memory_yz[-global_idx_event:]),
            "xy_images": np.array(memory_xy_images[-global_idx_event:]),
            "xz_images": np.array(memory_xz_images[-global_idx_event:]),
            "yz_images": np.array(memory_yz_images[-global_idx_event:]),
            "det": np.array(memory_detections[-global_idx_event:])}
        event_detections.append(_new_event)

    # --> Reset for GOOD
    global_idx_event, noise_count, signal_count, on_event == 0, 0, 0, False

    # # ============================================================
    # # ============================================================
    # # ========================   STORE ALL TRIGGERED EVENTS  (debug only)
    # if STORE_ALL_TRIGGERED_EVENT:
    #     np.savez("%s/AllTriggeredEvents_%04d.npz" % (str(STOREDIR), chunk_idx),
    #              events=event_detections)
    #     # It will save all the continuous data for plotting routines later on
    #     # ... suggested only for debugging porpuses
    #     np.savez(
    #        "%s/AllTriggeredEvents_%04d_PLOT.npz" % (str(STOREDIR), chunk_idx),
    #        ev_X=memory_x, ev_Y=memory_y, ev_R=memory_r, ev_det=memory_detections,
    #        ev_xy_img=memory_xy_images, ev_xz_img=memory_xz_images,
    #        ev_yz_img=memory_yz_images,
    #        ev_verdict=memory_verdicts, ev_verdict_new=memory_verdicts_new,
    #        ev_verdict_new_allresults=memory_verdicts_new_allresults,
    #        grid_x=xgr, grid_y=ygr, grid_z=zgr, gnn=heim_gnn, HG=HG,
    #        order=None, eq_on_grid=None,
    #        window_length=window_length, sliding=increment, overlap=overlap,
    #        heim_eq_ot=memory_windows_start_time[0], heim_eq_lon=None,
    #        heim_eq_lat=None, heim_eq_z_on_grid=None,
    #        heim_eq_mag=None, heim_eq_mag_err=None,
    #        mean_pdf3d=None, std_pdf3d=None,
    #        median_pdf3d=None, mad_pdf3d=None)

    endt = time.time()
    logger.info("Total time EVENT EXTRACTION:  %.2f minutes" % (
                                                        (endt-startt)/60.0))

    # -------------------------------------------- 5. LOCATE + [plot]
    for ev_idx, ev in enumerate(event_detections):
        event_store_idx = ev_idx + current_event
        #
        ((xy_raw_mean, xy_raw_median, xy_raw_mad, xy_num_outliers,
          xy_perc_outliers, xy_clean_mean, xy_clean_median, xy_clean_mad),
         (xz_raw_mean, xz_raw_median, xz_raw_mad, xz_num_outliers,
          xz_perc_outliers, xz_clean_mean, xz_clean_median, xz_clean_mad),
         (yz_raw_mean, yz_raw_median, yz_raw_mad, yz_num_outliers,
          yz_perc_outliers, yz_clean_mean, yz_clean_median, yz_clean_mad),
         heim_eq_lat, heim_eq_lon, heim_eq_x_on_grid, heim_eq_y_on_grid,
         heim_eq_z_on_grid, mean_pdf3d, median_pdf3d, std_pdf3d, mad_pdf3d,
         heim_eq_ot) = locate_triggered_events(ev, HG)

        mag_in_R = gutl.__merge_windows__(ev["R"], overlap, method="shift")
        mag_in_D = gutl.__merge_windows__(ev["det"], overlap, method="max")
        recording_stations = []
        for _name, _name_idx in heim_gnn["stations_order"].item().items():
            if np.max(mag_in_D[_name_idx, 0, :]) > THRESHOLD_STA_OBS_MAG:
                recording_stations.append(
                        (_name, mag_in_R[_name_idx, :, :])
                    )

        (heim_eq_mag, heim_eq_mag_err,
         heim_eq_mag_n_valid, heim_eq_mag_n_all) = HM.calculate_magnitude(
            heim_eq_lon, heim_eq_lat, recording_stations,
            event_time=heim_eq_ot, epi_thr=25, method="relative")
        logger.state(" HEIMDALL EVENT:  %.5f  %.5f  %.2f  Ml %.2f (uncert.  %.2f)  [%s]" % (
                      heim_eq_lon if heim_eq_lon else -9999.9,
                      heim_eq_lat if heim_eq_lat else -9999.9,
                      heim_eq_z_on_grid*10**-3 if heim_eq_z_on_grid else -9999.9,
                      heim_eq_mag if heim_eq_mag else -9999.9,
                      heim_eq_mag_err if heim_eq_mag_err else -9999.9,
                      heim_eq_ot))

        # ===============================================   STORING STATS
        np.savez("%s/Event_%04d_stats" % (str(STOREDIR_STATISTICS),
                                          event_store_idx),
                 eqtag="heim_%04d" % event_store_idx,
                 total_windows_event=ev["total_windows_event"],
                 #
                 xy_raw_mean=xy_raw_mean,
                 xy_raw_median=xy_raw_median,
                 xy_raw_mad=xy_raw_mad,
                 xy_num_outliers=xy_num_outliers,
                 xy_perc_outliers=xy_perc_outliers,
                 xy_clean_mean=xy_clean_mean,
                 xy_clean_median=xy_clean_median,
                 xy_clean_mad=xy_clean_mad,
                 #
                 xz_raw_mean=xz_raw_mean,
                 xz_raw_median=xz_raw_median,
                 xz_raw_mad=xz_raw_mad,
                 xz_num_outliers=xz_num_outliers,
                 xz_perc_outliers=xz_perc_outliers,
                 xz_clean_mean=xz_clean_mean,
                 xz_clean_median=xz_clean_median,
                 xz_clean_mad=xz_clean_mad,
                 #
                 yz_raw_mean=yz_raw_mean,
                 yz_raw_median=yz_raw_median,
                 yz_raw_mad=yz_raw_mad,
                 yz_num_outliers=yz_num_outliers,
                 yz_perc_outliers=yz_perc_outliers,
                 yz_clean_mean=yz_clean_mean,
                 yz_clean_median=yz_clean_median,
                 yz_clean_mad=yz_clean_mad,
                 #
                 heim_eq_lon=heim_eq_lon,
                 heim_eq_lat=heim_eq_lat,
                 heim_eq_x=heim_eq_x_on_grid,
                 heim_eq_y=heim_eq_y_on_grid,
                 heim_eq_z=heim_eq_z_on_grid,  # meters
                 heim_eq_ot=heim_eq_ot,
                 heim_eq_mag=heim_eq_mag,
                 heim_eq_mag_err=heim_eq_mag_err,
                 heim_eq_mag_n_valid=heim_eq_mag_n_valid,
                 heim_eq_mag_n_all=heim_eq_mag_n_all,
                 #
                 heim_eq_pdf_3Dmean=mean_pdf3d,
                 heim_eq_pdf_3Dmedian=median_pdf3d,
                 heim_eq_pdf_3Dstd=std_pdf3d,
                 heim_eq_pdf_3Dmad=mad_pdf3d,
                 #
                 proc_time_heim=proc_time)

        # ===============================================   STORE-PLOTS
        if MAKE_PLOTS:
            store_file_path = "%s/Event_%04d_PlotInfo.npz" % (
                                    str(STOREDIR_FIGURES), event_store_idx)
            #
            dict_epicenter = HG.sort_stations_by_distance(
                                    heim_eq_lon, heim_eq_lat,
                                    heim_gnn['stations_coordinate'].item())

            np.savez(
                store_file_path,
                ev_X=ev["X"], ev_Y=ev["Y"], ev_R=ev["R"], ev_det=ev["det"],
                ev_xy_img=ev["xy_images"], ev_xz_img=ev["xz_images"],
                ev_yz_img=ev["yz_images"],
                ev_verdict=ev["verdict"], ev_verdict_new=ev["verdict_new"],
                ev_verdict_new_allresults=ev["verdict_new_allresults"],
                grid_x=xgr, grid_y=ygr, grid_z=zgr, gnn=heim_gnn, HG=HG,
                order=dict_epicenter, eq_on_grid=[
                    heim_eq_x_on_grid, heim_eq_y_on_grid, heim_eq_z_on_grid],
                window_length=window_length, sliding=increment, overlap=overlap,
                heim_eq_ot=heim_eq_ot, heim_eq_lon=heim_eq_lon,
                heim_eq_lat=heim_eq_lat, heim_eq_z_on_grid=heim_eq_z_on_grid,
                heim_eq_mag=heim_eq_mag, heim_eq_mag_err=heim_eq_mag_err,
                mean_pdf3d=mean_pdf3d, std_pdf3d=std_pdf3d,
                median_pdf3d=median_pdf3d, mad_pdf3d=mad_pdf3d)

        # # ===============================================   MAKE-PLOTS (debug, on the fly)
        # if True:
        #     dict_epicenter = HG.sort_stations_by_distance(
        #                             heim_eq_lon, heim_eq_lat,
        #                             heim_gnn['stations_coordinate'].item())
        #     #
        #     r_ts = gutl.__merge_windows__(ev["R"], overlap, method="shift")
        #     y_ts = gutl.__merge_windows__(ev["Y"], overlap, method="shift")
        #     logger.info(" Plotting")
        #     try:
        #         _ = gplt.plot_heimdall_flow_continuous(
        #                 ev["X"], ev["Y"], ev["R"],
        #                 ev["det"], ev["xy_images"], ev["xz_images"], ev["yz_images"],
        #                 [xgr, ygr, zgr], ev["verdict"],
        #                 heim_gnn, HG, (r_ts, y_ts),  # merged windows
        #                 plot_stations=True,
        #                 order=dict_epicenter,
        #                 reference_locations_1=np.tile((heim_eq_x_on_grid,
        #                                                heim_eq_y_on_grid,
        #                                                heim_eq_z_on_grid),
        #                                               (Xplot.shape[0], 1)),
        #                 store_dir="%s/Event_%04d_figs" % (str(STOREDIR_FIGURES),
        #                                                   event_store_idx),
        #                 window_length=window_length,
        #                 sliding=increment,
        #                 suptitle="OT: %s  Ml: %.2f\nLON: %.4f  LAT: %.4f  DEP_KM: %.2f  PDF: %.2f (%.2f)" % (
        #                          heim_eq_ot, heim_eq_mag if heim_eq_mag else -9999.9,
        #                          heim_eq_lon, heim_eq_lat, heim_eq_z_on_grid*10**-3,
        #                          median_pdf3d, std_pdf3d)
        #                 )
        #     except:
        #         breakpoint()

    logger.state("Found  %3d  events" % len(event_detections))

    return len(event_detections)


def locate_triggered_events(ev, heim_locator):
    """
    Locate triggered events using plane statistics and triangulation.

    Args:
        ev (dict): Event dictionary containing extracted data.
        heim_locator (HeimdallLocator): Heimdall locator object.

    Returns:
        tuple: Event statistics, coordinates, and uncertainty estimates.
    """

    # ----------------  Once you have an event,
    # XY
    (xy_raw_mean, xy_raw_median, xy_raw_mad, xy_num_outliers, xy_perc_outliers,
     xy_clean_mean, xy_clean_median, xy_clean_mad, XY_ARR) = do_plane_statistics(
                                                                ev["xy"])
    # XZ
    (xz_raw_mean, xz_raw_median, xz_raw_mad, xz_num_outliers, xz_perc_outliers,
     xz_clean_mean, xz_clean_median, xz_clean_mad, XZ_ARR) = do_plane_statistics(
                                                                ev["xz"])
    # YZ
    (yz_raw_mean, yz_raw_median, yz_raw_mad, yz_num_outliers, yz_perc_outliers,
     yz_clean_mean, yz_clean_median, yz_clean_mad, YZ_ARR) = do_plane_statistics(
                                                                ev["yz"])

    # TRIANGULATE for BEST X/Y/Z / VAL estimate
    # Final 3D coordinates by averaging the two sources for each coordinate
    (heim_eq_x_on_grid, heim_eq_y_on_grid, heim_eq_z_on_grid,
     mean_pdf3d, median_pdf3d, std_pdf3d, mad_pdf3d) = triangulate_3d_point(
                                                XY_ARR, XZ_ARR, YZ_ARR)
    [(heim_eq_lat, heim_eq_lon, _), ] = heim_locator.grid.convert_cart_list(
                                                  [(heim_eq_x_on_grid,
                                                    heim_eq_y_on_grid,
                                                    heim_eq_z_on_grid),])

    # ORIGIN TIME  ---> Brutal estimation
    heim_eq_ot = ev["windows_start_time"][0]

    return (
        (xy_raw_mean, xy_raw_median, xy_raw_mad, xy_num_outliers,
         xy_perc_outliers, xy_clean_mean, xy_clean_median, xy_clean_mad),
        (xz_raw_mean, xz_raw_median, xz_raw_mad, xz_num_outliers,
         xz_perc_outliers, xz_clean_mean, xz_clean_median, xz_clean_mad),
        (yz_raw_mean, yz_raw_median, yz_raw_mad, yz_num_outliers,
         yz_perc_outliers, yz_clean_mean, yz_clean_median, yz_clean_mad),
        heim_eq_lat, heim_eq_lon, heim_eq_x_on_grid, heim_eq_y_on_grid,
        heim_eq_z_on_grid, mean_pdf3d, median_pdf3d, std_pdf3d, mad_pdf3d,
        heim_eq_ot
    )


def find_max_value_with_coordinates(matrix, x_grid, y_grid):
    """
    Given a 2D matrix and X/Y grid vectors, return the x, y, and the max value in the matrix.

    Parameters:
        matrix (2D array): The 2D matrix of values.
        x_grid (1D array): The vector representing the X-axis coordinates.
        y_grid (1D array): The vector representing the Y-axis coordinates.

    Returns:
        tuple: (x, y, max_value) - x, y coordinates corresponding to max value and the max value.
    """
    # Find the index of the maximum value in the matrix
    max_index = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)

    # Get the corresponding x and y coordinates
    try:
        # max_x = x_grid[max_index[1]]  # X is along the second axis (columns)
        # max_y = y_grid[max_index[0]]  # Y is along the first axis (rows)
        max_x = x_grid[max_index[0]]
        max_y = y_grid[max_index[1]]
    except:
        logger.warning("Probably you have loaded the wrong HeimGRID")
        breakpoint()

    # Get the maximum value
    max_value = matrix[max_index]

    return (max_x, max_y, max_value)


def do_plane_statistics(inarr, outlier_threshold=3.0, epsilon=1e-6):
    # Calculate the median of the points
    raw_mean = np.mean(inarr, axis=0)
    raw_median = np.median(inarr, axis=0)
    raw_mad = np.median(np.abs(inarr - raw_median), axis=0)

    # Small constant to avoid division by near-zero
    raw_mad = np.where(raw_mad < epsilon, epsilon, raw_mad)

    # Find the rows where the deviation exceeds the threshold for each column
    _deviations = np.abs(inarr - raw_median)

    # SI FILTRA BASANDOSI SOLO SULLA X/Y, non sulla PDF !!! (anche se alla fine rimuoviamo l'intera riga)
    _valid_mask = (_deviations[:, 0] < outlier_threshold * raw_mad[0]) & \
                  (_deviations[:, 1] < outlier_threshold * raw_mad[1])
    _valid_count = np.sum(_valid_mask)
    _outliers_count = inarr.shape[0] - _valid_count
    _perc_outliers = _outliers_count / inarr.shape[0]

    clean_measurements = inarr[_valid_mask]
    clean_mean = np.mean(clean_measurements, axis=0)
    clean_median = np.median(clean_measurements, axis=0)
    clean_mad = np.median(np.abs(clean_measurements - clean_median), axis=0)

    return (raw_mean, raw_median, raw_mad, _outliers_count, _perc_outliers,
            clean_mean, clean_median, clean_mad, clean_measurements)


def triangulate_3d_point(xy_points, xz_points, yz_points):
    """
    Combine the cleanest 2D points from XY, XZ, and YZ planes into a
    single 3D point (X, Y, Z).

    Parameters:
        xy_points (2D array): Clean points from the XY plane, shape (N, 2, N)
            where columns are (X, Y, PDF).
        xz_points (2D array): Clean points from the XZ plane, shape (N, 2, N)
            where columns are (X, Z, PDF).
        yz_points (2D array): Clean points from the YZ plane, shape (N, 2, N)
            where columns are (Y, Z, PDF).

    Returns:
        tuple: A single estimate for (X, Y, Z) and
            (3Derr mean, 3Derr median 3Derr std) -volumetric- of all valid plane measurement.
    """
    # Calculate median or mean for X, Y, Z from each plane
    # X: from xy_points and xz_points
    x_from_xy = np.median(xy_points[:, 0])
    x_from_xz = np.median(xz_points[:, 0])

    # Y: from xy_points and yz_points
    y_from_xy = np.median(xy_points[:, 1])
    y_from_yz = np.median(yz_points[:, 0])

    # Z: from xz_points and yz_points
    z_from_xz = np.median(xz_points[:, 1])
    z_from_yz = np.median(yz_points[:, 1])

    # Final 3D coordinates by averaging the two sources for each coordinate
    final_x = (x_from_xy + x_from_xz) / 2
    final_y = (y_from_xy + y_from_yz) / 2
    final_z = (z_from_xz + z_from_yz) / 2

    # Npow calculate the PDF stats
    de = np.concatenate((xy_points, xz_points, yz_points), axis=0)
    pdf3d = de[:, -1]
    mean_pdf3d, median_pdf3d = np.mean(pdf3d), np.median(pdf3d)
    std_pdf3d, mad_pdf3d = np.std(pdf3d), np.median(np.abs(pdf3d - median_pdf3d))

    return (final_x, final_y, final_z, mean_pdf3d, median_pdf3d, std_pdf3d, mad_pdf3d)


# ================================================================
# ================================================================
# ================================================================
# ================================================================
# ================================================================

def __extract_folder__(mseed_path):
    """
    Extract all MSEED files in a folder into a single Obspy stream.

    Args:
        mseed_path (Path): Path to folder containing .mseed files.

    Returns:
        obspy.Stream: Combined stream of all traces.
    """
    allnet_st = obspy.core.stream.Stream()
    for ms in mseed_path.glob("*.mseed"):
        logger.state("Reading %s" % str(ms.name))
        _st = obspy.read(ms)
        allnet_st += _st
    #
    return allnet_st


def __prepare_data_buffer__(inst, stat_dict_order, confs,
                            eq_tag="UNKNOWN"):
    """
    Prepare Heimdall-compatible NPZ buffer from Obspy stream.

    Args:
        inst (obspy.Stream): Input stream.
        stat_dict_order (dict): Station ordering.
        confs (AttributeDict): Configuration parameters.
        eq_tag (str): Earthquake tag.

    Returns:
        dict: NPZ dictionary with data ready for Heimdall.
    """
    logger.state("START:  %s  - END:  %s" % (
                    inst[0].stats.starttime, inst[0].stats.endtime))

    # Check time-length:
    _all_st_time_lengths = [
        (_tr.stats.endtime - _tr.stats.starttime) < 60.0 for _tr in inst]
    if (sum(_all_st_time_lengths)/len(inst)) > 0.95:
        logger.error("More than 95% traces in stream are less than a minute!")
        return None

    heim = gio.HeimdallDataLoader(inst, order=stat_dict_order)

    (X, Y, R) = heim.prepare_data_real(
                    downsample=confs.DOWNSAMPLE,
                    slicing=confs.SLICING,
                    create_labels=False,
                    debug_plot=False)

    # Store the data in a memory buffer instead of a file
    buffer = io.BytesIO()
    np.savez(buffer,
             X=X, Y=Y, R=R,
             sampling_rate=confs.DOWNSAMPLE['new_df'],
             order=stat_dict_order,
             slicing=confs.SLICING,
             downsample=confs.DOWNSAMPLE,
             cfconf=confs.CFCONF,
             start_date=str(heim.stream_start_date)[:-1],
             end_date=str(heim.stream_end_date)[:-1],
             eq_tag=eq_tag,
             version=heimdall.__version__)
    # Rewind the buffer so it can be read from the beginning
    buffer.seek(0)
    # Now you can load the NPZ data directly from the memory buffer without writing to disk
    npz = np.load(buffer, allow_pickle=True)
    return npz


def main(args):
    """
    Main driver function to run Heimdall detection pipeline on input data.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    StartProgram = UTC()
    logger.state("Starting HEIMDALL @  %s" % StartProgram.strftime("%Y-%m-%d %H:%M:%S"))

    # =====================================================
    # ====================================  IMPORT VARS

    heim_gnn = np.load(args.graph, allow_pickle=True)
    stats_coords = build_coord_matrix(heim_gnn["stations_coordinate"].item(),
                                      heim_gnn["stations_order"].item())  # (36,3)
    stats_coords_norm, stats = normalise_station_coords(stats_coords,
                                                        return_stats=True,
                                                        zero_one=False)

    heim_grid = np.load(args.grid, allow_pickle=True)
    CONFIG = gio.read_configuration_file(args.config, check_version=True)

    # =====================================================
    # =================================  Initialize  MODEL
    _temp_hg = glctr.HeimdallLocator(
                        heim_grid['boundaries'],
                        spacing_x=heim_grid['spacing_km'][0],
                        spacing_y=heim_grid['spacing_km'][1],
                        spacing_z=heim_grid['spacing_km'][2],
                        reference_point_lonlat=heim_grid['reference_point'])
    (xgr, ygr, zgr) = _temp_hg.get_grid()
    shapes = [(len(xgr), len(ygr)), (len(xgr), len(zgr)), (len(ygr), len(zgr))]

    logger.info(" Initializing MODELS")
    Heimdall = __init_model_heimdall__(HEIMDALL_WEIGHTS, shapes, stats_coords_norm)

    # =====================================================
    # ====================================  PREPARE DATA
    for fold in args.folders:
        st = __extract_folder__(fold)

        # ---> SLICE STREAM and create NPZ
        tot_events = 0
        for xx, window in enumerate(st.slide(window_length=STREAM_WIN_SLIDE+5,
                                             step=STREAM_WIN_SLIDE,
                                             include_partial_windows=True)):
            StartChunkTime = time.time()
            #
            npzfile = __prepare_data_buffer__(
                            window,
                            heim_gnn["stations_order"].item(),
                            CONFIG.PREPARE_DATA)
            store_dir = "%s_PREDS" % fold

            if not npzfile:
                # SKIP CHUNK, quality checks in __prepare_data_buffer__ failed!
                logger.error("SKIP CHUNK!")
                EndChunkTime = time.time()
                logger.state("CHUNK %d DONE --> Running Time:  %.2f min.\n\n" % (
                                xx, (EndChunkTime-StartChunkTime)/60.0))
                continue

            # =================================   PREDICT
            _out_ev = predict(heim_gnn, heim_grid, npzfile,
                              Heimdall, store_dir, tot_events, xx)
            tot_events += _out_ev
            #
            EndChunkTime = time.time()
            logger.state("CHUNK %d DONE --> Running Time:  %.2f min.\n\n" % (
                            xx, (EndChunkTime-StartChunkTime)/60.0))
    #
    EndProgram = UTC()
    logger.state("Finishing HEIMDALL @  %s" % EndProgram.strftime("%Y-%m-%d %H:%M:%S"))
    logger.state("Total running time:  %.2f hr" % ((EndProgram - StartProgram)/3600.0))


def build_coord_matrix(stations_coordinate: dict, stations_order: dict,
                       dtype=np.float32) -> np.ndarray:
    """
    Build ordered matrix of station coordinates from dictionaries.

    Args:
        stations_coordinate (dict): Station ID to (lon, lat, elev).
        stations_order (dict): Station ID to index mapping.
        dtype (type): Output NumPy dtype.

    Returns:
        np.ndarray: Ordered coordinate matrix [N, 3].
    """
    if len(stations_coordinate) != len(stations_order):
        raise ValueError("The two dictionaries list different numbers of stations.")

    n_sta = len(stations_order)
    coords = np.empty((n_sta, 3), dtype=dtype)

    for sta, idx in stations_order.items():
        try:
            coords[idx] = stations_coordinate[sta]
        except KeyError:
            raise KeyError(f"Coordinate for station '{sta}' not found.")

    return coords


def normalise_station_coords(coord_mat, *,
                             centre=True,
                             std_scale=True,
                             return_stats=False,
                             zero_one=False):
    """
    Normalize station coordinate matrix.

    Args:
        coord_mat (ndarray): [N, 3] coordinate matrix.
        centre (bool): Center coordinates to zero mean.
        std_scale (bool): Scale to unit standard deviation.
        return_stats (bool): Also return centering/scaling stats.
        zero_one (bool): Rescale to [0,1].

    Returns:
        tuple: Normalized coordinates, optionally stats dict.
    """

    lon = coord_mat[:, 0]
    lat = coord_mat[:, 1]
    z_m = coord_mat[:, 2]

    # 1) reference (mean) lat/lon
    lon0 = lon.mean()
    lat0 = lat.mean()

    # 2) degrees → kilometres, flat-earth
    km_per_deg_lat = 111.32
    km_per_deg_lon = km_per_deg_lat * np.cos(np.deg2rad(lat0))

    x_km = (lon - lon0) * km_per_deg_lon
    y_km = (lat - lat0) * km_per_deg_lat
    z_km = z_m / 1000.0

    coords = np.stack([x_km, y_km, z_km], axis=1).astype(np.float32)

    # 3) centre to zero mean (already true for x/y, do z if asked)
    if centre:
        coords -= coords.mean(axis=0, keepdims=True)

    # 4) scale to unit std-dev
    if std_scale:
        stds = coords.std(axis=0, keepdims=True) + 1e-9
        coords /= stds
    else:
        stds = np.ones((1, 3), dtype=np.float32)

    if zero_one:
        coords = (
            coords - coords.min(0, keepdim=True)[0]) / (
                coords.max(0, keepdim=True)[0] - coords.min(
                    0, keepdim=True)[0] + 1e-6)

    if return_stats:
        return coords, {'centroid': (lon0, lat0, z_m.mean()), 'std': stds.squeeze()}
    return coords


# ================================================================
# ================================================================
# ================================================================
# ================================================================
# ================================================================

if __name__ == "__main__":
    parser = _parse_cli()
    if len(sys.argv) == 1 or sys.argv[1].lower() in ("-h", "--help"):
        parser.print_help(sys.stdout)
        sys.exit()
    #
    args = parser.parse_args()
    logger.info("PARSER INPUTS")
    for k, v in vars(args).items():
        logger.info(f"{k}: {v}")
    #
    main(args)
