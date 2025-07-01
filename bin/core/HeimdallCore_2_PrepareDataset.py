#!/usr/bin/env python

import sys
import obspy
import numpy as np
import pandas as pd
import random
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed  # ProcessPoolExecutor,
import argparse
#
from heimdall import __version__
from heimdall import io as gio
from heimdall import locator as glctr
from heimdall import custom_logger as CL

logger = CL.init_logger(Path(sys.argv[0]).name, lvl="INFO",
                        log_file="PrepareDataset.log")

KM = 0.001
MT = 1000.0


# ==================================================================
# ==================================================================
# ==================================================================

def _parse_cli():
    """
    Parse command-line arguments.
    """
    p = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        description="Prepare Heimdall training dataset from MSEED files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core files / dirs -------------------------------------------------------
    p.add_argument("-graph", "-g", metavar="GRAPH", type=str,
                   help="Heimdall GNN graph file (.npz)")
    p.add_argument("-grid", "-grd", metavar="GRID", type=str,
                   help="Heimdall spatial grid file (.npz)")
    p.add_argument("-conf", "-c", dest="config", metavar="YAML", type=str,
                   help="Configuration YAML file")
    p.add_argument("-infold", "-in", dest="infolder", metavar="DIR", type=str,
                   help="Input folder with raw *.mseed data")
    p.add_argument("-outfold", "-out", dest="outfolder", metavar="DIR", type=str,
                   default="MSEED.TRAIN",
                   help="Output folder for processed *.npz files (auto-created)")
    p.add_argument("-picks", "-p", metavar="CSV", default="AllPicks.csv", type=str,
                   help="CSV containing all picks")

    # Detection
    p.add_argument("--picks-to-declare-event", metavar="N", type=int,
                   default=5,
                   help="Minimum number of picks in window required to declare an event label")
    p.add_argument("--mute-nodes", action="store_true",
                   help="Randomly zero out some stations during training")
    p.add_argument("--max-muted", metavar="N", type=int,
                   default=11,
                   help="Max number of stations to mute when --mute-nodes is set")
    p.add_argument("--seed", metavar="INT", type=int,
                   default=42,
                   help="Random seed for all stochastic steps")

    # Parallel processing -----------------------------------------------------
    p.add_argument("--parallel", action="store_true",
                   help="Process events in parallel")
    p.add_argument("--workers", metavar="N", type=int,
                   default=8,
                   help="Number of worker threads / processes for --parallel")

    return p


def __prepare_data__(inst, stat_dict_order, confs, pickdf,
                     save_tag, store_dir="PREDICTION", eq_tag="UNKNOWN"):

    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    heim = gio.HeimdallDataLoader(inst, order=stat_dict_order)

    (X, Y, R) = heim.prepare_data_real(
                    downsample=confs.DOWNSAMPLE,
                    slicing=confs.SLICING,
                    create_labels=pickdf,
                    debug_plot=False)

    np.savez("%s/predict_data.%s" % (str(store_dir), save_tag),
             X=X, Y=Y, R=R,
             sampling_rate=confs.DOWNSAMPLE['new_df'],
             order=stat_dict_order,
             slicing=confs.SLICING,
             downsample=confs.DOWNSAMPLE,
             # cfconf=confs.CFCONF,
             start_date=str(heim.stream_start_date)[:-1],
             end_date=str(heim.stream_end_date)[:-1],
             eq_tag=eq_tag,
             version=__version__)

    return True


# ==================================================================
# ==================================================================
# ==================================================================


def process_file(fp, grid_path,
                 randomly_mute_nodes, max_muted, picks_to_declare_event,
                 source_noise, seed):
    def __count_event_obs__(inmat, thr=0.9, chan=0):
        pcnt = np.any(inmat[:, chan, :] > thr, axis=1)
        return np.sum(pcnt)

    max_pcnt = max(source_noise.keys())

    # ==========================================================  START
    np.random.seed(seed)  # Ensure consistent randomness

    # 01234567890123
    # training_data.63.99195_-21.05188_4500.00_6.0_9.0_0.0_90.0_-180.0_RightStrikeSlip.ONLYNOISE.npz
    # fff = Path(fp).name[14:].split("_")
    # eq_fields = [float(_ele) for _x, _ele in enumerate(fff) if _x < len(fff)-1]
    de = np.load(fp, allow_pickle=True)
    heim_grid = np.load(grid_path, allow_pickle=True)

    big_x_det, big_y_det = [], []
    big_x_loc, big_y_loc = [], []
    big_x_cls, big_y_cls = [], []
    big_r, big_tag_geo, big_tag_xyz = [], [], []
    big_pick_count, big_sourcestd_noise = [], []

    for nwin in range(de["X"].shape[0]):
        _R = de["R"][nwin]

        # ---------  1a. DETECTOR
        _X_det = de["X"][nwin]
        _Y_det = de["Y"][nwin]

        assert _X_det.shape[0] == _Y_det.shape[0]  # nodes
        assert _X_det.shape[-1] == _Y_det.shape[-1]  # samples

        if randomly_mute_nodes:
            _nmute = np.random.randint(0, max_muted + 1)
            if _nmute != 0:
                _nmute_idx = np.random.choice(np.arange(_R.shape[0]),
                                              size=_nmute, replace=False)
                logger.debug("Muting %d nodes: %s" % (_nmute, _nmute_idx))
                _X_det[_nmute_idx, :, :] = 0.0
                _Y_det[_nmute_idx, :, :] = 0.0
                # # Spare the REALS to be able to merge at laterstage
                # _R[_nmute_idx, :, :] = 0.0

        # ---------  1b. DETECTOR countPicks + APPEND
        tot_pcnt = __count_event_obs__(_Y_det, thr=0.95, chan=0)
        if tot_pcnt > max_pcnt:
            tot_pcnt = max_pcnt
        #
        big_x_det.append(_X_det)
        big_y_det.append(_Y_det)
        big_r.append(_R)

        # -----------------------
        # ---------  LOCATOR
        HG = glctr.HeimdallLocator(heim_grid['boundaries'],
                                   spacing_x=heim_grid['spacing_km'][0],
                                   spacing_y=heim_grid['spacing_km'][1],
                                   spacing_z=heim_grid['spacing_km'][2],
                                   reference_point_lonlat=heim_grid['reference_point'])

        (xgr, ygr, zgr) = HG.get_grid()
        (reflon, reflat) = HG.get_grid_reference()

        # ['predict', 'data.GOOD', 'Ev000', '2018-12-01T11:06:59.401372',
        #  '-21.24090', '64.04967', '4.17', '1.39', '0.072.mseed.npz']
        LAT = float(Path(fp).name.split("_")[5])
        LON = float(Path(fp).name.split("_")[4])
        DEP = float(Path(fp).name.split("_")[6]) * 10**3  # meters
        MAG = float(Path(fp).name.split("_")[7])
        # SNR = float(Path(fp).name.split("_")[5])
        # STRIKE = float(Path(fp).name.split("_")[6])
        # DIP = float(Path(fp).name.split("_")[7])
        # RAKE = float(Path(fp).name.split("_")[8])

        (label_pdf, xy_image, xz_image, yz_image,
         (eqlon_ongrid, eqlat_ongrid, eqdep)) = HG.create_source_images(
                (LON, LAT, DEP*10**-3),
                std_err_km=source_noise[tot_pcnt][0],
                noise_max=source_noise[tot_pcnt][1],
                source_noise=source_noise[tot_pcnt][2],
                plot=False,
                additional_title_figure=" Picks: %d - %s" % (tot_pcnt,
                                                             source_noise[tot_pcnt]))

        try:
            assert eqlon_ongrid > min(xgr) and eqlon_ongrid < max(xgr)
            assert eqlat_ongrid > min(ygr) and eqlat_ongrid < max(ygr)
        except AssertionError:
            logger.error("Event outside boundaries:  [%.4f  %.4f  %.2f  %.2f]" % (
                            LON, LAT, DEP*KM, MAG))
            return False

        big_x_loc.append(_Y_det[:, :1, :])  # Only the EVENT channel
        big_y_loc.append([xy_image, xz_image, yz_image])
        big_tag_geo.append([LON, LAT, DEP*KM, MAG])  # , SNR, STRIKE, DIP, RAKE])
        big_tag_xyz.append([eqlon_ongrid, eqlat_ongrid, eqdep])
        big_pick_count.append(tot_pcnt)
        big_sourcestd_noise.append(source_noise[tot_pcnt])

        # -----------------------
        # ---------  CLASSIFIER
        big_x_cls.append([xy_image, xz_image, yz_image])
        big_y_cls.append(1 if tot_pcnt >= EVENT_DETECTION else 0)

    return (np.array(big_x_det, dtype=np.float32),
            np.array(big_y_det, dtype=np.float32),
            np.array(big_x_loc, dtype=np.float32),
            big_y_loc,  # different matrix shapes
            big_x_cls,  # different matrix shapes
            np.array(big_y_cls, dtype=np.float32),
            np.array(big_r, dtype=np.float32),
            np.array(big_tag_geo, dtype=np.float32),
            np.array(big_tag_xyz, dtype=np.float32),
            np.array(big_pick_count, dtype=np.float32),
            np.array(big_sourcestd_noise, dtype=np.float32))


# def merge_data_real(MSEED_DIR, GRID_PATH, randomly_mute_nodes=MUTE_NODES,
#                      max_muted=MAX_MUTED, num_events=None):

def merge_data_real(args, configs):
    MSEED_DIR = args.outfolder
    GRID_PATH = args.grid
    randomly_mute_nodes = args.randomly_mute_nodes
    max_muted = args.max_muted
    num_events = None
    source_errors_by_picks = configs.PREPARE_DATA.SOURCE_ERRORS_BY_PICKS
    picks_to_declare_event = args.picks_to_declare_event
    seed = args.seed
    #
    PARALLEL = args.parallel
    NWORKERS = args.workers

    total_files = sorted([fp for fp in MSEED_DIR.glob("*mseed.npz")])

    # Randomly select a subset of the pairs
    if num_events:
        np.random.seed(42)  # For reproducibility
        random_indices = random.sample(range(len(total_files)), num_events)
        assert len(set(random_indices)) == len(random_indices)  # make sure unique indexes
        selected_files = [total_files[idx] for idx in random_indices]
    else:
        selected_files = total_files

    big_x_det = []
    big_y_det = []
    big_x_loc = []
    big_y_loc = []
    big_x_cls = []
    big_y_cls = []
    big_r = []
    big_tag_geo = []
    big_tag_xyz = []
    big_pick_count = []
    big_sourcestd_noise = []

    logger.info("Looping on %d events: " % len(selected_files))

    if PARALLEL:
        # ============================================================
        # =======================================   MULTIPROC (error)
        # ============================================================
        logger.info("Working on PARALLEL mode (%d cores)" % NWORKERS)
        with ThreadPoolExecutor(max_workers=NWORKERS) as executor:
            futures = []
            for fp in selected_files:
                futures.append(executor.submit(process_file, fp, GRID_PATH,
                                               randomly_mute_nodes, max_muted,
                                               picks_to_declare_event,
                                               source_errors_by_picks,
                                               seed))
            for future in tqdm(as_completed(futures), total=len(futures)):
                (xdet, ydet,
                 xloc, yloc,
                 xcls, ycls,
                 r, eqgeo, eqxyz,
                 pcnt, sourcenoise) = future.result()
                #
                big_x_det.append(xdet)
                big_y_det.append(ydet)
                big_x_loc.append(xloc)
                big_y_loc.append(yloc)
                big_x_cls.append(xcls)
                big_y_cls.append(ycls)
                big_r.append(r)
                big_tag_geo.append(eqgeo)
                big_tag_xyz.append(eqxyz)
                big_pick_count.append(pcnt)
                big_sourcestd_noise.append(sourcenoise)
    else:
        # ============================================================
        # =======================================  SEQUENTIAL
        # ============================================================
        logger.info("Working in SEQUENTIAL mode")
        for fp in selected_files:
            _processed_output = process_file(fp, GRID_PATH,
                                             randomly_mute_nodes, max_muted,
                                             picks_to_declare_event,
                                             source_errors_by_picks,
                                             seed)
            if _processed_output:
                (xdet, ydet,
                 xloc, yloc,
                 xcls, ycls,
                 r, eqgeo, eqxyz,
                 pcnt, sourcenoise) = _processed_output
                #
                big_x_det.append(xdet)
                big_y_det.append(ydet)
                big_x_loc.append(xloc)
                big_y_loc.append(yloc)
                big_x_cls.append(xcls)
                big_y_cls.append(ycls)
                big_r.append(r)
                big_tag_geo.append(eqgeo)
                big_tag_xyz.append(eqxyz)
                big_pick_count.append(pcnt)
                big_sourcestd_noise.append(sourcenoise)

    # ============================================================
    # ---------- FINALIZE
    logger.info("Finalizing and concatenating ...")
    big_x_det = np.concatenate(big_x_det, axis=0)
    big_y_det = np.concatenate(big_y_det, axis=0)
    big_x_loc = np.concatenate(big_x_loc, axis=0)

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # ---> (big_y_loc and big_x_cls) are inhomogeneous lists.
    # We need to concatenate on first axis manually! And then transform
    # them into something NUMPYZ can handle
    de = []
    for xx in range(len(big_y_loc)):
        de.extend(big_y_loc[xx])
    big_y_loc = np.array(de, dtype=object)

    de = []
    for xx in range(len(big_x_cls)):
        de.extend(big_x_cls[xx])
    big_x_cls = np.array(de, dtype=object)
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    big_y_cls = np.concatenate(big_y_cls, axis=0)
    big_r = np.concatenate(big_r, axis=0)
    big_tag_geo = np.concatenate(big_tag_geo, axis=0)
    big_tag_xyz = np.concatenate(big_tag_xyz, axis=0)
    big_pick_count = np.concatenate(big_pick_count, axis=0)
    big_sourcestd_noise = np.concatenate(big_sourcestd_noise, axis=0)

    return (big_x_det,
            big_y_det,
            big_x_loc,
            big_y_loc,
            big_x_cls,
            big_y_cls,
            big_r,
            big_tag_geo,
            big_tag_xyz,
            big_pick_count,
            big_sourcestd_noise)


def main(args):

    # ------------------------------------  IMPORTING VARS
    logger.info("Importing VARS ...")
    pickdf = pd.read_csv(args.picks)
    heim_gnn = np.load(args.graph, allow_pickle=True)
    CONFIG = gio.read_configuration_file(args.config, check_version=True)

    # ------------------------------------  PREPROCESS
    startt = time.time()
    logger.info("Pre-Process MSEED")
    startt = time.time()
    for stfp in Path(args.infolder).glob("*.mseed"):
        logger.info("Reading: %s" % str(stfp))
        tagtype = stfp.name.split("_")[0]
        st = obspy.read(stfp)

        # TRIM ONLY PART OF IT
        _new_start = min([tr.stats.starttime for tr in st]) + 1
        _new_end = max([tr.stats.endtime for tr in st]) - 7
        st.trim(_new_start, _new_end)
        #
        _ = __prepare_data__(
                        st,
                        heim_gnn["stations_order"].item(),
                        CONFIG.PREPARE_DATA, pickdf,
                        stfp.name, store_dir=args.outfolder,
                        eq_tag=tagtype)
    endt = time.time()
    logger.info("Total running time PREPROCESS:  %.2f hrs." % ((endt - startt)/3600.0))

    # ------------------------------------  CONTINUOUS
    startt = time.time()
    (x_det,
     y_det,
     x_loc,
     y_loc,
     x_cls,
     y_cls,
     r,
     tag_geo,
     tag_xyz,
     pick_count,
     sourcestd_noise) = merge_data_real(args, CONFIG)

    logger.info("Saving training arrays")
    np.savez_compressed(
            "TrainingDataset_HEIM.npz",
            Xdet=x_det, Ydet=y_det,  # Xloc=x_loc, --> equal to  Ydet[:, :, :1, :]
            Yloc=y_loc, R=r,  # Xcls=x_cls, --> equal to Yloc
            Ycls=y_cls,
            sources=tag_geo, sources_grid=tag_xyz, pick_count=pick_count,
            sourcestd_noise=sourcestd_noise, allow_pickle=True)

    logger.info("DONE")
    endt = time.time()
    logger.info("Total running time CONTINUOUS:  %.2f hrs." % (
                (endt - startt)/3600.0))


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
