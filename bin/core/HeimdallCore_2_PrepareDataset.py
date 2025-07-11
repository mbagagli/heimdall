#!/usr/bin/env python
"""
Create **one** “fully-baked” *.npz* per input *.mseed* file.

The single function **prepare_single_event()** merges the old
`__prepare_data__` + `process_file` pipelines so that the whole
workflow can be launched in parallel (one worker per file).

*Variable names and the keys written to each NPZ are **unchanged**.*
"""

import sys
import obspy
import numpy as np
import pandas as pd
import random
import time
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

from heimdall import __version__
from heimdall import io as gio
from heimdall import locator as glctr
from heimdall import custom_logger as CL

logger = CL.init_logger(Path(sys.argv[0]).name, lvl="INFO",
                        log_file="PrepareDataset.log")

KM = 1e-3
MT = 1e3
EVENT_DETECTION = None        # will be set in main()


# ==================================================================
# ==================================================================
# ==================================================================

def _parse_cli():
    """
    Parse command-line arguments.
    """
    p = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        description="Prepare Heimdall training shards (one .npz per .mseed).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core files / dirs -------------------------------------------------
    p.add_argument("-graph", "-g", metavar="GRAPH", type=str,
                   required=True,
                   help="Heimdall GNN graph file (.npz)")
    p.add_argument("-grid", "-grd", metavar="GRID", type=str,
                   required=True,
                   help="Heimdall spatial grid file (.npz)")
    p.add_argument("-conf", "-c", dest="config", metavar="YAML", type=str,
                   required=True,
                   help="Configuration YAML file")
    p.add_argument("-infold", "-in", dest="infolder", metavar="DIR", type=str,
                   required=True,
                   help="Input folder with raw *.mseed data")
    p.add_argument("-outfold", "-out", dest="outfolder", metavar="DIR", type=str,
                   default="NPZ.SHARDS",
                   help="Output folder for per-event *.npz files (auto-created)")
    p.add_argument("-picks", "-p", metavar="CSV", default="AllPicks.csv", type=str,
                   help="CSV containing all picks")

    # Detection
    p.add_argument("--picks-to-declare-event", metavar="N", type=int,
                   default=5,
                   help="Minimum number of picks for a positive event label")
    p.add_argument("--mute-nodes", action="store_true",
                   help="Randomly zero out some stations during training")
    p.add_argument("--max-muted", metavar="N", type=int,
                   default=11,
                   help="Max number of stations to mute when --mute-nodes is set")
    p.add_argument("--seed", metavar="INT", type=int,
                   default=42,
                   help="Base random seed")

    # Parallel ----------------------------------------------------------
    p.add_argument("--parallel", action="store_true",
                   help="Run one worker per *.mseed* file")
    p.add_argument("--workers", metavar="N", type=int,
                   default=8,
                   help="Number of worker processes for --parallel")

    return p


# ----------------------------------------------------------------------
# Core + save (merges the old __prepare_data__ and process_file)
# ----------------------------------------------------------------------
def prepare_single_event(stfp: Path, args):
    """
    Run the full pipeline on **one** .mseed file and write a
    (not-compressed) .npz shard next to `args.outfolder/stfp.stem+'.npz'`.
    """
    # ------------------------------------------------------------------
    # Stage 0 – housekeeping
    # ------------------------------------------------------------------
    rnd = np.random.default_rng(args.seed + hash(stfp) % 2**16)
    tagtype = stfp.name.split("_")[0]

    # ------------------------------------------------------------------
    # Stage 1 – load and basic trim  (former __prepare_data__)
    # ------------------------------------------------------------------
    st = obspy.read(stfp)
    _new_start = min(tr.stats.starttime for tr in st) + 1
    _new_end = max(tr.stats.endtime for tr in st) - 7
    st.trim(_new_start, _new_end)

    heim_gnn = np.load(args.graph, allow_pickle=True)
    stat_dict_order = heim_gnn["stations_order"].item()

    CONFIG = gio.read_configuration_file(args.config, check_version=True)
    pickdf = pd.read_csv(args.picks)

    heim = gio.HeimdallDataLoader(st, order=stat_dict_order)
    (X, Y, R) = heim.prepare_data_real(
        downsample=CONFIG.PREPARE_DATA.DOWNSAMPLE,
        slicing=CONFIG.PREPARE_DATA.SLICING,
        create_labels=pickdf,
        debug_plot=False,
    )
    heim_grid = np.load(args.grid, allow_pickle=True)
    source_noise = CONFIG.PREPARE_DATA.SOURCE_ERRORS_BY_PICKS
    max_pcnt = max(source_noise.keys())

    # ------------------------------------------------------------------
    # Stage 2 – per-window feature build  (former process_file)
    # ------------------------------------------------------------------
    big_x_det, big_y_det, big_y_loc, big_y_cls = [], [], [], []
    big_r, big_tag_geo, big_tag_xyz = [], [], []
    big_pick_count, big_sourcestd_noise = [], []

    for nwin in range(X.shape[0]):
        _X_det = X[nwin].copy()
        _Y_det = Y[nwin].copy()
        _R = R[nwin].copy()

        # -------- random mute
        if args.mute_nodes:
            _nmute = rnd.integers(0, args.max_muted + 1)
            if _nmute:
                _nmute_idx = rnd.choice(np.arange(_R.shape[0]),
                                        size=_nmute, replace=False)
                _X_det[_nmute_idx] = 0.0
                _Y_det[_nmute_idx] = 0.0

        # -------- pick count
        pcnt = np.any(_Y_det[:, 0, :] > 0.95, axis=1).sum()
        pcnt = int(min(pcnt, max_pcnt))

        # -------- detector lists
        big_x_det.append(_X_det)
        big_y_det.append(_Y_det)
        big_r.append(_R)

        # -------- locator
        HG = glctr.HeimdallLocator(
            heim_grid['boundaries'],
            spacing_x=heim_grid['spacing_km'][0],
            spacing_y=heim_grid['spacing_km'][1],
            spacing_z=heim_grid['spacing_km'][2],
            reference_point_lonlat=heim_grid['reference_point']
        )

        LON = float(stfp.stem.split("_")[3])
        LAT = float(stfp.stem.split("_")[4])
        DEP = float(stfp.stem.split("_")[5])
        MAG = float(stfp.stem.split("_")[6])
        RMS = float(stfp.stem.split("_")[7])

        (label_pdf, xy_image, xz_image, yz_image,
         (eqlon, eqlat, eqdep)) = HG.create_source_images(
            (LON, LAT, DEP),
            std_err_km=source_noise[pcnt][0],
            noise_max=source_noise[pcnt][1],
            source_noise=source_noise[pcnt][2],
            plot=False,
            dep_in_km=True,
            additional_title_figure=f" Picks: {pcnt} - {source_noise[pcnt]}"
        )

        big_y_loc.append([xy_image, xz_image, yz_image])
        big_tag_geo.append([LON, LAT, DEP * MT, MAG])
        big_tag_xyz.append([eqlon, eqlat, eqdep])
        big_pick_count.append(pcnt)
        big_sourcestd_noise.append(source_noise[pcnt])

        # -------- classifier
        big_y_cls.append(1 if pcnt >= EVENT_DETECTION else 0)

    # ------------------------------------------------------------------
    # Stage 3 – pack + save
    # ------------------------------------------------------------------
    def _cat(seq, axis=0, dtype=np.float32):
        return np.array(seq, dtype=dtype) if seq and isinstance(seq[0], np.ndarray) else seq

    Xdet = _cat(big_x_det)
    Ydet = _cat(big_y_det)
    Yloc = np.array(big_y_loc, dtype=object)
    R_arr = _cat(big_r)
    Ycls = np.array(big_y_cls, dtype=np.float32)
    sources = np.array(big_tag_geo, dtype=np.float32)
    sources_grid = np.array(big_tag_xyz, dtype=np.float32)
    pick_count = np.array(big_pick_count, dtype=np.float32)
    sourcestd_noise = np.array(big_sourcestd_noise, dtype=np.float32)

    out_dir = Path(args.outfolder)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{stfp.stem}.npz"

    np.savez(
        npz_path,
        Xdet=Xdet,
        Ydet=Ydet,
        Yloc=Yloc,
        R=R_arr,
        Ycls=Ycls,
        sources=sources,
        sources_grid=sources_grid,
        pick_count=pick_count,
        sourcestd_noise=sourcestd_noise,
    )
    logger.state("Saved %s  (windows=%d)", npz_path.name, Xdet.shape[0])
    return True


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main(args):
    global EVENT_DETECTION
    EVENT_DETECTION = args.picks_to_declare_event

    random.seed(args.seed)
    np.random.seed(args.seed)

    mseed_files = sorted(Path(args.infolder).glob("*.mseed"))
    if not mseed_files:
        logger.error("No *.mseed files found in %s", args.infolder)
        sys.exit(1)

    logger.info("Processing %d events -> %s", len(mseed_files), args.outfolder)

    start_time = time.time()
    if args.parallel:
        logger.state("Running in PARALLEL mode (%d workers)", args.workers)
        with ProcessPoolExecutor(max_workers=args.workers) as exe:
            futures = [exe.submit(prepare_single_event, fp, args)
                       for fp in mseed_files]
            for _ in tqdm(as_completed(futures), total=len(futures)):
                pass
    else:
        logger.info("Running in SEQUENTIAL mode")
        for fp in tqdm(mseed_files, desc="Events"):
            prepare_single_event(fp, args)

    elapsed = (time.time() - start_time) / 3600.0
    logger.state("DONE – total wall-time: %.2f h", elapsed)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = _parse_cli()
    if len(sys.argv) == 1 or sys.argv[1].lower() in ("-h", "--help"):
        parser.print_help(sys.stdout)
        sys.exit()
    cli_args = parser.parse_args()
    logger.info("CLI ARGS:")
    for k, v in vars(cli_args).items():
        logger.info("  %s: %s", k, v)
    main(cli_args)
