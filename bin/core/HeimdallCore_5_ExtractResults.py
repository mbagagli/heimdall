#!/usr/bin/env python

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from scipy.signal import find_peaks, peak_widths
from obspy import UTCDateTime
from obspy.core.event import (Catalog, Event,
                              Origin, Magnitude, QuantityError,
                              OriginUncertainty, OriginQuality,
                              ResourceIdentifier,
                              Pick, Arrival,
                              WaveformStreamID)


from heimdall import utils as gut
from heimdall import custom_logger as CL

logger = CL.init_logger(Path(sys.argv[0]).name, lvl="INFO")


# =================================================================
# =================================================================
# =================================================================

def _parse_cli():
    """
    Parse command-line arguments for the Heimdall pipeline.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    p = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        description="Create 3 types of catalogs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-rootdir", "-d", metavar="PATH", dest="pathin", type=str,
                   help="Root dir for searching 'HeimdallResults.STATISTICS/*.npz' files.")
    p.add_argument("-pthreshold", "-p", metavar="P_THR", dest="pthr", type=float,
                   default=0.2, help="Probability threshold to declare P arrival.")
    p.add_argument("-sthreshold", "-s", metavar="S_THR", dest="sthr", type=float,
                   default=0.2, help="Probability threshold to declare S arrival.")
    p.add_argument("-sampling", "-f", metavar="DF", dest="df", type=float,
                   help="Sampling frequency of the timse-series.")
    p.add_argument("-overlap", "-o", metavar="OVERLAP", dest="overlap", type=float,
                   help="Overlapping seconds, needed to reconstruct waveforms.")
    # Optional
    p.add_argument("--min-magnitude", "-m", metavar="MIN_MAG", dest="min_magnitude", type=float,
                   default=-1.0, help="Minimum magnitude threshold for cleaned catalog.")
    p.add_argument("--min-p-picks", "-n", metavar="N_P_PICKS", dest="npicks", type=int,
                   default=4, help="Minimum number of P-picks required for PICK-filtered catalog.")
    p.add_argument("--extract-picks", dest="extract_picks", action="store_true",
                   help="If specified, also the individual picks associated with the events will be extracted.")
    return p


# =================================================================
# =================================================================
# =================================================================

def stats_to_plot_path(stats_path):
    """
    Converts a path from the statistics folder to the corresponding plot info path.

    Args:
        stats_path (Path): Path to the statistics .npz file.

    Returns:
        Path: Corresponding path in the figures directory.
    """
    # First, adjust the filename
    new_file = stats_path.name.replace('_stats.npz', '_PlotInfo.npz')

    # Swap the directory segment from STATISTICS to FIGURES
    new_dir = str(stats_path.parent).replace('HeimdallResults.STATISTICS', 'HeimdallResults.FIGURES')

    return Path(new_dir) / new_file


def __load_npz_to_df__(npz_file):
    """
    Loads a .npz file and converts it to a pandas DataFrame.

    Args:
        npz_file (str): Path to the .npz file.

    Returns:
        pd.DataFrame: DataFrame containing the .npz contents.
    """
    data = np.load(npz_file, allow_pickle=True)
    # Convert the NPZ file into a dictionary, then into a DataFrame
    df = pd.DataFrame({key: data[key] for key in data.files})
    return df


def __extract_picks__(ts, thr=0.2, min_distance=50, smooth=True):
    """
    Extract peak information from a 1D time series using find_peaks.

    Args:
        ts (np.ndarray): Input time series.
        thr (float): Threshold for peak height.
        min_distance (int): Minimum distance between peaks.
        smooth (bool): Apply smoothing filter.

    Returns:
        tuple: (peak indices, peak widths, peak amplitudes, smoothed trace)
    """

    if smooth:
        smoothing_filter = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]  # remove rapid oscillations
        ts = np.convolve(ts, smoothing_filter, mode='same')

    # Ensure no NaNs
    assert not np.isnan(ts).any(), "NaNs found in the time series."

    peaks, properties = find_peaks(ts, height=thr, distance=min_distance)
    ampl = properties["peak_heights"]  # peak heights
    widths = peak_widths(ts, peaks, rel_height=0.5)[0]
    return peaks, widths, ampl, ts


def __load_npz_to_picklist__(npz_file, overlap, df,
                             p_thr=0.2, s_thr=0.2):
    """
    Load and process picks from a .npz PlotInfo file.

    Args:
        npz_file (str): Path to the PlotInfo .npz file.
        overlap (int): Overlap in samples.
        df (float): Sampling frequency.
        p_thr (float): Threshold for P-pick.
        s_thr (float): Threshold for S-pick.

    Returns:
        dict: Dictionary of picks per station.
    """
    npz = np.load(npz_file, allow_pickle=True)
    data = gut.__merge_windows__(npz["ev_det"], overlap, method="max")
    all_picks = {}
    for _stat, _idx in npz["order"].item().items():
        p_peaks, p_widths, p_ampl, _ = __extract_picks__(
                                                data[_idx][1], thr=p_thr)
        pt = [npz["heim_eq_ot"].item() + _p/df for _p in p_peaks]
        pw = [_p/df for _p in p_widths]
        s_peaks, s_widths, s_ampl, _ = __extract_picks__(
                                                data[_idx][2], thr=s_thr)
        st = [npz["heim_eq_ot"].item() + _s/df for _s in s_peaks]
        sw = [_s/df for _s in s_widths]

        all_picks[_stat] = (
                (pt, pw, p_ampl),
                (st, sw, s_ampl)
            )
    return all_picks


def create_obspy_catalog(dfs, picks):
    """
    Create an ObsPy Catalog from statistics and pick data.

    Args:
        dfs (List[pd.DataFrame]): List of DataFrames, each representing event-level metadata
            loaded from Heimdall .npz statistic files.
        picks (List[dict]): List of dictionaries, each containing picks for one event.
            The dictionary maps station codes to ((P_times, P_widths, P_amplitudes), (S_times, S_widths, S_amplitudes)).

    Returns:
        Tuple[Catalog, pd.DataFrame]:
            - ObsPy Catalog containing Event, Origin, Magnitude, Pick, and Arrival objects.
            - A Pandas DataFrame with combined metadata and pick counts (P_PICKS and S_PICKS).
    """
    def __create_compact(indf, _number):
        out_df = indf[[
                'eqtag', 'heim_eq_ot', 'heim_eq_lon', 'heim_eq_lat', 'heim_eq_z',
                'heim_eq_mag', 'heim_eq_mag_err', 'heim_eq_pdf_3Dmedian',
                'heim_eq_pdf_3Dstd', 'total_windows_event']].copy()

        out_df['heim_eq_z'] = out_df['heim_eq_z']*10**-3
        out_df['heim_eq_ot'] = out_df['heim_eq_ot'].apply(lambda x: x.datetime)
        out_df['heim_eq_ot'] = pd.to_datetime(out_df['heim_eq_ot'])
        out_df = out_df.sort_values(by='heim_eq_ot')
        out_df.reset_index(drop=True, inplace=True)
        out_df['eqtag'] = out_df['heim_eq_ot'].dt.strftime(
                                        "heim_%Y%m%d%H%M%S")
        out_df['eqtag'] = out_df['eqtag'].astype(str) + f"_{_number:04d}"
        out_df = out_df.drop_duplicates()
        out_df.reset_index(drop=True, inplace=True)
        out_df = out_df.rename(
                        columns={'eqtag': "EVID",
                                 'heim_eq_ot': "ORIGIN_TIME",
                                 'heim_eq_lon': "LONGITUDE",
                                 'heim_eq_lat': "LATITUDE",
                                 'heim_eq_z': "DEPTH(km)",
                                 'heim_eq_mag': "MAGNITUDE",
                                 'heim_eq_mag_err': "MAGNITUDE_ERROR",
                                 'heim_eq_pdf_3Dmedian': "RMS",
                                 'heim_eq_pdf_3Dstd': "RMS_std",
                                 'total_windows_event': "TOTAL_WINDOWS"})
        return out_df

    # --------------------------------------------------------------
    cat = Catalog()
    all_headers = []
    all_pickCounts = {}
    all_pickCounts["P_PICKS"] = []
    all_pickCounts["S_PICKS"] = []
    for _xx, (_statistics, _picks) in enumerate(zip(dfs, picks)):
        header_df = __create_compact(_statistics, _xx)
        header = header_df.iloc[0]

        # ---------- Origin
        ot = UTCDateTime(header["ORIGIN_TIME"])
        lon = float(header["LONGITUDE"])
        lat = float(header["LATITUDE"])
        depth_m = float(header["DEPTH(km)"]) * 1e3
        mag_val = float(
            header["MAGNITUDE"] if not pd.isna(header["MAGNITUDE"]) else -9.99)
        mag_err = float(
            header["MAGNITUDE_ERROR"]) if not pd.isna(header["MAGNITUDE_ERROR"]) else -9.99
        rms = float(header["RMS"]) if not pd.isna(header["RMS"]) else None
        rms_std = float(header["RMS_std"]) if not pd.isna(header["RMS_std"]) else None

        origin = Origin(time=ot,
                        latitude=lat,
                        longitude=lon,
                        depth=depth_m,
                        depth_type="operator assigned")

        # Add origin quality
        origin.quality = OriginQuality(
            standard_error=rms,
            azimuthal_gap=None,
            minimum_distance=None,
            maximum_distance=None,
            rms=rms,
            used_phase_count=None
        )

        # Add uncertainty (you can modify with your own error values if available)
        origin.origin_uncertainty = OriginUncertainty(
            preferred_description="confidence ellipsoid",
            max_horizontal_uncertainty=rms_std  # or whatever uncertainty you have
        )

        # Add ID
        origin_id = ResourceIdentifier(id=f"smi:local/heimdall/{header['EVID']}")  # or use a timestamp
        origin.resource_id = origin_id

        # ................................  initialize EVENT
        event = Event()
        event.origins.append(origin)

        if mag_val is not None:
            magnitude = Magnitude(
                mag=mag_val,
                magnitude_type="Mr",
                origin_id=origin.resource_id
            )

            if mag_err is not None:
                magnitude.mag_errors = QuantityError(uncertainty=mag_err)

            event.magnitudes.append(magnitude)

        # Optionally: add custom tags using .extra
        event.extra = {
            "total_windows": {
                "value": int(header["TOTAL_WINDOWS"]),
                "namespace": "heimdall",
                "type": "attribute"
            }
        }

        # ---------- Picks
        p_count, s_count = 0, 0
        for _station, _picks in _picks.items():
            (pt, pw, pa), (st, sw, sa) = _picks

            # Loop over P picks
            for i in range(len(pt)):
                p_count += 1
                pick = Pick(
                    time=UTCDateTime(pt[i]),
                    waveform_id=WaveformStreamID(station_code=_station),
                    phase_hint="P",
                    method_id=ResourceIdentifier(id="smi:local/heimdall/picker"),
                    onset="impulsive",
                    evaluation_mode="automatic",
                    evaluation_status="preliminary",
                    time_errors=QuantityError(uncertainty=float(pw[i]),
                                              confidence_level=float(pa[i])),
                )
                pick.resource_id = ResourceIdentifier(
                    id=f"smi:local/pick/{header['EVID']}/P/{_station}/{i}")
                event.picks.append(pick)

            # Loop over S picks
            for i in range(len(st)):
                s_count += 1
                pick = Pick(
                    time=UTCDateTime(st[i]),
                    waveform_id=WaveformStreamID(station_code=_station),
                    phase_hint="S",
                    method_id=ResourceIdentifier(id="smi:local/heimdall/picker"),
                    onset="emergent",
                    evaluation_mode="automatic",
                    evaluation_status="preliminary",
                    time_errors=QuantityError(uncertainty=float(sw[i]),
                                              confidence_level=float(sa[i]))
                )
                pick.resource_id = ResourceIdentifier(
                    id=f"smi:local/pick/{header['EVID']}/S/{_station}/{i}")
                event.picks.append(pick)

        # -------------------------------  Associate ARRIVALS
        for i in range(len(pt)):
            arrival = Arrival(
                pick_id=ResourceIdentifier(
                    id=f"smi:local/pick/{header['EVID']}/P/{_station}/{i}"),
                phase="P",
                time_weight=1.0  # Optional: adjust or compute based on pw[i] if meaningful
            )
            origin.arrivals.append(arrival)

        # Link S picks to origin via arrivals
        for i in range(len(st)):
            arrival = Arrival(
                pick_id=ResourceIdentifier(
                    id=f"smi:local/pick/{header['EVID']}/S/{_station}/{i}"),
                phase="S",
                time_weight=1.0  # Optional: adjust or compute based on sw[i]
            )
            origin.arrivals.append(arrival)

        #
        cat.events.append(event)
        all_headers.append(header_df)
        all_pickCounts["P_PICKS"].append(p_count)
        all_pickCounts["S_PICKS"].append(s_count)

    # END
    all_headers = pd.concat(all_headers, ignore_index=True)
    pick_df = pd.DataFrame(all_pickCounts)
    full_df = pd.concat([all_headers, pick_df], axis=1)
    return (cat, full_df)


def write_catalog_picks_csv(catalog, outfolder="PICKS"):
    """
    Write a text file for each event in the catalog containing origin and pick information.

    Args:
        catalog (Catalog): ObsPy Catalog object with events containing picks and origin info.
        outfolder (str, optional): Path to the output directory where pick files will be written.
            Defaults to "PICKS".

    Returns:
        bool: True if all files are written successfully.
    """
    logger.info("Writing picks in folder: %s" % outfolder)
    save_fold = Path(outfolder)
    save_fold.mkdir(exist_ok=True, parents=True)

    for event in tqdm(catalog):
        origin = event.preferred_origin() or event.origins[0]
        magnitude = event.preferred_magnitude() or (event.magnitudes[0] if event.magnitudes else None)

        evid = origin.resource_id.id.split("/")[-1]
        tag = evid  # can be changed to other format if desired

        file_path = save_fold / f"{tag}_picks.txt"

        with open(file_path, "w") as PICK:
            PICK.write("Origin\n")
            PICK.write("ot %s\n" % origin.time.datetime)
            PICK.write("lon %.5f\n" % origin.longitude)
            PICK.write("lat %.5f\n" % origin.latitude)
            PICK.write("dep %.2f\n" % (origin.depth / 1e3))  # m → km
            if magnitude:
                PICK.write("mag %.2f\n" % magnitude.mag)
                err = magnitude.mag_errors.uncertainty if magnitude.mag_errors else -9999.99
                PICK.write("mag_err %.2f\n" % err)
            else:
                PICK.write("-9999.99\n-9999.99\n")

            if hasattr(origin.quality, "standard_error") and origin.quality.standard_error is not None:
                PICK.write("pdf %.2f\n" % origin.quality.standard_error)
            else:
                PICK.write("-9999.99\n")
            if origin.origin_uncertainty and origin.origin_uncertainty.max_horizontal_uncertainty is not None:
                PICK.write("pdf_err %.2f\n" % origin.origin_uncertainty.max_horizontal_uncertainty)
            else:
                PICK.write("-9999.99\n")

            PICK.write("Phases\n")

            for pick in event.picks:
                stat = pick.waveform_id.station_code
                phase = pick.phase_hint
                time = pick.time.datetime
                amp = pick.time_errors.confidence_level if pick.time_errors else -9999.99
                width = pick.time_errors.uncertainty if pick.time_errors else -9999.99
                PICK.write("%10s %s %s %.2f %.2f\n" % (
                    stat, phase, time, amp, width))
    return True


def main(args):
    """
    Main entry point of the catalog building pipeline.

    Loads .npz statistic and pick data, processes them into ObsPy Catalog and
    DataFrame representations, writes full and filtered catalogs to disk,
    and optionally writes per-event pick text files.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    search_path = Path(args.pathin) if args.pathin else Path(".")
    logger.state("Searching on:  %s" % str(search_path))
    folder_list = sorted(search_path.rglob('HeimdallResults.STATISTICS/*.npz'))
    stat_list = []
    pick_list = []
    OVERLAP = int(args.overlap*args.df)+1
    for _ddd, npz_file in enumerate(tqdm(folder_list)):
        npz_file_plot = stats_to_plot_path(npz_file)

        try:
            stat_list.append(__load_npz_to_df__(str(npz_file)))
        except Exception as e:
            logger.error("ERROR in STATS:  %s --> %s" % (str(npz_file), e))
            breakpoint()

        try:
            pick_list.append(__load_npz_to_picklist__(
                    str(npz_file_plot), OVERLAP, args.df,
                    p_thr=args.pthr, s_thr=args.sthr,
                ))
        except Exception as e:
            logger.error("ERROR in PICKS:  %s --> %s" % (str(npz_file_plot), e))
            breakpoint()
    #
    assert len(stat_list) == len(pick_list)
    logger.info("... merging stats and filtering")
    (full_cat, full_df) = create_obspy_catalog(stat_list, pick_list)
    full_cat.write("Heimdall_Catalog.FULL.xml", format="QUAKEML")
    full_df.to_csv("Heimdall_Catalog.FULL.csv", index=False)

    if args.extract_picks:
        try:
            write_catalog_picks_csv(full_cat, outfolder="PICKS")
        except Exception as e:
            logger.error("ERROR in PICKS:  %s --> %s" % (str(npz_file_plot), e))
            breakpoint()

    # -------> Filtering CATALOG
    filtered_df = full_df[full_df["MAGNITUDE"].notna() & (
        full_df["MAGNITUDE"] >= args.min_magnitude)].copy()
    filtered_df = filtered_df[filtered_df["P_PICKS"] >= args.npicks].copy()
    filtered_df.reset_index(drop=True, inplace=True)
    filtered_df.to_csv("Heimdall_Catalog.FILTERED.csv", index=False)

    # Extract EVIDs and filter catalog
    filtered_evids = set(filtered_df["EVID"])
    filtered_cat = Catalog(events=[
        event for event in full_cat
        if event.origins and event.origins[0].resource_id.id.split("/")[-1] in filtered_evids
    ])
    filtered_cat.write("Heimdall_Catalog.FILTERED.xml", format="QUAKEML")


# =================================================================
# =================================================================
# =================================================================

if __name__ == "__main__":
    parser = _parse_cli()
    if len(sys.argv) == 1 or sys.argv[1].lower() in ("-h", "--help"):
        parser.print_help(sys.stdout)
        sys.exit()
    #
    args = parser.parse_args()
    logger.info("PARSER INPUTS")
    for k, v in vars(args).items():
        logger.info(f"    {k}: {v}")
    #
    main(args)
    logger.info("DONE")
