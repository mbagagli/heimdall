"""
This module provides classes and functions for managing seismic data processing and analysis within the HEIMDALL framework.

It includes utilities for:
    - Reading and validating YAML configuration files.
    - Loading and preprocessing seismic data from MiniSEED files.
    - Computing characteristic functions such as PCA, energy, higher-order statistics (HOS), and others from seismic waveforms.
    - Preparing data matrices for machine learning models, including slicing and normalization operations.

Dependencies:
    obspy, numpy, scipy, pandas, matplotlib, tqdm, yaml, heimdall

Classes:
    RecursiveAttributeDict: A dictionary supporting attribute-style access.
    HeimdallDataLoader: Handles loading, processing, and slicing of seismic data.
    HeimdallProcessing: Computes characteristic functions and transforms from seismic data.
    FBSummary: Summarizes band-filtered data using filter picker statistics.

Functions:
    read_configuration_file(config_path, check_version=True): Reads a YAML configuration file into a RecursiveAttributeDict object, optionally checking version compatibility.
"""


import sys
import yaml
import time
import copy
import obspy
from obspy.signal.filter import bandpass  # for FilterPicker
from scipy.signal import lfilter  # for FilterPicker
import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor  # ThreadPoolExecutor, as_completed

from pathlib import Path
from tqdm import tqdm
#
import heimdall
from heimdall import utils as GUT
from heimdall import custom_logger as CL

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger = CL.init_logger(Path(sys.argv[0]).name, lvl="INFO")


# ==================================================================
# =========================================================   SETUP
# ==================================================================


def read_configuration_file(config_path, check_version=True):
    """Read a HEIMDALL YAML configuration file.

    Args:
        config_path (str | Path): Absolute or relative path to the YAML
            configuration file.
        check_version (bool, optional): If ``True`` (default), fail when the
            file’s ``version`` entry differs from :pydataattr:`heimdall.__version__`.

    Returns:
        RecursiveAttributeDict: Parsed parameters with attribute-style access.

    Raises:
        ValueError: The file lacks a ``version`` key or its value is
            incompatible with the running HEIMDALL version.
    """

    with open(str(config_path), 'r') as yaml_file:
        configs = yaml.load(yaml_file, Loader=yaml.FullLoader)

    if check_version:
        # Check if the version in the file matches the required version
        file_version = configs.get('version', None)
        if file_version is None:
            raise ValueError("Version not found in the configuration file.")
        elif file_version != heimdall.__version__:
            raise ValueError(
                "Configuration file version %r doesn't match the current "
                "HEIMDALL version (%r)" % (file_version, heimdall.__version__))

    # Convert the dictionary to a SimpleNamespace object
    configs = RecursiveAttributeDict(configs)
    return configs


class RecursiveAttributeDict(dict):
    """
    A dictionary that allows accessing its elements both as keys and attributes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, item):
        if item in self:
            value = self[item]
            if isinstance(value, dict):
                return RecursiveAttributeDict(value)
            return value
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        del self[item]

# ==================================================================
# =========================================================   DATA
# ==================================================================


class HeimdallDataLoader(object):
    """Load, preprocess, and organise MiniSEED data.

    Args:
        mseed_path (Path | str | obspy.core.stream.Stream): MiniSEED file path
            *or* an already loaded :class:`obspy.core.stream.Stream`.
        order (dict[str, int] | None, optional): Mapping
            ``{"NET.STA": row_index}``.  When *None* (default) the order is
            derived from the stream.
        trim (tuple[UTCDateTime, UTCDateTime] | bool, optional): If a tuple,
            the stream is trimmed between these start/stop times.  Pass
            ``False`` (default) to disable trimming.
        tag (str, optional): Identifier for log messages. Defaults to
            ``"HeimdallDataLoader"``.

    Attributes:
        stream (obspy.core.stream.Stream): De-trended, gap-free stream with
            channels ordered *Z, N, E*.
        order (dict[str, int]): Station-to-row index map.
        tag (str): Loader identifier carried into log output.
    """

    def __init__(self, mseed_path, order=None,
                 trim=False,  # (list of UTCDateTime --> start,end)
                 tag="HeimdallDataLoader"):
        # self.mseed_path = Path(mseed_path)
        self.stream = self.__load__(mseed_path, trim=trim)
        self.__update_class_stream_meta__()
        self.order = self.__set_order__(order)
        self.tag = tag
        self.epsilon = 10**-6

    def __update_class_stream_meta__(self):
        """ Update all necessary information for the stream-class
        """
        (self.stream_start_date,
         self.stream_end_date,
         self.stream_max_npts) = (
                np.min([_t.stats.starttime for _t in self.stream]),
                np.max([_t.stats.endtime for _t in self.stream]),
                np.max([_t.stats.npts for _t in self.stream])
            )

    def __load__(self, fp, trim):
        """Return a processed stream, optionally trimmed.

        Args:
            fp (Path | str | obspy.core.stream.Stream): Data source.
            trim (tuple[UTCDateTime, UTCDateTime] | bool): See
                :pymeth:`HeimdallDataLoader.__init__`.

        Returns:
            obspy.core.stream.Stream: Demeaned, merged, and channel-sorted
            stream.

        Raises:
            ValueError: *fp* is neither a path nor a Stream.
            AssertionError: Gaps remain after merge.
        """
        starttm = time.time()
        #
        if isinstance(fp, Path):
            logger.info("Reading file name: %s" % fp.name)
            _st = obspy.read(str(fp))
        elif isinstance(fp, obspy.core.stream.Stream):
            logger.info("File is already a stream")
            _st = fp
        else:
            raise ValueError("Must be either Path or Stream object")
        #
        if trim:
            assert isinstance(trim, (list, tuple))
            logger.info("Trimming between  %s and %s" % (trim[0], trim[1]))
            _st.trim(trim[0], trim[1])
        #
        logger.info("... processing ... (demean+fillValue)")

        # always remove mean/detrend before merging!
        _st.detrend("demean")
        _st.detrend("linear")

        _st.merge(fill_value=0)
        _st.sort(keys=['channel', ], reverse=True)  # make sure ZNE order
        for tr in _st:
            if tr.stats.npts <= 1:
                _st.remove(tr)

        assert len(_st.get_gaps()) == 0  # important
        endtm = time.time()
        logger.info("--> Total Time:  %.2f min." % ((endtm-starttm)/60.0))
        return _st

    def __set_order__(self, value):
        """
        Sets the order of stations if provided, or generates a default order.

        Args:
            value (dict or None): The custom or default order of stations.

        Returns:
            dict: The order of stations.
        """
        if not value:
            outval = {}
            for _x, (_net, _stat) in enumerate(self.__stat_net_in_stream__):
                outval[_net+"."+_stat] = _x
        elif value:
            assert isinstance(value, dict)
            outval = value
        else:
            raise ValueError("Wrong or unexpected value")
        return outval

    def change_order(self, value=None):
        """
        Changes the order of stations in the data loader.

        Args:
            value (dict, optional): A new order for the stations.
        """
        # In order it must:
        # - check if order already exist --> throw warning and re-prepare the ordered matrix --> run set_order + ordered
        # - run __set_order__
        pass

    def change_tag(self, value="HeimdallDataLoader"):
        """
        Changes the tag of the data loader.

        Args:
            value (str): A new tag for the data loader.
        """
        assert isinstance(value, str)
        self.tag = value

    @property
    def __stat_net_in_stream__(self):
        """ Set containing all the stations reported in the self.stream object
        NB: It's not the official order! Use self.order dict instead
        for that """
        return set([(_tr.stats.network, _tr.stats.station)
                    for _tr in self.stream])

    @property
    def __stat_net_in_order__(self):
        """ Set containing all the stations reported in the self.order dict
        NB: It's not the official order! Use self.orderdict instead
        for that """
        return set([(kk.split(".")[0], kk.split(".")[1])
                    for kk in self.order.keys()])

    def prepare_data_synth(self,
                           downsample=False,
                           calculate_cf=False,
                           slicing=False,
                           create_labels=False,
                           debug_plot=False):

        def __align_traces__(stream):
            # Determine the earliest start time and latest end time
            earliest_start = min([trace.stats.starttime for trace in stream])
            latest_end = max([trace.stats.endtime for trace in stream])

            # Create a new stream to hold aligned traces
            aligned_stream = obspy.core.Stream()

            for trace in stream:
                # Calculate the number of samples to pad at the start
                start_padding = int((trace.stats.starttime - earliest_start) /
                                    trace.stats.delta)
                end_padding = int((latest_end - trace.stats.endtime) /
                                  trace.stats.delta)

                # Copy the trace to avoid modifying the original stream
                new_trace = trace.copy()

                # Pad the trace at the start and end with zeros
                new_trace.data = np.pad(new_trace.data,
                                        (start_padding, end_padding),
                                        'constant')

                # Update the starttime to the earliest start time
                new_trace.stats.starttime = earliest_start

                # Append the new trace to the aligned stream
                aligned_stream += new_trace

            return aligned_stream

        def __fix_mixed_dt__(stream):
            logger.error("Fixing MIXED DT ... you should never be here!")
            breakpoint()

        def __fix_mixed_npts__(stream):
            logger.warning("Fixing MIXED NPTS")
            stream = __align_traces__(stream)

            if len(set([tr.stats.npts for tr in stream])) != 1:

                # ... still not there ...
                if (max(set([tr.stats.npts for tr in stream])) -
                   min(set([tr.stats.npts for tr in stream]))) >= 50:
                    logger.error("More than 1 second differences after "
                                 "trace alignment...something's wrong!")
                    breakpoint()

                # Go with the final HARDCUT
                hard_cut = min(set([tr.stats.npts for tr in stream]))
                for tr in stream:
                    tr.data = tr.data[:hard_cut]

            # double_check once again ...
            try:
                assert len(
                    set([tr.stats.npts for tr in stream])
                ) == 1
            except AssertionError:
                breakpoint()

            return stream

        def __fix_mixed_time__(stream):
            logger.warning("Fixing MIXED Start/End TIMES")
            _all_starts = list(set([
                            float(tr.stats.starttime) for tr in stream]))
            _all_ends = list(set([
                            float(tr.stats.endtime) for tr in stream]))
            try:
                assert np.all(np.abs(np.diff(_all_starts)) < 0.02)
                assert np.all(np.abs(np.diff(_all_ends)) < 0.02)
                assert len(set([tr.stats.npts for tr in stream])) == 1

            except AssertionError:
                logger.error("Not all STARTS DIFF are below 1/df !!! "
                             "Should not happen")
                breakpoint()

        # -------------------------------
        # --- 0. CHECKS
        if len(self.__stat_net_in_stream__ -
               self.__stat_net_in_order__) > 0:
            logger.warning("There are stations that will not be "
                           "extracted and processed:  %r" % (
                                    self.__stat_net_in_stream__ -
                                    self.__stat_net_in_order__))

        # -------------------------------
        # --- 1. DOWNSAMPLE STREAM
        if downsample and isinstance(downsample, dict):
            logger.info("Downsampling Stream ...")
            starttm = time.time()
            self.stream = GUT.downsampling_stream(
                             self.stream, **downsample,
                             copy=False)
            endtm = time.time()
            #
            logger.info("--> Total Time:  %.2f min." % ((endtm-starttm)/60.0))

        # -------------------------------
        # --- 2. QUALITY  CHECKS
        try:
            assert len(
                set([tr.stats.sampling_rate for tr in self.stream])
            ) == 1
            delta = 1.0/self.stream[0].stats.sampling_rate
        except AssertionError:
            __fix_mixed_dt__(self.stream)

        try:
            assert len(
                set([tr.stats.npts for tr in self.stream])
            ) == 1
        except AssertionError:
            self.stream = __fix_mixed_npts__(self.stream)
            self.__update_class_stream_meta__()

        try:
            assert len(
                set([tr.stats.starttime.time for tr in self.stream])
            ) == 1
        except AssertionError:
            __fix_mixed_time__(self.stream)
            self.__update_class_stream_meta__()
        self.__update_class_stream_meta__()

        # -------------------------------
        # --- 3. WORK
        #  (A.) Preparing Ordered matrix
        self.data_matrix_ordered = np.zeros((
                len(self.__stat_net_in_order__),
                3,  # channel
                self.stream[0].stats.npts
            ))

        for _net, _stat in self.__stat_net_in_order__:

            _st = self.stream.select(station=_stat, network=_net)
            if len(_st) == 0:
                logger.warning("Station:  %s.%s  can't be found on stream" %
                               (_net, _stat))
                continue

            try:
                assert len([_t.stats.channel[-1] for _t in _st]) == 3
                assert [_t.stats.channel[-1] for _t in _st] == ["Z", "N", "E"]
            except AssertionError:
                # leaving that channel equal to 0
                logger.warning("Station: %s  doesn't have 3 clean channels...skipping" % (
                                _net+"."+_stat))
                continue

            _mat = [_t.data for _t in _st]
            _mat_idx = self.order[_net+"."+_stat]
            self.data_matrix_ordered[_mat_idx, :, :] = _mat

        #  (B.) Label data (Y)
        if create_labels is not None and create_labels is not False:
            # ------------------------------------------
            # CREATE LABELS is a DataFrame with the following keys:
            #   eventId, stationId_noChan, isotime_P, isotime_S,
            #   lowerUncertainty_P, lowerUncertainty_S,
            #   upperUncertainty_P, upperUncertainty_S
            # Mandatory are:
            #   eventId, isotime_P, isotime_S, stationId_noChan (i.e. 2C.BIT06)
            if isinstance(create_labels, (Path, str)):
                create_labels = pd.read_csv(create_labels)
            elif isinstance(create_labels, pd.DataFrame):
                pass
            else:
                raise TypeError("`create_labels` must be either a path to CSV "
                                "or pandas dataframe object")

            # ------------------------------------------
            # ... initializing matrix (Y)
            self.data_labels_ordered = np.zeros(
                                            (self.data_matrix_ordered.shape[0],
                                             self.data_matrix_ordered.shape[-1]))
            self.data_picks_P_ordered = np.zeros(
                                            (self.data_matrix_ordered.shape[0],
                                             self.data_matrix_ordered.shape[-1]))
            self.data_picks_S_ordered = np.zeros(
                                            (self.data_matrix_ordered.shape[0],
                                             self.data_matrix_ordered.shape[-1]))

            assert isinstance(create_labels, pd.DataFrame)
            logger.info("Creating labels")
            create_labels['isotime_P'] = pd.to_datetime(create_labels['isotime_P'])
            create_labels['isotime_S'] = pd.to_datetime(create_labels['isotime_S'])

            init_date = self.stream_start_date.strftime("%Y-%m-%d %H:%M:%S")
            end_date = self.stream_end_date.strftime("%Y-%m-%d %H:%M:%S")

            # Filter DataFrame based on date range and that contains
            # both P and S per event
            create_labels = create_labels[
                                (create_labels['isotime_P'] >= init_date) &
                                (create_labels['isotime_P'] < end_date) &
                                (create_labels['isotime_S'] >= init_date) &
                                (create_labels['isotime_S'] < end_date) &
                                (create_labels['isotime_P'].notnull()) &
                                (create_labels['isotime_S'].notnull())]
            logger.debug("Total picks contained in stream")
            logger.debug(create_labels)
            if len(create_labels) > 0:
                # If there are picks in the time-frame of the miniseed
                for kk, vv in self.order.items():
                    _df = create_labels[create_labels["stationId_noChan"] == kk]
                    self.__label_data__(kk, vv, _df, delta)
        else:
            # ... initializing matrix (Y) to 0s anyway
            self.data_labels_ordered = np.zeros(
                                            (self.data_matrix_ordered.shape[0],
                                             self.data_matrix_ordered.shape[-1]))
            self.data_picks_P_ordered = np.zeros(
                                            (self.data_matrix_ordered.shape[0],
                                             self.data_matrix_ordered.shape[-1]))
            self.data_picks_S_ordered = np.zeros(
                                            (self.data_matrix_ordered.shape[0],
                                             self.data_matrix_ordered.shape[-1]))

        #  (C.) Create CFs
        HPRO = HeimdallProcessing(self.data_matrix_ordered, delta)
        self.cf_arr = HPRO.calculate_cf(**calculate_cf)

        #  (D.) Slicing
        (X, Y, R, PP, PS) = self.__slicing__(delta=delta, **slicing)

        Y = Y[:, :, np.newaxis, :]    # SLICES, NODES, CHAN, VALUES
        PP = PP[:, :, np.newaxis, :]  # SLICES, NODES, CHAN, VALUES
        PS = PS[:, :, np.newaxis, :]  # SLICES, NODES, CHAN, VALUES

        #  (E.) Normalizing each channel
        for _chan in range(X.shape[2]):
            if _chan < 2:
                # # Only for ENERGY and PCA
                X[:, :, _chan, :] = self.__normalize_cf_windows_BC_parallel__(
                                                X[:, :, _chan, :],
                                                nproc=8, chunk_size=500)

            X[:, :, _chan, :] = self.__normalize_cf_windows__(
                                            X[:, :, _chan, :], by="amplitude_chan")

        #  (F.) Plotting
        if debug_plot:
            for batch in range(X.shape[0]):
                sta_idx = 32
                Z, N, E = R[batch, sta_idx]
                cf_arr = X[batch, sta_idx, :, :]
                label_plot = Y[batch, sta_idx, 0]
                picks_plot_p = PP[batch, sta_idx, 0]
                picks_plot_s = PS[batch, sta_idx, 0]
                #
                fig = plt.figure(figsize=(9, 4.5))
                ax1 = fig.add_subplot(211)
                ax1.plot(N+E, color="darkgray", label="N+E", alpha=0.5, lw=1.5)
                ax1.plot(Z, color="black", label="Z", alpha=0.5, lw=1.5)
                ax1.set_title([key for key, val in self.order.items() if
                               val == sta_idx])
                plt.legend(loc="upper left")

                # Create the second y-axis sharing the same x-axis
                ax1a = ax1.twinx()
                _ = ax1a.plot(
                    picks_plot_p, label="picks_P", lw=1.5, alpha=0.7, color="blue")
                _ = ax1a.plot(
                    picks_plot_s, label="picks_S", lw=1.5, alpha=0.7, color="red")
                _ = ax1a.plot(
                    label_plot, label="label", lw=1.5, alpha=0.5, color="black")
                ax1a.set_ylim((-0.1, 1.1))
                plt.legend(loc="upper right")
                #
                ax2 = fig.add_subplot(212, sharex=ax1)

                for _n in range(cf_arr.shape[0]):
                    ax2.plot(cf_arr[_n],
                             label="CF_%1d" % (_n+1), lw=1.5, alpha=0.5)

                ax2.set_title("CFs")
                # ax2.set_ylim((-0.1, 1.1))
                plt.tight_layout()
                plt.legend()
                plt.show()

        return (X, Y, R)

    def prepare_data_real(self,
                          downsample=False,
                          slicing=False,
                          create_labels=False,
                          debug_plot=False):

        def __align_traces__(stream):
            # Determine the earliest start time and latest end time
            earliest_start = min([trace.stats.starttime for trace in stream])
            latest_end = max([trace.stats.endtime for trace in stream])

            # Create a new stream to hold aligned traces
            aligned_stream = obspy.core.Stream()

            for trace in stream:
                # Calculate the number of samples to pad at the start
                start_padding = int((trace.stats.starttime - earliest_start) /
                                    trace.stats.delta)
                end_padding = int((latest_end - trace.stats.endtime) /
                                  trace.stats.delta)

                # Copy the trace to avoid modifying the original stream
                new_trace = trace.copy()

                # Pad the trace at the start and end with zeros
                new_trace.data = np.pad(new_trace.data,
                                        (start_padding, end_padding),
                                        'constant')

                # Update the starttime to the earliest start time
                new_trace.stats.starttime = earliest_start

                # Append the new trace to the aligned stream
                aligned_stream += new_trace

            return aligned_stream

        def __fix_mixed_dt__(stream):
            logger.error("Fixing MIXED DT ... you should never be here!")
            breakpoint()

        def __fix_mixed_npts__(stream):
            logger.warning("Fixing MIXED NPTS")
            stream = __align_traces__(stream)

            if len(set([tr.stats.npts for tr in stream])) != 1:

                # ... still not there ...
                if (max(set([tr.stats.npts for tr in stream])) -
                   min(set([tr.stats.npts for tr in stream]))) >= 50:
                    logger.error("More than 1 second differences after "
                                 "trace alignment...something's wrong!")
                    breakpoint()

                # Go with the final HARDCUT
                hard_cut = min(set([tr.stats.npts for tr in stream]))
                for tr in stream:
                    tr.data = tr.data[:hard_cut]

            # double_check once again ...
            try:
                assert len(
                    set([tr.stats.npts for tr in stream])
                ) == 1
            except AssertionError:
                breakpoint()

            return stream

        def __fix_mixed_time__(stream):
            logger.warning("Fixing MIXED Start/End TIMES")
            _all_starts = list(set([
                            float(tr.stats.starttime) for tr in stream]))
            _all_ends = list(set([
                            float(tr.stats.endtime) for tr in stream]))
            try:
                assert np.all(np.abs(np.diff(_all_starts)) < 0.02)
                assert np.all(np.abs(np.diff(_all_ends)) < 0.02)
                assert len(set([tr.stats.npts for tr in stream])) == 1

            except AssertionError:
                logger.error("Not all STARTS DIFF are below 1/df !!! "
                             "Should not happen")
                breakpoint()

        # -------------------------------
        # --- 0. CHECKS
        if len(self.__stat_net_in_stream__ -
               self.__stat_net_in_order__) > 0:
            logger.warning("There are stations that will not be "
                           "extracted and processed:  %r" % (
                                    self.__stat_net_in_stream__ -
                                    self.__stat_net_in_order__))

        # -------------------------------
        # --- 1. DOWNSAMPLE STREAM
        if downsample and isinstance(downsample, dict):
            logger.info("Downsampling Stream ...")
            starttm = time.time()
            self.stream = GUT.downsampling_stream(
                             self.stream, **downsample,
                             copy=False)
            endtm = time.time()
            #
            logger.info("--> Total Time:  %.2f min." % ((endtm-starttm)/60.0))

        # =============================================
        # =============================================
        # -------------------------------
        # --- 2. QUALITY  CHECKS
        try:
            assert len(
                set([tr.stats.sampling_rate for tr in self.stream])
            ) == 1
            delta = 1.0/self.stream[0].stats.sampling_rate
        except AssertionError:
            __fix_mixed_dt__(self.stream)
            self.__update_class_stream_meta__()

        try:
            assert len(
                set([tr.stats.npts for tr in self.stream])
            ) == 1
        except AssertionError:
            self.stream = __fix_mixed_npts__(self.stream)
            self.__update_class_stream_meta__()

        try:
            assert len(
                set([tr.stats.starttime.time for tr in self.stream])
            ) == 1
        except AssertionError:
            __fix_mixed_time__(self.stream)
            self.__update_class_stream_meta__()
        #
        self.__update_class_stream_meta__()  # do it one more time

        # -------------------------------
        # --- 3. WORK
        #  (A.) Preparing Ordered matrix
        self.data_matrix_ordered = np.zeros((
                len(self.__stat_net_in_order__),
                3,  # channel
                self.stream[0].stats.npts
            ))

        for _net, _stat in self.__stat_net_in_order__:

            _st = self.stream.select(station=_stat, network=_net)
            if len(_st) == 0:
                logger.warning("Station:  %s.%s  can't be found on stream" %
                               (_net, _stat))
                continue

            try:
                assert len([_t.stats.channel[-1] for _t in _st]) == 3
                assert [_t.stats.channel[-1] for _t in _st] == ["Z", "N", "E"]
            except AssertionError:
                # leaving that channel equal to 0
                logger.warning("Station: %s  doesn't have 3 clean channels...skipping" % (
                                _net+"."+_stat))
                continue

            _mat = [_t.data for _t in _st]
            _mat_idx = self.order[_net+"."+_stat]
            self.data_matrix_ordered[_mat_idx, :, :] = np.array(_mat)

        #  (B.) Label data (Y)
        if create_labels is not None and create_labels is not False:
            # ------------------------------------------
            # CREATE LABELS is a DataFrame with the following keys:
            #   eventId, stationId_noChan, isotime_P, isotime_S,
            #   lowerUncertainty_P, lowerUncertainty_S,
            #   upperUncertainty_P, upperUncertainty_S
            # Mandatory are:
            #   eventId, isotime_P, isotime_S, stationId_noChan (i.e. 2C.BIT06)
            if isinstance(create_labels, (Path, str)):
                create_labels = pd.read_csv(create_labels)
            elif isinstance(create_labels, pd.DataFrame):
                pass
            else:
                raise TypeError("`create_labels` must be either a path to CSV "
                                "or pandas dataframe object")

            # ------------------------------------------
            # ... initializing matrix (Y)
            self.data_labels_ordered = np.zeros(
                                            (self.data_matrix_ordered.shape[0],
                                             self.data_matrix_ordered.shape[-1]))
            self.data_picks_P_ordered = np.zeros(
                                            (self.data_matrix_ordered.shape[0],
                                             self.data_matrix_ordered.shape[-1]))
            self.data_picks_Perr_ordered = np.zeros(
                                            (self.data_matrix_ordered.shape[0],
                                             self.data_matrix_ordered.shape[-1]))
            self.data_picks_S_ordered = np.zeros(
                                            (self.data_matrix_ordered.shape[0],
                                             self.data_matrix_ordered.shape[-1]))
            self.data_picks_Serr_ordered = np.zeros(
                                            (self.data_matrix_ordered.shape[0],
                                             self.data_matrix_ordered.shape[-1]))

            assert isinstance(create_labels, pd.DataFrame)
            logger.info("Creating labels")
            create_labels['isotime_P'] = pd.to_datetime(create_labels['isotime_P'])
            create_labels['isotime_S'] = pd.to_datetime(create_labels['isotime_S'])

            init_date = self.stream_start_date.strftime("%Y-%m-%d %H:%M:%S")
            end_date = self.stream_end_date.strftime("%Y-%m-%d %H:%M:%S")

            # Filter DataFrame based on date range and that contains
            # both P and S per event
            create_labels = create_labels[
                                (create_labels['isotime_P'] >= init_date) &
                                (create_labels['isotime_P'] < end_date) &
                                (create_labels['isotime_S'] >= init_date) &
                                (create_labels['isotime_S'] < end_date) &
                                (create_labels['isotime_P'].notnull()) &
                                (create_labels['isotime_S'].notnull())]
            logger.debug("Total picks contained in stream")
            logger.debug(create_labels)
            if len(create_labels) > 0:
                # If there are picks in the time-frame of the miniseed
                for kk, vv in self.order.items():
                    _df = create_labels[create_labels["stationId_noChan"] == kk]
                    self.__label_data__(kk, vv, _df, delta)
        else:
            # ... initializing matrix (Y) to 0s anyway
            self.data_labels_ordered = np.zeros(
                                            (self.data_matrix_ordered.shape[0],
                                             self.data_matrix_ordered.shape[-1]))
            self.data_picks_P_ordered = np.zeros(
                                            (self.data_matrix_ordered.shape[0],
                                             self.data_matrix_ordered.shape[-1]))
            self.data_picks_S_ordered = np.zeros(
                                            (self.data_matrix_ordered.shape[0],
                                             self.data_matrix_ordered.shape[-1]))

        #  (D.) Slicing
        (X, Y, R, PP, PS) = self.__slicing__(delta=delta, **slicing)
        _Y = Y[:, :, np.newaxis, :]    # SLICES, NODES, CHAN, VALUES
        PP = PP[:, :, np.newaxis, :]  # SLICES, NODES, CHAN, VALUES
        PS = PS[:, :, np.newaxis, :]  # SLICES, NODES, CHAN, VALUES
        Y = np.concatenate([_Y, PP, PS], axis=2)  # SLICES, NODES, 3, VALUES

        #  (E.) Normalizing each channel
        for _chan in range(X.shape[2]):
            # X[:, :, _chan, :] = self.__normalize_cf_windows__(
            #                             X[:, :, _chan, :], by="amplitude_chan")
            X[:, :, _chan, :] = self.__normalize_cf_windows__(
                                        X[:, :, _chan, :], by="std_chan")

        #  (F.) Plotting
        if debug_plot:
            for batch in range(X.shape[0]):
                sta_idx = 1
                Z, N, E = R[batch, sta_idx]
                cf_arr = X[batch, sta_idx, :, :]
                label_plot = Y[batch, sta_idx, 0]
                if not np.sum(label_plot) > 0: continue
                picks_plot_p = Y[batch, sta_idx, 1]
                picks_plot_s = Y[batch, sta_idx, 2]
                #
                fig = plt.figure(figsize=(9, 4.5))
                ax1 = fig.add_subplot(211)
                ax1.plot(N+E, color="darkgray", label="N+E", alpha=0.5, lw=1.5)
                ax1.plot(Z, color="black", label="Z", alpha=0.5, lw=1.5)
                ax1.set_title([key for key, val in self.order.items() if
                               val == sta_idx])
                plt.legend(loc="upper left")

                # Create the second y-axis sharing the same x-axis
                ax1a = ax1.twinx()
                _ = ax1a.plot(
                    picks_plot_p, label="picks_P", lw=1.5, alpha=0.7, color="blue")
                _ = ax1a.plot(
                    picks_plot_s, label="picks_S", lw=1.5, alpha=0.7, color="red")
                _ = ax1a.plot(
                    label_plot, label="label", lw=1.5, alpha=0.5, color="black")
                ax1a.set_ylim((-0.1, 1.1))
                plt.legend(loc="upper right")
                #
                ax2 = fig.add_subplot(212, sharex=ax1)

                for _n in range(cf_arr.shape[0]):
                    ax2.plot(cf_arr[_n],
                             label="CF_%1d" % (_n+1), lw=1.5, alpha=0.5)

                ax2.set_title("CFs")
                # ax2.set_ylim((-0.1, 1.1))
                plt.tight_layout()
                plt.legend()
                plt.show()

        return (X, Y, R)

    # ==============================================================
    # ==============================================================
    # ==================  PARALLEL
    # ==============================================================

    def __normalize_cf_windows_BC_parallel__(
                self, data, nproc=8, chunk_size=500):
        """Box-Cox normalise CF windows using a process pool.

        Args:
            data (np.ndarray): Array of shape ``(B, N, F)`` containing
                characteristic-function values (batches × stations × samples).
            nproc (int, optional): Worker process count. Default is ``8``.
            chunk_size (int, optional): Number of batches each worker
                receives at once. Default is ``500``.

        Returns:
            np.ndarray: The normalised array with identical shape.
        """
        logger.info("Normalizing BC --> multi CPU")

        B, N, F = data.shape

        def gen_chunks(array, size):
            for i in range(0, array.shape[0], size):
                yield i, array[i:i+size]

        out = np.empty_like(data)

        _stime = time.time()
        with ProcessPoolExecutor(max_workers=nproc) as executor:
            futures = []
            # Submit chunks
            for start_idx, batch_chunk in gen_chunks(data, chunk_size):
                futures.append(
                    executor.submit(self._process_chunk, batch_chunk, start_idx)
                )

            # Collect results
            for fut in futures:
                idx, result_chunk = fut.result()
                out[idx:idx+result_chunk.shape[0]] = result_chunk
        _etime = time.time()
        logger.info("Total Time BC parallel:  %.2f" % ((_etime - _stime)/60.0))
        return out

    # --> Worker function for multiprocessing
    def transform_batch(self, batch):
        """
        batch shape: (N, F)
        Apply station-wise transform for one batch (or chunk).
        Returns the transformed batch of shape (N, F).
        """
        N, F = batch.shape
        out = np.empty_like(batch)
        for station in range(N):
            row = batch[station]
            rmax, rmin = np.max(row), np.min(row)
            if rmax > 0 and (rmax - rmin) > 0.01:
                row[row == 0] = 0.01
                row_bc, _ = scipy.stats.boxcox(row)
                row_bc[row_bc < 0] = 0.0
                out[station] = row_bc
            else:
                out[station] = row
        return out

    def _process_chunk(self, chunk, start_idx):
        """
        Worker helper that transforms each row in 'chunk'.
        chunk shape: (chunk_size, N, F)
        """
        # Transform each (N, F) sub-array
        Bc, N, F = chunk.shape
        chunk_out = np.empty_like(chunk)
        for b_idx in range(Bc):
            batch_transformed = self.transform_batch(chunk[b_idx])
            chunk_out[b_idx] = batch_transformed
        return start_idx, chunk_out

    # ==============================================================
    # ==============================================================
    # ==============================================================

    def __normalize_cf_windows__(self, inmat, by="area", value=None):
        """In-place normalisation of CF windows.

        Args:
            inmat (np.ndarray): CF matrix of shape ``(B, N, F)``.
            by (str, optional): Normalisation keyword. One of::

                    {"area_chan", "amp_chan", "std_chan", "sqrt_chan",
                     "boxcox_chan", "amp_mat"}

                Defaults to ``"area_chan"``.
            value (float | None, optional): Extra parameter required by some
                methods (currently unused). Defaults to ``None``.

        Returns:
            np.ndarray: The same *inmat* instance, after scaling.
        """

        def __normalize_by_area__(time_series):
            area = np.trapz(time_series)  # Calculate the area under the curve
            if area != 0.0:
                # normalized_series = time_series / (area + self.epsilon)
                normalized_series = time_series / area
                return normalized_series
            else:
                # Do nothing
                return time_series

        logger.info("normalizing single windows by: %s" % by.upper())
        for _mat in tqdm(inmat):
            # For every window (1st dimension)
            if by.lower() in ("area_chan", "ar_chan"):
                for _stat in range(_mat.shape[0]):
                    # For every node (station, 2nd dimension)
                    _mat[_stat, :] = __normalize_by_area__(_mat[_stat, :])

            elif by.lower() in ("amplitude_chan", "amp_chan"):
                for _stat in range(_mat.shape[0]):
                    if not np.all(_mat[_stat, :] == 0):
                        # For every node (station, 2nd dimension)
                        _mat[_stat, :] /= np.nanmax(_mat[_stat, :])

            elif by.lower() in ("standarddeviation_chan", "std_chan"):
                for _stat in range(_mat.shape[0]):
                    if not np.all(_mat[_stat, :] == 0):
                        # For every node (station, 2nd dimension)
                        _mat[_stat, :] /= (np.nanstd(_mat[_stat, :]) + 1e-8)

            elif by.lower() in ("squareroot_chan", "sqrt_chan"):
                for _stat in range(_mat.shape[0]):
                    if not np.all(_mat[_stat, :] == 0):
                        # For every node (station, 2nd dimension)
                        _mat[_stat, :] /= np.sqrt(_mat[_stat, :])

            elif by.lower() in ("boxcox_chan", "bc_chan"):
                for _stat, row in enumerate(_mat):
                    row_max = np.nanmax(row)
                    row_min = np.nanmin(row)
                    if row_max > 0 and (row_max - row_min) > 0.01:
                        row[row == 0] = 0.01
                        try:
                            row_bc, _ = scipy.stats.boxcox(row)
                            row_bc[row_bc < 0] = 0
                            _mat[_stat] = row_bc
                        except ValueError:
                            logger.warning("Box-Cox failed on row %d. Possibly near-constant data.", _stat)

            elif by.lower() in ("amplitude_mat", "amp_mat"):
                _mat /= np.nanmax(_mat)

            else:
                logger.warning("No valid method specified for normalization! "
                               "Check source code!")

        return inmat

    def __slicing__(self,
                    delta,
                    wlen_seconds=None,
                    slide_seconds=None,
                    randomly_mute_nodes=False,
                    max_muted=5,
                    **kwargs):
        """Cut continuous data into overlapping windows.

        Args:
            delta (float): Sample interval (seconds).
            wlen_seconds (float): Window length (seconds).
            slide_seconds (float): Step between successive windows (seconds).
            randomly_mute_nodes (bool, optional): If ``True``, randomly set up
                to *max_muted* station traces to zero in each window.
                Defaults to ``False``.
            max_muted (int, optional): Maximum number of nodes to mute when
                *randomly_mute_nodes* is enabled. Defaults to ``5``.

        Returns:
            tuple[np.ndarray, ...]: ``(X, Y, R, picks_P, picks_S)`` where

            * **X** – windows used as input (shape ``W x N x C x F``)
            * **Y** – event label mask (``W x N x C x F``)
            * **R** – raw three-component windows (``W x N x 3 x F``)
            * **picks_P**, **picks_S** – Gaussian pick probability masks
              (each ``W x N x F``).
        """
        logger.info("Slicing Data (X and Y)")
        num_samples = self.data_matrix_ordered.shape[-1]

        # ---> 10 seconds are actually 501 seconds! (add +1)
        wlwn_idx = int(wlen_seconds / delta) + 1
        slide_idx = int(slide_seconds / delta)
        num_windows = int((num_samples - wlwn_idx) // slide_idx + 1)

        # windows_x_chan1, windows_x_chan2, windows_y, windows_real = [], [], [], []
        windows_x, windows_y, windows_real = [], [], []
        windows_picks_p, windows_picks_s = [], []

        for i in tqdm(range(num_windows)):
            start_index = i * slide_idx
            end_index = start_index + wlwn_idx

            # # X
            # _window = self.cf_arr[:, :, start_index:end_index]
            # windows_x.append(_window)

            # Y (even if not extracted/calculated before..is still a 0-matrix)
            _window = self.data_labels_ordered[:, start_index:end_index]
            windows_y.append(_window)

            # REAL
            _window = self.data_matrix_ordered[:, :, start_index:end_index]
            windows_real.append(_window)
            windows_x.append(_window)

            # PICKS (even if not extracted/calculated before..is still a 0-matrix)
            _window = self.data_picks_P_ordered[:, start_index:end_index]
            windows_picks_p.append(_window)

            # PICKS (even if not extracted/calculated before..is still a 0-matrix)
            _window = self.data_picks_S_ordered[:, start_index:end_index]
            windows_picks_s.append(_window)

        if randomly_mute_nodes:
            for win_family in (windows_x, windows_y, windows_real,
                               windows_picks_p, windows_picks_s):
                for win in win_family:
                    _nmute = np.random.randint(1, max_muted+1)
                    _nmute_idx = np.random.choice(np.arange(win.shape[0]),  # number of channels
                                                  size=_nmute, replace=False)
                    logger.debug("Muting %d nodes:  %s" % (_nmute, _nmute_idx))
                    win[_nmute_idx, :, :] = 0.0

        return (np.array(windows_x),
                np.array(windows_y),
                np.array(windows_real),
                np.array(windows_picks_p),
                np.array(windows_picks_s))

    def __label_data__(self, netstat, yidx, dfpicks, dt):
        """ Take care of populating the Y (labels) matrix.
            Works station-wise
        """

        if len(dfpicks) == 0:
            # No picks --> no label ... skip all!
            return False

        # -----------------------------------------------
        def __gaussian_kernel__(size, sigma):
            """ Create a Gaussian kernel. """
            kernel = np.linspace(-size // 2, size // 2, size)
            kernel = np.exp(-kernel**2 / (2 * sigma**2))
            kernel /= np.sum(kernel)
            return kernel

        def __scale_array__(arr, min_val, max_val):
            arr_min = np.min(arr)
            arr_max = np.max(arr)
            return (arr - arr_min) / (arr_max - arr_min) * (max_val - min_val) + min_val

        # -----------------------------------------------
        KERNEL_SIZE = 10
        KERNEL_SIGMA = 5
        df = 1/dt

        signal = self.data_labels_ordered[yidx]
        picks_p = self.data_picks_P_ordered[yidx]
        picks_s = self.data_picks_S_ordered[yidx]

        time_vector = np.arange(0, len(self.data_labels_ordered[yidx]))
        for _de, (xx, row) in enumerate(dfpicks.iterrows()):
            if _de > 0:
                logger.warning("More than one event in window !!! "
                               "Not allowed !!! Keeping the first only")
                break
            #
            P_time = obspy.UTCDateTime(row["isotime_P"])
            P_abserr = row["lowerUncertainty_P"] + row["upperUncertainty_P"]  # seconds
            S_time = obspy.UTCDateTime(row["isotime_S"])
            S_abserr = row["lowerUncertainty_S"] + row["upperUncertainty_S"]  # seconds
            ev_id = row["eventId"]

            # --- Create Label
            if (P_time >= self.stream_start_date and S_time <= self.stream_end_date):
                END_time = S_time + (S_time-P_time)*0.5
                P_idx = int((P_time - self.stream_start_date) * df)
                S_idx = int((S_time - self.stream_start_date) * df)
                END_IDx = int((END_time - self.stream_start_date) * df)

                # Create the square signal
                signal[(time_vector >= P_idx-KERNEL_SIZE/2) & (time_vector <= END_IDx+KERNEL_SIZE/2)] = 1  # With END
                picks_p[P_idx] = 1
                picks_s[S_idx] = 1

            elif (P_time <= self.stream_end_date and S_time > self.stream_end_date):
                END_time = S_time + (S_time-P_time)*0.5
                P_idx = int((P_time - self.stream_start_date) * df)  #
                S_idx = int((S_time - self.stream_start_date) * df)
                END_IDx = self.pca[yidx].shape[0]

                # Create the square signal
                signal[(time_vector >= P_idx-KERNEL_SIZE/2) & (time_vector <= END_IDx+KERNEL_SIZE/2)] = 1  # With END
                picks_p[P_idx] = 1
                picks_s[S_idx] = 1

            elif (P_time <= self.stream_start_date and S_time > self.stream_start_date):
                END_time = S_time + (S_time-P_time)*0.5
                P_idx = 0
                S_idx = int((S_time - self.stream_start_date) * df)
                END_IDx = int((END_time - self.stream_start_date) * df)

                # Create the square signal
                signal[(time_vector >= P_idx-KERNEL_SIZE/2) & (time_vector <= END_IDx+KERNEL_SIZE/2)] = 1  # With END
                picks_p[P_idx] = 1
                picks_s[S_idx] = 1

            else:
                # most likey, the pick analised is out of bounds!!
                # Therefore skip and move on
                logger.error("Pick for EvID: %d  @  Station: %s is out of bounds"  % (
                    ev_id, netstat))
                continue

        # --- End loop picks
        # Smooth with a Gaussian kernel --> maybe at the end of all
        kernel = __gaussian_kernel__(KERNEL_SIZE, KERNEL_SIGMA)  # OFFICIAL
        signal = scipy.signal.convolve(signal, kernel, mode='same')

        if sum(picks_p) > 0.0:
            P_abserr_samples = int(round(P_abserr*df + 1e-6))
            kernel_p = __gaussian_kernel__(
                            6*P_abserr_samples, P_abserr_samples)
            picks_p = scipy.signal.convolve(picks_p, kernel_p, mode='same')
            picks_p /= max(picks_p + 1e-6)  # ensure range 0-1

        if sum(picks_s) > 0.0:
            S_abserr_samples = int(round(S_abserr*df + 1e-6))
            kernel_s = __gaussian_kernel__(
                            6*S_abserr_samples, S_abserr_samples)
            picks_s = scipy.signal.convolve(picks_s, kernel_s, mode='same')
            picks_s /= max(picks_s + 1e-6)  # ensure range 0-1

        # --- ALLOCATE
        self.data_labels_ordered[yidx] = signal  # populate label matrix
        self.data_picks_P_ordered[yidx] = picks_p  # populate picks matrix
        self.data_picks_S_ordered[yidx] = picks_s  # populate picks matrix

# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================
# ==================================================================


class HeimdallProcessing(object):
    """ Handling CFs calculation and similar stuff.
        !!! DO NOT HANDLE LABEL nor SLICING HERE !!!
    """

    def __init__(self, R, delta, normalize="std"):
        """ The input R must be a 3-channel stream, no gaps,
            with always ZNE order.

            Normalization is done channel-wise!
        """
        self.R = R.copy()   # [nodes, channels, nsamples]
        self.delta = delta
        self.epsilon = 10**-6
        #
        if normalize:
            logger.debug("Normalizing real matrix before processing: %s" % normalize)
            self.R = self.R - np.mean(self.R, axis=2, keepdims=True)

            if normalize.lower() in ("std", "standard_deviation"):
                self.R = self.R / (
                    np.std(self.R, axis=2, keepdims=True) + self.epsilon
                    )

            elif normalize and normalize.lower() in ("max", "maximum"):
                self.R = self.R / (
                    np.max(self.R, axis=2, keepdims=True) + self.epsilon
                    )
            else:
                raise ValueError("Normalizing should be either 'std' or 'max'")

    # ----------------------------------------------------
    # -------------------------------------------  UTILS

    def __derivative__(self, inarr):
        """ calculate derivatives of input data -->
            done at station level 3 channel
        """
        inarr[:, :, 1:] -= inarr[:, :, :-1]  # Calculate differences along the second axis
        inarr[:, :, 0] = 0.0  # Set the first element of each array to 0
        return inarr

    def __normalize__(self, inarr, thr=None):
        """ Perform normalization based on maximum value/clipping value
        considering the values of all 3 channels at once """
        for i in range(inarr.shape[0]):
            # Normalize each row based on the maximum value of each station
            trmax = np.max(np.abs(inarr[i]))
            if isinstance(thr, (float, int)):
                # Normalize data only if the absolute maxima is larger
                # than a certain input threshold
                if trmax >= thr:
                    inarr[i] /= trmax
            else:
                # Normalize data by the absolute data maxima
                if trmax > 0:
                    inarr[i] /= trmax
        return inarr

    def __normalize_gpu__(inarr, thr=None):
        """Perform normalization based on maximum value/clipping value
        considering the values of all 3 channels at once.

        Args:
            inarr (torch.Tensor): Input tensor of shape (N, C, L) on CUDA device.
            thr (float, int, or None): Threshold for normalization.

        Returns:
            torch.Tensor: Normalized tensor on the same device.
        """
        # Compute the maximum absolute value for each station (row)
        trmax = torch.max(torch.abs(inarr), dim=-1, keepdim=True).values  # Shape: (N, C, 1)

        if thr is not None:
            # Apply threshold condition
            mask = (trmax >= thr)  # Shape: (N, C, 1)
            trmax = trmax * mask  # Zero out values below threshold

        # Avoid division by zero
        trmax[trmax == 0] = 1.0

        # Normalize the input array
        inarr = inarr / trmax

        return inarr

    def __hilbert__(self, trace):
        tracef = np.fft.fft(trace)
        nsta, nfreq = np.shape(tracef)
        freqs = np.fft.fftfreq(nfreq, self.delta)
        traceh = -1j*np.sign(freqs).T*tracef
        trace = trace+1j*np.fft.ifft(traceh).real
        return trace

    def __hilbert_transform_gpu__(self, trace):
        """
        GPU-compatible Hilbert transform using PyTorch.

        Args:
            trace (torch.Tensor): Input signal tensor of shape (nsta, nsamples), moved to GPU.

        Returns:
            torch.Tensor: Hilbert-transformed signal tensor of the same shape, on GPU.
        """
        # Perform FFT
        tracef = torch.fft.fft(trace, dim=-1)
        nsta, nfreq = tracef.shape
        freqs = torch.fft.fftfreq(nfreq, d=self.delta).to(trace.device)

        # Hilbert transform via multiplication with -1j * sign(freq)
        traceh = -1j * torch.sign(freqs) * tracef

        # Add original signal to inverse FFT of transformed signal
        trace_hilbert = trace + 1j * torch.fft.ifft(traceh, dim=-1).real

        return trace_hilbert

    # ----------------------------------------------------
    # ----------------------------------------------------

    def __pca_loki__(self,
                     derivative=False,
                     normalize=False,
                     sta_lta=False, **kwargs):

        logger.info("Calculating PCA")
        starttm = time.time()
        (nstation, nchan, ns) = self.R.shape
        mat = copy.deepcopy(self.R)
        #
        if derivative:
            mat = self.__derivative__(mat)
        if normalize:
            mat = self.__normalize__(mat, normalize)
        #
        obs_dataH = np.zeros([nstation, ns])
        obs_dataV = np.zeros([nstation, ns])

        obs_dataH1 = self.__hilbert__(mat[:, 2, :])  # East  (x)  Eq 1. (Vidale)
        obs_dataH2 = self.__hilbert__(mat[:, 1, :])  # North (y)
        obs_dataH3 = self.__hilbert__(mat[:, 0, :])  # Depth (z)

        obs_dataH1C = np.conjugate(obs_dataH1)
        obs_dataH2C = np.conjugate(obs_dataH2)
        obs_dataH3C = np.conjugate(obs_dataH3)
        xx = obs_dataH1*obs_dataH1C
        xy = obs_dataH1*obs_dataH2C
        xz = obs_dataH1*obs_dataH3C
        yx = obs_dataH2*obs_dataH1C
        yy = obs_dataH2*obs_dataH2C
        yz = obs_dataH2*obs_dataH2C
        zx = obs_dataH3*obs_dataH1C
        zy = obs_dataH3*obs_dataH2C
        zz = obs_dataH3*obs_dataH3C
        for i in range(nstation):
            for j in range(ns):
                cov3d = np.array([[xx[i, j], xy[i, j], xz[i, j]], [
                                  yx[i, j], yy[i, j], yz[i, j]], [zx[i, j], zy[i, j], zz[i, j]]])
                cov2d = np.array([[xx[i, j], xy[i, j]], [yx[i, j], yy[i, j]]])
                U2d, s2d, V2d = np.linalg.svd(cov2d, full_matrices=True)
                U3d, s3d, V3d = np.linalg.svd(cov3d, full_matrices=True)
                obs_dataV[i, j] = (s3d[0]**2)*(np.abs(V3d[0][2]))
                obs_dataH[i, j] = (s2d[0]**2)*(1-np.abs(V3d[0][2]))

            if abs(np.max(obs_dataH[i, :])) > 0:
                obs_dataH[i, :] = (
                    obs_dataH[i, :]/np.max(obs_dataH[i, :]))+0.00001
            if abs(np.max(obs_dataV[i, :])) > 0:
                obs_dataV[i, :] = (obs_dataV[i, :]/np.max(obs_dataV[i, :]))

        endtm = time.time()
        logger.info("--> Total Time:  %.2f min." % ((endtm-starttm)/60.0))

        # ---- STA/LTA
        if sta_lta and isinstance(sta_lta, list):
            for obs_data_pca in (obs_dataH, obs_dataV):
                for _row in range(obs_data_pca.shape[0]):
                    obs_data_pca[_row, :] = GUT.recursive_stalta(
                                                obs_data_pca[_row, :], self.delta,
                                                sta_lta[0], sta_lta[1], norm=False)
        return (obs_dataV, obs_dataH)

    def __pca__(self,
                derivative=False,
                normalize=False,
                sta_lta=False,
                pca_type="full", **kwargs):
        """
        Normalize can be either a number for normalizing to clipping threshold
        or simply True to normalize on maximum
        """

        logger.info("Calculating PCA --> %s" % pca_type)
        starttm = time.time()
        (nstation, nchan, ns) = self.R.shape
        mat = copy.deepcopy(self.R)  # MB: prob overflow
        #
        if derivative:
            mat = self.__derivative__(mat)
        if normalize:
            mat = self.__normalize__(mat, normalize)
        #
        obs_data_pca = np.zeros([nstation, ns])
        #
        obs_dataH1 = self.__hilbert__(mat[:, 2, :])  # East  (x)  Eq 1. (Vidale)
        obs_dataH2 = self.__hilbert__(mat[:, 1, :])  # North (y)
        obs_dataH3 = self.__hilbert__(mat[:, 0, :])  # Depth (z)
        rectilinearity = np.zeros(obs_dataH3.shape)
        obs_dataH1C = np.conjugate(obs_dataH1)
        obs_dataH2C = np.conjugate(obs_dataH2)
        obs_dataH3C = np.conjugate(obs_dataH3)
        xx = obs_dataH1*obs_dataH1C                                    # Eq. 2 (Vidale)
        xy = obs_dataH1*obs_dataH2C
        xz = obs_dataH1*obs_dataH3C
        yx = obs_dataH2*obs_dataH1C
        yy = obs_dataH2*obs_dataH2C
        yz = obs_dataH2*obs_dataH2C
        zx = obs_dataH3*obs_dataH1C
        zy = obs_dataH3*obs_dataH2C
        zz = obs_dataH3*obs_dataH3C
        for i in range(nstation):
            for j in range(ns):

                if pca_type.lower() in ('full', '3d', 'f'):
                    covmat = np.array([[xx[i, j], xy[i, j], xz[i, j]], [
                                        yx[i, j], yy[i, j], yz[i, j]], [
                                        zx[i, j], zy[i, j], zz[i, j]]])
                elif pca_type.lower() in ('horizontal', '2d', 'h'):
                    covmat = np.array([[xx[i, j], xy[i, j]],
                                      [yx[i, j], yy[i, j]]])
                else:
                    raise ValueError("Wrong PCA type specified ('3d' / '2d')")

                U, s, V = np.linalg.svd(covmat, full_matrices=True)
                eigenvalues = s**2

                obs_data_pca[i, j] = eigenvalues[0]
                rectilinearity[i, j] = (
                    1 - (eigenvalues[-1] / (eigenvalues[0]+self.epsilon))
                )

            if normalize and abs(np.max(obs_data_pca[i, :])) > 0:
                obs_data_pca[i, :] = (obs_data_pca[i, :] /
                                      np.max(obs_data_pca[i, :])) + self.epsilon
        #
        endtm = time.time()
        logger.info("--> Total Time:  %.2f min." % ((endtm-starttm)/60.0))

        # ---- STA/LTA
        if sta_lta and isinstance(sta_lta, list):
            for _row in range(obs_data_pca.shape[0]):
                obs_data_pca[_row, :] = GUT.recursive_stalta(
                                            obs_data_pca[_row, :], self.delta,
                                            sta_lta[0], sta_lta[1], norm=False)
        return (obs_data_pca, rectilinearity)

    def __pca_gpu__(self,
                    derivative=False,
                    normalize=False,
                    sta_lta=False,
                    pca_type="full",
                    **kwargs):

        logger.info("Calculating PCA - GPU --> %s" % pca_type)
        starttm = time.time()
        mat = torch.tensor(self.R, device=DEVICE)
        (nstation, nchan, ns) = mat.shape

        if derivative:
            mat = self.__derivative__(mat)
        if normalize:
            mat = self.__normalize_gpu__(mat, normalize)

        obs_data_pca = torch.zeros((nstation, ns), device=DEVICE)
        rectilinearity = torch.zeros((nstation, ns), device=DEVICE)

        # Hilbert Transform (use your GPU-compatible method here if available)
        obs_dataH1 = self.__hilbert_transform_gpu__(mat[:, 2, :])  # East
        obs_dataH2 = self.__hilbert_transform_gpu__(mat[:, 1, :])  # North
        obs_dataH3 = self.__hilbert_transform_gpu__(mat[:, 0, :])  # Depth

        obs_dataH1C = torch.conj(obs_dataH1)
        obs_dataH2C = torch.conj(obs_dataH2)
        obs_dataH3C = torch.conj(obs_dataH3)
        xx = obs_dataH1 * obs_dataH1C                   # Eq. 2 (Vidale)
        xy = obs_dataH1 * obs_dataH2C
        xz = obs_dataH1 * obs_dataH3C
        yx = obs_dataH2 * obs_dataH1C
        yy = obs_dataH2 * obs_dataH2C
        yz = obs_dataH2 * obs_dataH3C
        zx = obs_dataH3 * obs_dataH1C
        zy = obs_dataH3 * obs_dataH2C
        zz = obs_dataH3 * obs_dataH3C

        for i in range(nstation):
            for j in range(ns):
                if pca_type.lower() in ('full', '3d', 'f'):
                    covmat = torch.tensor([
                        [xx[i, j], xy[i, j], xz[i, j]],
                        [yx[i, j], yy[i, j], yz[i, j]],
                        [zx[i, j], zy[i, j], zz[i, j]]
                    ], device=DEVICE)
                elif pca_type.lower() in ('horizontal', '2d', 'h'):
                    covmat = torch.tensor([
                        [xx[i, j], xy[i, j]],
                        [yx[i, j], yy[i, j]]
                    ], device=DEVICE)
                else:
                    raise ValueError("Wrong PCA type specified ('3d' / '2d')")

                U, S, V = torch.svd(covmat)
                eigenvalues = S**2

                obs_data_pca[i, j] = eigenvalues[0]
                rectilinearity[i, j] = (
                    1 - (eigenvalues[-1] / (eigenvalues[0] + self.epsilon))
                )

            if normalize and torch.max(obs_data_pca[i, :]) > 0:
                obs_data_pca[i, :] = (obs_data_pca[i, :] /
                                      torch.max(obs_data_pca[i, :])) + self.epsilon
        #
        endtm = time.time()
        logger.info("--> Total Time:  %.2f min." % ((endtm-starttm)/60.0))

        # ---- STA/LTA
        if sta_lta and isinstance(sta_lta, list):
            for _row in range(obs_data_pca.shape[0]):
                obs_data_pca[_row, :] = GUT.recursive_stalta_gpu(
                                            obs_data_pca[_row, :], self.delta,
                                            sta_lta[0], sta_lta[1], norm=False)

        return obs_data_pca.cpu().numpy(), rectilinearity.cpu().numpy()

    def __incidence__(self):
        """Compute instantaneous incidence angle and modulus.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                *incidence*, *modulus* – both arrays have shape ``(N, F)``:

                * **incidence** – Normalised angle in ``[-1, 1]``
                  (−1 = down, 0 = horizontal, 1 = up).
                * **modulus** – Particle-motion magnitude.
        """
        logger.info("Calculating INCIDENCE+MODULUS")
        starttm = time.time()
        (nstation, nchan, ns) = self.R.shape
        mat = copy.deepcopy(self.R)  # MB: prob overflow
        #
        out_incidence = np.zeros([nstation, ns])
        out_modulus = np.zeros([nstation, ns])
        for i in range(nstation):
            vertical, north, east = mat[i, 0], mat[i, 1], mat[i, 2]
            hxy = np.hypot(north, east)
            modulus = np.hypot(hxy, vertical)
            # if no horizontal signal, set incidence to 0
            if np.max(hxy) > np.max(vertical) / 1000.0:
                incidence = np.arctan2(vertical, hxy)   # -pi (down) -> 0 (horiz) -> pi (up)
                incidence = incidence / (np.pi / 2.0)   # -1.0 (down) -> 0 (horiz) -> 1.0 (up)  # BUGFIX
            else:
                incidence = np.zeros_like(vertical)
            #
            out_incidence[i, :] = incidence
            out_modulus[i, :] = modulus
        #
        endtm = time.time()
        logger.info("--> Total Time:  %.2f min." % ((endtm-starttm)/60.0))
        return (out_incidence, out_modulus)

    def __z_energy__(self, sta_lta=False, **kwargs):

        logger.info("Calculating ENERGY --> Z-channel")
        starttm = time.time()
        #
        outarr = self.R[:, 0, :]**2  # LOKI original
        # ---- STA/LTA
        if sta_lta and isinstance(sta_lta, list):
            for _row in range(outarr.shape[0]):
                outarr[_row, :] = GUT.recursive_stalta(
                                            outarr[_row, :], self.delta,
                                            sta_lta[0], sta_lta[1],
                                            norm=False)
        #
        endtm = time.time()
        logger.info("--> Total Time:  %.2f min." % ((endtm-starttm)/60.0))
        return outarr

    def __hos__(self, **kwargs):
        def __kurtosis__(arr, szwin):
            # arr must be numpy.array
            # Precompute squared and quartic arrays
            arr2 = arr**2
            arr4 = arr2**2

            # Rolling window sum using NumPy's convolve
            sum4 = np.convolve(arr4, np.ones(szwin), mode='valid') / szwin
            sum2 = np.convolve(arr2, np.ones(szwin), mode='valid') / szwin

            # Initialize the kurtosis coefficient array with zeros
            kcf = np.zeros_like(arr, dtype=float)

            # Calculate kurtosis only for indices where the full window applies
            kcf[szwin-1:] = sum4 / (sum2**2 + self.epsilon)
            return kcf

        def __skewness__(arr, szwin):
            # arr must be numpy.array
            # Precompute squared and quartic arrays
            arr3 = arr**3
            arr4 = arr**4

            # Rolling window sum using NumPy's convolve
            sum4 = np.convolve(arr4, np.ones(szwin), mode='valid') / szwin
            sum3 = np.convolve(arr3, np.ones(szwin), mode='valid') / szwin

            # Initialize the kurtosis coefficient array with zeros
            scf = np.zeros_like(arr, dtype=float)

            # Calculate kurtosis only for indices where the full window applies
            scf[szwin-1:] = sum4 / (sum3**2 + self.epsilon)
            return scf

        logger.info("Calculating HOS")
        starttm = time.time()
        (nstation, nchan, ns) = self.R.shape
        mat = copy.deepcopy(self.R)  # MB: prob overflow
        out_kurt, out_skew = np.zeros(mat.shape), np.zeros(mat.shape)
        #
        kurt3, skew3 = np.zeros((nstation, ns)), np.zeros((nstation, ns))
        for i in range(nstation):
            for j in range(nchan):
                if np.any(mat[i, j, :]):

                    # Multi-window approach
                    kurt_window, skew_window = [], []
                    for _win in kwargs["windows"]:
                        kurt_vals = __kurtosis__(
                                        mat[i, j, :], int(_win/self.delta))
                        kurt_window.append(kurt_vals)

                        skew_vals = __skewness__(
                                        mat[i, j, :], int(_win/self.delta))
                        skew_window.append(skew_vals)

                    # Combine the results of multiple windows (MEDIAN)
                    out_kurt[i, j, :] = np.median(kurt_window, axis=0)
                    out_skew[i, j, :] = np.median(skew_window, axis=0)

            # Combine the three channels:
            # The RMS value is a measure of the magnitude of the vector
            # components, averaged over the three dimensions.
            kurt3[i, :] = np.sqrt((out_kurt[i, 0, :]**2 +
                                   out_kurt[i, 1, :]**2 +
                                   out_kurt[i, 2, :]**2) / 3)
            kurt3[i, :] -= 3
            kurt3[kurt3 <= 0] = 0.0

            skew3[i, :] = np.sqrt((out_skew[i, 0, :]**2 +
                                  out_skew[i, 1, :]**2 +
                                  out_skew[i, 2, :]**2) / 3)
            # SKEW can be negative, therefore do not filter out negative values!

            if kwargs['normalize_log']:
                skew3[i, :] = np.log10(skew3[i, :]+self.epsilon)
                skew3[skew3 <= 0] = 0.0
                kurt3[i, :] = np.log10(kurt3[i, :]+self.epsilon)
                kurt3[kurt3 <= 0] = 0.0
        #
        endtm = time.time()
        logger.info("--> Total Time:  %.2f min." % ((endtm-starttm)/60.0))
        return (kurt3, skew3)

    def __filter_picker__(self, **kwargs):

        def __calculate_fpcfs__(arr, **kwargs):
            # 1) Run FBsummary
            summary = FBSummary(arr,
                                npts=arr.shape[0],
                                sampling_rate=1/self.delta,
                                **kwargs)
            return np.array(summary.summary)

        logger.info("Calculating FILTERPICKER")
        starttm = time.time()
        (nstation, nchan, ns) = self.R.shape
        mat = copy.deepcopy(self.R)
        out_fp, fp3 = np.zeros(mat.shape), np.zeros((nstation, ns))

        for i in range(nstation):
            for j in range(nchan):
                if np.any(mat[i, j, :]):
                    out_fp[i, j, :] = __calculate_fpcfs__(mat[i, j, :])
                    # out_fp[i, j, :] /= np.max(out_fp[i, j, :])
                    out_fp[i, j, :] /= np.std(out_fp[i, j, :])
            # The RMS value is a measure of the magnitude of the vector
            # components, averaged over the three dimensions.
            fp3[i, :] = np.sqrt((out_fp[i, 0, :]**2 +
                                 out_fp[i, 1, :]**2 +
                                 out_fp[i, 2, :]**2) / 3)
        #
        endtm = time.time()
        logger.info("--> Total Time:  %.2f min." % ((endtm-starttm)/60.0))
        return fp3

    def __tkeo__(self, **kwargs):
        """Apply the Teager–Kaiser Energy Operator to the vertical channel.

        Returns:
            np.ndarray: TKEO energy for each station (shape ``N × F``) with
            negative values clipped to zero.
        """
        def __calculate_tkeo__(signal):
            # Initialize an output array with the same length as the input signal
            tkeo = np.zeros_like(signal)
            # Apply TKEO to each point, where applicable
            # Avoid the first and last point to prevent out-of-bounds indexing
            tkeo[1:-1] = signal[1:-1]**2 - signal[:-2] * signal[2:]
            return tkeo
        #
        logger.info("Calculating TKEO")
        (nstation, nchan, ns) = self.R.shape
        mat = copy.deepcopy(self.R)  # MB: prob overflow

        out_tkeo = np.zeros(mat.shape)
        tkeo3 = np.zeros((nstation, ns))
        for i in range(nstation):
            for j in range(nchan):
                if np.any(mat[i, j, :]):
                    out_tkeo[i, j, :] = __calculate_tkeo__(mat[i, j, :])
            #
            # tkeo3[i, :] = np.sqrt((out_tkeo[i, 0, :]**2 +
            #                        out_tkeo[i, 1, :]**2 +
            #                        out_tkeo[i, 2, :]**2) / 3)
            tkeo3[i, :] = out_tkeo[i, 0, :]
            tkeo3[tkeo3 <= 0] = 0.0
        #
        return tkeo3

    def calculate_cf(self, **kwargs):

        chan1 = self.__z_energy__(**kwargs['z_energy'])
        (chan2, _) = self.__pca__(**kwargs['pca'])
        # (chan2, _) = self.__pca_gpu__(**kwargs['pca'])
        (chan3, _) = self.__hos__(**kwargs['hos'])
        chan4 = self.__filter_picker__(**kwargs['fp'])

        # Add axis for channeld division + concatenate
        chan1 = chan1[:, np.newaxis, :]  # NODES, CHAN, VALUES
        chan2 = chan2[:, np.newaxis, :]  # NODES, CHAN, VALUES
        chan3 = chan3[:, np.newaxis, :]  # NODES, CHAN, VALUES
        chan4 = chan4[:, np.newaxis, :]  # NODES, CHAN, VALUES
        XX = np.concatenate((chan1, chan2, chan3, chan4), axis=1)

        return XX


class FBSummary():
    """Compute filter-bank characteristic functions (A.Lomax FilterPicker).

    Args:
        data (np.ndarray): One-dimensional waveform.
        npts (int, optional): Number of samples in *data*. Default ``501``.
        sampling_rate (float, optional): Samples per second. Default ``50``.
        t_long (int, optional): Long-term window length (seconds). Default ``4``.
        freqmin (float, optional): Minimum centre frequency (Hz). Default ``1``.
        corner (int, optional): Butterworth filter corners. Default ``1``.
        perc_taper (float, optional): Proportion of cosine taper. Default ``0.1``.
        mode (str, optional): Statistic mode ``"rms"`` or ``"std"``.
            Default ``"rms"``.
        t_ma (int, optional): Moving-average window (samples). Default ``20``.
        nsigma (int, optional): Sigma threshold multiplier. Default ``6``.
        t_up (float, optional): Up-trigger level (fraction). Default ``0.78``.
        nr_len (int, optional): Noise-reduction window (seconds). Default ``2``.
        nr_coeff (int, optional): Noise-reduction factor. Default ``2``.
        pol_len (int, optional): Polarity window (samples). Default ``10``.
        pol_coeff (int, optional): Polarity coefficient. Default ``10``.
        uncert_coeff (int, optional): Uncertainty coefficient. Default ``3``.

    Attributes:
        FC (np.ndarray): Characteristic functions, shape ``(n_bands, npts)``.
        BF (np.ndarray): Band-filtered traces, shape ``(n_bands, npts)``.
        summary (np.ndarray): Max‐over-bands FC, shape ``(npts,)``.
    """

    def __init__(self,
                 data,
                 npts=501,
                 sampling_rate=50.0,
                 t_long=4,
                 freqmin=1,
                 corner=1,
                 perc_taper=0.1,
                 mode='rms',
                 #
                 t_ma=20,
                 nsigma=6,
                 t_up=0.78,
                 nr_len=2,
                 nr_coeff=2,
                 pol_len=10,
                 pol_coeff=10,
                 uncert_coeff=3):

        self.data = data
        self.npts = npts
        self.sampling_rate = sampling_rate
        self.delta = 1/self.sampling_rate
        self.summary = None  # to make sure we reset everytime the array MB

        # --------------------------------
        self.t_long = t_long
        self.freqmin = freqmin
        self.cnr = corner
        self.perc_taper = perc_taper
        self.statistics_mode = mode
        self.t_ma = t_ma
        self.nsigma = nsigma
        self.t_up = t_up
        self.nr_len = nr_len
        self.nr_coeff = nr_coeff
        self.pol_len = pol_len
        self.pol_coeff = pol_coeff
        self.uncert_len = self.t_ma
        self.uncert_coeff = uncert_coeff
        # --------------------------------

        self.FC, self.BF = self._statistics_decay()
        self.summary = np.amax(self.FC, axis=0)

    def _rms(self, x, axis=None):
        """ Function to calculate the root mean square value of an array.
        """
        return np.sqrt(np.mean(x**2, axis=axis))

    def _N_bands(self):
        """ Determine number of band n_bands in term of sampling rate.
        """
        Nyquist = self.sampling_rate / 2.0
        n_bands = int(np.log2(Nyquist / 1.5 / self.freqmin)) + 1
        return n_bands

    def filter(self):
        """ Filter data for each band.
        """
        n_bands = self._N_bands()
        # create zeros 2D array for BF
        BF = np.zeros(shape=(n_bands, self.npts))

        for j in range(n_bands):
            octave_high = (self.freqmin + self.freqmin * 2.0) / 2.0 * (2**j)
            octave_low = octave_high / 2.0
            BF[j] = bandpass(
                            self.data, octave_low, octave_high,
                            self.sampling_rate,
                            corners=self.cnr,
                            zerophase=False)
            # BF[j] = cosine_taper(self.npts, self.perc_taper) * BF[j]

        return BF

    def get_summary(self):
        return self.summary

    def _statistics_decay(self):
        """ Calculate statistics for each band.
        """
        n_bands = self._N_bands()

        npts_t_long = int(self.t_long / self.delta)
        decay_const = 1.0 - (1.0 / npts_t_long)
        # one_minus_decay_const = 1.0 - decay_const
        decay_factor = self.delta / self.t_long
        decay_const = 1.0 - decay_factor

        # BF: band filtered data
        BF = self.filter()

        # E: the instantaneous energy
        E = np.power(BF, 2)

        # create zeros 2D array for rmsE, aveE and sigmaE
        aveE = np.zeros(shape=(n_bands, self.npts))
        if self.statistics_mode == 'rms':  # ALomax #
            rmsE = np.zeros(shape=(n_bands, self.npts))
        elif self.statistics_mode == 'std':  # ALomax #
            sigmaE = np.zeros(shape=(n_bands, self.npts))

        # range starts from 1, not 0, because recursive-decay algorithm requires previous value
        if self.statistics_mode == 'rms':  # ALomax #
            E_2 = np.power(E, 2)
            # lfilter(b, a, x, axis=- 1, zi=None)
            # a[0]*y[n] = b[0]*x[n] + b[1]*x[n-1]
            aveE = lfilter([decay_factor], [1.0, -decay_const], E_2, axis=1)
            sqrt_aveE = np.sqrt(aveE)
            rmsE = lfilter([decay_factor], [1.0, -decay_const], sqrt_aveE, axis=1)
        elif self.statistics_mode == 'std':  # ALomax #
            raise NotImplementedError(
                self.__class__.__name__ + "._statistics_decay(statistics_mode=='std')")

        # calculate statistics
        if self.statistics_mode == 'rms':
            FC = np.abs(E)/(rmsE + 1.0e-6)
        elif self.statistics_mode == 'std':
            FC = np.abs(E-aveE)/(sigmaE)

        # reassign FC values for the very beginning couple samples to avoid
        # unreasonable large FC from poor sigmaE
        S = self.t_long
        L = int(round(S/self.delta,0))
        for k in range(L):
            FC[:, k] = 0

        return FC, BF    #ALomax#
