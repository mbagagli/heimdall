"""
This module provides a detection class, HeimdallScanner, that processes model predictions
to identify potential seismic events and prepares data for further location analysis.

!!!  UNDER DEVELOPMENT  !!!

Classes:
    HeimdallScanner: Manages the detection of seismic events from prediction data, handles
                     cleaning, detection, and preparation of the data for a location algorithm.

"""

import sys
import copy as cp
import numpy as np
import scipy
from pathlib import Path
from heimdall import custom_logger as CL
logger = CL.init_logger(Path(sys.argv[0]).name, lvl="INFO")


class HeimdallScanner(object):

    """
    A class to manage the detection of seismic events from predictions, handling the cleaning, detection,
    and preparation of data for location algorithms.

    Attributes:
        predictions (numpy.ndarray): Input prediction data.
        start_time (datetime): Start time of the prediction data.
        df (float): The sampling rate of the data.
        fake_network_detections (list): Stores detections deemed unreliable.
        detections_at_network (list): Stores valid network-wide detections.
        detections_at_stations (dict): Stores detections on a per-station basis.
        window_cuts (dict): Stores sliced data around detections for location processing.
        window_cuts_index (dict): Stores the indices of the slices in window_cuts.
        window_cuts_locator (dict): Stores processed slices for locator input.

    Methods:
        __init__(self, predictions, start_time, df, **kwargs): Initializes the detector with data and settings.
        __clean_all_detections__(self): Resets all detections and related attributes to an empty state.
        __seeker__(self, array, threshold, sustain, ref_idx=False): Scans an array to find sequences exceeding a threshold.
        __cutter__(self, start_cut, end_cut, buffer=False): Cuts the prediction data around detected events.
        network_detections(self, threshold=0.5, sustain=50): Detects events across the network using summed predictions.
        detect(self, min_stations=4, sustain_sec=1, station_threshold=0.5, slice_buffer_sec=1): Orchestrates the detection process.
        prepare_cuts_for_locator(self): Prepares the detected events for localization by smoothing and filtering.
        get_stations_detections(self, copy=False): Retrieves the detected events at station level.
        get_network_detections(self, copy=False): Retrieves the network-wide detected events.
        get_windows_detections(self, copy=False): Retrieves the data windows around detected events for localization.
        get_windows_detections_locator(self, copy=False): Retrieves processed data windows for the locator.
        get_windows_detections_index(self, copy=False): Retrieves the indices of data windows used for localization.
    """

    def __init__(self, predictions, start_time, df, **kwargs):
        self.predictions = predictions
        self.start_time = start_time
        self.df = df
        #
        self.fake_network_detections = []
        self.detections_at_network = []
        self.detections_at_stations = {}
        self.window_cuts = {}
        self.window_cuts_index = {}

    def __clean_all_detections__(self):
        """
        Resets all detection lists and dictionaries to their initial empty state.

        This method is used to clear the current state of detections, including network-wide,
        per-station detections, and associated metadata. It should be called to reset the detector
        state before beginning a new detection process.

        Note:
            Total detections: fake_network_detections+detections_at_network
        """
        self.fake_network_detections = []
        self.detections_at_network = []
        self.detections_at_stations = {}
        self.window_cuts = {}
        self.window_cuts_index = {}
        self.window_cuts_locator = {}

    def __seeker__(self, array, threshold, sustain, ref_idx=False):
        """
        Identifies sequences within an array where values exceed a
        specified threshold for a sustained duration.

        Args:
            array (numpy.ndarray): The array to analyze, expected to be
                a 1D numpy array.
            threshold (float or int): The value above which a sequence
                is considered significant.
            sustain (int): The minimum number of consecutive samples
                above threshold required to consider a sequence
                significant.
            ref_idx (bool, optional): If True, adjusts the returned
                indices based on the original array indices.
                Defaults to False.

        Returns:
            list of tuples: Each tuple contains two integers representing the start and end indices of sequences
                            that meet the threshold and sustain criteria.
        """

        above_threshold = array >= threshold
        above_threshold_diff = np.diff(np.concatenate(
                                        ([0], above_threshold.astype(int), [0])
                               ))

        starts = np.where(above_threshold_diff == 1)[0]
        ends = np.where(above_threshold_diff == -1)[0] - 1

        if ref_idx:
            sequence_indices = [(start+ref_idx, end+ref_idx)
                                for start, end in zip(starts, ends)
                                if end - start + 1 >= sustain]
        else:
            sequence_indices = [(start, end)
                                for start, end in zip(starts, ends)
                                if end - start + 1 >= sustain]
        return sequence_indices

    def __cutter__(self, start_cut, end_cut, buffer=False):
        """
        Extracts a subsection from the prediction data, optionally adding a buffer before and after the specified indices.

        Args:
            start_cut (int): The starting index from which to begin cutting the data.
            end_cut (int): The ending index up to which to cut the data.
            buffer (int, optional): The number of samples to include before and after the start and end indices as a buffer. Defaults to 0.

        Returns:
            list: A list of numpy arrays corresponding to the subsection of each station's data, with buffer if specified.
        """

        # Slice network over precise index.
        # It returns the first chunk of the cutter, then we go for
        # station detection, make sure the buffer here is enough for coda, and
        # BUFFER IN SAMPLE!!!
        buffer = int(buffer)
        if buffer:
            mat = [stat[:, (start_cut-buffer):(end_cut+buffer)]
                   for _, stat in enumerate(self.predictions)]
        else:
            mat = [stat[:, start_cut:end_cut]
                   for _, stat in enumerate(self.predictions)]
        return mat

    def network_detections(self, threshold=0.5, sustain=50):
        """
        Detects events across the network by analyzing summed predictions and applying a threshold and sustain criteria.

        Args:
            threshold (float, optional): The threshold above which a detection is considered valid. Defaults to 0.5.
            sustain (int, optional): The minimum number of consecutive samples that must exceed the threshold. Defaults to 50.

        Returns:
            numpy.ndarray: The summed tensor of predictions used to detect events, for verification or further analysis.
        """
        summed_tensor = np.sum(self.predictions[:, 0, :],
                               axis=0, keepdims=True)
        self.detections_at_network = self.__seeker__(
                                            summed_tensor[0],
                                            threshold, sustain)
        return summed_tensor

    def detect(self, min_stations=4, sustain_sec=1,
               station_threshold=0.5, slice_buffer_sec=1):
        """
        Main method to control the detection process, applying network-wide
        and per-station detection criteria.

        Args:
            min_stations (int, optional): Minimum number of stations that
                must confirm a detection for it to be considered valid.
                Defaults to 4.
            sustain_sec (int, optional): Minimum duration in seconds for
                which predictions must be sustained to qualify as a detection.
                Defaults to 1.
            station_threshold (float, optional): Threshold for predictions
                at individual stations to qualify as detections.
                Defaults to 0.5.
            slice_buffer_sec (int, optional): Number of seconds to
                include as a buffer before and after each detected event
                when slicing the data. Defaults to 1.

        Returns:
            None: This method modifies the detector's state by populating detection lists and dictionaries.
        """

        # ------------------------------  Checks
        if not (len(self.predictions.shape) == 3 and
                self.predictions.shape[1] == 1):
            logger.error("Prediction has more than 1 channel !!!")
            sys.exit()

        # ----------------------------------------

        # Make sure to clean first ...
        self.__clean_all_detections__()

        # 1. STACK the predictions
        _net_sum_tensor = self.network_detections(threshold=min_stations,
                                                  sustain=sustain_sec*self.df)
        # plt.plot(_net_sum_tensor[0]); plt.show()

        # 2. See if there are predictions at NETWORK level
        if self.detections_at_network:
            logger.info("Network's detections:  %d" % len(
                                                self.detections_at_network))
            # 3. ... slice events
            for (xx, _det) in enumerate(self.detections_at_network):
                #
                if slice_buffer_sec:
                    _ot = int(_det[0] - slice_buffer_sec*self.df)
                else:
                    _ot = int(_det[0])
                _mat = self.__cutter__(_det[0], _det[1],
                                       buffer=slice_buffer_sec*self.df)

                self.detections_at_stations[xx] = [
                    (_s, self.__seeker__(stat[0],  # prediction channel
                                         station_threshold,
                                         sustain_sec*self.df,
                                         ref_idx=_ot))  # To refer to the init stream
                    for _s, stat in enumerate(_mat)
                ]

                # !!! Avoid FAKE network detections !!!
                if len([_z for _z in dict(self.detections_at_stations[xx]).values()
                        if _z]) < min_stations:
                    logger.warning("Event detected at NETWORK's level, but NOT at ANY STATIONS !!! "
                                   "Possibly fake or spike signal!")
                    self.fake_network_detections.append((xx, _det))
                    self.window_cuts[xx] = []
                    self.window_cuts_index[xx] = []
                else:
                    # ... go on ... PREPARE FOR LOCATOR !!!
                    _earliest_index = int(min([det[1][0][0] for det in
                                               self.detections_at_stations[xx]
                                               if det[1]]))
                    self.window_cuts[xx] = (self.predictions[
                                                :, :,
                                                int(_earliest_index-2*self.df):
                                                int(_earliest_index+8*self.df)+1])
                    self.window_cuts_index[xx] = (int(_earliest_index-2*self.df),
                                                  int(_earliest_index+8*self.df)+1)

            # 3. CLEAN UP FOR FAKE DETECTIONS
            if len(self.fake_network_detections) > 0:
                to_be_removed = [_ir for (_ir, _vr) in self.fake_network_detections]
                self.detections_at_network = [_val for _iii, _val in enumerate(
                                                    self.detections_at_network)
                                              if _iii not in to_be_removed]
                self.detections_at_stations = {_kk: _vv for _kk, _vv in
                                               self.detections_at_stations.items()
                                               if _kk not in to_be_removed}
                self.window_cuts = {_kk: _vv for _kk, _vv in
                                    self.window_cuts.items()
                                    if _kk not in to_be_removed}
                self.window_cuts_index = {_kk: _vv for _kk, _vv in
                                          self.window_cuts_index.items()
                                          if _kk not in to_be_removed}

            # 4. Prepare data for LOCATOR
            self.prepare_cuts_for_locator()

        else:
            logger.warning("Found no events in the network ...")

        return None

    def prepare_cuts_for_locator(self):
        """
        Prepares the slices of detection data for location analysis by
        applying a Gaussian kernel to smooth the data.

        This method processes the window cuts of detected events,
        applying a smoothing operation to facilitate more accurate
        location determination by the locator algorithm.

        Returns:
            None: The processed data is stored in the `window_cuts_locator`
            attribute for further use.
        """

        # -----------------------------------------------
        def __gaussian_kernel__(size, sigma):
            """ Create a Gaussian kernel. """
            kernel = np.linspace(-size // 2, size // 2, size)
            kernel = np.exp(-kernel**2 / (2 * sigma**2))
            kernel /= np.sum(kernel)
            return kernel
        # def __scale_array__(arr, min_val, max_val):
        #     arr_min = np.min(arr)
        #     arr_max = np.max(arr)
        #     return (arr - arr_min) / (arr_max - arr_min) * (max_val - min_val) + min_val
        # -----------------------------------------------
        KERNEL_SIZE = 10
        KERNEL_SIGMA = 5
        for (_ev, _det) in self.detections_at_stations.items():
            signal = np.zeros(self.window_cuts[_ev].shape)
            start_idx = self.window_cuts_index[_ev][0]
            # Fill detections with ones
            for ss in range(signal.shape[0]):
                for idxr in _det[ss][1]:
                    signal[ss][:, (idxr[0]-start_idx):(idxr[1]-start_idx)] = 1
                    # Smooth with a Gaussian kernel --> maybe at the end of all
                    kernel = __gaussian_kernel__(KERNEL_SIZE, KERNEL_SIGMA)
                    signal[ss, 0] = scipy.signal.convolve(
                                            signal[ss, 0], kernel, mode='same')
            # Store
            self.window_cuts_locator[_ev] = signal
            # import matplotlib.pyplot as plt
            # for ss in range(signal.shape[0]):
            #     plt.plot(signal[ss, 0, :])
            # plt.show()
            # breakpoint()
        #
        return None

    # ============================================================
    # ========================================  Getter/Setter

    def get_stations_detections(self, copy=False):
        """
        Retrieves the detections made at individual stations.

        Args:
            copy (bool, optional): If True, returns a deep copy of the detections to prevent modification of the original data. Defaults to False.

        Returns:
            dict or None: The dictionary containing detections at each station if available; otherwise, logs a warning and returns None if no detections have been made.
        """
        if self.detections_at_stations:
            if copy:
                return cp.deepcopy(self.detections_at_stations)
            else:
                return self.detections_at_stations
        else:
            logger.warning("No detections yet..run `detect` class-method before-hand!")

    def get_network_detections(self, copy=False):
        """
        Retrieves the detections identified across the network.

        Args:
            copy (bool, optional): If True, returns a deep copy of the network detections to prevent modification of the original data. Defaults to False.

        Returns:
            list or None: A list of detected events across the network if available; otherwise, logs a warning and returns None if no detections have been made.
        """
        if self.detections_at_network:
            if copy:
                return cp.deepcopy(self.detections_at_network)
            else:
                return self.detections_at_network
        else:
            logger.warning("No detections yet..run `detect` class-method before-hand!")

    def get_windows_detections(self, copy=False):
        """
        Retrieves the window cuts of the detected events, which are prepared
        slices of data around detected events.

        Args:
            copy (bool, optional): If True, returns a deep copy of the window cuts to prevent modification of the original data. Defaults to False.

        Returns:
            dict or None: A dictionary containing the sliced data around detected events if available; otherwise, logs a warning and returns None if no detections have been made.
        """
        if self.window_cuts:
            if copy:
                return cp.deepcopy(self.window_cuts)
            else:
                return self.window_cuts
        else:
            logger.warning("No detections yet..run `detect` class-method before-hand!")

    def get_windows_detections_locator(self, copy=False):
        """
        Retrieves the locator-specific detection windows, which are processed
        data windows used by the locator.

        Args:
            copy (bool, optional): If True, returns a deep copy of the detection windows for the locator to prevent modification of the original data. Defaults to False.

        Returns:
            dict or None: A dictionary containing processed data windows for the locator if available; otherwise, logs a warning and returns None if no detections have been made.
        """
        if self.window_cuts_locator:
            if copy:
                return cp.deepcopy(self.window_cuts_locator)
            else:
                return self.window_cuts_locator
        else:
            logger.warning("No detections yet..run `detect` class-method before-hand!")

    def get_windows_detections_index(self, copy=False):
        """
        Retrieves the indices of the detection windows, useful for understanding the temporal positioning of the detected events within the original data stream.

        Args:
            copy (bool, optional): If True, returns a deep copy of the detection window indices to prevent modification of the original data. Defaults to False.

        Returns:
            dict or None: A dictionary containing the indices of detection windows if available; otherwise, logs a warning and returns None if no detections have been made.
        """
        if self.window_cuts_index:
            if copy:
                return cp.deepcopy(self.window_cuts_index)
            else:
                return self.window_cuts_index
        else:
            logger.warning("No detections yet..run `detect` class-method before-hand!")
