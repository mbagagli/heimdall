"""
Magnitude utilities for the HEIMDALL project.

This module contains a thin wrapper around ObsPy routines that lets you
compute local or relative magnitudes from station traces, a station
meta-data inventory, and a dictionary of reference amplitudes.

Main class
----------
HeimdallMagnitude
    Loads an ObsPy inventory and, if available, a dictionary that maps
    each network.station code to a reference magnitude and reference
    peak-to-peak amplitude.  Provides the `calculate_magnitude()`
    method to derive an event magnitude from multiple stations.

Functions are embedded as private helpers inside the class.

Dependencies
------------
* numpy
* obspy
* tqdm
* heimdall.custom_logger
"""

import sys
from pathlib import Path
from tqdm import tqdm
#
import numpy as np
import obspy
from obspy.core.inventory import Inventory
from obspy.signal.invsim import estimate_magnitude
#
from obspy.geodetics.base import degrees2kilometers, locations2degrees
from heimdall import custom_logger as CL

logger = CL.init_logger(Path(sys.argv[0]).name, lvl="INFO")


class HeimdallMagnitude(object):
    """Compute local or relative event magnitudes.

    The class supports two strategies:

    * **Relative** – Use a pre-computed dictionary that maps
      ``"NET.STA"`` to ``(M_abs, A_abs)`` where *M_abs* is a reference
      magnitude and *A_abs* its associated peak-to-peak amplitude.
      Each new station measurement is scaled against that pair.

    * **Absolute** – Convert raw counts to ground motion using the
      station response (ObsPy ``Inventory``) and apply
      `obspy.signal.invsim.estimate_magnitude()`.

    Parameters
    ----------
    opinv : str | pathlib.Path | obspy.core.inventory.Inventory
        Path to a StationXML file or an already-loaded inventory.
    ref_amps_dict : str | pathlib.Path | dict | None, optional
        Either a ``.npz`` file produced by HEIMDALL or a mapping
        ``{"NET.STA": [M_abs, A_abs], ...}``.  Required for the
        *relative* method.  Default is ``None`` (only *absolute* can be
        used).

    Attributes
    ----------
    inv : obspy.core.inventory.Inventory
        Station meta-data.
    reference_amplitudes : dict | None
        The reference amplitude dictionary or ``None`` when not
        supplied.
    """
    def __init__(self, opinv, ref_amps_dict=None):
        """Instantiate the magnitude helper.

        Args:
            opinv (str | Path | Inventory): Inventory file path or object.
            ref_amps_dict (str | Path | dict | None, optional): Reference
                amplitude mapping used by the *relative* magnitude
                method.  Can be a ``.npz`` file or an in-memory dict.
        """
        self.inv = self.__load_inv__(opinv)

        # The next dict contains the reference amplitude for the reference
        # magnitudes acalculations. Key: Net.Stat Val: [magnitudeAbs, Max Hampl]
        if ref_amps_dict:
            self.reference_amplitudes = self.__load_ref_amps__(ref_amps_dict)  # {"NET.STAT": [float, float], ...}
        else:
            logger.warning("Reference Amplitudes missing ... Only ABSOLUTE "
                           "method will be used in calculations")
            self.reference_amplitudes = None

    def __load_inv__(self, inv):
        """Load and return an ObsPy inventory.

        Args:
            inv (str | Path | Inventory): StationXML path or already
                loaded inventory.

        Returns:
            Inventory: Parsed station meta-data.

        Raises:
            TypeError: When *inv* is neither a path nor an ``Inventory``.
        """
        if isinstance(inv, (str, Path)):
            return obspy.read_inventory(inv)
        elif isinstance(inv, Inventory):
            return inv
        else:
            raise TypeError("Erroneous type of input for INV:  %s" % type(inv))

    def __load_ref_amps__(self, ind):
        """Load the reference amplitude dictionary.

        Args:
            ind (str | Path | dict): ``.npz`` file created by HEIMDALL or
                a dict of the form
                ``{"NET.STA": [M_abs, A_abs], ...}``.

        Returns:
            dict: The reference amplitude mapping.

        Raises:
            TypeError: If *ind* is neither a path nor a dict.
        """
        if isinstance(ind, (str, Path)):
            _de = np.load(ind, allow_pickle=True)
            return _de["mag_dict"].item()
        elif isinstance(ind, dict):
            return ind
        else:
            raise TypeError("Erroneous type of input for AMPS:  %s" % type(ind))

    def __calculate_epicentral_distance__(self, lon1, lat1, lon2, lat2):
        """Return great-circle distance in kilometres.

        Args:
            lon1 (float): Longitude of point 1 (degrees).
            lat1 (float): Latitude of point 1 (degrees).
            lon2 (float): Longitude of point 2 (degrees).
            lat2 (float): Latitude of point 2 (degrees).

        Returns:
            float: Distance in kilometres.
        """
        dist = degrees2kilometers(locations2degrees(lat1, lon1, lat2, lon2))
        return dist

    def __event_magnitude__(self, arraw, mad_thr=3):
        """Robust average of per-station magnitudes.

        A Median Absolute Deviation (MAD) filter removes outliers.

        Args:
            arraw (np.ndarray): Vector of station magnitudes.
            mad_thr (int, optional): Keep values within
                ``mad_thr * MAD`` of the median.  Default is ``3``.

        Returns:
            tuple:
                * ((mean_cln, std_cln, median_cln, mad_cln),
                * (mean_raw, std_raw, median_raw, mad_raw),
                * valid_obs_count,
                * total_obs_count)
        """
        if len(arraw) < 4:
            return ((None, None, None, None),
                    (None, None, None, None), len(arraw), len(arraw))
        inarr = arraw[np.isfinite(arraw)]
        mn = np.mean(inarr)
        std = np.std(inarr)
        mdn = np.median(inarr)
        mad = np.median(np.abs(inarr - np.median(inarr)))

        # Filter elements that are >= mad_thr * mad
        cleaned_arr = inarr[np.abs(inarr - mdn) < mad_thr * mad]

        # Calculate mean, std, median, and mad of the cleaned array
        cleaned_mean = np.mean(cleaned_arr)
        cleaned_std = np.std(cleaned_arr)
        cleaned_median = np.median(cleaned_arr)
        cleaned_mad = np.median(np.abs(cleaned_arr - cleaned_median))

        return ((cleaned_mean, cleaned_std, cleaned_median, cleaned_mad),
                (mn, std, mdn, mad), len(cleaned_arr), len(inarr))

    def calculate_magnitude(self, evlon, evlat, stations,
                            event_time=None, epi_thr=9999.9,
                            method="relative",  # "absolute"
                            trace_df=50.0):
        """Estimate the event magnitude from multiple stations.

        Args:
            evlon (float): Event longitude (degrees).
            evlat (float): Event latitude (degrees).
            stations (list): Sequence ``[(net_sta, data), ...]`` where
                *net_sta* is ``"NET.STA"`` and *data* is a
                ``3 x N`` NumPy array (Z, N, E channels in counts).
            event_time: Event origin time.  Required when the inventory
                contains epoch-dependent response information.
            epi_thr (float, optional): Maximum epicentral distance
                (kilometres) for stations to be used.  Default 9999.9.
            method (str, optional): ``"relative"`` or ``"absolute"``.
                Default ``"relative"``.
            trace_df (float, optional): Sampling rate of *data*
                (samples per second).  Default 50.0.

        Returns:
            tuple[float | None, float | None, int, int]:
                *median_mag*, *mad_mag*, *valid_station_count*,
                *total_station_count*.

                Returns ``(None, None, 0, total)`` when estimation fails.

        Raises:
            ValueError: If *method* is not ``"relative"`` or ``"absolute"``.
        """
        def __extract_channel_amplitude__(inarr, df=50.0):

            array = np.abs(inarr)
            amp1, amp2 = np.sort(array)[-2:]
            p2p = amp1+amp2+1e-6
            index1, index2 = np.argsort(array)[-2:]
            time_span_seconds = np.abs(index1 - index2) / df

            if time_span_seconds >= 2.0:
                logger.debug("peak-to-peak distance is higher than 2 seconds! Rejecting ...")
                return (False, False)

            return (p2p, time_span_seconds)

        if method.lower() in ("r", "rel", "relative"):
            if not self.reference_amplitudes:
                logger.error("Reference Amplitudes Dict MISSING! Abort ...")
                return (False, False, False)
            logger.info("Estimating MAGNITUDE (MRel)")
            _stations_magnitude = []
            for _x, (_netstat, _arr) in tqdm(enumerate(stations)):
                # Get reference mag
                _net, _stat = _netstat.split(".")
                Mabs, Aabs = self.reference_amplitudes[_netstat]
                _stat_inv = self.inv.select(
                    station=_stat, network=_net, time=event_time)
                _stat_epidist = self.__calculate_epicentral_distance__(
                                        evlon, evlat,
                                        _stat_inv[0][0].longitude,
                                        _stat_inv[0][0].latitude)

                if _stat_epidist < epi_thr:
                    # CALC MAGNITUDE
                    _amp_N, _ = __extract_channel_amplitude__(_arr[1, :],
                                                              df=trace_df)
                    _amp_E, _ = __extract_channel_amplitude__(_arr[2, :],
                                                              df=trace_df)

                    if _amp_E and _amp_N:
                        Ar = (_amp_N+_amp_E)/2
                        # Mrel = (Mabs * np.log10(Ar)) / np.log10(Aabs)  # old
                        Mrel = Mabs + np.log10(Ar/Aabs)
                        _stations_magnitude.append(Mrel)

                    else:
                        logger.warning("Station %s no valid Horizontal Measurement!" % _netstat)

                else:
                    logger.warning("Station %s is too far! (%.1f km / %.1f km)" % (
                                    _netstat, _stat_epidist, epi_thr))

        elif method.lower() in ("a", "abs", "absolute"):
            logger.info("Estimating MAGNITUDE (MLv)")
            _stations_magnitude = []
            for _x, (_netstat, _arr) in tqdm(enumerate(stations)):
                # _arr shape  row ==> components
                _net, _stat = _netstat.split(".")
                _stat_inv = self.inv.select(
                    station=_stat, network=_net, time=event_time)
                # _stat_resp_Z = _stat_inv.select(channel="*Z")[0][0][0].response
                _stat_resp_N = _stat_inv.select(channel="*N")[0][0][0].response
                _stat_resp_E = _stat_inv.select(channel="*E")[0][0][0].response

                _stat_epidist = self.__calculate_epicentral_distance__(
                                        evlon, evlat,
                                        _stat_inv[0][0].longitude,
                                        _stat_inv[0][0].latitude)

                if _stat_epidist < epi_thr:
                    # CALC MAGNITUDE
                    assert _stat_resp_N == _stat_resp_E
                    amp_N, h_N = __extract_channel_amplitude__(_arr[1, :],
                                                               df=trace_df)
                    amp_E, h_E = __extract_channel_amplitude__(_arr[2, :],
                                                               df=trace_df)
                    amp_list = []
                    step_list = []
                    if amp_N and h_N:
                        amp_list.append(amp_N)
                        step_list.append(h_N)
                    if amp_E and h_E:
                        amp_list.append(amp_E)
                        step_list.append(h_E)
                    #
                    if len(amp_list) > 0 and len(step_list) > 0:
                        _stat_mag = estimate_magnitude(_stat_resp_N,
                                                       amp_list, step_list,
                                                       _stat_epidist)
                        _stations_magnitude.append(_stat_mag)
                    else:
                        logger.warning("Station %s no valid Horizontal Measurement!" % _netstat)

                else:
                    logger.warning("Station %s is further than threshold! (%.1f / %.1f)" % (
                                    _netstat, _stat_epidist, epi_thr))

        else:
            raise ValueError("Method type can only be 'relative' or 'absolute'!")
        #
        (_cleaned, _raw, valid_obs, all_obs) = self.__event_magnitude__(np.array(
                                                        _stations_magnitude))
        (_, _, cln_median, cln_mad) = _cleaned
        (_, _, raw_median, raw_mad) = _raw
        return (cln_median, cln_mad, valid_obs, all_obs)
