"""
Noise synthesis and signal-plus-noise mixing helpers for the HEIMDALL
framework.

The module offers one public class, :class:`NoiseBrewery`, that can

* read a directory of MiniSEED noise traces,
* extract instrument sensitivities from a StationXML inventory,
* generate stochastic noise with random phase while preserving the
  station-specific amplitude spectrum,
* cut random time windows from the noise, and
* mix the crafted noise with a signal stream at a user-defined signal-to-noise
  ratio (SNR).

Dependencies
    numpy, scipy, obspy, matplotlib, tqdm, heimdall

"""

import sys
import copy
import obspy
import scipy
import numpy as np

from heimdall import utils as UT

from pathlib import Path
import matplotlib.pyplot as plt

from heimdall import custom_logger as CL
logger = CL.init_logger(Path(sys.argv[0]).name, lvl="INFO")


class NoiseBrewery(object):
    """Create realistic stochastic noise and mix it with seismic signals.

    Args:
        noise_stream (str | pathlib.Path | obspy.core.stream.Stream):
            Path to a folder containing ``*.mseed`` files **or** an ObsPy
            :class:`~obspy.core.stream.Stream` that already holds noise traces.
        opinventory (str | pathlib.Path):
            StationXML file from which to read instrument responses.
            The inventory is required to remove sensitivity and to keep
            consistent amplitudes across stations.

    Attributes:
        nst (obspy.core.stream.Stream): In-memory noise traces.
        inventory (obspy.core.inventory.Inventory): Parsed StationXML object.
        sensitivity (dict[str, float]): Mapping
            ``"NET.STA..Z" → instrument_sensitivity`` (counts per metre).
        avg_sensitivity (float): Mean sensitivity used as fallback when a
            specific channel code is missing in *sensitivity*.
    """
    def __init__(self, noise_stream, opinventory):
        logger.info("Importing NOISE")
        self.nst = self.import_noise(noise_stream)
        if isinstance(opinventory, (str, Path)):
            logger.info("Reading INVENTORY")
            self.inventory = obspy.read_inventory(str(opinventory))
        #
        logger.info("Extracting RESPONSES")
        (self.sensitivity, self.avg_sensitivity) = self.__get_sensitivity_full_network__()

    def __get_sensitivity_full_network__(self):
        """Extract sensitivities for all channels in the inventory.

        Returns:
            tuple[dict[str, float], float]:
                * Mapping ``"NET.STA..X" → sensitivity`` where *X* is the
                  channel code suffix **Z, N, E** (one value per component).
                * Mean sensitivity of the whole network, used as a default.
        """
        sensitivity = {}
        avg_sens = []
        for net in self.inventory:
            for sta in net:
                for cha in sta:
                    # invcode = f"{net.code}.{sta.code}.{cha.location_code}.{cha.code}"  # FG
                    invcode = f"{net.code}.{sta.code}..{cha.code[-1]}"  # MB --> to avoid conflict on site
                    try:
                        if cha.response and cha.response.instrument_sensitivity:
                            sens = cha.response.instrument_sensitivity.value
                            sensitivity[invcode] = sens
                            avg_sens.append(sens)
                        else:
                            print(f"No sensitivity information for {invcode}")
                    except Exception as e:
                        print(f"Error extracting sensitivity for {invcode}: {e}")

        return (sensitivity, np.mean(avg_sens) if avg_sens else 1)

    def __get_sensitivity__(self, invcode):
        """Return the sensitivity for a given channel.

        Args:
            invcode (str): Key of the form ``"NET.STA..X"`` where *X* is
                the channel component.

        Returns:
            float: Sensitivity in counts per metre.  Falls back to
            *avg_sensitivity* when the key is missing.
        """
        if invcode in self.sensitivity.keys():
            sensitivity = self.sensitivity[invcode]
        else:
            logger.warning(f"Missing sensitivity for {invcode}")
            sensitivity = self.avg_sensitivity
        return sensitivity

    def __select_trace_from_stream__(self, trace_id, copy=False):
        """Return a sub-stream that matches *trace_id*.

        Args:
            trace_id (str): ObsPy trace id, for example
                ``"XX.SSS..Z"`` or with wildcards.
            copy (bool, optional): When ``True`` the returned stream is
                deep-copied so that later edits do not affect *self.nst*.
                Default is ``False``.

        Returns:
            obspy.core.stream.Stream: Stream containing at least one trace.

        Raises:
            ValueError: If no trace matches *trace_id*.
        """
        wst = self.nst.select(id=trace_id)
        if not wst:
            raise ValueError("No noise-waveforms for %s" % trace_id)
        if copy:
            return wst.copy()
        else:
            return wst

    def __process_stream__(self, opst):
        """Demean, taper, and high-pass filter a stream in place.

        Args:
            opst (obspy.core.stream.Stream): Input stream.

        Returns:
            obspy.core.stream.Stream: The same stream after processing.
        """
        opst.detrend("demean")
        opst.taper(0.05, type='hann')
        opst.filter("highpass", freq=2.0, zerophase=True)
        return opst

    def __generate_stochastic_noise__(self, trace_id,
                                      remove_sensitivity=False,
                                      return_stats=False,
                                      window_size_seconds=None):
        """Create phase-randomised noise that matches a reference trace.

        Args:
            trace_id (str): Trace id (may contain wildcards) used to pick
                one noise trace from *self.nst*.
            remove_sensitivity (str | None, optional): Channel code whose
                sensitivity should be removed from the noise
                (``"NET.STA..X"``).  When ``None`` the raw counts are kept.
            return_stats (bool, optional): Reserved for future use;
                currently ignored.
            window_size_seconds (float | None, optional): Desired length of
                the output trace in seconds.  ``None`` or a non-positive
                value returns the full length of the reference trace.

        Returns:
            obspy.core.trace.Trace: New trace with the same header as the
            reference but containing stochastic noise.
        """
        tr = self.__select_trace_from_stream__(trace_id, copy=True)
        tr = self.__process_stream__(tr)
        assert len(tr) == 1, "Function expects exactly one trace after processing."
        tr = tr[0]

        if remove_sensitivity:
            logger.debug(f"Removing sensitivity of  {remove_sensitivity}")
            _noisens = self.__get_sensitivity__(remove_sensitivity)
            tr.data /= _noisens

        # ------------------------------------------------------------

        original_data = tr.data
        original_len = len(original_data)

        if window_size_seconds is None or window_size_seconds <= 0:
            window_size = original_len
        else:
            window_size = int(window_size_seconds / tr.stats.delta)

        # Zero-pad (or truncate)
        if window_size == original_len:
            padded_data = original_data
        else:
            padded_data = np.zeros(window_size, dtype=original_data.dtype)
            if window_size < original_len:
                padded_data[:] = original_data[:window_size]
            else:
                padded_data[:original_len] = original_data

        # SCIPY APPROACH
        fourier = scipy.fftpack.fft(padded_data)
        rnd_phase = np.exp(1j * np.random.uniform(-np.pi, np.pi,   # 0, 2*np.pi
                                                  len(fourier)))
        # Apply the random phase to the Fourier Transform
        randomized_transform = np.abs(fourier) * rnd_phase
        noise_ts = scipy.fftpack.ifft(randomized_transform)
        tr.data = noise_ts.real.astype(original_data.dtype, copy=False)

        # # NUMPY APPROACH
        # fourier = np.fft.rfft(padded_data)
        # rnd_phase = -np.pi + 2*np.pi*np.random.rand(np.size(padded_data))
        # noise_ts = np.abs(fourier) * np.exp(-1j * rnd_phase)
        # noise_ts = np.fft.irfft(noise_ts)
        # tr.data = noise_ts.astype(original_data.dtype, copy=False)

        return tr

    def __extract_random_window__(self, optr, window_size):
        """Crop a random time window from a trace, in place.

        Args:
            optr (obspy.core.trace.Trace): Trace whose data will be
                truncated.
            window_size (int): Number of samples to keep.  Must be smaller
                than ``len(optr.data)``.

        Returns:
            obspy.core.trace.Trace: The same trace instance with shortened
            data.

        Raises:
            ValueError: If *window_size* is not smaller than the trace
            length.
        """
        if window_size >= len(optr.data):
            raise ValueError("window_size must be smaller than the length of the data")

        start_index = np.random.randint(0, len(optr.data) - window_size)
        optr.data = optr.data[start_index:start_index+window_size]
        return optr

    # ============================================  PUBLIC
    def import_noise(self, noise_ref):
        """Load noise traces from a directory or return a given stream.

        Args:
            noise_ref (str | Path | obspy.core.stream.Stream):
                * **str / Path** – Folder that contains ``*.mseed`` files.
                * **Stream** – Already loaded noise traces.

        Returns:
            obspy.core.stream.Stream: Combined noise stream.

        Raises:
            TypeError: If *noise_ref* is of an unsupported type.
        """
        logger.info("Importing NOISE")
        if isinstance(noise_ref, (str, Path)):
            nst = obspy.core.stream.Stream()
            for fff in Path(noise_ref).glob("*.mseed"):
                nst += obspy.read(fff, format="MSEED")
            return nst
        elif isinstance(noise_ref, obspy.core.stream.Stream()):
            return nst
        else:
            raise TypeError("NoiseStream must be either one of "
                            "str/path/obspy.stream type object!")

    def get_noise_stream(self):
        """Return the cached noise stream.

        Returns:
            obspy.core.stream.Stream | None: The noise stream, or ``None``
            when it has not been loaded.
        """
        if self.nst:
            return self.nst

    def combine_noise_signal(self, opst, snr=None, plot=False):
        """Add synthetic noise to a signal stream.

        The function loops over each trace in *opst*, generates a matching
        noise trace, rescales it to reach the requested SNR, and returns
        both the mixed signal and the pure noise.

        Args:
            opst (obspy.core.stream.Stream): Signal stream to which noise
                will be added.  Each trace must have correct network,
                station, and channel codes.
            snr (float | int | None, optional): Desired signal-to-noise
                ratio.  When ``None`` the noise is added without scaling.
            plot (bool, optional): If ``True`` show a three-panel Matplotlib
                figure of original signal, noise, and mixed result for each
                trace.

        Returns:
            tuple[obspy.core.stream.Stream, obspy.core.stream.Stream]:
                *mixed_stream*, *noise_only_stream*.

        Notes:
            * The same SNR is applied to every station.
            * High-pass filtering and down-sampling of the noise are handled
              automatically to match each signal trace.
        """
        synthXst = opst.copy()
        noiseXst = opst.copy()

        for (_sig_tr, _onlynoise_tr) in zip(synthXst, noiseXst):
            _trace_id = ".".join([_sig_tr.stats.network,
                                  _sig_tr.stats.station,
                                  "*", "*"+_sig_tr.stats.channel[-1][-1]])
            _noise_sens_id = ".".join([_sig_tr.stats.network,
                                       _sig_tr.stats.station,
                                       "", _sig_tr.stats.channel[-1][-1]])

            _noi_tr = self.__generate_stochastic_noise__(
                    _trace_id, remove_sensitivity=_noise_sens_id,
                    window_size_seconds=(
                        (_sig_tr.stats.endtime - _sig_tr.stats.starttime) + 1))

            # -----------------  1. Adjust SamplingRate
            try:
                assert _sig_tr.stats.sampling_rate == _noi_tr.stats.sampling_rate
            except AssertionError:
                _noi_tr = UT.downsampling_trace(
                                _noi_tr, _sig_tr.stats.sampling_rate,
                                hp_freq=2.0,
                                taper=True,
                                copy=False)
            #
            assert _sig_tr.stats.sampling_rate == _noi_tr.stats.sampling_rate

            # -----------------  2. Adjust VectorLength
            try:
                assert len(_sig_tr.data) == len(_noi_tr.data)
            except AssertionError:
                self.__extract_random_window__(
                        _noi_tr, len(_sig_tr.data))
            #
            assert len(_sig_tr.data) == len(_noi_tr.data)

            # -----------------  3. Adjust Amplitude to SNR
            if snr and isinstance(snr, (int, float)):
                logger.warning(f"Scaling NOISE by SNR: {snr}")
                _noise_max = np.max(np.abs(_noi_tr.data))
                _sig_max = np.max(np.abs(_sig_tr.data))
                _noise_ref = _sig_max/snr
                # Scale the noise array to the new maximum value
                _scaled_noise_data = [(a / _noise_max) * _noise_ref for a in _noi_tr.data]
            else:
                _scaled_noise_data = _noi_tr.data

            if plot:
                _sig_tr_data_original = copy.deepcopy(_sig_tr.data)

            _sig_tr.data = _sig_tr.data + np.array(_scaled_noise_data)  # Combine the signals
            _onlynoise_tr.data = np.array(_scaled_noise_data)           # Return only a noise stream

            # -----------------  4. Optionally plot
            if plot:
                fig = plt.figure()
                ax1 = fig.add_subplot(3, 1, 1)
                ax1.plot(_sig_tr_data_original)
                ax2 = fig.add_subplot(3, 1, 2)
                ax2.sharex(ax1)
                ax2.plot(_scaled_noise_data)
                ax3 = fig.add_subplot(3, 1, 3)
                ax3.plot(_sig_tr.data)
                ax3.sharex(ax1)
                plt.show()
        #
        return (synthXst, noiseXst)
