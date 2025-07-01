"""
Generic utility functions used across the HEIMDALL project.

Content
    Time-series processing
        * `downsampling_trace`
        * `downsampling_stream`
        * `recursive_stalta`
        * `recursive_stalta_gpu`
        * `envelope_from_wavelet`

    Array conversion helpers
        * `__numpy_array_to_pytorch_tensor__`
        * `__pytorch_array_to_numpy__`

    Data I/O
        * `pickle_data`
        * `unpickle_data`

    Window merging
        * `__merge_windows__`

All code is NumPy / SciPy based (CUDA only for the *_gpu* version of
recursive STA/LTA).
"""


import sys
import pickle
import torch
import numpy as np
import scipy
import pywt
#
from pathlib import Path
from heimdall import custom_logger as CL
from tqdm import tqdm

logger = CL.init_logger(Path(sys.argv[0]).name, lvl="INFO")


# =========================================================  MISC
# ===================================================================

def __merge_windows__(matrix, overlap_size, method="avg"):
    """Stitch a batch of overlapping windows into one long record.

    Args:
        matrix (np.ndarray): Array with shape
            ``(num_windows, num_nodes, num_channels, win_len)``.
        overlap_size (int): Number of samples by which adjacent windows
            overlap.
        method (str, optional): Strategy for the overlap region:

            * ``"avg"`` – average the two windows (default)
            * ``"max"`` – take the element-wise maximum
            * ``"shift"`` – keep the later window untouched

    Returns:
        np.ndarray: Merged array with shape
        ``(num_nodes, num_channels, new_len)``.

    Raises:
        SystemExit: If *method* is not one of the recognised keywords.
    """
    X, M, C, N = matrix.shape
    new_N = N + (X - 1) * (N - overlap_size)
    merged_matrix = np.zeros((M, C, new_N))

    for i in tqdm(range(X)):
        start_idx = i * (N - overlap_size)
        end_idx = start_idx + N

        if i == 0:
            merged_matrix[:, :, :N] = matrix[i][:, :, :]  #.squeeze()
        else:
            if method.lower() in ("avg", " average", "mean"):
                _merged = (merged_matrix[:, :, start_idx:start_idx+overlap_size]+
                           matrix[i][:, :, :overlap_size]) / 2.0
                _merged = np.concatenate([_merged, matrix[i][:, :, overlap_size:]], axis=-1)
                merged_matrix[:, :, start_idx:end_idx] = _merged

            elif method.lower() in ("maximum", "max"):
                _merged = np.maximum(
                            merged_matrix[:, :, start_idx:start_idx+overlap_size],
                            matrix[i][:, :, :overlap_size])
                _merged = np.concatenate([_merged, matrix[i][:, :, overlap_size:]], axis=-1)
                merged_matrix[:, :, start_idx:end_idx] = _merged

            elif method.lower() in ("shift", "slide"):
                merged_matrix[:, :, start_idx:end_idx] = matrix[i][:, :, :]

            else:
                logger.error("Method must be either: 'avg', 'max', 'shift'")
                sys.exit()
        #
    return merged_matrix


def __numpy_array_to_pytorch_tensor__(nparr, dtype="float64"):
    """Convert a NumPy array to a PyTorch tensor.

    Args:
        nparr (np.ndarray | torch.Tensor): Source array.  If a tensor is
            passed a warning is logged and the input is returned unchanged.
        dtype (str, optional): Target dtype keyword.  One of
            ``{"int32","int64","uint8","float32","float64","bool"}``.
            Default is ``"float64"``.

    Returns:
        torch.Tensor: Tensor on the CPU with the requested dtype.
    """
    if isinstance(nparr, torch.Tensor):
        logger.warning("input is already a pytorch tensor!")
        return nparr
    #
    dtype_map = {
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "float32": torch.float32,
        "float64": torch.float64,
        "bool": torch.bool
    }
    return torch.from_numpy(nparr).to(dtype=dtype_map[dtype])


def __pytorch_array_to_numpy__(ptarr, dtype="float64", clone=False):
    """Convert a PyTorch tensor to a NumPy array.

    Args:
        ptarr (torch.Tensor | np.ndarray): Source tensor.  If an array is
            passed a warning is logged and the input is returned unchanged.
        dtype (str, optional): Destination dtype keyword.  Same options as
            in :pyfunc:`__numpy_array_to_pytorch_tensor__`.  Default
            ``"float64"``.
        clone (bool, optional): When ``True`` return a detached copy;
            when ``False`` share memory where possible.

    Returns:
        np.ndarray: Array on the CPU with the requested dtype.
    """
    if isinstance(ptarr, np.ndarray):
        logger.warning("input is already a numpy array!")
        return ptarr
    #
    dtype_map = {
        "int32": np.int32,
        "int64": np.int64,
        "uint8": np.uint8,
        "float32": np.float32,
        "float64": np.float64,
        "bool": np.bool_
    }
    if clone:
        return ptarr.clone().detach().cpu().numpy().astype(dtype_map[dtype])
    else:
        return ptarr.detach().cpu().numpy().astype(dtype_map[dtype])


# =======================================================  TIME-PROC
# ===================================================================

def downsampling_trace(
                 tr, new_df,
                 hp_freq=2.0,
                 taper=True,
                 copy=False):
    """Band-limit and resample a single ObsPy trace.

    Args:
        tr (obspy.core.trace.Trace): Input trace.
        new_df (float): Target sampling rate (samples per second).
        hp_freq (float, optional): High-pass corner frequency before
            resampling.  Default 2.0 Hz.
        taper (bool, optional): Apply a 0.5 % Hann taper before filtering.
            Default ``True``.
        copy (bool, optional): Operate on a deep copy of *tr*.
            Default ``False``.

    Returns:
        obspy.core.trace.Trace: The processed trace (original or copy).
    """
    aaff = new_df / 2.0
    if copy:
        work_tr = tr.copy()
    else:
        work_tr = tr

    # --- BandPass
    if taper:
        work_tr.taper(0.005, type='hann')
    #
    work_tr.filter(
                "bandpass",
                freqmax=aaff,
                freqmin=hp_freq,
                corners=2,
                zerophase=True)

    # --- Downsample
    work_tr.resample(
        sampling_rate=new_df,
        window="hann",
        no_filter=False,  # MB: leave this on! somehow is supernecessary
        strict_length=False)

    return work_tr


def downsampling_stream(
                 st, new_df=50.0,
                 hp_freq=1.5,
                 copy=False):
    """Down-sample every trace in a stream that exceeds *new_df*.

    Args:
        st (obspy.core.stream.Stream): Input stream.
        new_df (float, optional): Target sampling rate.  Default 50 Hz.
        hp_freq (float, optional): High-pass corner for the anti-alias
            filter.  Default 1.5 Hz.
        copy (bool, optional): Work on a deep copy of *st*.  Default
            ``False``.

    Returns:
        obspy.core.stream.Stream: Stream with uniform sampling rate.
    """
    if copy:
        work_st = st.copy()
    else:
        work_st = st
    #
    for _tr in tqdm(work_st):
        if _tr.stats.sampling_rate > new_df:
            _tr = downsampling_trace(
                             _tr, new_df,
                             hp_freq=hp_freq,
                             copy=False)
        else:
            logger.debug("Trace %s.%s.%s.%s has already samplingrate less than %.1f" % (
                         _tr.stats.network, _tr.stats.station, "", _tr.stats.channel, new_df))
            continue
    #
    return work_st


def pickle_data(data, filename):
    """Serialise *data* to a pickle file.

    Args:
        data: Any pickle-compatible Python object.
        filename (str | Path): Destination path.

    Notes:
        Errors are caught and logged; no exception is raised to the caller.
    """
    try:
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        logger.info(f"Data saved to {filename} successfully.")
    except Exception as e:
        logger.info(f"An error occurred while pickling data: {e}")


def unpickle_data(filename):
    """Load a Python object from a pickle file.

    Args:
        filename (str | Path): Path to the pickle.

    Returns:
        Any | None: The un-pickled object or ``None`` on failure.
    """
    try:
        with open(filename, 'rb') as file:
            data = pickle.load(file)
        logger.info(f"Data loaded from {filename} successfully.")
        return data
    except Exception as e:
        logger.info(f"An error occurred while unpickling data: {e}")
        return None


def recursive_stalta(obs_data, dt, tshort, tlong, norm=False):
    """Recursive STA/LTA detector (CPU NumPy version).

    Args:
        obs_data (np.ndarray): One-dimensional signal array.
        dt (float): Sample interval in seconds.
        tshort (float): Short window length in seconds.
        tlong (float): Long window length in seconds.
        norm (bool, optional): Divide the output by its maximum value.
            Default ``False``.

    Returns:
        np.ndarray: STA/LTA ratio for each sample.
    """
    # After LOKI ALGORITHM
    nsamples = obs_data.shape[0]
    sw = int(round(tshort / dt))
    lw = int(round(tlong / dt))

    ks = 1/sw
    kl = 1/lw

    h = sw + lw
    eps = 1.0e-07
    stalta = np.zeros(nsamples)

    # Evaluation of the LTA
    lta0 = np.sum(obs_data[:lw]) * dt / tlong

    # Evalutation of the STA
    sta0 = np.sum(obs_data[lw:h]) * dt / tshort

    # Evaluation of the STA LTA ratio for the first h samples
    stalta[:h] = sta0 / (lta0 + eps)

    # Recursive STALTA
    stltmax = eps
    for j in range(h, nsamples):
        sta0 = ks * obs_data[j] + (1. - ks) * sta0
        lta0 = kl * obs_data[j - sw] + (1. - kl) * lta0
        stalta[j] = sta0 / (lta0 + eps)
        stltmax = max(stltmax, stalta[j])

    if norm:
        stalta /= stltmax

    return stalta


def recursive_stalta_gpu(obs_data, dt, tshort, tlong, norm=False):
    """Calculate the recursive STA/LTA ratio on CUDA.

    Args:
        obs_data (torch.Tensor): Input signal vector on CUDA device.
        dt (float): Sampling interval.
        tshort (float): Short time window length.
        tlong (float): Long time window length.
        norm (bool): Whether to normalize the output by its maximum value.

    Returns:
        torch.Tensor: STA/LTA ratio vector on CUDA device.
    """
    nsamples = obs_data.shape[0]
    sw = int(round(tshort / dt))
    lw = int(round(tlong / dt))

    ks = 1 / sw
    kl = 1 / lw

    h = sw + lw
    eps = 1.0e-07
    stalta = torch.zeros(nsamples, device=obs_data.device)

    # Compute the initial LTA and STA
    lta0 = torch.sum(obs_data[:lw]) * dt / tlong
    sta0 = torch.sum(obs_data[lw:h]) * dt / tshort

    # Initialize STA/LTA for the first h samples
    stalta[:h] = sta0 / (lta0 + eps)

    # Recursive STA/LTA calculation
    for j in range(h, nsamples):
        sta0 = ks * obs_data[j] + (1.0 - ks) * sta0
        lta0 = kl * obs_data[j - sw] + (1.0 - kl) * lta0
        stalta[j] = sta0 / (lta0 + eps)

    # Normalize if requested
    if norm:
        stltmax = torch.max(stalta)
        stalta /= stltmax

    return stalta


def envelope_from_wavelet(signal, wavelet='db4', level=4,
                          use_coeffs=[0, 1], boxcox=False):
    """Compute a wavelet-based signal envelope.

    Args:
        signal (np.ndarray): Input 1-D signal.
        wavelet (str, optional): PyWavelets wavelet name.  Default ``"db4"``.
        level (int, optional): Maximum decomposition level.  Default ``4``.
        use_coeffs (Sequence[int], optional): Indices of the wavelet
            coefficients to keep when reconstructing the envelope
            (0 = approximation, 1 = first detail, etc.).  Default ``(0, 1)``.
        boxcox (bool, optional): Apply Box-Cox transform to the envelope.
            Default ``False``.

    Returns:
        np.ndarray: Positive envelope, length equal to *signal*.
    """
    # Boundary handling mode (e.g., 'symmetric', 'smooth', 'constant')
    coeffs = pywt.wavedec(signal, wavelet, level=level, mode='symmetric')

    # Initialize an empty list for the coefficients to be used in reconstruction
    selected_coeffs = [None] * (level + 1)

    # Select the approximation and detail coefficients based on the indices provided
    for idx in use_coeffs:
        selected_coeffs[idx] = coeffs[idx]

    # Reconstruct the signal using the selected approximation and detail coefficients
    envelope = pywt.waverec(selected_coeffs, wavelet)

    # Ensure the reconstructed envelope matches the original length
    envelope = envelope[:len(signal)]
    envelope[envelope <= 0.0] = np.finfo(float).eps

    if boxcox:
        envelope, _ = scipy.stats.boxcox(envelope)
        envelope[envelope <= 0.0] = np.finfo(float).eps

    return envelope
