#!/usr/bin/env python

"""
End‑to‑end training script for HEIMDALL detector‑locator (GNN‑based)
===============================================================================
This script is the **last stage** of the Heimdall processing chain:
1.  It loads pre‑computed
    * a station **graph** describing spatial relationships between stations,
    * a 3‑D **grid** where location probabilities are evaluated,
    * an **HDF5 archive** containing input windows (detector char‑functions) and
      associated labels / targets prepared by the *PrepareDataset* stage, and
    * a YAML **configuration** file.
2.  It instantiates the joint **detector+locator** neural network (defined in
    `heimdall.models.HEIMDALL`), optionally loading pretrained weights or
    freezing the encoder layers as requested in the YAML.
3.  Using three `DataLoader`s (train / test / val) that **stream directly from
    disk**, it trains the network with early‑stopping and a ReduceLROnPlateau
    scheduler while keeping track of a wealth of metrics (loss histories,
    per‑phase confusion matrices and F‑scores, locator losses per‑plane, etc.).
4.  After training it evaluates on the hold‑out set and, **optionally**, plots
    pretty diagnostic figures every _N_ batches.
5.  All metrics are stored in `refining_training_metrics.npz` and the best model
    weights are saved to `HEIMDALL.refined.pt`.

Key design choices & implementation highlights
---------------------------------------------
* **Memory‑light dataset** – `HeimdallH5Dataset` opens the HDF5 file lazily in
  each worker so that multiple subprocesses do not fight over the file handle.
* **Evenising** – class imbalance (signal vs noise windows) is fixed *before*
  touching the heavy waveforms by looking only at the `pick_count` vector.
* **Augmentations** – cheap time‑series transforms (jitter, scaling, signed‑log,
  inversion) are applied *on‑the‑fly* to **half** of every batch so that the
  effective batch size doubles when `AUGMENTATION.enabled = True`.
* **Composite loss** – the total loss is:
  ```
  L_total = ALPHA·L_detector  +  BETA·L_locator/3  +  GAMMA·L_coord
             [from CF CNN]        [3 raster BCE]     [smooth‑L1 on XYZ coords]
  ```
  where the factors come from `COMPOSITE_LOSS` in the YAML.
* **Early stopping** – if `OPTIMISATION.epochs` is `null`, training will proceed
  indefinitely **until** the validation loss fails to improve by at least
  `delta` for `patience` epochs.
* **Heavy plotting** is *disabled* by default because it slows training down a
  lot – flip `PLOTS.make_plots` if you really need it.

The rest of the file is organised as follows (search for the headers):
----------------------------------------------------------------------
0.  **Imports and logging**
1.  `__init_model__` - build model + optimiser
2.  *Utility helpers* (evenising, augmentations, scaling)
3.  `HeimdallH5Dataset` + streaming `DataLoader`s
4.  A batch of metric helpers (precision/recall, pick matching, PDF→XYZ…)
5.  `__training_flow_ALL__` - the actual training loop
6.  Thin wrappers: `training_flow`, `validate_flow`, plotting helpers
7.  CLI entry‑point `main()` + some geo/normalisation utilities

All edits are **inline** below - look for blocks starting with
`# - EXPL:` for additional explanations that were *not* present in the
original source.
"""

import sys
import copy
import time
from tqdm import tqdm
import numpy as np
import random
from itertools import combinations
#
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#
from heimdall import io as gio
from heimdall import plot as gplt
from heimdall import models as gmdl
from heimdall import locator as glctr
from heimdall import custom_logger as CL

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
#
from scipy.signal import find_peaks, peak_widths

import argparse
import h5py
from torch.utils.data import Dataset  # <-- plain PyTorch (works with PyG collate)

import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


__log_name__ = "HeimdallTraining.log"
logger = CL.init_logger(__log_name__, lvl="INFO", log_file=__log_name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@  MODELS  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def __init_model__(shapes, stations, confs):
    """Build a Heimdall model instance and its Adam optimizer.

    Args:
        shapes (list[tuple[int, int]]): Raster sizes for the XY, XZ and YZ
            locator planes.
        stations (torch.Tensor): Normalised station coordinates shaped
            ``(N_sta, 3)`` on **CPU**.
        freeze_encoder (bool, optional): If ``True`` the convolutional
            encoder is frozen. Defaults to ``FREEZE_ENCODER``.
        weights (str, optional): Path to a ``.pt`` file with pretrained
            weights.  Leave empty to start from scratch.

    Returns:
        tuple[torch.nn.Module, torch.optim.Adam]: The Heimdall network (on
        the global *device*) and an optimizer that updates only
        parameters with ``requires_grad=True``.
    """

    freeze_encoder = confs.MODEL.freeze_encoder
    weights = confs.MODEL.pretrained_weights

    stations = torch.as_tensor(stations, dtype=torch.float32)   # on CPU
    model = gmdl.HEIMDALL(stations_coords=stations,
                          location_output_sizes=shapes)
    if weights:
        logger.state("Loading model weights:  %s" % weights)
        model.load_state_dict(torch.load(weights, map_location=device))

    # Freeze encoder weights
    if freeze_encoder:
        logger.state("Freezing ENCODER")
        for param in model.encoder.parameters():
            param.requires_grad = False

    # Only optimize parameters that require gradients
    _optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=confs.OPTIMISATION.learning_rate)
    #
    # model = torch.compile(model, mode="reduce-overhead")
    model.to(device)
    return (model, _optimizer)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@  WORK  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def __evenize_classes__(totpickcount, min_pick_signal=3,
                        shuffle=False, reduce_data=False,
                        signal_perc=0.5, noise_perc=0.5, seed=42):
    """Return balanced indices for signal/noise classes.

    A trace is deemed *signal* when its pick count exceeds
    ``min_pick_signal``.

    Args:
        totpickcount (np.ndarray): Per-trace pick counts.
        min_pick_signal (int, optional): Threshold above which a trace
            is classed as signal. Defaults to ``3``.
        shuffle (bool, optional): Shuffle the final index array.
            Defaults to ``False``.
        reduce_data (float | bool, optional): Fraction (0–1) of the
            balanced set to drop symmetrically.  Defaults to ``False``.
        signal_perc (float, optional): Target proportion of signals.
        noise_perc (float, optional): Target proportion of noise.
        seed (int, optional): RNG seed. Defaults to ``42``.

    Returns:
        np.ndarray: Balanced indices.

    Raises:
        AssertionError: If ``signal_perc + noise_perc`` is not **1.0**.
    """
    def round_to_nearest_even(number):
        # Round the float to the nearest integer
        rounded_num = round(number)

        # If the rounded number is divisible by 2, return it
        if rounded_num % 2 == 0:
            return rounded_num
        else:
            # If it's not divisible by 2, adjust it to the nearest even number
            return rounded_num + 1 if rounded_num % 2 != 0 else rounded_num

    # If a seed is provided, set it for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Identify indices of each class
    class_0_indices = np.where(totpickcount > min_pick_signal)[0]  # Indices for class 0
    class_1_indices = np.where(totpickcount <= min_pick_signal)[0]  # Indices for class 1
    # Number of samples for class 0 and class 1
    M = len(class_0_indices)  # Number of class 0 samples
    N = len(class_1_indices)  # Number of class 1 samples

    # ----------------------------------------------------
    # Check the class imbalance and balance the dataset
    if N > M:
        # Case 1: More class 1 samples than class 0
        # Randomly select M indices from class 1
        selected_class_1_indices = np.random.choice(class_1_indices, M, replace=False)
        # Combine class 0 indices and the selected class 1 indices
        balanced_indices = np.concatenate((class_0_indices,
                                           selected_class_1_indices))

    elif M > N:
        # Case 2: More class 0 samples than class 1
        # Randomly select N indices from class 0
        selected_class_0_indices = np.random.choice(class_0_indices, N, replace=False)
        # Combine class 1 indices and the selected class 0 indices
        balanced_indices = np.concatenate((selected_class_0_indices,
                                           class_1_indices))

    else:
        # Case 3: Classes are already balanced
        logger.info("... Classes are already balanced")
        # Use all indices as they are already balanced
        balanced_indices = np.arange(len(totpickcount))

    # ----------------------------------------------------
    if reduce_data:
        logger.info("Reducing DATA of %.02f" % reduce_data)
        # The first X are class 0 the last X are 1
        total_reduction = round_to_nearest_even(
                            len(balanced_indices)*reduce_data)
        # Remove index from start,
        _half_red_0 = np.random.choice(
                        np.arange(0, int(total_reduction/2)),
                        int(total_reduction/2),
                        replace=False)
        # Remove index from reverse
        _half_red_1 = -1*_half_red_0

        _remove = np.concatenate((_half_red_0, _half_red_1))
        balanced_indices = np.delete(balanced_indices, _remove)

    # ----------------------------------------------------
    # Recalculate the indices for signals (class 0) and noise (class 1)
    # from the balanced data
    class_0_indices_balanced = np.where(totpickcount[balanced_indices] >
                                        min_pick_signal)[0]
    class_1_indices_balanced = np.where(totpickcount[balanced_indices] <=
                                        min_pick_signal)[0]
    num_signals = len(class_0_indices_balanced)
    num_noise = len(class_1_indices_balanced)

    # Apply signal and noise percentage constraints
    if signal_perc and noise_perc:
        assert signal_perc + noise_perc == 1.0, "Signal and noise percentages must sum to 1.0"

        # Calculate the number of samples to keep for signals (class 0) and noise (class 1)
        num_signals = int(len(class_0_indices_balanced) * signal_perc)
        num_noise = int(len(class_1_indices_balanced) * noise_perc)

        # Randomly sample the required number of signals and noise from the recalculated indices
        selected_signals = np.random.choice(class_0_indices_balanced,
                                            num_signals, replace=False)
        selected_noise = np.random.choice(class_1_indices_balanced,
                                          num_noise, replace=False)

        # Combine the selected signals and noise into balanced indices
        balanced_indices = np.concatenate((balanced_indices[selected_signals],
                                           balanced_indices[selected_noise]))

    # ----------------------------------------------------
    if shuffle:
        # Shuffle after applying signal and noise percentages
        logger.info("Shuffling DATA")
        np.random.shuffle(balanced_indices)

    logger.info("We now have  %d SIGNALS and  %d NOISES" %
                (num_signals, num_noise))

    return balanced_indices


class Augmentations(object):
    """Random time-series augmentations used during training.

    Available operators:
        * ``jt`` – Gaussian jitter
        * ``sc`` – Global amplitude scaling
        * ``lg`` – Signed logarithmic compression
        * ``inv`` – Amplitude inversion
    """

    def __init__(self):
        self.tag = "AUGMENTATIONS"
        self.elements = ["jt", "sc", "lg", "inv"]  # 'tw'
        self.combinations = self.__create_all_combos__()

    def __create_all_combos__(self):
        # Generate all possible unique combinations with sorting
        all_combinations = set()
        for r in range(1, len(self.elements) + 1):
            for subset in combinations(self.elements, r):
                all_combinations.add(tuple(sorted(subset)))  # Ensure sorting for uniqueness

        # Convert to a sorted list for consistent order
        return sorted(all_combinations)

    def __time_warp__(self, x, warp_factor_range=(0.8, 1.2), time_dim=-1):
        """
        Randomly warps the time dimension by a factor in [warp_factor_range].
        We up-sample (or down-sample) the signal, then resample it back to the
        original length to create a local time distortion.

        Args:
            x (torch.Tensor): Shape (B, N, C, F).
            warp_factor_range (tuple): (min_factor, max_factor) for random warping.
            time_dim (int): The dimension corresponding to time (default -1).

        Returns:
            torch.Tensor: Time-warped tensor with the same shape as x.
        """
        # We'll assume time is the last dimension (-1).
        original_len = x.size(time_dim)
        warp_factor = random.uniform(*warp_factor_range)
        new_len = int(round(original_len * warp_factor))

        # Reshape to (B*N*C, 1, F) so we can use 1D interpolation easily.
        # (If your shape is truly (B, N, C, F), we'll flatten B*N*C into one dimension.)
        b, n, c, f = x.shape
        x_reshaped = x.view(b*n*c, 1, f)  # shape: (B*N*C, 1, F)

        # Step 1: Interpolate to new length (stretch or compress)
        # mode='linear' for 1D signals
        if new_len > 1:  # avoid corner case of new_len=0
            x_warped = F.interpolate(x_reshaped, size=new_len, mode='linear', align_corners=False)
        else:
            # If the warp factor is extremely small, just skip it or clamp
            x_warped = x_reshaped

        # Step 2: Interpolate back to original length (F)
        x_back = F.interpolate(x_warped, size=original_len, mode='linear', align_corners=False)

        # Reshape to original (B, N, C, F)
        x_out = x_back.view(b, n, c, original_len)
        return x_out

    def __log_scale__(self, x):
        """
        Applies a sign-preserving logarithmic transformation to enhance small amplitudes.
        For negative values, we do sign(x) * log1p(abs(x)).

        Args:
            x (torch.Tensor): Any shape, can have both positive/negative values.

        Returns:
            torch.Tensor: Log-scaled version of x with the same shape.
        """
        # sign(x) * log(1 + |x|)
        return torch.sign(x) * torch.log1p(torch.abs(x))

    def __time_flip__(self, x, time_dim=-1):
        """
        Flips the tensor x along the given time dimension.
        x: Torch tensor (B, N, C, F) (or any shape)
        time_dim: The dimension to flip (default = -1)
        """
        return x.flip(dims=[time_dim])

    def augment_time_series(self, x,
                            jitter_sigma=0.09,
                            scale_min=0.3, scale_max=1.7,
                            normalize=True):  # to be consistent with the -1/1 scaling

        """Apply a random combination of augmentations.
          1. Time-warp (random up/down sampling)
          2. Jitter (Gaussian noise)
          3. Random amplitude scaling
          4. Logarithmic scaling (optional)

        Args:
            x (torch.Tensor): Input tensor shaped ``(B, N, C, F)``.
            jitter_sigma (float, optional): Std-dev for Gaussian noise.
            scale_min (float, optional): Lower bound for amplitude
                scaling. Defaults to ``0.3``.
            scale_max (float, optional): Upper bound for amplitude
                scaling. Defaults to ``1.7``.
            normalize (bool, optional): Rescale each trace into
                **[-1, 1]** after augmentation.  Defaults to ``True``.

        Returns:
            torch.Tensor: Augmented tensor with the same shape as *x*.
        """

        out = x.clone()
        combo = random.choice(self.combinations)  # Randomly select a combination

        # # 1. Time Warp
        # if "tw" in combo:
        #     out = self.__time_warp__(out, warp_factor_range=(0.8, 1.2), time_dim=-1)

        # 2. Jitter
        if "jt" in combo:
            noise = torch.randn_like(out) * jitter_sigma
            out = out + noise

        # 3. Random amplitude scaling
        if "sc" in combo:
            scale_factor = random.uniform(scale_min, scale_max)
            out = out * scale_factor

        # 4. Logarithmic scaling
        if "lg" in combo:
            out = self.__log_scale__(out)

        # 5. Inverse
        if "inv" in combo:
            out = -1*out

        if normalize:
            min_vals = torch.min(out, dim=-1, keepdim=True).values
            max_vals = torch.max(out, dim=-1, keepdim=True).values
            out = 2 * (out - min_vals) / (max_vals - min_vals + 1e-8) - 1

        return out


# ==========================================================================
# ==========================================================================
# ==========================================================================
# ==========================================================================
# ========================================================   ALL
# ==========================================================================
# ==========================================================================
# ==========================================================================
# ==========================================================================

# --------------------------------------------------------------------------- I/O
def scale_minusone_one(t, dim=-1):
    """Rescale tensor to [-1,1] along `dim` (same math used before)."""
    min_v = t.min(dim=dim, keepdim=True).values
    max_v = t.max(dim=dim, keepdim=True).values
    return 2 * (t - min_v) / (max_v - min_v + 1e-8) - 1

def scale_zero_one(mat, dimension=-1):
    min_vals = torch.min(mat, dim=dimension, keepdim=True).values
    max_vals = torch.max(mat, dim=dimension, keepdim=True).values
    mat = (mat - min_vals) / (max_vals - min_vals + 1e-8)
    return mat

class HeimdallH5Dataset(Dataset):
    """
    Memory-light replacement for the former NumPy-in-RAM archive.
    Each __getitem__ returns exactly one `torch_geometric.data.Data`
    object, constructed on the fly from the HDF5 slice.
    """
    # def __init__(self, h5_path, indices, heim_gnn, augment=False):
    #     self.h5 = h5py.File(h5_path, "r", swmr=True)      # read-only, thread-safe

    def __init__(self, h5_path, indices, heim_gnn, augment=False):
        self.h5_path = h5_path          # keep only the path
        self.h5 = None                  # will be opened in the worker
        self.idx = np.asarray(indices, dtype=np.int64)
        self.edges = torch.as_tensor(heim_gnn["edges"], dtype=torch.long)
        self.eattr = torch.as_tensor(heim_gnn["weights"], dtype=torch.float32)

    def __len__(self):
        return len(self.idx)

    def _lazy_open(self):
        if self.h5 is None:             # first call inside *this* worker
            self.h5 = h5py.File(self.h5_path, "r", swmr=True)

    def __getitem__(self, i):
        self._lazy_open()
        j = int(self.idx[i])

        # ---------- detector inputs / labels ---------------------------------
        x_det = torch.from_numpy(self.h5["Xdet"][j]).float()      # (N,C,F)
        y_det = torch.from_numpy(self.h5["Ydet"][j]).float()

        x_det = scale_minusone_one(x_det)                         # same scaling

        # ---------- locator inputs & targets ---------------------------------
        x_loc = x_det[:, :1, :]                                   # keep CF#0
        yl1   = torch.from_numpy(self.h5["Yloc_XY"][j]).float().unsqueeze(0)
        yl2   = torch.from_numpy(self.h5["Yloc_XZ"][j]).float().unsqueeze(0)
        yl3   = torch.from_numpy(self.h5["Yloc_YZ"][j]).float().unsqueeze(0)

        # ---------- misc tensors ---------------------------------------------
        r      = torch.from_numpy(self.h5["R"][j]).float()
        src_xyz = torch.from_numpy(self.h5["sources_grid"][j]).float().unsqueeze(0)

        return Data(
            x=x_det,
            yd=y_det,
            xl=x_loc,
            yl1=yl1, yl2=yl2, yl3=yl3,
            r=r, sources_xyz=src_xyz,
            edge_index=self.edges,
            edge_attr=self.eattr,
            num_nodes=x_det.shape[0]
        )

def prepare_h5_loaders(h5_path, heim_gnn, confs):
    """
    Light-memory splitter that works directly on the HDF5 file.

    `evenize_params` is applied by
    loading **only the 1-D `pick_count` dataset** once, computing the
    balanced indices, and never touching the heavy waveforms.
    """

    batch_size = confs.DATASET.batch_size
    evenize_params = confs.DATASET.evenize
    test_split = confs.SPLIT.test
    val_split = confs.SPLIT.val
    rnd_seed = confs.RANDOM_SEED
    augment = confs.AUGMENTATION.enabled

    with h5py.File(h5_path, "r") as h5:
        pick_count = h5["pick_count"][:]                        # small (≤ int32)
        all_idx    = np.arange(len(pick_count))

    if evenize_params:
        all_idx = __evenize_classes__(pick_count, **evenize_params)

    # train / test / val split (unchanged logic)
    train_idx, rem_idx = train_test_split(
        all_idx, test_size=test_split + val_split, random_state=rnd_seed)
    test_idx,  val_idx = train_test_split(
        rem_idx, test_size=test_split / (test_split + val_split),
        random_state=rnd_seed)

    ds_train = HeimdallH5Dataset(h5_path, train_idx, heim_gnn, augment=augment)
    ds_test  = HeimdallH5Dataset(h5_path, test_idx,  heim_gnn, augment=False)
    ds_val   = HeimdallH5Dataset(h5_path, val_idx,  heim_gnn, augment=False)

    train_loader = DataLoader(ds_train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=confs.DATASET.n_work,
                              pin_memory=True,
                              persistent_workers=True,
                              #
                              prefetch_factor=confs.DATASET.n_work)
    test_loader  = DataLoader(ds_test,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=confs.DATASET.n_work,
                              pin_memory=True,
                              persistent_workers=True,
                              #
                              prefetch_factor=confs.DATASET.n_work)
    val_loader   = DataLoader(ds_val,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=confs.DATASET.n_work,
                              pin_memory=True,
                              persistent_workers=True,
                              #
                              prefetch_factor=confs.DATASET.n_work)
    return train_loader, test_loader, val_loader

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def __calculate_scores__(_tp, _tn, _fp, _fn):
    """Compute precision, recall and F1 score from confusion counts."""
    prec = _tp/(_tp+_fp+1e-8)
    rec = _tp/(_tp+_fn+1e-8)
    fone = 2*((prec*rec)/(prec+rec+1e-8))
    return (prec, rec, fone)


def __batch_stats__(
    pred: torch.Tensor,
    label: torch.Tensor,
    channel: int = 0,
    threshold: float = 0.5,
    overlap_threshold: float = 0.5
):
    """Vectorised detector confusion counts for a whole batch.

    Computes event-level confusion stats for a batch of shape (B*N, C, F):
      - B*N = number of nodes across all graphs in the batch
      - C   = number of channels [event, P, S]
      - F   = number of time steps

    Logic:
      -  For each node, compare prediction vs label at given channel
      -  If predicted and labeled overlap >= overlap_threshold -> TP
         Else if labeled exists but low overlap >= FN
         Else if prediction exists with no label >= FP
         Else -> TN
    """

    # 1) Select the desired channel => shape (B*N, F)
    pred_selected = pred[:, channel, :]
    label_selected = label[:, channel, :]

    # 2) Binarize at threshold => bool tensors
    pred_bin = (pred_selected > threshold)
    label_bin = (label_selected > threshold)

    # 3) Count labeled, predicted, and overlapping samples per node
    label_count   = label_bin.sum(dim=1)             # [B*N]
    pred_count    = pred_bin.sum(dim=1)              # [B*N]
    overlap_count = (label_bin & pred_bin).sum(dim=1)

    # 4) Has labeled event
    has_label_event = (label_count > 0)

    # 5) Overlap fraction
    overlap_fraction = torch.zeros_like(label_count, dtype=torch.float)
    valid_mask = has_label_event
    overlap_fraction[valid_mask] = overlap_count[valid_mask].float() / label_count[valid_mask].float()

    # 6) Confusion logic
    tp_mask = has_label_event & (overlap_fraction >= overlap_threshold)
    fn_mask = has_label_event & (overlap_fraction < overlap_threshold)
    fp_mask = (~has_label_event) & (pred_count > 0)
    tn_mask = (~has_label_event) & (pred_count == 0)

    # 7) Aggregate stats
    tp = tp_mask.sum().item()
    fn = fn_mask.sum().item()
    fp = fp_mask.sum().item()
    tn = tn_mask.sum().item()

    return (tp, tn, fp, fn)


def __extract_picks__(ts, thr=0.2, min_distance=50, smooth=True):
    """Detect peaks in a 1-D time-series.
    Extract peaks from a 1D array `ts` using scipy's find_peaks.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Indices,
        widths, amplitudes and (optionally smoothed) trace.
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


def __compare_picks__(peaks_model, peaks_ref, thr=25):
    """Compare predicted vs. reference picks within +/- ``thr`` samples.

    If a model peak is within `thr` of a reference peak => TP
          - Reference peaks not matched => FN
          - Model peaks not matched => FP

    Returns:
        tuple[int, int, int]: ``(TP, FP, FN)``.
    """

    TP, FP, FN = 0, 0, 0

    # Keep track of which model peaks were matched
    matched_model_peaks = set()

    # Find TPs (model peak within `thr` of a ref peak)
    for pf in peaks_ref:
        match_found = False
        for pm in peaks_model:
            if pm in matched_model_peaks:
                continue
            if abs(pm - pf) <= thr:
                TP += 1
                matched_model_peaks.add(pm)
                match_found = True
                break
        if not match_found:
            FN += 1

    # Any unmatched model peak => FP
    for pm in peaks_model:
        if pm not in matched_model_peaks:
            FP += 1

    return TP, FP, FN


def __batch_stats_picks__(
    pred: torch.Tensor,
    label: torch.Tensor,
    channel: int,
    peak_threshold: float = 0.3,
    min_distance: int = 25,  # samples @ 50 Hz
    match_threshold: float = 25.0,
    smooth: bool = True
):
    """
    Compute station-level confusion statistics (TP, TN, FP, FN) by
    comparing Gaussian-like peaks in *pred* and *label*.

    The function extracts peaks from both tensors for the chosen
    *channel*, matches them within ``match_threshold`` samples, and
    returns the aggregate counts of true/false positives and
    true/false negatives.

    Args:
        pred (torch.Tensor): Predicted waveforms of shape ``(B*N, C, F)``.
        label (torch.Tensor): Ground-truth waveforms of identical shape.
        channel (int): Channel index to evaluate (0 = event, 1 = P, 2 = S).
        peak_threshold (float, optional): Amplitude threshold passed to
            :func:`scipy.signal.find_peaks`. Defaults to ``0.3``.
        min_distance (int, optional): Minimum distance (in samples) between
            successive peaks. Defaults to ``25``.
        match_threshold (float, optional): Maximum temporal distance (in
            samples) for a predicted peak to count as a match. Defaults
            to ``25``.
        smooth (bool, optional): If ``True`` apply a 3-point moving-average
            filter before peak detection. Defaults to ``True``.

    Returns:
        tuple[int, int, int, int]: ``(TP, TN, FP, FN)``.
    """

    # 1) Select the desired channel => shape (B*N, F)
    pred_selected = pred[:, channel, :].detach().cpu().numpy()
    label_selected = label[:, channel, :].detach().cpu().numpy()

    # 2) For each row => extract picks, compare
    TP_total, FP_total, FN_total, TN_total = 0, 0, 0, 0

    for i in range(pred_selected.shape[0]):
        pred_peaks, _, _, _ = __extract_picks__(
            pred_selected[i],
            thr=peak_threshold,
            min_distance=min_distance,
            smooth=smooth
        )
        label_peaks, _, _, _ = __extract_picks__(
            label_selected[i],
            thr=peak_threshold,
            min_distance=min_distance,
            smooth=smooth
        )

        # If both have no peaks => that is a TN
        if len(pred_peaks) == 0 and len(label_peaks) == 0:
            TN_total += 1
            continue

        # Otherwise, compare picks
        TP, FP, FN = __compare_picks__(pred_peaks, label_peaks, thr=match_threshold)
        TP_total += TP
        FP_total += FP
        FN_total += FN

    return (TP_total, TN_total, FP_total, FN_total)


def pdfs_to_coords_argmax(xy, xz, yz):
    """
    Convert three locator PDFs to normalised coordinates by taking the
    argmax of each plane.

    Args:
        xy (torch.Tensor): Tensor of shape ``(B, 304, 330)`` containing
            XY plane probability maps.
        xz (torch.Tensor): Tensor of shape ``(B, 304, 151)`` containing
            XZ plane probability maps.
        yz (torch.Tensor): Tensor of shape ``(B, 330, 151)`` containing
            YZ plane probability maps.

    All tensors must be of dtype ``float32`` and reside on the same
    device.

    Returns:
        torch.Tensor: Normalised coordinates tensor of shape ``(B, 3)``,
        with columns ``(x_norm, y_norm, z_norm)`` and values in ``[0, 1]``.
    """

    B = xy.shape[0]

    # --- xy -----------------------------------------------------------
    # argmax over the flattened map, then unravel to 2-D indices
    xy_flat_idx = torch.argmax(xy.view(B, -1), dim=1)          # [B]
    xy_row = xy_flat_idx // xy.shape[2]                        # X index [0..303]
    xy_col = xy_flat_idx %  xy.shape[2]                        # Y index [0..329]

    # --- xz -----------------------------------------------------------
    xz_flat_idx = torch.argmax(xz.view(B, -1), dim=1)
    xz_row = xz_flat_idx // xz.shape[2]                        # X again
    xz_col = xz_flat_idx %  xz.shape[2]                        # Z index [0..150]

    # --- yz -----------------------------------------------------------
    yz_flat_idx = torch.argmax(yz.view(B, -1), dim=1)
    yz_row = yz_flat_idx // yz.shape[2]                        # Y again
    yz_col = yz_flat_idx %  yz.shape[2]                        # Z again

    # --- reconcile (average just in case peaks differ by ±1 pixel) ---
    x_idx = (xy_row + xz_row) / 2.0
    y_idx = (xy_col + yz_row) / 2.0
    z_idx = (xz_col + yz_col) / 2.0

    # --- normalise to 0-1 --------------------------------------------
    x_norm = x_idx / (304 - 1)
    y_norm = y_idx / (330 - 1)
    z_norm = z_idx / (151 - 1)

    coords_norm = torch.stack([x_norm, y_norm, z_norm], dim=1)  # [B,3]
    return coords_norm


def __training_flow_ALL__(train_loader_ALL, test_loader_ALL, model, gnn,
                          optimizer, confs, out_save_path="HeimdallModel_LOCATOR_refined.pt"):
    """
    Fine-tune the Heimdall model on the provided training and test data loaders.

    This function performs epoch-based training and evaluation of the Heimdall
    model, logging losses and confusion metrics, and applying optional data
    augmentation and early stopping.

    Args:
        train_loader_ALL (DataLoader): DataLoader for training batches.
        test_loader_ALL (DataLoader): DataLoader for test batches.
        model (torch.nn.Module): Heimdall model instance to be fine-tuned.
        gnn (dict): Graph data and related metadata.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        out_save_path (str, optional): Path to store the refined model weights.
            Defaults to "HeimdallModel_LOCATOR_refined.pt".

    Returns:
        tuple: Tuple containing:
            - model (torch.nn.Module): Trained model.
            - list[float]: Heimdall training loss history.
            - list[float]: Heimdall test loss history.
            - list[float]: Detector training loss history.
            - list[float]: Detector test loss history.
            - list[tuple]: Detector confusion metrics history (event channel).
            - list[tuple]: Detector confusion metrics history (P channel).
            - list[tuple]: Detector confusion metrics history (S channel).
            - list[float]: Locator training loss history.
            - list[float]: Locator training XY-plane loss history.
            - list[float]: Locator training XZ-plane loss history.
            - list[float]: Locator training YZ-plane loss history.
            - list[float]: Locator test loss history.
            - list[float]: Locator test XY-plane loss history.
            - list[float]: Locator test XZ-plane loss history.
            - list[float]: Locator test YZ-plane loss history.
            - int: Final training epoch number.
    """

    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, mode='min', factor=0.1,
                            patience=3, threshold=0.001, verbose=True)
    if not confs.OPTIMISATION.epochs or confs.OPTIMISATION.epochs == 0:
        early_stopping = EarlyStopping(patience=confs.OPTIMISATION.early_stopping.patience,
                                       delta=confs.OPTIMISATION.early_stopping.delta,
                                       verbose=True)

    augmentin = Augmentations()

    # -------------------------------------  DETECTOR
    HEIM_train_loss_history = []
    HEIM_test_loss_history = []

    # -------------------------------------  DETECTOR
    DET_train_loss_history = []
    DET_test_loss_history = []
    DET_test_confusion_history_event = []
    DET_test_confusion_history_p = []
    DET_test_confusion_history_s = []

    # -------------------------------------  LOCATOR
    # Training and Testing Loop
    LOC_train_loss_history = []
    LOC_train_loss_xy_history = []
    LOC_train_loss_xz_history = []
    LOC_train_loss_yz_history = []

    LOC_test_loss_history = []
    LOC_test_loss_xy_history = []
    LOC_test_loss_xz_history = []
    LOC_test_loss_yz_history = []

    print_sizes = True
    for epoch in range(confs.OPTIMISATION.epochs
                       if confs.OPTIMISATION.epochs else 9999):

        # ============================>   TRAINING
        model.train()

        HEIM_running_train_loss = 0.0
        DET_running_train_loss = 0.0
        LOC_running_train_loss = 0.0
        LOC_running_train_loss_xy = 0.0
        LOC_running_train_loss_xz = 0.0
        LOC_running_train_loss_yz = 0.0

        for _bidx, batch in enumerate(tqdm(train_loader_ALL,
                                           total=len(train_loader_ALL))):

            det_inputs = batch.x.to(device)
            det_targets = batch.yd.to(device)
            loc_inputs = batch.xl.to(device)
            loc_target1 = torch.stack([g.yl1.squeeze(0) for g in batch.to_data_list()]).to(device)  # shape: [B, 304, 330]
            loc_target2 = torch.stack([g.yl2.squeeze(0) for g in batch.to_data_list()]).to(device)  # shape: [B, 304, 151]
            loc_target3 = torch.stack([g.yl3.squeeze(0) for g in batch.to_data_list()]).to(device)  # shape: [B, 330, 151]
            edges = batch.edge_index.to(device)
            weights = batch.edge_attr.to(device)
            batch_vector = batch.batch.to(device)
            with torch.no_grad():                       # no gradient needed
                coord_targets = pdfs_to_coords_argmax(
                                    loc_target1, loc_target2, loc_target3)

            if confs.AUGMENTATION.enabled:
                # If only 2
                det_inputs_aug = augmentin.augment_time_series(det_inputs)
                det_inputs = torch.cat([det_inputs, det_inputs_aug], dim=0)
                det_targets = torch.cat([det_targets, det_targets], dim=0)
                #
                loc_inputs = torch.cat([loc_inputs, loc_inputs], dim=0)
                loc_target1 = torch.cat([loc_target1, loc_target1], dim=0)
                loc_target2 = torch.cat([loc_target2, loc_target2], dim=0)
                loc_target3 = torch.cat([loc_target3, loc_target3], dim=0)
                # BatchVector
                offset = batch_vector.max() + 1
                batch_vector_aug = batch_vector + offset
                batch_vector = torch.cat([batch_vector, batch_vector_aug], dim=0)
                coord_targets = torch.cat([coord_targets, coord_targets], dim=0)

            if _bidx == 0 and print_sizes:
                print_sizes = False
                logger.info("INPUT SIZES")
                logger.info(f"batch.x.shape: {det_inputs.shape!r}")
                logger.info(f"batch.yd.shape: {det_targets.shape!r}")
                logger.info(f"batch.xl.shape: {loc_inputs.shape!r}")
                logger.info(f"batch.yl1.shape:{loc_target1.shape!r}")
                logger.info(f"batch.yl2.shape: {loc_target2.shape!r}")
                logger.info(f"batch.yl3.shape: {loc_target3.shape!r}")
                logger.info(f"batch.edge_index.shape: {edges.shape!r}")
                logger.info(f"batch.edge_attr.shape: {weights.shape!r}")
                logger.info(f"batch.shape: {batch_vector.shape!r}")

            # ===================================  GO WITH THE FLOW
            # --->  0. Reset gradients
            optimizer.zero_grad()

            # --->  1. Run MODEL
            (evpk, (location_xy, location_xz, location_yz), coords) = model(
                                    det_inputs, edges, weights, batch_vector)

            # ================================== DETECTOR
            det_loss = criterion(evpk, det_targets)
            DET_running_train_loss += det_loss.item()

            # ================================== LOCATOR
            loss_xy = criterion(location_xy, loc_target1)
            loss_xz = criterion(location_xz, loc_target2)
            loss_yz = criterion(location_yz, loc_target3)
            loc_loss_weighted = (confs.LOCATOR_LOSS.xy * loss_xy +
                                 confs.LOCATOR_LOSS.xz  * loss_xz +
                                 confs.LOCATOR_LOSS.yz  * loss_yz)

            LOC_running_train_loss += loc_loss_weighted.item()
            LOC_running_train_loss_xy += loss_xy.item()
            LOC_running_train_loss_xz += loss_xz.item()
            LOC_running_train_loss_yz += loss_yz.item()

            # # ================================== COMPILE LOSS
            # heim_loss = ALPHA*det_loss + BETA*loc_loss_weighted

            # ================================== COMPILE LOSS (new coords)
            # coordinate constraint
            loss_coord = F.smooth_l1_loss(coords, coord_targets)   # or MSE / Huber
            heim_loss = (confs.COMPOSITE_LOSS.alpha*det_loss +
                         confs.COMPOSITE_LOSS.beta*(loc_loss_weighted/3) +
                         confs.COMPOSITE_LOSS.gamma*loss_coord)

            # ================================== Backpropagation and optimization
            HEIM_running_train_loss += heim_loss.item()
            heim_loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()
        #
        HEIM_train_loss = HEIM_running_train_loss / len(train_loader_ALL)
        HEIM_train_loss_history.append(HEIM_train_loss)

        DET_train_loss = DET_running_train_loss / len(train_loader_ALL)
        DET_train_loss_history.append(DET_train_loss)

        LOC_train_loss = LOC_running_train_loss / len(train_loader_ALL)
        LOC_train_loss_history.append(LOC_train_loss)

        LOC_train_loss_xy = LOC_running_train_loss_xy / len(train_loader_ALL)
        LOC_train_loss_xy_history.append(LOC_train_loss_xy)

        LOC_train_loss_xz = LOC_running_train_loss_xz / len(train_loader_ALL)
        LOC_train_loss_xz_history.append(LOC_train_loss_xz)

        LOC_train_loss_yz = LOC_running_train_loss_yz / len(train_loader_ALL)
        LOC_train_loss_yz_history.append(LOC_train_loss_yz)

        # ============================>   TESTING
        model.eval()
        HEIM_running_test_loss = 0.0
        DET_running_test_loss = 0.0
        LOC_running_test_loss = 0.0
        LOC_running_test_loss_xy = 0.0
        LOC_running_test_loss_xz = 0.0
        LOC_running_test_loss_yz = 0.0

        with torch.no_grad():
            for _bidx, batch in enumerate(tqdm(test_loader_ALL,
                                               total=len(test_loader_ALL))):
                #
                TP_E, TN_E, FP_E, FN_E = 0, 0, 0, 0
                TP_P, TN_P, FP_P, FN_P = 0, 0, 0, 0
                TP_S, TN_S, FP_S, FN_S = 0, 0, 0, 0
                #
                det_inputs = batch.x.to(device)
                det_targets = batch.yd.to(device)
                loc_inputs = batch.xl.to(device)
                loc_target1 = torch.stack([g.yl1.squeeze(0) for g in batch.to_data_list()]).to(device)  # shape: [B, 304, 330]
                loc_target2 = torch.stack([g.yl2.squeeze(0) for g in batch.to_data_list()]).to(device)  # shape: [B, 304, 151]
                loc_target3 = torch.stack([g.yl3.squeeze(0) for g in batch.to_data_list()]).to(device)  # shape: [B, 330, 151]
                edges = batch.edge_index.to(device)
                weights = batch.edge_attr.to(device)
                batch_vector = batch.batch.to(device)
                coord_targets = pdfs_to_coords_argmax(
                                    loc_target1, loc_target2, loc_target3)

                # ===================================  GO WITH THE FLOW
                # --->  1. Run MODEL
                (evpk, (location_xy, location_xz, location_yz), coords) = model(
                                    det_inputs, edges, weights, batch_vector)

                # ================================== DETECTOR
                det_loss = criterion(evpk, det_targets)
                DET_running_test_loss += det_loss.item()

                # ================================== LOCATOR
                loss_xy = criterion(location_xy, loc_target1)
                loss_xz = criterion(location_xz, loc_target2)
                loss_yz = criterion(location_yz, loc_target3)
                loc_loss_weighted = (confs.LOCATOR_LOSS.xy * loss_xy +
                                     confs.LOCATOR_LOSS.xz * loss_xz +
                                     confs.LOCATOR_LOSS.yz * loss_yz)

                LOC_running_test_loss += loc_loss_weighted.item()
                LOC_running_test_loss_xy += loss_xy.item()
                LOC_running_test_loss_xz += loss_xz.item()
                LOC_running_test_loss_yz += loss_yz.item()

                # # ================================== COMPILE LOSS
                # heim_loss = ALPHA*det_loss + BETA*loc_loss_weighted

                # ================================== COMPILE LOSS (new coords)
                # coordinate constraint
                loss_coord = F.smooth_l1_loss(coords, coord_targets)   # or MSE / Huber
                heim_loss = (confs.COMPOSITE_LOSS.alpha*det_loss +
                             confs.COMPOSITE_LOSS.beta*(loc_loss_weighted/3) +
                             confs.COMPOSITE_LOSS.gamma*loss_coord)
                #
                HEIM_running_test_loss += heim_loss.item()
                torch.cuda.empty_cache()

                # =============================== ACCURACY / F1
                (_tpe, _tne, _fpe, _fne) = __batch_stats__(
                    evpk, det_targets, channel=0, threshold=0.5, overlap_threshold=0.5)
                TP_E += _tpe; TN_E += _tne; FP_E += _fpe; FN_E += _fne

                if det_targets.shape[1] > 1:
                    (_tpp, _tnp, _fpp, _fnp) = __batch_stats_picks__(
                        evpk, det_targets, channel=1)
                    TP_P += _tpp; TN_P += _tnp; FP_P += _fpp; FN_P += _fnp

                    (_tps, _tns, _fps, _fns) = __batch_stats_picks__(
                        evpk, det_targets, channel=2)
                    TP_S += _tps; TN_S += _tns; FP_S += _fps; FN_S += _fns

            # ---> End batches
            HEIM_test_loss = HEIM_running_test_loss / len(test_loader_ALL)
            HEIM_test_loss_history.append(HEIM_test_loss)

            DET_test_loss = DET_running_test_loss / len(test_loader_ALL)
            DET_test_loss_history.append(DET_test_loss)

            LOC_test_loss = LOC_running_test_loss / len(test_loader_ALL)
            LOC_test_loss_history.append(LOC_test_loss)

            LOC_test_loss_xy = LOC_running_test_loss_xy / len(test_loader_ALL)
            LOC_test_loss_xy_history.append(LOC_test_loss_xy)

            LOC_test_loss_xz = LOC_running_test_loss_xz / len(test_loader_ALL)
            LOC_test_loss_xz_history.append(LOC_test_loss_xz)

            LOC_test_loss_yz = LOC_running_test_loss_yz / len(test_loader_ALL)
            LOC_test_loss_yz_history.append(LOC_test_loss_yz)

        # ---> End EVALUATION process
        # ============================= >>> STATS (detector)
        (PRECISION_E, RECALL_E, FONE_E) = __calculate_scores__(TP_E, TN_E, FP_E, FN_E)
        DET_test_confusion_history_event.append((TP_E, TN_E, FP_E, FN_E,
                                                 PRECISION_E, RECALL_E, FONE_E))
        if det_targets.shape[1] > 1:
            # Append all ACCURACY TEST --> p-phase
            (PRECISION_P, RECALL_P, FONE_P) = __calculate_scores__(TP_P, TN_P, FP_P, FN_P)
            DET_test_confusion_history_p.append((TP_P, TN_P, FP_P, FN_P,
                                                 PRECISION_P, RECALL_P, FONE_P))
            # Append all ACCURACY TEST --> p-phase
            (PRECISION_S, RECALL_S, FONE_S) = __calculate_scores__(TP_S, TN_S, FP_S, FN_S)
            DET_test_confusion_history_s.append((TP_S, TN_S, FP_S, FN_S,
                                                 PRECISION_S, RECALL_S, FONE_S))
        else:
            DET_test_confusion_history_p.append((0, 0, 0, 0, None, None, None))
            DET_test_confusion_history_s.append((0, 0, 0, 0, None, None, None))

        # ============================= >>> PRINTS
        logger.info(
            f"Epoch {epoch}/{confs.OPTIMISATION.epochs}, "
            f"HEIM TRAIN Loss: {HEIM_train_loss:.5f} "
            f"HEIM TEST Loss: {HEIM_test_loss:.5f}")
        logger.info(
            f"    DET - Train: {DET_train_loss:.5f}, Test: {DET_test_loss:.5f}, "
            f"F1_E: {FONE_E:.4f} / "
            f"F1_P: {FONE_P:.4f} / "
            f"F1_S: {FONE_S:.4f}")
        logger.info(
            f"    LOC - Train: {LOC_train_loss:.5f}, Test: {LOC_test_loss:.5f}")
        logger.info(
            f"        XY: {LOC_train_loss_xy:.4f} / {LOC_test_loss_xy:.4f}")
        logger.info(
            f"        XZ: {LOC_train_loss_xz:.4f} / {LOC_test_loss_xz:.4f}")
        logger.info(
            f"        YZ: {LOC_train_loss_yz:.4f} / {LOC_test_loss_yz:.4f}")

        # ============================= >>> SCHEDULER + EARLY STOPPING
        scheduler.step(HEIM_test_loss)
        if not confs.OPTIMISATION.epochs:
            early_stopping(HEIM_test_loss, model, epoch)
            if early_stopping.early_stop:
                logger.state(f"Early stopping triggered at epoch {epoch}")
                FINAL_TRAIN_EPOCH = early_stopping.restore_best(model)
                break
        else:
            FINAL_TRAIN_EPOCH = confs.OPTIMISATION.epochs

    # Save the trained model weights
    logger.info("Storing model in:  %s  After  %d  epochs" % (
        out_save_path,  FINAL_TRAIN_EPOCH))
    torch.save(model.state_dict(), out_save_path)

    return (model,
            #
            HEIM_train_loss_history,
            HEIM_test_loss_history,
            #
            DET_train_loss_history,
            DET_test_loss_history,
            DET_test_confusion_history_event,
            DET_test_confusion_history_p,
            DET_test_confusion_history_s,
            #
            LOC_train_loss_history,
            LOC_train_loss_xy_history,
            LOC_train_loss_xz_history,
            LOC_train_loss_yz_history,
            LOC_test_loss_history,
            LOC_test_loss_xy_history,
            LOC_test_loss_xz_history,
            LOC_test_loss_yz_history,
            FINAL_TRAIN_EPOCH)


def training_flow(train_load_ALL, test_load_ALL, Model, heim_gnn, optimizer, confs):
    """
    Wrapper function to run fine-tuning of the Heimdall model and store training metrics.

    Calls `__training_flow_ALL__` to perform training and evaluation, then saves all
    returned training and evaluation metrics to a compressed `.npz` file.

    Args:
        train_load_ALL (DataLoader): Training data loader.
        test_load_ALL (DataLoader): Test data loader.
        Model (torch.nn.Module): Heimdall model to fine-tune.
        Classifier (torch.nn.Module): Unused classifier instance (ignored).
        heim_gnn (dict): Graph data and related metadata.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        confs (AttributeDict): configuration dictionary containing the training parameter

    Returns:
        tuple: Tuple containing:
            - Model_ref (torch.nn.Module): Fine-tuned Heimdall model.
            - Classifier (torch.nn.Module): Unchanged classifier instance.
    """
    (Model_ref,
     HEIM_train_loss_history,
     HEIM_test_loss_history,
     #
     DET_train_loss_history,
     DET_test_loss_history,
     DET_test_confusion_history_event,
     DET_test_confusion_history_p,
     DET_test_confusion_history_s,
     #
     LOC_train_loss_history,
     LOC_train_loss_xy_history,
     LOC_train_loss_xz_history,
     LOC_train_loss_yz_history,
     LOC_test_loss_history,
     LOC_test_loss_xy_history,
     LOC_test_loss_xz_history,
     LOC_test_loss_yz_history, final_train_epoch) = (
        __training_flow_ALL__(
                    train_load_ALL, test_load_ALL, Model,
                    heim_gnn, optimizer, confs, out_save_path="HEIMDALL.refined.pt"))

    # Save all metrics to an .npz file
    np.savez('refining_training_metrics.npz',
             train_loss_history_heim=HEIM_train_loss_history,
             test_loss_history_heim=HEIM_test_loss_history,
             #
             train_loss_history_det=DET_train_loss_history,
             test_loss_history_det=DET_test_loss_history,
             test_confusion_history_det_event=DET_test_confusion_history_event,
             test_confusion_history_det_s=DET_test_confusion_history_p,
             test_confusion_history_det_p=DET_test_confusion_history_s,
             #
             train_loss_history_loc=LOC_train_loss_history,
             test_loss_history_loc=LOC_test_loss_history,
             train_loss_history_xy_loc=LOC_train_loss_xy_history,
             train_loss_history_xz_loc=LOC_train_loss_xz_history,
             train_loss_history_yz_loc=LOC_train_loss_yz_history,
             test_loss_history_xy_loc=LOC_test_loss_xy_history,
             test_loss_history_xz_loc=LOC_test_loss_xz_history,
             test_loss_history_yz_loc=LOC_test_loss_yz_history,
             #
             batch_size=confs.DATASET.batch_size,  # the effective one, with or without augmentation
             rnd_seed=confs.RANDOM_SEED,
             #
             learning_rate=confs.OPTIMISATION.learning_rate,
             epochs=final_train_epoch,
             early_stop_patience=(confs.OPTIMISATION.early_stopping.patience
                                  if not confs.OPTIMISATION.epochs else None),
             early_stop_delta=(confs.OPTIMISATION.early_stopping.delta
                               if not confs.OPTIMISATION.epochs else None)
    )
    return Model_ref


def unflatten_BN(tensor, B, N):
    """
    Reshape a flattened tensor back to [B, N, C, F].

    Args:
        tensor (torch.Tensor): Input tensor of shape [B*N, C, F].
        B (int): Batch size.
        N (int): Number of nodes.

    Returns:
        torch.Tensor: Reshaped tensor of shape [B, N, C, F].
    """
    return tensor.view(B, N, *tensor.shape[1:])


def validate_flow(model, val_load_ALL, heim_gnn, heim_locator, confs):
    """
    Run validation loop on provided data loader, evaluating detection
    and location performance, computing losses, confusion statistics,
    and optionally generating figures.

    Args:
        model: The main HEIMDALL model.
        val_load_ALL: Validation data loader.
        heim_gnn: Dictionary containing graph and station information.
        heim_locator: Locator module with grid information and conversion utilities.
        confs (dict): Attribute dict configuration file with TRAINING_PARAMETERS

    Returns:
        bool: True if validation completed successfully.
    """
    make_figures = confs.PLOTS.make_plots
    every = confs.PLOTS.every_batches

    model.eval()
    criterion = nn.BCELoss()
    HEIM_running_val_loss = 0.0
    DET_running_val_loss = 0.0
    LOC_running_val_loss = 0.0
    LOC_running_val_loss_xy = 0.0
    LOC_running_val_loss_xz = 0.0
    LOC_running_val_loss_yz = 0.0
    #
    with torch.no_grad():
        for _bidx, batch in enumerate(tqdm(val_load_ALL, total=len(val_load_ALL))):

            # # Skip last one to avoid conflicts
            # if _bidx == len(val_load_ALL)-1:
            #     logger.error("Skipping LASTONE")
            #     continue
            #
            TP_E, TN_E, FP_E, FN_E = 0, 0, 0, 0
            TP_P, TN_P, FP_P, FN_P = 0, 0, 0, 0
            TP_S, TN_S, FP_S, FN_S = 0, 0, 0, 0
            #
            det_inputs = batch.x.to(device)
            det_targets = batch.yd.to(device)
            loc_inputs = batch.xl.to(device)
            loc_target1 = torch.stack([g.yl1.squeeze(0) for g in batch.to_data_list()]).to(device)  # shape: [B, 304, 330]
            loc_target2 = torch.stack([g.yl2.squeeze(0) for g in batch.to_data_list()]).to(device)  # shape: [B, 304, 151]
            loc_target3 = torch.stack([g.yl3.squeeze(0) for g in batch.to_data_list()]).to(device)  # shape: [B, 330, 151]
            edges = batch.edge_index.to(device)
            weights = batch.edge_attr.to(device)
            batch_vector = batch.batch.to(device)
            reals = batch.r.to(device)
            coord_targets = pdfs_to_coords_argmax(
                                loc_target1, loc_target2, loc_target3)

            # ===================================  GO WITH THE FLOW
            # --->  1. Run MODEL
            (evpk, (location_xy, location_xz, location_yz), coords) = model(
                                    det_inputs, edges, weights, batch_vector)

            verdict = torch.zeros(location_xy.shape[0]).to(device)

            # ================================== DETECTOR
            det_loss = criterion(evpk, det_targets)
            DET_running_val_loss += det_loss.item()

            # ================================== LOCATOR
            loss_xy = criterion(location_xy, loc_target1)
            loss_xz = criterion(location_xz, loc_target2)
            loss_yz = criterion(location_yz, loc_target3)
            loc_loss_weighted = (confs.LOCATOR_LOSS.xy * loss_xy +
                                 confs.LOCATOR_LOSS.xz * loss_xz +
                                 confs.LOCATOR_LOSS.yz * loss_yz)

            LOC_running_val_loss += loc_loss_weighted.item()
            LOC_running_val_loss_xy += loss_xy.item()
            LOC_running_val_loss_xz += loss_xz.item()
            LOC_running_val_loss_yz += loss_yz.item()

            # # ================================== COMPILE LOSS Backpropagation and optimization
            # heim_loss = ALPHA*det_loss + BETA*loc_loss_weighted

            # ================================== COMPILE LOSS (new coords)
            # coordinate constraint
            loss_coord = F.smooth_l1_loss(coords, coord_targets)   # or MSE / Huber
            heim_loss = (confs.COMPOSITE_LOSS.alpha*det_loss +
                         confs.COMPOSITE_LOSS.beta*(loc_loss_weighted/3) +
                         confs.COMPOSITE_LOSS.gamma*loss_coord)

            HEIM_running_val_loss += heim_loss.item()
            torch.cuda.empty_cache()

            # =============================== ACCURACY / F1
            (_tpe, _tne, _fpe, _fne) = __batch_stats__(
                evpk, det_targets, channel=0, threshold=0.5, overlap_threshold=0.5)
            TP_E += _tpe; TN_E += _tne; FP_E += _fpe; FN_E += _fne

            if det_targets.shape[1] > 1:
                (_tpp, _tnp, _fpp, _fnp) = __batch_stats_picks__(
                    evpk, det_targets, channel=1)
                TP_P += _tpp; TN_P += _tnp; FP_P += _fpp; FN_P += _fnp

                (_tps, _tns, _fps, _fns) = __batch_stats_picks__(
                    evpk, det_targets, channel=2)
                TP_S += _tps; TN_S += _tns; FP_S += _fps; FN_S += _fns

            # =================================  PLOTS !!!
            B = batch.num_graphs
            N = batch.num_nodes // B

            if make_figures and _bidx % every == 0:
                logger.info(" Plotting")
                stations_latlon = [(vv[1], vv[0]) for kk, vv in
                                   heim_gnn['stations_coordinate'].item().items()]
                stations_xyz = heim_locator.grid.convert_geo_list(stations_latlon)
                #
                Xplot = unflatten_BN(det_inputs.cpu(), B, N).numpy()
                Yplot = unflatten_BN(det_targets.cpu(), B, N).numpy()
                Rplot = unflatten_BN(reals.cpu(), B, N).numpy()
                detections = unflatten_BN(evpk.cpu(), B, N).numpy()
                #
                location_xy = location_xy.cpu().numpy()
                location_xz = location_xz.cpu().numpy()
                location_yz = location_yz.cpu().numpy()
                verdict = verdict.cpu().numpy()

                # -----------  Normalize 0-1
                for _mat in Xplot:
                    for _stat in range(_mat.shape[0]):
                        _mat[_stat, :] /= np.nanmax(_mat[_stat, :])
                # ------------------------------
                _ = gplt.plot_heimdall_flow(
                        Xplot, Yplot, Rplot, detections,
                        location_xy, location_xz, location_yz,
                        heim_locator.get_grid(), verdict,
                        stations=stations_xyz,
                        reference_locations=batch.sources_xyz.cpu().numpy(),
                        store_dir="HeimdallResults/Batch_%03d" % _bidx)
                #
                _ = plot_all_stations(
                        Xplot, Yplot, Rplot, detections,
                        store_dir="HeimdallResults/Batch_%03d" % _bidx)

        # ---> End batches
        HEIM_val_loss = HEIM_running_val_loss / len(val_load_ALL)
        DET_val_loss = DET_running_val_loss / len(val_load_ALL)
        LOC_val_loss = LOC_running_val_loss / len(val_load_ALL)
        LOC_val_loss_xy = LOC_running_val_loss_xy / len(val_load_ALL)
        LOC_val_loss_xz = LOC_running_val_loss_xz / len(val_load_ALL)
        LOC_val_loss_yz = LOC_running_val_loss_yz / len(val_load_ALL)

    # ---> End EVALUATION process
    # ============================= >>> STATS (detector)
    (PRECISION_E, RECALL_E, FONE_E) = __calculate_scores__(TP_E, TN_E, FP_E, FN_E)
    if det_targets.shape[1] > 1:
        (PRECISION_P, RECALL_P, FONE_P) = __calculate_scores__(TP_P, TN_P, FP_P, FN_P)
        (PRECISION_S, RECALL_S, FONE_S) = __calculate_scores__(TP_S, TN_S, FP_S, FN_S)
    else:
        (PRECISION_P, RECALL_P, FONE_P) = 0.0, 0.0, 0.0
        (PRECISION_S, RECALL_S, FONE_P) = 0.0, 0.0, 0.0

    # ============================= >>> PRINTS
    logger.info(
        f"HEIM VALIDATION Loss: {HEIM_val_loss:.5f}")
    logger.info(
        f"    DET - Loss: {DET_val_loss:.5f}   "
        f"F1_E: {FONE_E:.4f} / "
        f"F1_P: {FONE_P:.4f} / "
        f"F1_S: {FONE_S:.4f}")
    logger.info(
        f"    LOC - Loss: {LOC_val_loss:.5f}   "
        f"XY: {LOC_val_loss_xy:.4f} / "
        f"XZ: {LOC_val_loss_xz:.4f} / "
        f"YZ: {LOC_val_loss_yz:.4f}")

    return True


def plot_all_stations(xmat, ymat, rmat, dets,
                      store_dir="BatchResults"):
    """
    Plot time series, labels, real data, and predictions for all stations in a batch.

    Args:
        xmat (np.ndarray): Array of input features (char functions) of shape [B, N, C, F].
        ymat (np.ndarray): Array of ground-truth labels of shape [B, N, C, F].
        rmat (np.ndarray): Array of real observed waveforms of shape [B, N, C, F].
        dets (np.ndarray): Array of predicted detection outputs of shape [B, N, C, F].
        store_dir (str): Directory path to store resulting plots.

    Returns:
        list[matplotlib.figure.Figure]: List of figures created for each batch.
    """
    # Inputs msut be NUMPY array already
    fig_list = []
    for ii in tqdm(range(xmat.shape[0])):
        # Create the figure
        # fig = plt.figure(figsize=(8, 8))
        fig = plt.figure(figsize=(19, 7))

        # Set up the grid
        # gs = GridSpec(6, 5, figure=fig)
        gs = GridSpec(6, 12, figure=fig)

        _x, _y, _r, _p = xmat[ii], ymat[ii], rmat[ii], dets[ii]
        condition = np.any(_y >= 0.95, axis=-1)
        _tot_picks = np.sum(np.any(condition, axis=1))

        # Plot the timeseries on the left (rectangular panels) in columns 1-3
        for cc in range(4):
            # 4 columns
            for rr in range(6):
                xx = (cc*6)+rr
                ax = fig.add_subplot(gs[rr, (0+cc*3):(3+cc*3)])
                lf_ax2 = ax.twinx()

                # # REAL
                # ax.plot(_r[xx][1] + _r[xx][2], color="darkgray", alpha=0.6, label="real_NE")
                # ax.plot(_r[xx][0], color="black", alpha=0.6, label="real_Z")

                # INPUTS (CHARFUNCTION)
                for _cfs in range(_x[xx].shape[0]):
                    # lf_ax2.plot(_x[xx][_cfs], alpha=0.7, color=f"C{_cfs}", label=f"CF_{_cfs}")
                    ax.plot(_x[xx][_cfs], alpha=0.7, color=f"C{_cfs}", label=f"CF_{_cfs}")

                # LABELS
                lf_ax2.plot(_y[xx][0], alpha=0.7, label="Ye", color="darkgray", ls="-")
                lf_ax2.plot(_y[xx][1], alpha=0.7, label="Yp", color="darkgray", ls="--")
                lf_ax2.plot(_y[xx][2], alpha=0.7, label="Ys", color="darkgray", ls=":")

                # PREDICTIONS
                lf_ax2.plot(_p[xx][0], alpha=0.7, label="prediction", color="purple")
                if _p[xx].shape[0] > 1:
                    # picker_type
                    lf_ax2.plot(_p[xx][1], alpha=0.7, label="prediction_P", color="darkblue")
                    lf_ax2.plot(_p[xx][2], alpha=0.7, label="prediction_S", color="darkred")

                lf_ax2.set_ylim([-0.2, 1.2])
                # ax.set_ylabel(sorted_indices_ylabel[idx])  # station name
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


class EarlyStopping:
    """
    Early stopping utility to halt training when validation loss fails to improve.

    Attributes:
        patience (int): Number of epochs to wait after last improvement.
        delta (float): Minimum change in validation loss to qualify as improvement.
        verbose (bool): If True, prints messages when counter increases.
    """
    def __init__(self, patience=5, delta=5e-4, verbose=False):
        """Initialize early stopping parameters."""
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_weights = None
        self.verbose = verbose
        logger.state("Early stop initialized. PATIENCE:  %d  DELTA:  %.6f" %
                     (self.patience, self.delta))

    def __call__(self, val_loss, model, epoch_num, reverse=True):
        """Check if validation loss improved, update best model state, and increment counter otherwise.
        Args:
            val_loss (float): Current epoch validation loss.
            model (torch.nn.Module): Model being trained.
            epoch_num (int): Current epoch number.
            reverse (bool): If True, assumes minimizing loss.
        """

        if reverse:
            score = -val_loss  # Minimizing loss => maximizing -val_loss
        else:
            score = val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.best_model_epoch = epoch_num
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                logger.warning(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # Store the *best* weights
            self.best_model_weights = copy.deepcopy(model.state_dict())
            self.best_model_epoch = epoch_num
            self.counter = 0

    def restore_best(self, model):
        """Restore model weights from epoch with best validation loss.

        Args:
            model (torch.nn.Module): Model to restore.

        Returns:
            int: Epoch number corresponding to best validation loss.
        """
        if self.best_model_weights is not None:
            model.load_state_dict(self.best_model_weights)
        return self.best_model_epoch


def main(heim_gnn_path, heim_grid_path, config_path, file_path_train):
    startt = time.time()
    logger.info("HEIMDALL starting ...")

    # ---------- 0. Load MSEED / Config
    logger.state("Loading GNN:   %s" % heim_gnn_path)
    heim_gnn = np.load(heim_gnn_path, allow_pickle=True)
    stats_coords = build_coord_matrix(heim_gnn["stations_coordinate"].item(),
                                      heim_gnn["stations_order"].item())  # (36,3)
    stats_coords_norm, stats = normalise_station_coords(stats_coords,
                                                        return_stats=True,
                                                        zero_one=False)

    logger.state("Loading GRID:  %s" % heim_grid_path)
    heim_grid = np.load(heim_grid_path, allow_pickle=True)

    logger.state("Loading CONFIG:  %s" % config_path)
    CONFIG = gio.read_configuration_file(config_path, check_version=True)
    TP = CONFIG.TRAINING_PARAMETERS          # AttributeDict level

    # ---------- 1. Initialize  LOCATOR
    HG = glctr.HeimdallLocator(heim_grid['boundaries'],
                               spacing_x=heim_grid['spacing_km'][0],
                               spacing_y=heim_grid['spacing_km'][1],
                               spacing_z=heim_grid['spacing_km'][2],
                               reference_point_lonlat=heim_grid['reference_point'])
    (xgr, ygr, zgr) = HG.get_grid()
    (reflon, reflat) = HG.get_grid_reference()
    SHAPES = [(len(xgr), len(ygr)), (len(xgr), len(zgr)), (len(ygr), len(zgr))]

    # ---------- 1a. Initialize  MODEL
    logger.info(" Initializing MODELS")
    (Heimdall, optimizer) = __init_model__(SHAPES, stats_coords_norm, TP)

    # ---------- Read Training Data --> July2025
    # We now have saved with everything to save time n an unique NPZ

    logger.info("Preparing HDF5 loaders")
    (train_load, test_load, val_load) = prepare_h5_loaders(
                                            file_path_train, heim_gnn, TP)

    logger.state(" T R A I N I N G   S T A R T S   !!!")
    Heimdall_ref = training_flow(
                train_load, test_load, Heimdall, heim_gnn, optimizer, TP)

    validate_flow(Heimdall_ref, val_load, heim_gnn, HG, TP)

    endt = time.time()
    logger.info("DONE!  Running Time:  %.2f hr." % ((endt-startt)/3600.0))


def build_coord_matrix(stations_coordinate: dict, stations_order: dict,
                       dtype=np.float32) -> np.ndarray:
    """
    Build a [N_sta, 3] NumPy array with rows as (lon, lat, elev), ordered by `stations_order`.

    Args:
        stations_coordinate (dict): Mapping {station_id: (lon, lat, elev)}.
        stations_order (dict): Mapping {station_id: index} specifying output row order.
        dtype (np.dtype, optional): Data type of output array (default: np.float32).

    Raises:
        KeyError: If any station in `stations_order` is missing in `stations_coordinate`.
        ValueError: If the two dicts have different numbers of stations.

    Returns:
        np.ndarray: Array of shape [N_sta, 3] with station coordinates.
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


def normalise_station_coords(coord_mat,
                             centre=True,
                             std_scale=True,
                             return_stats=False,
                             zero_one=False):
    """
    Normalize station coordinates from (lon, lat, elev) to (x, y, z) in kilometers.

    Args:
        coord_mat (np.ndarray): Array [N,3] with columns (lon_deg, lat_deg, elev_m).
        centre (bool): If True, subtract centroid before scaling.
        std_scale (bool): If True, divide each channel by its std deviation.
        return_stats (bool): If True, also return (centroid, std) dict.
        zero_one (bool): If True, scale coordinates to [0,1] range.

    Returns:
        np.ndarray: Normalized coordinates [N,3] in float32.
        dict (optional): {'centroid': tuple, 'std': array} if `return_stats` is True.
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


def _parse_cli():
    parser = argparse.ArgumentParser(
        prog=Path(sys.argv[0]).name,
        description="Train HEIMDALL with a single HDF5 set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-g", "--graph", metavar="GRAPH", type=str, dest="gnn",
                        required=True, help="Heimdall GNN graph file (.npz)")
    parser.add_argument("-grd", "--grid", metavar="GRID", type=str, dest="grid",
                        required=True, help="pre-built grid (npz) (default: %(default)s)")
    parser.add_argument("-f", "--file", metavar="H5TRAIN", type=str, dest="h5train",
                        required=True, help="*.h5 with the training data")
    parser.add_argument("-c", "--conf", dest="config", metavar="YAML", type=str,
                        required=True, help="Configuration YAML file")
    return parser

if __name__ == "__main__":

    p = _parse_cli()
    args = p.parse_args()
    # confs = args.TRAINING_PARAMETERS
    #
    args.gnn = str(Path(args.gnn).expanduser().resolve())
    args.grid = str(Path(args.grid).expanduser().resolve())
    args.config = str(Path(args.config).expanduser().resolve())
    args.h5train = str(Path(args.h5train).expanduser().resolve())
    #
    logger.state("Inputs:")
    logger.state(f"    gnn: {args.gnn}")
    logger.state(f"    grid: {args.grid}")
    logger.state(f"    config: {args.config}")
    logger.state(f"    h5train: {args.h5train}")
    main(args.gnn, args.grid, args.config, args.h5train)

