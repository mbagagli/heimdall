#!/usr/bin/env python

"""
End-to-end training script for HEIMDALL detector-locator
(GNN-based).  It loads seismic datasets, applies optional data
augmentation, trains the joint detector + locator network with early
stopping, and finally evaluates on a hold-out set.
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

__log_name__ = "HeimdallTraining.log"
logger = CL.init_logger(__log_name__, lvl="INFO", log_file=__log_name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@  SET GLOBALS  @@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def apply_training_parameters(tp):
    """Utility for setting-up training global parameters

    Args:
        tp (AttributeDict): extract from configuration YAML file
    """
    global MAKE_PLOTS, EVERY, HOWMANY, DO_AUGMENTATIONS, BATCH_SIZE_ALL
    global BATCH_SIZE, RND_SEED, EVENIZE_ALL, LEARNING_RATE
    global NUM_EPOCHS, ES_PATIENCE_ALL, ES_DELTA_ALL
    global TEST_SPLIT, VAL_SPLIT
    global W1_XY, W2_XZ, W3_YZ, ALPHA, BETA, GAMMA
    global MODEL_WEIGHTS, FREEZE_ENCODER

    # ------ 1. plots / logging
    MAKE_PLOTS = tp.PLOTS.make_plots
    EVERY      = tp.PLOTS.every_batches

    # ------ 2. data / augm.
    HOWMANY          = tp.DATASET.how_many
    EVENIZE_ALL      = dict(tp.DATASET.evenize)
    RND_SEED         = tp.RANDOM_SEED
    DO_AUGMENTATIONS = tp.AUGMENTATION.enabled
    BATCH_SIZE_ALL   = tp.AUGMENTATION.batch_size_all
    BATCH_SIZE       = BATCH_SIZE_ALL // 2 if DO_AUGMENTATIONS else BATCH_SIZE_ALL

    # ------ 3. split & optimiser
    TEST_SPLIT = tp.SPLIT.test
    VAL_SPLIT  = tp.SPLIT.val
    LEARNING_RATE   = tp.OPTIMISATION.learning_rate
    NUM_EPOCHS      = tp.OPTIMISATION.epochs
    ES_PATIENCE_ALL = tp.OPTIMISATION.early_stopping.patience
    ES_DELTA_ALL    = tp.OPTIMISATION.early_stopping.delta

    # ------ 4. loss weights & model init
    W1_XY = tp.LOCATOR_LOSS.xy
    W2_XZ = tp.LOCATOR_LOSS.xz
    W3_YZ = tp.LOCATOR_LOSS.yz
    ALPHA = tp.COMPOSITE_LOSS.alpha
    BETA  = tp.COMPOSITE_LOSS.beta
    GAMMA = tp.COMPOSITE_LOSS.gamma

    MODEL_WEIGHTS  = tp.MODEL.pretrained_weights
    FREEZE_ENCODER = tp.MODEL.freeze_encoder
    return True


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@  MODELS  @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def __init_model__(shapes, stations, freeze_encoder=FREEZE_ENCODER, weights=""):
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
    _optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=LEARNING_RATE)
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
        if "jw" in combo:
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

def __unpack_data_LOCATOR__(X):
    """Split locator targets into three numpy arrays.

    Args:
        X (list[np.ndarray]): Each element contains three 2-D planes
            (XY, XZ, YZ).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Arrays with shapes
        ``(E, 304, 330)``, ``(E, 304, 151)``, ``(E, 330, 151)``.
    """
    # Initialize empty lists to store the arrays
    data1_list = []
    data2_list = []
    data3_list = []

    # Iterate through each element in the first dimension of X
    for i in range(X.shape[0]):
        data1_list.append(X[i][0])  # (62, 67)
        data2_list.append(X[i][1])  # (62, 31)
        data3_list.append(X[i][2])  # (67, 31)

    # Convert the lists to numpy arrays with the desired shape
    data1 = np.array(data1_list)  # Shape will be (131379, 62, 67)
    data2 = np.array(data2_list)  # Shape will be (131379, 62, 31)
    data3 = np.array(data3_list)  # Shape will be (131379, 67, 31)

    return (data1, data2, data3)


def __prepare_data_ALL__(data_det, label_det, data_loc,
                         label1, label2, label3, reals, sources_xyz, heim_gnn,
                         test_size=TEST_SPLIT, val_size=VAL_SPLIT,
                         batch_size=32, rnd_seed=RND_SEED,
                         adding_path=""):
    """Create train/val/test DataLoaders and apply optional injection."""

    assert len(data_det) == len(label_det) == len(data_loc) == len(label1) == len(label2) == len(label3) == len(sources_xyz) == len(reals), \
        "All inputs must have the same length"

    # Generate indices
    indices = np.arange(len(data_det))

    # First split: Train vs. Remaining (Test + Validation)
    train_idx, remaining_idx = train_test_split(
        indices, test_size=test_size + val_size, random_state=rnd_seed)

    # Second split: Remaining (Test + Validation) into Test vs. Validation
    test_idx, val_idx = train_test_split(
        remaining_idx, test_size=test_size / (test_size + val_size),
        random_state=rnd_seed)

    assert len(set(train_idx) & set(remaining_idx)) == 0, \
           "Train and remaining indices must be disjoint"
    assert len(set(test_idx) & set(val_idx)) == 0, \
           "Test and validation indices must be disjoint"

    # Indexing instead of copying large arrays
    (train_data_det, train_label_det, train_data_loc,
     train_label1, train_label2, train_label3,
     train_reals, train_sources_xyz) = (data_det[train_idx], label_det[train_idx],
                                        data_loc[train_idx], label1[train_idx],
                                        label2[train_idx], label3[train_idx],
                                        reals[train_idx], sources_xyz[train_idx])

    (test_data_det, test_label_det, test_data_loc,
     test_label1, test_label2, test_label3,
     test_reals, test_sources_xyz) = (data_det[test_idx], label_det[test_idx],
                                      data_loc[test_idx], label1[test_idx],
                                      label2[test_idx], label3[test_idx],
                                      reals[test_idx], sources_xyz[test_idx])

    (val_data_det, val_label_det, val_data_loc,
     val_label1, val_label2, val_label3,
     val_reals, val_sources_xyz) = (data_det[val_idx], label_det[val_idx],
                                    data_loc[val_idx], label1[val_idx],
                                    label2[val_idx], label3[val_idx],
                                    reals[val_idx], sources_xyz[val_idx])

    if adding_path:
        logger.warning("Adding training dataset injection !!! [%s]" % adding_path)
        adding = np.load(adding_path, allow_pickle=True)
        add_pick_count = adding["pick_count"]
        add_data_det = adding["Xdet"].astype(np.float32)
        add_labels_det = adding["Ydet"].astype(np.float32)
        add_labels_loc = adding["Yloc"]  # convert to float32 later
        add_reals = adding["R"].astype(np.float32)
        add_sources_xyz = adding["sources_grid"].astype(np.float32)

        add_data_loc = add_labels_det[:, :, :1, :]  # keep only the first channel for training
        (_label1, _label2, _label3) = __unpack_data_LOCATOR__(add_labels_loc)
        add_label1 = _label1.astype(np.float32)
        add_label2 = _label2.astype(np.float32)
        add_label3 = _label3.astype(np.float32)

        train_data_det = np.concatenate(
                        [train_data_det.astype(np.float32),
                         add_data_det.astype(np.float32)], axis=0)
        train_label_det = np.concatenate(
                        [train_label_det.astype(np.float32),
                         add_labels_det.astype(np.float32)], axis=0)
        train_data_loc = np.concatenate(
                        [train_data_loc.astype(np.float32),
                         add_data_loc.astype(np.float32)], axis=0)
        train_label1 = np.concatenate(
                        [train_label1.astype(np.float32),
                         add_label1.astype(np.float32)], axis=0)
        train_label2 = np.concatenate(
                        [train_label2.astype(np.float32),
                         add_label2.astype(np.float32)], axis=0)
        train_label3 = np.concatenate(
                        [train_label3.astype(np.float32),
                         add_label3.astype(np.float32)], axis=0)
        train_reals = np.concatenate(
                        [train_reals.astype(np.float32),
                         add_reals.astype(np.float32)], axis=0)
        train_sources_xyz = np.concatenate(
                        [train_sources_xyz.astype(np.float32),
                         add_sources_xyz.astype(np.float32)], axis=0)

    # ==========================================

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

    train_data_list = [
            # shape: [N, C, T]
            Data(
                x=__scale_minusone_one(
                    torch.tensor(train_data_det[idx], dtype=torch.float32)),
                yd=torch.tensor(train_label_det[idx], dtype=torch.float32),
                xl=torch.tensor(train_data_loc[idx], dtype=torch.float32),
                yl1=torch.tensor(train_label1[idx], dtype=torch.float32).unsqueeze(0),
                yl2=torch.tensor(train_label2[idx], dtype=torch.float32).unsqueeze(0),
                yl3=torch.tensor(train_label3[idx], dtype=torch.float32).unsqueeze(0),
                r=torch.tensor(train_reals[idx], dtype=torch.float32),
                sources_xyz=torch.tensor(train_sources_xyz[idx], dtype=torch.float32).unsqueeze(0),
                edge_index=torch.tensor(heim_gnn["edges"], dtype=torch.long),
                edge_attr=torch.tensor(heim_gnn["weights"], dtype=torch.float32),
                num_nodes=train_data_det[idx].shape[0]
            )
            for idx in range(train_data_det.shape[0])     # E = num events
        ]
    train_loader = DataLoader(
                    train_data_list, batch_size=batch_size, shuffle=True)

    test_data_list = [
            Data(
                x=__scale_minusone_one(
                    torch.tensor(test_data_det[idx], dtype=torch.float32)),
                yd=torch.tensor(test_label_det[idx], dtype=torch.float32),
                xl=torch.tensor(test_data_loc[idx], dtype=torch.float32),
                yl1=torch.tensor(test_label1[idx], dtype=torch.float32).unsqueeze(0),
                yl2=torch.tensor(test_label2[idx], dtype=torch.float32).unsqueeze(0),
                yl3=torch.tensor(test_label3[idx], dtype=torch.float32).unsqueeze(0),
                r=torch.tensor(test_reals[idx], dtype=torch.float32),
                sources_xyz=torch.tensor(test_sources_xyz[idx], dtype=torch.float32).unsqueeze(0),
                edge_index=torch.tensor(heim_gnn["edges"], dtype=torch.long),
                edge_attr=torch.tensor(heim_gnn["weights"], dtype=torch.float32),
                num_nodes=test_data_det[idx].shape[0]
            )
            for idx in range(test_data_det.shape[0])     # E = num events
        ]
    test_loader = DataLoader(
                    test_data_list, batch_size=batch_size, shuffle=False)

    val_data_list = [
            Data(
                x=__scale_minusone_one(
                    torch.tensor(val_data_det[idx], dtype=torch.float32)),                      # shape: [N, C, T]
                yd=torch.tensor(val_label_det[idx], dtype=torch.float32),
                xl=torch.tensor(val_data_loc[idx], dtype=torch.float32),
                yl1=torch.tensor(val_label1[idx], dtype=torch.float32).unsqueeze(0),
                yl2=torch.tensor(val_label2[idx], dtype=torch.float32).unsqueeze(0),
                yl3=torch.tensor(val_label3[idx], dtype=torch.float32).unsqueeze(0),
                r=torch.tensor(val_reals[idx], dtype=torch.float32),
                sources_xyz=torch.tensor(val_sources_xyz[idx], dtype=torch.float32).unsqueeze(0),
                edge_index=torch.tensor(heim_gnn["edges"], dtype=torch.long),
                edge_attr=torch.tensor(heim_gnn["weights"], dtype=torch.float32),
                num_nodes=val_data_det[idx].shape[0]
            )
            for idx in range(val_data_det.shape[0])     # E = num events
        ]
    val_loader = DataLoader(
                    val_data_list, batch_size=BATCH_SIZE_ALL, shuffle=False)

    return (train_loader, test_loader, val_loader)


def prepare_ALL(npz, heim_gnn, how_many=None, evenize={}, adding_path=""):
    """Load an ``.npz`` archive, balance classes, and create loaders.

    Args:
        npz (np.lib.npyio.NpzFile): Pre-saved training dataset.
        heim_gnn (np.lib.npyio.NpzFile): GNN description (edges, etc.).
        how_many (int | None, optional): If set, keep only the first
            *how_many* events.
        evenize (dict | None, optional): Parameters to pass to
            :func:`__evenize_classes__`.
        adding_path (str, optional): Additional ``.npz`` to inject.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: Train, test, val loaders.
    """
    logger.info("Preparing DATA")

    # =========================================================
    pick_count = npz["pick_count"]
    data_det = npz["Xdet"].astype(np.float32)
    labels_det = npz["Ydet"].astype(np.float32)
    labels_loc = npz["Yloc"]  # convert to float32 later
    reals = npz["R"].astype(np.float32)
    sources_xyz = npz["sources_grid"].astype(np.float32)

    if evenize:
        logger.info(" Evenizing classes")
        selected_index = __evenize_classes__(
            pick_count,
            seed=RND_SEED, shuffle=False,  # The shuffle is done by train_test_split function!
            **evenize)

        logger.info("Extracting  %6d elements" % len(selected_index))
        pick_count = pick_count[selected_index]
        data_det = data_det[selected_index]
        labels_det = labels_det[selected_index]
        labels_loc = labels_loc[selected_index]
        reals = reals[selected_index]
        sources_xyz = sources_xyz[selected_index]
        logger.info("DONE")

    if how_many and isinstance(how_many, int):
        logger.info("Selecting the first %5d samples" % how_many)
        data_det = data_det[:how_many]
        labels_det = labels_det[:how_many]
        labels_loc = labels_loc[:how_many]
        reals = reals[:how_many]
        sources_xyz = sources_xyz[:how_many]
        reals = reals[:how_many]
        sources_xyz = sources_xyz[:how_many]

    data_loc = labels_det[:, :, :1, :]  # keep only the first channel for training
    (label1, label2, label3) = __unpack_data_LOCATOR__(labels_loc)
    label1 = label1.astype(np.float32)
    label2 = label2.astype(np.float32)
    label3 = label3.astype(np.float32)

    logger.info("... Doing _ALL_")
    (train_data_loader_ALL, test_data_loader_ALL, val_data_loader_ALL) = (
                                __prepare_data_ALL__(
                                        data_det, labels_det,
                                        data_loc, label1, label2, label3,
                                        reals, sources_xyz,
                                        heim_gnn, batch_size=BATCH_SIZE,
                                        adding_path=adding_path))

    return (train_data_loader_ALL, test_data_loader_ALL, val_data_loader_ALL)


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


def __training_flow_ALL__(train_loader_ALL, test_loader_ALL, model, gnn, optimizer,
                          out_save_path="HeimdallModel_LOCATOR_refined.pt"):
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
    if not NUM_EPOCHS or NUM_EPOCHS == 0:
        early_stopping = EarlyStopping(patience=ES_PATIENCE_ALL, delta=ES_DELTA_ALL,
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
    for epoch in range(NUM_EPOCHS if NUM_EPOCHS else 9999):

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

            if DO_AUGMENTATIONS:
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
            loc_loss_weighted = (W1_XY * loss_xy +
                                 W2_XZ * loss_xz +
                                 W3_YZ * loss_yz)

            LOC_running_train_loss += loc_loss_weighted.item()
            LOC_running_train_loss_xy += loss_xy.item()
            LOC_running_train_loss_xz += loss_xz.item()
            LOC_running_train_loss_yz += loss_yz.item()

            # # ================================== COMPILE LOSS
            # heim_loss = ALPHA*det_loss + BETA*loc_loss_weighted

            # ================================== COMPILE LOSS (new coords)
            # coordinate constraint
            loss_coord = F.smooth_l1_loss(coords, coord_targets)   # or MSE / Huber
            heim_loss = ALPHA*det_loss + BETA*(loc_loss_weighted/3) + GAMMA*loss_coord

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
                loc_loss_weighted = (W1_XY * loss_xy +
                                     W2_XZ * loss_xz +
                                     W3_YZ * loss_yz)

                LOC_running_test_loss += loc_loss_weighted.item()
                LOC_running_test_loss_xy += loss_xy.item()
                LOC_running_test_loss_xz += loss_xz.item()
                LOC_running_test_loss_yz += loss_yz.item()

                # # ================================== COMPILE LOSS
                # heim_loss = ALPHA*det_loss + BETA*loc_loss_weighted

                # ================================== COMPILE LOSS (new coords)
                # coordinate constraint
                loss_coord = F.smooth_l1_loss(coords, coord_targets)   # or MSE / Huber
                heim_loss = ALPHA*det_loss + BETA*(loc_loss_weighted/3) + GAMMA*loss_coord
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
            f"Epoch {epoch}/{NUM_EPOCHS}, "
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
        if not NUM_EPOCHS:
            early_stopping(HEIM_test_loss, model, epoch)
            if early_stopping.early_stop:
                logger.state(f"Early stopping triggered at epoch {epoch}")
                FINAL_TRAIN_EPOCH = early_stopping.restore_best(model)
                break
        else:
            FINAL_TRAIN_EPOCH = NUM_EPOCHS

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


def training_flow(train_load_ALL, test_load_ALL, Model, heim_gnn, optimizer):
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

    Returns:
        tuple: Tuple containing:
            - Model_ref (torch.nn.Module): Fine-tuned Heimdall model.
            - Classifier (torch.nn.Module): Unchanged classifier instance.
    """
    logger.state("Fine-Tuning ALL")

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
                    heim_gnn, optimizer, out_save_path="HEIMDALL.refined.pt"))

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
             batch_size=BATCH_SIZE_ALL,  # the effective one, with or without augmentation
             rnd_seed=RND_SEED,
             #
             learning_rate=LEARNING_RATE,
             epochs=final_train_epoch,
             early_stop_patience=ES_PATIENCE_ALL if not NUM_EPOCHS else None,
             early_stop_delta=ES_DELTA_ALL if not NUM_EPOCHS else None)
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


def validate_flow(model, classifier, val_load_ALL,
                  heim_gnn, heim_locator,
                  make_figures=True, every=5):
    """
    Run validation loop on provided data loader, evaluating detection
    and location performance, computing losses, confusion statistics,
    and optionally generating figures.

    Args:
        model: The main HEIMDALL model.
        classifier: Classifier module operating on location outputs.
        val_load_ALL: Validation data loader.
        heim_gnn: Dictionary containing graph and station information.
        heim_locator: Locator module with grid information and conversion utilities.
        make_figures (bool): Whether to generate plots during validation.
        every (int): Frequency (in batches) for generating plots.

    Returns:
        bool: True if validation completed successfully.
    """
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
            loc_loss_weighted = (W1_XY * loss_xy +
                                 W2_XZ * loss_xz +
                                 W3_YZ * loss_yz)

            LOC_running_val_loss += loc_loss_weighted.item()
            LOC_running_val_loss_xy += loss_xy.item()
            LOC_running_val_loss_xz += loss_xz.item()
            LOC_running_val_loss_yz += loss_yz.item()

            # # ================================== COMPILE LOSS Backpropagation and optimization
            # heim_loss = ALPHA*det_loss + BETA*loc_loss_weighted

            # ================================== COMPILE LOSS (new coords)
            # coordinate constraint
            loss_coord = F.smooth_l1_loss(coords, coord_targets)   # or MSE / Huber
            heim_loss = ALPHA*det_loss + BETA*(loc_loss_weighted/3) + GAMMA*loss_coord

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


def main(heim_gnn_path, heim_grid_path, config_path,
         file_path_train, file_path_train_additional=""):
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
    logger.state("... setting globals")
    _ = apply_training_parameters(TP)

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
    (Heimdall, optimizer) = __init_model__(SHAPES,
                                           stats_coords_norm,
                                           weights=MODEL_WEIGHTS)

    # ---------- Read Training Data --> March2025
    # We now have saved with everything to save time n an unique NPZ
    logger.info("Loading ALL npz")
    npz_all = np.load(file_path_train, allow_pickle=True)
    logger.info("Splitting FINETUNING dataset")
    (train_load, test_load, val_load) = prepare_ALL(npz_all,
                                                    heim_gnn,
                                                    how_many=HOWMANY,
                                                    evenize=EVENIZE_ALL,
                                                    adding_path=file_path_train_additional)

    logger.info("Starting FINETUNING")
    Heimdall_ref = training_flow(
                train_load, test_load, Heimdall, heim_gnn, optimizer)

    validate_flow(Heimdall_ref, val_load, heim_gnn, HG,
                  make_figures=MAKE_PLOTS, every=EVERY)

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


if __name__ == "__main__":
    if not (5 <= len(sys.argv) <= 6):
        logger.error(
            "%s  GNN.npz  GRID.npz  CONFIG.yml  TRAIN.npz  [ADDTRAIN.npz]",
            Path(sys.argv[0]).name,
        )
        sys.exit()

    # Pass the optional argument only if provided
    if len(sys.argv) == 5:                       # no ADDTRAIN given
        main(*sys.argv[1:5])                     # 4 params
    else:                                        # ADDTRAIN present
        main(*sys.argv[1:6])                     # 5 params
