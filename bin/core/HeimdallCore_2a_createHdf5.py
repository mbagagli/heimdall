#!/usr/bin/env python
"""
Aggregate *.npz shards into a single HDF5 file – FAST version.

1. Scan once to know TOTAL number of events and create fixed-size datasets.
2. Write straight into the correct slice (no per-file `resize()` calls).
3. Use the much faster `lzf` compressor.

"""

from __future__ import annotations
import argparse
import glob
import os
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Tuple

# --------------------------------------------------------------------- helpers
def first_shapes(sample: np.lib.npyio.NpzFile) -> Tuple[dict, Tuple[int]]:
    """Return dict{key: shape[1:]} + the 3 locator plane shapes."""
    sh = {k: sample[k].shape[1:] for k in sample.files if k != "Yloc"}
    # Yloc is an object-array of length n with [XY,XZ,YZ] planes
    xy, xz, yz = sample["Yloc"][0]
    return sh, xy.shape, xz.shape, yz.shape

def split_yloc_vectorised(yloc_obj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     plane1 = np.array([yloc_obj[ii][0] for ii in range(yloc_obj.shape[0])], dtype="float32")
#     plane2 = np.array([yloc_obj[ii][1] for ii in range(yloc_obj.shape[0])], dtype="float32")
#     plane3 = np.array([yloc_obj[ii][2] for ii in range(yloc_obj.shape[0])], dtype="float32")
#     return (plane1, plane2, plane3)
    return (np.array([yloc_obj[ii][0] for ii in range(yloc_obj.shape[0])], dtype="float32"),
            np.array([yloc_obj[ii][1] for ii in range(yloc_obj.shape[0])], dtype="float32"),
            np.array([yloc_obj[ii][2] for ii in range(yloc_obj.shape[0])], dtype="float32"))

def create(h5, name, shape_tail, N, chunks, dtype=np.float32, **kw):
    """Fixed-size dataset with chunked layout."""
    return h5.create_dataset(
        name,
        shape=(N, *shape_tail),
        maxshape=(N, *shape_tail),      # no further growing
        chunks=(chunks, *shape_tail),
        compression="lzf",              # << change here instant speed-up
        dtype=dtype,
        **kw)

# ----------------------------------------------------------------------- main
def main(in_dir: str, out_file: str, chunk_len: int = 512) -> None:
    print("... set-up archive:  %s" % out_file)
    print("... chunk-length:    %d" % chunk_len)
    files = sorted(glob.glob(os.path.join(in_dir, "*.npz")))
    if not files:
        raise RuntimeError(f"No .npz files in {in_dir}")
    # ---------- phase 0 – metadata scan --------------------------------------
    n_per_file = []
    for f in files:
        with np.load(f, mmap_mode="r", allow_pickle=True) as z:
            n_per_file.append(z["Xdet"].shape[0])
    Ntot = int(np.sum(n_per_file))

    # ---------- phase 1 – create H5 ------------------------------------------
    with np.load(files[0], allow_pickle=True) as s0:
        sh, sh_xy, sh_xz, sh_yz = first_shapes(s0)

    # with h5py.File(out_file, "w") as h5:
    with h5py.File(out_file, "w", libver="latest") as h5:
        d_xdet = create(h5,"Xdet",   sh["Xdet"],   Ntot, chunk_len)
        d_ydet = create(h5,"Ydet",   sh["Ydet"],   Ntot, chunk_len)
        d_r    = create(h5,"R",      sh["R"],      Ntot, chunk_len)
        d_ycls = create(h5,"Ycls",   sh["Ycls"],   Ntot, chunk_len)
        d_src  = create(h5,"sources",sh["sources"],Ntot, chunk_len)
        d_srcg = create(h5,"sources_grid",sh["sources_grid"],Ntot, chunk_len)
        d_pcnt = create(h5,"pick_count",sh["pick_count"],Ntot, chunk_len)
        d_std  = create(h5,"sourcestd_noise",sh["sourcestd_noise"],Ntot, chunk_len)
        # locator planes
        d_xy = create(h5,"Yloc_XY", sh_xy, Ntot, chunk_len)
        d_xz = create(h5,"Yloc_XZ", sh_xz, Ntot, chunk_len)
        d_yz = create(h5,"Yloc_YZ", sh_yz, Ntot, chunk_len)

        # ---------- phase 2 – streaming write --------------------------------
        cursor = 0
        for fn, n_ev in tqdm(zip(files, n_per_file),
                             total=len(files), desc="Packing",
                             bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            with np.load(fn, allow_pickle=True) as z:
                sl = slice(cursor, cursor + n_ev)      # where to write

                # plain copies
                d_xdet[sl] = z["Xdet"]
                d_ydet[sl] = z["Ydet"]
                d_r[sl] = z["R"]
                d_ycls[sl] = z["Ycls"]
                d_src[sl] = z["sources"]
                d_srcg[sl] = z["sources_grid"]
                d_pcnt[sl] = z["pick_count"]
                d_std[sl] = z["sourcestd_noise"]

                # locator planes – vectorised
                xy, xz_, yz = split_yloc_vectorised(z["Yloc"])
                d_xy[sl], d_xz[sl], d_yz[sl] = xy, xz_, yz

            cursor += n_ev

        h5.attrs["n_events"] = Ntot
    print(f"Finished. {Ntot} events written to {out_file}")

# --------------------------------------------------------------------------- CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("out_h5")
    ap.add_argument("--chunk", type=int, default=1,
                    help=("rows per HDF5 chunk – tune for your I/O. "
                          "If disk-space is not a problem, leave it to 1")
                    )
    args = ap.parse_args()
    Path(args.out_h5).parent.mkdir(parents=True, exist_ok=True)
    main(args.in_dir, args.out_h5, args.chunk)
