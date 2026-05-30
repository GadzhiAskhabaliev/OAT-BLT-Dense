"""
Create a small LIBERO zarr subset for quick smoke validation.
"""

import argparse
import os
import shutil

import numpy as np
import zarr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Source zarr path")
    parser.add_argument("--dst", required=True, help="Destination zarr path")
    parser.add_argument("--episodes", type=int, default=32, help="Number of episodes to keep")
    args = parser.parse_args()

    if os.path.exists(args.dst):
        shutil.rmtree(args.dst)

    src_root = zarr.open(args.src, mode="r")
    ends = np.array(src_root["meta"]["episode_ends"])
    keep_episodes = min(args.episodes, len(ends))
    cutoff = int(ends[keep_episodes - 1])

    dst_root = zarr.open(args.dst, mode="w")
    meta = dst_root.create_group("meta")
    data = dst_root.create_group("data")

    for key, arr in src_root["meta"].items():
        value = np.array(arr)
        if value.ndim >= 1 and value.shape[0] == len(ends):
            value = value[:keep_episodes]
        meta.array(name=key, data=value, chunks=value.shape if value.ndim > 0 else None)

    for key, arr in src_root["data"].items():
        value = np.array(arr[:cutoff])
        chunks = arr.chunks
        if chunks is not None and len(chunks) > 0:
            chunks = (min(chunks[0], value.shape[0]),) + tuple(chunks[1:])
        data.array(name=key, data=value, chunks=chunks)

    print("created", args.dst)
    print("episodes", keep_episodes, "cutoff", cutoff)
    print("meta keys", list(meta.keys()))
    print("data keys", list(data.keys()))


if __name__ == "__main__":
    main()
