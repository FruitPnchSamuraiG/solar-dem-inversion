"""
Stage bp/enet hofdeconv_full inversions to zarr format for training.

Uses same train/val/test timestamps as lp_AIA_notrunc_DS.
Noisy realizations supply DEM; AIA is always read from the clean (noisy=0) file.

Usage:
    launchfast python3 scripts/stage_hofdeconv_full.py --config bp_AIA_hofdeconv_full
    launchfast python3 scripts/stage_hofdeconv_full.py --config bp_AIAXRT_hofdeconv_full
    launchfast python3 scripts/stage_hofdeconv_full.py --config enet_AIA_hofdeconv_full
    launchfast python3 scripts/stage_hofdeconv_full.py --config enet_AIAXRT_hofdeconv_full
"""

import os
import gc
import argparse
import multiprocessing
import random
import numpy as np
import zarr
from numcodecs import Blosc

_HERE = os.path.dirname(os.path.abspath(__file__))

BLOCK_SIZE   = 256
H = W        = 4096
PER_IMAGE    = 64
N_AIA        = 6
N_BINS       = 26
N_WORKERS    = 64

REF_DS       = _HERE
DATA_ROOT    = '/scratch/vp2435/workspace/dem/data'

EXPECTED_TRAIN = 917
EXPECTED_VAL   = 153
EXPECTED_TEST  = 153


def read_timestamps(path):
    with open(path) as f:
        return [l.strip() for l in f if l.strip()]


def extract_timestamp(fn):
    base = os.path.basename(fn).replace('.npz', '')
    if '_noise' in base:
        return base.split('_noise')[0]
    return base


def filter_npzs(all_npzs, timestamps):
    ts_set = set(timestamps)
    return [fn for fn in all_npzs if extract_timestamp(fn) in ts_set]


def handle(t):
    X, Y, i, fn, numBlocksPerH, numBlocksPerW, numBlocksPerImage, blockSize, perImage, src = t
    compressor = Blosc(cname='zstd', clevel=4, shuffle=2)
    random.seed(i)

    print(f"staging {i}: {os.path.basename(fn)}", flush=True)
    timestamp = extract_timestamp(fn)

    try:
        with np.load(fn, mmap_mode='r') as d:
            if 'DEMCube' not in d:
                print(f"  warning: no DEMCube in {fn}")
                return
            raw = d['DEMCube'].item() if d['DEMCube'].ndim == 0 else d['DEMCube'].copy()
            shape = tuple(d['DEMCubeShape'])
        DEMData = np.frombuffer(compressor.decode(raw), dtype=np.float32).reshape(shape)
        del raw
    except Exception as e:
        print(f"  error loading DEM: {e}")
        return

    aia_file = os.path.join(src, f"{timestamp}.npz")
    if not os.path.exists(aia_file):
        print(f"  warning: AIA file not found: {aia_file}")
        return
    try:
        with np.load(aia_file, mmap_mode='r') as d:
            if 'AIACube' not in d:
                print(f"  warning: no AIACube in {aia_file}")
                return
            raw = d['AIACube'].item() if d['AIACube'].ndim == 0 else d['AIACube'].copy()
            shape = tuple(d['AIACubeShape'])
        AIA = np.frombuffer(compressor.decode(raw), dtype=np.float32).reshape(shape)[:N_AIA]
        del raw
    except Exception as e:
        print(f"  error loading AIA: {e}")
        return

    # store Y at native DEM resolution (no NaN inflation); dataloader upsamples on the fly
    dec = AIA.shape[1] // DEMData.shape[1]
    dem_block = blockSize // dec

    blocks = [(bi * blockSize, bj * blockSize)
              for bi in range(numBlocksPerH) for bj in range(numBlocksPerW)]
    if perImage != -1:
        random.shuffle(blocks)
        blocks = blocks[:perImage]

    for c, (sy, sx) in enumerate(blocks):
        ind = i * numBlocksPerImage + c
        X[:, :, :, ind] = AIA[:, sy:sy+blockSize, sx:sx+blockSize]
        Y[:, :, :, ind] = DEMData[:, sy//dec:sy//dec+dem_block, sx//dec:sx//dec+dem_block]

    del AIA, DEMData, blocks
    gc.collect()


def _peek_dec(src, npzs):
    """Read one file to determine AIA/DEM spatial decimation factor."""
    with np.load(os.path.join(src, npzs[0]), mmap_mode='r') as d:
        return int(d['AIACubeShape'][1]) // int(d['DEMCubeShape'][1])


def stage(src, npzs, target, phase, block_size, per_image):
    n_h = H // block_size
    n_w = W // block_size
    n_per = per_image if per_image != -1 else n_h * n_w
    n_total = n_per * len(npzs)

    dec       = _peek_dec(src, npzs)
    dem_block = block_size // dec
    print(f"  dec={dec}, AIA block={block_size}x{block_size}, DEM block={dem_block}x{dem_block}")

    tobytes    = zarr.codecs.BytesCodec()
    compressor = zarr.codecs.BloscCodec(cname='zstd', clevel=4,
                                        shuffle=zarr.codecs.BloscShuffle.bitshuffle)

    X = zarr.open(os.path.join(target, f'{phase}_x.zarr'), mode='w',
                  shape=(N_AIA,  block_size, block_size, n_total),
                  chunks=(N_AIA, block_size, block_size, 1),
                  dtype='<f4', codecs=[tobytes, compressor])
    Y = zarr.open(os.path.join(target, f'{phase}_y.zarr'), mode='w',
                  shape=(N_BINS, dem_block, dem_block, n_total),
                  chunks=(N_BINS, dem_block, dem_block, 1),
                  dtype='<f4', codecs=[tobytes, compressor])

    jobs = [(X, Y, i, os.path.join(src, fn), n_h, n_w, n_per, block_size, per_image, src)
            for i, fn in enumerate(npzs)]

    pool = multiprocessing.Pool(N_WORKERS)
    pool.map(handle, jobs)
    pool.close()
    pool.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True,
                        help='e.g. bp_AIA_hofdeconv_full or enet_AIAXRT_hofdeconv_full')
    args = parser.parse_args()

    src    = os.path.join(DATA_ROOT, args.config)
    target = os.path.join(DATA_ROOT, args.config + '_DS')

    train_ts = read_timestamps(os.path.join(REF_DS, 'train_datetimes.txt'))
    val_ts   = read_timestamps(os.path.join(REF_DS, 'val_datetimes.txt'))
    test_ts  = read_timestamps(os.path.join(REF_DS, 'test_datetimes.txt'))

    print(f"config:  {args.config}")
    print(f"src:     {src}")
    print(f"target:  {target}")
    print(f"train timestamps: {len(train_ts)}, val: {len(val_ts)}, test: {len(test_ts)}")

    all_npzs = sorted(fn for fn in os.listdir(src) if fn.endswith('.npz'))
    print(f"found {len(all_npzs)} npz files in src")

    train_npzs = filter_npzs(all_npzs, train_ts)
    val_npzs   = filter_npzs(all_npzs, val_ts)
    test_npzs  = filter_npzs(all_npzs, test_ts)

    print(f"train: {len(train_npzs)}, val: {len(val_npzs)}, test: {len(test_npzs)}")

    assert len(train_npzs) == EXPECTED_TRAIN, f"expected {EXPECTED_TRAIN} train, got {len(train_npzs)}"
    assert len(val_npzs)   == EXPECTED_VAL,   f"expected {EXPECTED_VAL} val, got {len(val_npzs)}"
    assert len(test_npzs)  == EXPECTED_TEST,  f"expected {EXPECTED_TEST} test, got {len(test_npzs)}"

    os.makedirs(target, exist_ok=True)

    for phase, npzs, timestamps in [
        ('train', train_npzs, train_ts),
        ('val',   val_npzs,   val_ts),
        ('test',  test_npzs,  test_ts),
    ]:
        print(f"\n=== staging {phase} ({len(npzs)} files) ===")
        stage(src, npzs, target, phase, BLOCK_SIZE, PER_IMAGE)
        with open(os.path.join(target, f'{phase}_datetimes.txt'), 'w') as f:
            f.write('\n'.join(timestamps) + '\n')
        print(f"done {phase}")

    print(f"\nCompleted staging {args.config} -> {target}")


if __name__ == '__main__':
    main()
