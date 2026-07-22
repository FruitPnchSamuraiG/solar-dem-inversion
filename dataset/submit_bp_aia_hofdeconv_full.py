"""
submit_bp_aia_hofdeconv_full.py — generate job list and submit SLURM array for
bp_AIA_hofdeconv_full inversions (LP, AIA-only, Hofmeister deconv, full noise).

Uses the same train/val/test splits as lp_AIA_notrunc_DS.
"""

import os
import argparse
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))

SPLIT_FILES = {
    'train': os.path.join(_HERE, 'train_datetimes.txt'),
    'val':   os.path.join(_HERE, 'val_datetimes.txt'),
    'test':  os.path.join(_HERE, 'test_datetimes.txt'),
}

SRC_DIR    = '/scratch/vp2435/workspace/dem/data/xrtSource'
OUT_DIR    = '/scratch/vp2435/workspace/dem/data/bp_AIA_hofdeconv_full'
JOB_LIST   = os.path.join(_HERE, 'jobs_bp_aia_hofdeconv_full.txt')
SLURM_SCRIPT = os.path.join(_HERE, 'inv_bp_aia_hofdeconv_full.sh')

POINTING    = os.path.join(_HERE, 'aia_pointing_master_2014-2016.ecsv')
COMMON_ARGS = (f'--deconvolve hofmeister --errorfn full --notrunc --extendto8 '
               f'--decimate 2 --zerochill --fitfn lp --parallel 16 '
               f'--pointing_file {POINTING}')
N_NOISE     = 5   # noise indices 0..4
BATCH_SIZE  = 4500


def load_timestamps():
    timestamps = []
    for split, path in SPLIT_FILES.items():
        with open(path) as f:
            for line in f:
                ts = line.strip()
                if ts:
                    timestamps.append(ts)
    return sorted(set(timestamps))


def noise_suffix(noise_idx):
    if noise_idx == 0:   return ''
    if noise_idx == 1:   return '_noise'
    return f'_noise{noise_idx}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry_run', action='store_true', help='generate job list only, do not submit')
    args = parser.parse_args()

    timestamps = load_timestamps()
    print(f'Timestamps: {len(timestamps)}  x  {N_NOISE} noise realizations = {len(timestamps)*N_NOISE} jobs')

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(JOB_LIST), exist_ok=True)
    os.makedirs('logs/inv_hof', exist_ok=True)

    jobs = []
    for ts in timestamps:
        src = os.path.join(SRC_DIR, ts)
        for ni in range(N_NOISE):
            out = os.path.join(OUT_DIR, f'{ts}{noise_suffix(ni)}.npz')
            jobs.append(f'{src} {out} --noisy {ni} {COMMON_ARGS}')

    with open(JOB_LIST, 'w') as f:
        for j in jobs:
            f.write(j + '\n')
    print(f'Wrote {len(jobs)} jobs to {JOB_LIST}')

    if args.dry_run:
        print('--dry_run: not submitting.')
        print(f'To submit: sbatch --array=0-{min(BATCH_SIZE,len(jobs))-1} '
              f'--export=JOB_LIST_FILE={JOB_LIST},BATCH_END={min(BATCH_SIZE,len(jobs))-1},'
              f'NEXT_START={BATCH_SIZE if len(jobs)>BATCH_SIZE else -1},'
              f'TOTAL_JOBS={len(jobs)},BASE_JOB_NAME=inv_hof,SLURM_SCRIPT={SLURM_SCRIPT} {SLURM_SCRIPT}')
        return

    # submit first batch (SLURM max array = 4500)
    total = len(jobs)
    batch_end = min(BATCH_SIZE, total) - 1
    next_start = BATCH_SIZE if total > BATCH_SIZE else -1

    cmd = [
        'sbatch',
        f'--array=0-{batch_end}',
        f'--export=JOB_LIST_FILE={JOB_LIST},BATCH_END={batch_end},'
        f'NEXT_START={next_start},TOTAL_JOBS={total},'
        f'BASE_JOB_NAME=inv_hof,SLURM_SCRIPT={SLURM_SCRIPT}',
        SLURM_SCRIPT,
    ]
    print('Submitting:', ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f'Submitted: {result.stdout.strip()}')
    else:
        print(f'Failed: {result.stderr.strip()}')


if __name__ == '__main__':
    main()
