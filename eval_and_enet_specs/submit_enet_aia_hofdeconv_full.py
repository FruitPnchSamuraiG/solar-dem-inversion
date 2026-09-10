"""
submit_enet_aia_hofdeconv_full.py — generate job list and submit SLURM array for
enet_AIA_hofdeconv_full inversions (ElasticNet, AIA-only, Hofmeister deconv, full noise).
"""

import os, argparse, subprocess

SPLIT_FILES = {
    'train': '/scratch/vp2435/workspace/dem/data/lp_AIA_notrunc_DS/train_datetimes.txt',
    'val':   '/scratch/vp2435/workspace/dem/data/lp_AIA_notrunc_DS/val_datetimes.txt',
    'test':  '/scratch/vp2435/workspace/dem/data/lp_AIA_notrunc_DS/test_datetimes.txt',
}

SRC_DIR      = '/scratch/vp2435/workspace/dem/data/xrtSource'
OUT_DIR      = '/scratch/vp2435/workspace/dem/data/enet_AIA_hofdeconv_full'
JOB_LIST     = '/scratch/vp2435/workspace/dem/demdemo/slurm/jobs_enet_aia_hofdeconv_full.txt'
SLURM_SCRIPT = '/scratch/vp2435/workspace/dem/demdemo/slurm/inv_bp_aia_hofdeconv_full.sh'
POINTING     = '/scratch/vp2435/workspace/dem/demdemo/optimScripts/aia_pointing_master_2014-2016.ecsv'

COMMON_ARGS  = (f'--deconvolve hofmeister --errorfn full --notrunc --extendto8 '
                f'--decimate 2 --zerochill --fitfn elasticnet '
                f'--fitlinearalpha 0.001 --fitlinearl1ratio 0.5 --parallel 16 '
                f'--pointing_file {POINTING}')
N_NOISE      = 5
BATCH_SIZE   = 1000


def load_timestamps():
    timestamps = []
    for path in SPLIT_FILES.values():
        with open(path) as f:
            for line in f:
                ts = line.strip()
                if ts:
                    timestamps.append(ts)
    return sorted(set(timestamps))


def noise_suffix(i):
    return '' if i == 0 else ('_noise' if i == 1 else f'_noise{i}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    timestamps = load_timestamps()
    print(f'Timestamps: {len(timestamps)} x {N_NOISE} = {len(timestamps)*N_NOISE} jobs')

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(JOB_LIST), exist_ok=True)
    os.makedirs('logs/inv_hof', exist_ok=True)

    jobs = []
    for ts in timestamps:
        for ni in range(N_NOISE):
            out = os.path.join(OUT_DIR, f'{ts}{noise_suffix(ni)}.npz')
            jobs.append(f'{os.path.join(SRC_DIR, ts)} {out} --noisy {ni} {COMMON_ARGS}')

    with open(JOB_LIST, 'w') as f:
        f.write('\n'.join(jobs) + '\n')
    print(f'Wrote {len(jobs)} jobs to {JOB_LIST}')

    if args.dry_run:
        print('--dry_run: not submitting.')
        return

    total = len(jobs)
    batch_end = min(BATCH_SIZE, total) - 1
    next_start = BATCH_SIZE if total > BATCH_SIZE else -1

    cmd = ['sbatch', f'--array=0-{batch_end}',
           f'--export=JOB_LIST_FILE={JOB_LIST},BATCH_END={batch_end},'
           f'NEXT_START={next_start},BATCH_SIZE={BATCH_SIZE},TOTAL_JOBS={total},'
           f'BASE_JOB_NAME=inv_enet,SLURM_SCRIPT={SLURM_SCRIPT}',
           SLURM_SCRIPT]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip() if result.returncode == 0 else f'Failed: {result.stderr.strip()}')


if __name__ == '__main__':
    main()
