"""
submit_bp_aia_hofdeconv_full.py — generate job list and submit SLURM array for
AIA-only, Hofmeister-deconvolved DEM label generation.

Solver, output tree, and number of noise realizations are all flags now. Defaults
are clean-only (--n_noise 1) with tolerance relaxation enabled, matching the
2026-07-28 scaling plan: labels are evaluation-only under unsupervised training,
and the stored noise realizations only earn their cost if the uncertainty head
stays in scope.

Uses the same train/val/test splits as lp_AIA_notrunc_DS.

    python submit_bp_aia_hofdeconv_full.py --fitfn lp        --dry_run
    python submit_bp_aia_hofdeconv_full.py --fitfn elasticnet --dry_run
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

DEFAULT_SRC_DIR = '/projects/rps/dff6142/fouheylab/solar_dem/xrtSource'
SLURM_SCRIPT = os.path.join(_HERE, 'inv_bp_aia_hofdeconv_full.sh')

POINTING    = os.path.join(_HERE, 'aia_pointing_master_2014-2016.ecsv')
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
    parser.add_argument('--src_dir', default=DEFAULT_SRC_DIR, help='shared AIA/XRT source tree')
    parser.add_argument('--out_dir', default=None,
                        help='where the .npz land; defaults to $SCRATCH/dem/data/<fitfn>_AIA_hofdeconv_full')
    parser.add_argument('--fitfn', default='lp', choices=['lp', 'elasticnet'],
                        help='which solver generates the labels')
    parser.add_argument('--n_noise', type=int, default=1,
                        help='noise realizations per timestamp; 1 = clean only (default). '
                             'Only raise this if the uncertainty head is in scope.')
    parser.add_argument('--zerochill', action='store_true',
                        help='disable BP tolerance relaxation, leaving infeasible pixels NaN')
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(
        os.environ['SCRATCH'], 'dem', 'data', f'{args.fitfn}_AIA_hofdeconv_full')
    job_list = os.path.join(_HERE, f'jobs_{args.fitfn}_aia_hofdeconv_full.txt')

    common_args = (f'--deconvolve hofmeister --errorfn full --notrunc --extendto8 '
                   f'--decimate 2 --fitfn {args.fitfn} --parallel 16 '
                   f'--pointing_file {POINTING}'
                   + (' --zerochill' if args.zerochill else ''))

    timestamps = load_timestamps()
    print(f'Timestamps: {len(timestamps)}  x  {args.n_noise} noise realizations '
          f'= {len(timestamps)*args.n_noise} jobs  ({args.fitfn})')
    print(f'  src: {args.src_dir}\n  out: {out_dir}')

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs('logs/inv_hof', exist_ok=True)

    jobs = []
    for ts in timestamps:
        src = os.path.join(args.src_dir, ts)
        for ni in range(args.n_noise):
            out = os.path.join(out_dir, f'{ts}{noise_suffix(ni)}.npz')
            jobs.append(f'{src} {out} --noisy {ni} {common_args}')

    with open(job_list, 'w') as f:
        for j in jobs:
            f.write(j + '\n')
    print(f'Wrote {len(jobs)} jobs to {job_list}')

    if args.dry_run:
        print('--dry_run: not submitting.')
        print(f'To submit: sbatch --array=0-{min(BATCH_SIZE,len(jobs))-1} '
              f'--export=JOB_LIST_FILE={job_list},BATCH_END={min(BATCH_SIZE,len(jobs))-1},'
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
        f'--export=JOB_LIST_FILE={job_list},BATCH_END={batch_end},'
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
