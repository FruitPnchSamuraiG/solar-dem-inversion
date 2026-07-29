# Meeting brief — 2026-07-28 — scaling pipeline handover

Prepared after validating Samuel's `dataset/` pipeline end-to-end on our own Torch
allocation (user `hsr3649`, account `torch_pr_41_tandon_advanced`).

---

## 1. What I ran

Samuel's `dataset/fullBP.py` on a real timestamp from the shared source data, both a
64×64 crop and a full disk, at `--decimate 2`, AIA-only, **without** Hofmeister
deconvolution (PSF files are not in the repo — see blockers).

```
python dataset/fullBP.py \
    /projects/rps/dff6142/fouheylab/solar_dem/xrtSource/20140101_060356 \
    $SCRATCH/dem/data/bp_smoke_test/20140101_060356_fulldisk.npz \
    --errorfn full --notrunc --extendto8 \
    --decimate 2 --zerochill --fitfn lp --parallel 16 \
    --pointing_file dataset/aia_pointing_master_2014-2016.ecsv
```

**It works.** Output verified physically sensible: peak logT 6.3–6.5 across sampled
pixels (textbook quiet-sun / AR corona, ~2–3 MK).

---

## 2. Measured numbers (full disk, 16 cores, no deconvolution)

| Quantity | Measured |
|---|---|
| Wall time per timestamp | **4 min 2 s** (BP itself 3 min 10 s) |
| Throughput | 22,127 px/s at 16 cores; 1,383 px/s per core |
| Output shape | `(26, 2048, 2048)` — 26 DEM bins per pixel |
| File size (clean, `--noisy 0`) | **715 MB** (DEM + AIA + AIAErrors) |
| **NaN (infeasible) pixels** | **6.6%** (278,183 / 4,194,304) |

NaN on the small on-disk crop was only 2.4%.

### 2b. With Hofmeister deconvolution (same timestamp, 16 cores, no GPU)

PSFs obtained from the Harvard Dataverse (doi:10.7910/DVN/DYT4ZL, 3.2 GB zip, seven
512 MB 8192x8192 PSFs — 304A included but unused), staged on scratch and symlinked into
`dataset/hofmeister_psf/`.

| Quantity | No deconvolution | Hofmeister |
|---|---|---|
| Wall time per timestamp | 4 min 2 s | **7 min 57 s** |
| BP portion | 3 min 10 s | 3 min 42 s |
| **NaN (infeasible) pixels** | 6.6% | **10.2%** |

**Deconvolution runs on CPU, not GPU.** `fullBP.py:776` hardcodes `use_gpu=False` and the
sbatch script requests no GPU. Cost is ~4 min/job on 16 cores, comfortably inside the
2 h limit. (An earlier version of this brief claimed it needed a GPU; it does not. The
error-table caching fix in 4.3 still stands — compute nodes have no internet regardless.)

**Deconvolution verified to be doing the right thing.** Median relative change vs the
undeconvolved AIA: 94A 32%, 335A 35%, 171A 12%, 193A 8%, 211A 8%. The largest corrections
land on the two faintest channels, which is exactly where scattered light from the bright
channels dominates — the correct physical signature, and a much larger effect than
`aiapy`'s standard PSF deconvolution.

### Extrapolation to the full run

On the reduced evaluation-only job list (see `scaling_plan_20260728.md` section 2b —
roughly 400 jobs rather than 6,115), at ~8 min/job with deconvolution that is about
**53 job-hours** and well under 500 GB. Cost is no longer a constraint. For reference,
the original full 6,115-job plan would have been ~815 job-hours and ~2.1 TB, against a
5 TB quota on a filesystem currently 97% full with a 60-day no-access purge.

---

## 3. Bug found and fixed (already pushed)

`dataset/fullBP.py` line 5 imported `aiapy.calibrate.util`, which does not exist in
aiapy ≥ 0.12 (it is `aiapy.calibrate.utils`). Fixed and pushed as `eb9a8d6`.

---

## 4. Blockers / decisions needed from Samuel

### 4.1 Hofmeister PSF files — RESOLVED
Downloaded from the Dataverse link (doi:10.7910/DVN/DYT4ZL) to
`$SCRATCH/dem/data/hofmeister_psf/` and symlinked into `dataset/hofmeister_psf/`, which is
where `fullBP.py:768` looks. Filenames matched the expected pattern exactly. Symlinks are
gitignored (they point at personal scratch). `--deconvolve hofmeister` now runs
end to end — see section 2b.

### 4.2 NaN policy for infeasible pixels — 6.6% of every image
`fullBP.py:478` initialises the DEM cube to NaN; `--zerochill` disables the tolerance
relaxation, so any pixel whose LP is infeasible at the tight tolerance **stays NaN in
the saved output**. `nnInterpNaN` exists but is only applied on the *visualisation*
path, never to the saved `.npz`.

`stage_hofdeconv_full.py` has **no NaN handling at all** (grepped — nothing). So 6.6%
of every training patch will carry NaN labels straight into the Zarr, which silently
destroys training (NaN loss → NaN gradients).

**With deconvolution this rises to 10.2%**, and the lost pixels are *not* harmless empty
sky — they concentrate on-disk:

| | NaN rate |
|---|---|
| On-disk | **13.4%** |
| Off-limb | 6.3% |

**Mechanism (measured).** NaN rate binned by 171A brightness decile:

| decile | 3 | 5 | 7 | 9 | 10 |
|---|---|---|---|---|---|
| median DN | 10.7 | 69.5 | 140.6 | 263.6 | 487.1 |
| NaN rate | 2.1% | 11.0% | 15.3% | **17.7%** | 10.8% |

An 8x rise with brightness. This is consistent with photon noise (sigma ~ sqrt(counts)),
which makes the *relative* tolerance band `+/- t.sigma/o` shrink as pixels brighten. Once
that band is narrower than the mismatch between the response matrix R and real coronal
plasma, no non-negative DEM fits and the LP is infeasible. Faint pixels have wide relative
bands and are easy to satisfy. Deconvolution raises contrast in bright structure, which
tightens things further — hence 6.6% -> 10.2%.

Two candidate causes were tested and **ruled out**: the deconvolution's positivity clamp
(66% of pixels have a zeroed channel, but `P(NaN | zeroed)` is 10.5% vs 9.6% otherwise —
no relationship), and low-signal off-limb regions (the rate is *higher* on-disk).
Unexplained: the turnover in the brightest decile (17.7% -> 10.8%).

**So infeasibility is model mismatch at high SNR, not bad pixels.** BP's escalating
tolerance schedule (1.4 -> 2.0 -> 2.8 -> 5.0) exists precisely to handle this, and
`--zerochill` disables it. **Recommendation: drop `--zerochill`**, or raise the tolerance
floor, and re-measure. He chose it deliberately so there will be a reason, but it is worth
confirming he measured the rate *after* deconvolution.

**If NaNs remain, mask rather than interpolate.** Carry a validity mask into the Zarr and
exclude those pixels from the loss. Interpolation invents labels BP never produced, and
the NN would learn to imitate `nnInterpNaN` rather than the solver.

### 4.3 Error table is fetched over the network on every job
`fullBP.py:790` calls `get_error_table(source="SSW")`, which downloads. The pointing
table and correction table are both already cached to local files (`--pointing_file`,
`--corr_table`) precisely to avoid this; the error table was missed.

Two consequences: it will **fail outright on GPU compute nodes** (no internet, and
Hofmeister deconvolution requires a GPU), and 6,115 jobs hammering the SSW server will
likely get rate-limited. Should be cached the same way as the other two.

### 4.4 Storage and ownership
~2.1 TB of `.npz` plus the Zarr. Whose scratch does it land on? Scratch is 97% full
cluster-wide and purges after 60 days without access — is `/projects/rps/.../solar_dem/`
the right home for something we want to keep?

### 4.5 Path fix in the submit script
`submit_bp_aia_hofdeconv_full.py` has `SRC_DIR = /scratch/vp2435/workspace/dem/data/xrtSource`.
The real shared location is `/projects/rps/dff6142/fouheylab/solar_dem/xrtSource`.

### 4.6 Are the existing inversions reusable?
`/projects/rps/dff6142/fouheylab/solar_dem/` already contains `xrtData_lp_full`,
`xrtData_lp_AIA_noisy`, `lp_AIA_notrunc_noisy_full`. Does the Hofmeister re-run
supersede all of these, or is some of it still usable?

### 4.7 Who runs the 6,115 jobs
We are all on the same allocation. Do we run them, does he, or do we split by split
(train / val / test)?

### 4.8 ENet labels too?
`stage_hofdeconv_full.py` already has config entries for `enet_AIA_hofdeconv_full` and
`enet_AIAXRT_hofdeconv_full`. Is ENet part of this generation round, or BP only for now?

---

## 5. Modelling questions this raises

### 5.1 The 26-bin grid — 8 bins are structurally zero for AIA-only
`--extendto8` pads bins for logT 7.3–8.0, but AIA has no response there, so for
AIA-only runs those 8 bins are **exactly zero by construction** (`fullBP.py:1013–1019`
literally `np.vstack` a zero block). Our current models output 18 bins.

**Question:** for the AIA-only models, do we train an 18-bin head and only go to 26 when
XRT is actually in the input? Training a 26-bin head where 8 outputs are always exactly
zero wastes capacity and makes the sparsity metrics misleading (the Hoyer ratio changes
denominator).

### 5.2 Input/output resolution mismatch
The saved AIA is at **full 4096×4096** while the DEM is at **2048×2048** (decimated).
The staging script comments that the dataloader "upsamples on the fly". Our patch CNN
maps a 9×9 AIA neighbourhood → one DEM pixel, so we need the exact alignment convention
(does one DEM pixel correspond to a 2×2 AIA block, and which corner is the anchor?)
before writing the Zarr dataloader.

### 5.3 Noise realisations vs our uncertainty plan
5 realisations per timestamp (index 0 clean, 1–4 noisy, `--noisescale 0.5` i.e. half
noise). That is the distribution the uncertainty head is meant to reproduce. Worth
confirming 4 draws is enough to fit a per-bin distribution, and why half noise rather
than full.

---

## 6. Where the science stands (our side)

Both questions from the 2026-07-10 meeting are answered — see `CLAUDE.md` and
`results/findings_log.docx`:

- **Bimodality is not an urgent failure mode.** Rare (5–9% of pixels), mostly noise
  degeneracy (only 6/120 studied pixels stable under perturbation), and those cluster in
  flare timestamps where multi-thermal plasma is physically expected. At those 6 pixels
  the CNN's unimodal answer has *lower* barrier loss than BP's bimodal one in 4/6 cases.
  Distribution head → low priority, flare-targeted.
- **Leave-one-timestamp-out: overfitting not confirmed**, sparsity gap within ±0.2 for
  3/4 folds. The weak fold is the held-out X1.6 flare (cnn +0.47, mlp6 +0.27) — only one
  other flare image in training. **The CNN's in-sample edge over mlp6 disappears on
  unseen images** (mlp6 wins 2 folds incl. the flare, ties 1, cnn wins 1). Caveat: 4
  folds is a small sample.

Both point the same direction: **scale the data, track cnn vs mlp6 as it grows**. This
pipeline is exactly that. Flare timestamps specifically are the gap — worth checking
whether the 2014–2016 window is flare-rich enough, since that was our weakest fold.
