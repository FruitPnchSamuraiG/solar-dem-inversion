# dem (solar-dem-inversion) on NYU Torch HPC — Handoff

> Handoff doc for another agent/collaborator picking up the **dem** project
> (solar-dem-inversion) on NYU's Torch GPU cluster.
> Owner: Hriday Ranka (NetID `hsr3649`). Written 2026-06-29.
> Describes the cluster, the setup we did, what had to be fixed, and what's left.

---

## 1. The cluster in one paragraph

NYU's HPC GPU cluster is literally named **Torch** (separate from PyTorch). SSH to a **login node**
(`login.torch.hpc.nyu.edu` → `torch-login-*`): editing / git / installs / downloads only — **no GPU,
limited RAM, shared, has internet**. Compute is allocated **per-job by SLURM** onto **compute nodes**
(`ga*`=A100, `gl*`=L40S, `gh*`=H100/H200, `gr*`=RTX Pro 6000). **Compute nodes have NO internet.**

- **Auth:** NetID password + Duo MFA. **SSH keys not supported** for Torch login. NYU network/VPN required.
- **SLURM account (required on every job):** `--account=torch_pr_41_tandon_advanced` (or `torch_pr_41_general`). No `--account` → job errors.
- No `--partition`; use `--gres=gpu:1 --constraint=<a100|l40s|h100|h200>`. Drop the constraint + `--comment="preemption=yes;requeue=true"` to grab any free GPU fast when busy.

---

## 2. Storage

| FS | Path | Quota | Notes |
|----|------|-------|-------|
| Home | `$HOME` (`/home/hsr3649`) | 50 GB / **30k inodes** | code + configs only. NEVER venvs/caches. |
| Scratch | `$SCRATCH` (`/scratch/hsr3649`) | 5 TB / 5M inodes | data, checkpoints, venvs, caches. Purged after 60 days no-access, no backup. |

Shared across projects (in `~/.bashrc`): `UV_CACHE_DIR=$SCRATCH/.uv-cache`, `UV_PYTHON_INSTALL_DIR=$SCRATCH/.uv-python`. Per-project: code dir, scratch subtree, venv, `env.sh`.

---

## 3. The repo

- **Repo:** `git@github.com:FruitPnchSamuraiG/solar-dem-inversion.git` → `~/projects/dem`
- **Toolchain:** **uv WITH `uv.lock`** → install with `uv sync` (reproducible, deterministic).
- **`requires-python = ">=3.13"`** ← important: the cluster's shared managed Python is 3.12.13, which does NOT satisfy this. We installed managed **3.13.14** for this project.
- **Domain stack:** solar physics — sunpy, astropy, aiapy, xrtpy, drms, reproject, sunpy, zarr, dask, scikit-image. Plus torch.
- **torch 2.12.0+cu130** (CUDA 13 build) — pulled automatically by `uv sync`, confirmed working on GPU.

---

## 4. Exact setup we ran (all worked, one-shot)

```bash
PROJ=dem
# 1. clone + CHECK PYTHON REQUIREMENT FIRST
cd $HOME/projects
git clone git@github.com:FruitPnchSamuraiG/solar-dem-inversion.git $PROJ && cd $PROJ
grep -n "requires-python" pyproject.toml          # -> ">=3.13"  (so 3.12.13 won't work!)

# 2. managed python 3.13 (on scratch, parity across nodes) + venv
uv python install 3.13                              # installs 3.13.14 to $SCRATCH/.uv-python
mkdir -p $SCRATCH/$PROJ/{data,checkpoints,output,runs,env}
uv venv --python 3.13.14 $SCRATCH/$PROJ/env/.venv

# 3. env.sh (per-project; source every session + in job.sbatch)
cat > $HOME/projects/$PROJ/env.sh <<'EOF'
#!/usr/bin/env bash
[ -n "${VIRTUAL_ENV:-}" ] && deactivate 2>/dev/null   # avoid double-activation prompt
export PROJ=dem
export PROJ_HOME=$HOME/projects/$PROJ
export PROJ_SCRATCH=$SCRATCH/$PROJ
export UV_PROJECT_ENVIRONMENT=$PROJ_SCRATCH/env/.venv
export VIRTUAL_ENV=$PROJ_SCRATCH/env/.venv
source "$VIRTUAL_ENV/bin/activate"
export HF_HOME=$PROJ_SCRATCH/.hf
cd "$PROJ_HOME"
echo "[$PROJ] venv=$VIRTUAL_ENV  scratch=$PROJ_SCRATCH"
EOF

# 4. install + symlink data/output into the repo (scripts expect ./data, ./output)
source env.sh
uv sync                                            # 112 packages, clean
ln -sfn $SCRATCH/$PROJ/data   ./data
ln -sfn $SCRATCH/$PROJ/output ./output
```

### Verification (the correct two-stage way)
```bash
# LOGIN node — checks the BUILD type (cuda.is_available() is always False on login: no GPU there)
python -c "import torch; print(torch.__version__); print('cuda build:', torch.version.cuda)"
# -> 2.12.0+cu130 / cuda build: 13.0   ✅ CUDA-enabled build

# GPU node — checks ACCESS
srun --account=torch_pr_41_tandon_advanced --gres=gpu:1 --comment="preemption=yes;requeue=true" \
     --cpus-per-task=4 --mem=32GB --time=0:15:00 --pty /bin/bash
cd ~/projects/dem && source env.sh
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# -> True / NVIDIA RTX PRO 6000 Blackwell Server Edition   ✅
```

---

## 5. What had to be fixed / watch out for

1. **`requires-python >=3.13`** — shared 3.12.13 is incompatible. Installed managed **3.13.14** (lives on scratch, visible from all nodes → still parity). LESSON: always `grep requires-python pyproject.toml` before creating the venv.
2. **Verify torch build vs access separately:** on login use `torch.version.cuda` (build); `torch.cuda.is_available()` is always False on login (no GPU) — check access on a GPU node.
3. **Home inode/space (50GB/30k):** venv + caches on scratch via env.sh + bashrc.
4. **Python parity (login 3.12.12 vs GPU 3.12.9 system pythons):** never use the system python; use uv-managed (3.13.14 here) so login/GPU match and uv never rebuilds the venv on node switch.
5. **uv only on login (internet); on compute just `source env.sh` (activates venv).** No `uv sync`/`uv run` on compute nodes (no internet → hangs).
6. **Double `(.venv) (.venv)` prompt:** harmless — srun inherits the active venv, then env.sh re-activates. The `deactivate` guard line in env.sh (above) prevents it.

---

## 6. Status (2026-06-29)

✅ DONE: clone, managed py 3.13.14, `uv sync` (112 pkgs), env.sh, data/output symlinks, torch build (cu130) + GPU access both confirmed.

⏸️ REMAINING (project-specific, not cluster setup):
- **Data acquisition:** `$SCRATCH/dem/data` is ready & symlinked to `./data`, but nothing downloaded yet. Solar data likely comes via sunpy/drms (SDO/AIA, Hinode/XRT) or a provided dataset — TBD by the project's scripts. Do any downloads on the LOGIN node (internet). If large, mind the login-node memory cap (see the HF-xet gotcha in other handoffs) and use tmux.
- **Smoke / training run:** no GPU training run done yet — env is verified but the actual inversion/training entrypoint hasn't been exercised. Identify the main script and do a short run on a GPU node, then write a `job.sbatch`.
- **`job.sbatch`:** not written yet. Copy the template from www-jepa/jepa-physical: `--account=torch_pr_41_tandon_advanced`, `--gres=gpu:1 --constraint=l40s`, `--output=/scratch/hsr3649/dem/runs/%x_%j.out`, `source env.sh` then the run command.

---

## 7. Run cheat sheet

```bash
# build/update env (LOGIN, internet)
cd ~/projects/dem && source env.sh && uv sync

# interactive GPU
srun --account=torch_pr_41_tandon_advanced --gres=gpu:1 --constraint=l40s \
     --cpus-per-task=8 --mem=64GB --time=1:00:00 --pty /bin/bash
cd ~/projects/dem && source env.sh && python <script>

# batch (fire-and-forget)
sbatch ~/projects/dem/job.sbatch ; squeue --me
tail -f /scratch/hsr3649/dem/runs/<job>_<id>.out
```

---

## 8. Open questions / suggestions welcome

- Where does the solar data come from (sunpy Fido query? a provided archive?) and how big — so we place it in `$SCRATCH/dem/data` correctly.
- Main entrypoint / training or inversion script + its expected `./data`, `./output` layout.
- Whether to commit `env.sh` + `job.sbatch` to the repo.
- Does it actually need a GPU for the core DEM inversion, or mostly CPU (sunpy/dask)? Sizing depends on this.

---

*General cluster reference: `~/nyu-torch-hpc-setup.md`. Sibling handoffs: `~/Projects/WebWorldModels/www-jepa-handoff.md`, `~/Projects/DL_Project/jepa-physical-handoff.md`.*
