# Narval Setup Guide

Environment setup for running Mothra experiments on the Narval cluster (Digital Research Alliance of Canada).

## Cluster Overview

Narval uses the **Slurm** scheduler. Jobs cannot run directly on the login node — they must be submitted via `sbatch` (batch) or `salloc` (interactive).

**GPU nodes:** AMD Milan CPUs, 4× A100-40GB per node, connected via NVLink.
Recommended: ≤12 CPU cores and ~46 GB RAM per GPU requested.

---

## Python Environment

**Location:** `~/mothra_env`  
**Python:** 3.11.5  
**Key packages:** torch 2.11.0, ultralytics 8.4.24, opencv 4.9.0, numpy 1.26.4 (pinned <2.0 for opencv compatibility)

### Activating the environment

Always load modules **before** activating the virtualenv (especially opencv):

```bash
module load python/3.11.5 gcc opencv/4.9.0
source ~/mothra_env/bin/activate
```

### Rebuilding the environment from scratch

```bash
module load python/3.11.5 gcc opencv/4.9.0
virtualenv --no-download ~/mothra_env
source ~/mothra_env/bin/activate
pip install --no-index torch torchvision
pip install --no-index ultralytics tqdm tensorboard PyYAML
pip install --no-index "numpy==1.26.4"   # pin to 1.x for opencv compatibility
```

All packages install from Alliance's CVMFS wheelhouse (`--no-index`), no internet required.

---

## Running Jobs

### Submitting a batch job

```bash
sbatch my_job.sh
```

### Checking job status

```bash
sq                        # list your jobs (R=running, PD=pending)
scancel <jobid>           # cancel a job
```

Output goes to `slurm-<jobid>.out` in the submission directory by default.

### Example GPU job script

```bash
#!/bin/bash
#SBATCH --account=def-ichiro
#SBATCH --gpus-per-node=a100:1    # 1× A100-40GB
#SBATCH --cpus-per-task=12        # max recommended per GPU on Narval
#SBATCH --mem=46000M
#SBATCH --time=0-12:00            # DD-HH:MM
#SBATCH --output=%j.out

module load python/3.11.5 gcc opencv/4.9.0
source ~/mothra_env/bin/activate

cd ~/projects/def-ichiro/wyma/mothra
python scripts_v11/train_mothra.py --config configs/mothra_base11.yaml
```

### Requesting less GPU (MIG slices — shorter queue wait)

If the full A100 is not needed, request a MIG instance instead:

| Slurm specifier | VRAM | Use when |
|----------------|------|----------|
| `a100_1g.5gb`  | 5 GB | small debug runs |
| `a100_2g.10gb` | 10 GB | medium models |
| `a100_3g.20gb` | 20 GB | most training runs |
| `a100:1`       | 40 GB | full model / large batch |

```bash
#SBATCH --gpus-per-node=a100_3g.20gb:1
```

---

## Storage

| Path | Quota | Notes |
|------|-------|-------|
| `~/` (`/home/wyma`) | 50 GB | virtualenv lives here |
| `/scratch/wyma` | 20 TB | large temp outputs; purged after 60 days |
| `/project/def-ichiro` | 1 TB (shared) | ~930 GB used — nearly full, use carefully |

For large training outputs, write to `/scratch/wyma/` and copy important results back to `/project`.
