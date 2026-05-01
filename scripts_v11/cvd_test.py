"""
Diagnose whether CUDA_VISIBLE_DEVICES isolation actually maps a process to
the expected physical GPU. Allocates a tiny tensor, then asks nvidia-smi
which physical GPU has this PID.
"""
import os
import subprocess
import sys
import time

import torch

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
print(f"  env CUDA_VISIBLE_DEVICES = {visible}", flush=True)
print(f"  torch.cuda.device_count() = {torch.cuda.device_count()}", flush=True)

if torch.cuda.device_count() == 0:
    print("  no GPU visible", flush=True)
    sys.exit(0)

x = torch.zeros(1).cuda()
time.sleep(2)

apps_out = subprocess.check_output(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"]
).decode()
pid = os.getpid()
my_uuid = None
for line in apps_out.strip().split("\n"):
    if line and str(pid) in line:
        my_uuid = line.split(",")[0].strip()
        break

if not my_uuid:
    print(f"  PID {pid} not yet visible to nvidia-smi (race?)", flush=True)
    sys.exit(0)

idx_out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"]
).decode()
phys = "?"
for l in idx_out.strip().split("\n"):
    if my_uuid in l:
        phys = l.split(",")[0].strip()
        break

print(f"  -> PROCESS IS ON PHYSICAL GPU {phys}  (uuid={my_uuid[:24]}...)", flush=True)
