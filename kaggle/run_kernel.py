#!/usr/bin/env python3
"""Kaggle bootstrapper: only code_file is deployed, so clone the repo, then run.

Usage: python3 kaggle/run_kernel.py [lr]  (no lr -> pre-flight probes only)
Local: runs in-place from the repo root (no clone).
"""

import os
import subprocess
import sys

REPO = "https://github.com/arnabwithab/complex-diffusion.git"


def run(cmd, gpu=None, cwd=None):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, cwd=cwd, check=True)


def spawn(cmd, gpu, cwd):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("+", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, env=env, cwd=cwd)


def main():
    lr = sys.argv[1] if len(sys.argv) > 1 else "3e-4"  # pre-flight LR pick per spec 5
    root = "." if os.path.exists(os.path.join("src", "train.py")) else "repo"
    if root == "repo" and not os.path.exists(os.path.join("repo", "src", "train.py")):
        run(["git", "clone", "--depth", "1", REPO, "repo"])
    py = sys.executable
    try:
        run([py, "-c", "import transformers, datasets"], cwd=root)
    except subprocess.CalledProcessError:
        run([py, "-m", "pip", "install", "-q", "transformers", "datasets"], cwd=root)
    run([py, "-m", "pytest", "src/smoke_test.py", "-x", "-q"], cwd=root)
    if not os.path.exists(os.path.join(root, "data", "train.pt")):
        run([py, "src/data.py", "--out", "data"], cwd=root)
    if lr is None:
        for mode in ("throughput", "lrs"):
            ps = [spawn([py, "src/train.py", "--variant", v, "--data", "data",
                         "--mode", mode], g, root) for v, g in (("A", 0), ("B", 1))]
            for p in ps:
                assert p.wait() == 0
        print("Pick LR per spec section 5 rule, then rerun with <lr>.")
        return
    ps = [spawn([py, "src/train.py", "--variant", v, "--data", "data",
                 "--lr", lr], g, root) for v, g in (("A", 0), ("B", 1))]
    for p in ps:
        assert p.wait() == 0
    for v in ("A", "B"):
        run([py, "src/probes.py", "--ckpt", f"checkpoints/ckpt_{v}.pt",
             "--variant", v], cwd=root)
        run([py, "src/generate.py", "--ckpt", f"checkpoints/ckpt_{v}.pt",
             "--variant", v], cwd=root)
    print("DONE")


if __name__ == "__main__":
    main()
