#!/usr/bin/env python3
"""Kaggle 2xT4 entrypoint: A/B in parallel, one GPU each, then joint eval.

Usage: python3 kaggle/run_kernel.py [lr]  (no lr -> probes only, then exit)
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = next(
    c for c in (os.path.join(HERE, "..", "src"), os.path.join(HERE, "src"))
    if os.path.isdir(c)
)
os.chdir(os.path.dirname(SRC))  # kaggle runs script.py from elsewhere
sys.path.insert(0, SRC)


def run(cmd, gpu=None):
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def main():
    lr = sys.argv[1] if len(sys.argv) > 1 else None
    run([sys.executable, "-m", "pytest", "src/smoke_test.py", "-x", "-q"])
    if not os.path.exists("data/train.pt"):
        run([sys.executable, "src/data.py", "--out", "data"])
    if lr is None:
        ps = [subprocess.Popen([sys.executable, "src/train.py", "--variant", v,
                                "--data", "data", "--mode", m],
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)})
              for (v, g, m) in (("A", 0, "throughput"), ("B", 1, "throughput"))]
        for p in ps:
            p.wait()
            assert p.returncode == 0
        ps = [subprocess.Popen([sys.executable, "src/train.py", "--variant", v,
                                "--data", "data", "--mode", "lrs"],
                               env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)})
              for v, g in (("A", 0), ("B", 1))]
        for p in ps:
            p.wait()
            assert p.returncode == 0
        print("Pick LR per spec section 5 rule, then rerun with <lr>.")
        return
    ps = [subprocess.Popen([sys.executable, "src/train.py", "--variant", v,
                            "--data", "data", "--lr", lr],
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": str(g)})
          for v, g in (("A", 0), ("B", 1))]
    for p in ps:
        p.wait()
        assert p.returncode == 0
    for v in ("A", "B"):
        run([sys.executable, "src/probes.py", "--ckpt", f"checkpoints/ckpt_{v}.pt",
             "--variant", v])
        run([sys.executable, "src/generate.py", "--ckpt", f"checkpoints/ckpt_{v}.pt",
             "--variant", v])
    print("DONE")


if __name__ == "__main__":
    main()
