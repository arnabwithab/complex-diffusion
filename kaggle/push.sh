#!/bin/bash
# Push this repo as a Kaggle kernel script. Dry-run: ./kaggle/push.sh --dry-run
set -e
if [ "$1" = "--dry-run" ]; then
  echo "dry-run: files to push:"; ls kaggle/run_kernel.py src/*.py pyproject.toml; exit 0
fi
rm -rf /tmp/opencode/kernel && mkdir -p /tmp/opencode/kernel
cp kaggle/run_kernel.py kaggle/kernel-metadata.json /tmp/opencode/kernel/
cp -r src pyproject.toml /tmp/opencode/kernel/
kaggle kernels push -p /tmp/opencode/kernel
