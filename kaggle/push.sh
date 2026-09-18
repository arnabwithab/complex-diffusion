#!/bin/bash
# Push this repo as a Kaggle kernel script. Dry-run: ./kaggle/push.sh --dry-run
set -e
if [ "$1" = "--dry-run" ]; then
  echo "dry-run: files to push:"; ls kaggle/run_all.sh src/*.py pyproject.toml; exit 0
fi
mkdir -p /tmp/opencode/kernel && cp kaggle/run_all.sh kaggle/kernel-metadata.json /tmp/opencode/kernel/
cp -r src pyproject.toml /tmp/opencode/kernel/
kaggle kernels push -p /tmp/opencode/kernel
