#!/bin/bash
# Kaggle 2xT4 session: A/B in parallel, one GPU each, then joint eval.
# Usage: bash kaggle/run_all.sh [lr]  (lr empty -> run probes to select it)
set -e
LR=${1:-}
export HF_HUB_OFFLINE=0

python3 -m pytest src/smoke_test.py -x -q
[ -f data/train.pt ] || python3 src/data.py --out data

if [ -z "$LR" ]; then
  CUDA_VISIBLE_DEVICES=0 python3 src/train.py --variant A --data data --mode throughput &
  CUDA_VISIBLE_DEVICES=1 python3 src/train.py --variant B --data data --mode throughput &
  wait
  # 500-step x 2-LR x both-variants probe, one GPU per variant
  CUDA_VISIBLE_DEVICES=0 python3 src/train.py --variant A --data data --mode lrs &
  CUDA_VISIBLE_DEVICES=1 python3 src/train.py --variant B --data data --mode lrs &
  wait
  echo "Pick LR per spec section 5 rule, then: bash kaggle/run_all.sh <lr>"
  exit 0
fi

CUDA_VISIBLE_DEVICES=0 python3 src/train.py --variant A --data data --lr "$LR" &
CUDA_VISIBLE_DEVICES=1 python3 src/train.py --variant B --data data --lr "$LR" &
wait

for v in A B; do
  python3 src/probes.py --ckpt checkpoints/ckpt_$v.pt --variant $v > results/probes_$v.txt
  python3 src/generate.py --ckpt checkpoints/ckpt_$v.pt --variant $v
done
echo DONE
