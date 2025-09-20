#!/usr/bin/env bash
# 极简版训练脚本：不做 GPU/多进程智能分支，只做最小参数透传
# 用法：bash simple_fine.sh --train_json data.json --num_train_epochs 0.1 --no_mask
set -euo pipefail

# 若需要 conda 激活请自行取消注释
# if [ "${CONDA_DEFAULT_ENV:-}" != "qwen" ]; then
#   eval "$(conda shell.bash hook)" && conda activate qwen
# fi

BASE_ARGS=(
  --data_mode conversation
  --freeze_whisper
  --use_lora
  --train_qwen3_layers 0
)

python finetune_audio.py "${BASE_ARGS[@]}" "$@"
