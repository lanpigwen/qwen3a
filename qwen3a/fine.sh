#!/usr/bin/env bash
###############################################################################
# Fine-tune launcher
# 功能:
#  1. 自动激活 conda 环境: qwen
#  2. 设置显存分配策略 (expandable_segments)
#  3. 自动检测多 GPU (>=2) 时使用 accelerate launch (Plan B)
#  4. 单卡则直接 python 运行 (Plan A)
#  5. 允许用户在执行时追加自定义参数: sh fine.sh --no_mask --num_train_epochs 1
#
# 你可以通过环境变量覆盖默认参数:
#   FT_LR, FT_EPOCHS, FT_BATCH, FT_ACC_STEPS, FT_DEVICES (如 "0,1")
# 例如:
#   FT_LR=1e-4 FT_EPOCHS=1 FT_DEVICES=0,1 sh fine.sh --no_mask
###############################################################################
# 兼容 /bin/sh 执行: 如果用户误用 `sh fine.sh` (dash 不支持 pipefail) 则降级
if [ -n "${BASH_VERSION:-}" ]; then
  set -euo pipefail
else
  echo "[fine.sh] 警告: 建议使用 bash 运行 (bash fine.sh 或 ./fine.sh)，当前不是 bash，禁用 pipefail" >&2
  set -eu
fi

# --- 1. 激活 conda 环境 ------------------------------------------------------
if [ "${CONDA_DEFAULT_ENV:-}" != "qwen" ]; then
  if command -v conda >/dev/null 2>&1; then
    # 初始化 conda shell 钩子 (避免 'conda: command not found')
    eval "$(conda shell.bash hook)"
    conda activate qwen || { echo "无法激活 conda 环境 qwen" >&2; exit 1; }
  else
    echo "未检测到 conda，请先安装/加载含有 qwen 环境的 conda" >&2
    exit 1
  fi
fi

# --- 2. 默认超参 (可被环境变量覆盖) -----------------------------------------
LR=${FT_LR:-2e-4}
EPOCHS=${FT_EPOCHS:-3}
BATCH=${FT_BATCH:-1}
ACC_STEPS=${FT_ACC_STEPS:-4}
DEVICES_ENV=${FT_DEVICES:-}

# --- 3. 显存分配策略 --------------------------------------------------------
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 若用户显式指定了设备则尊重; 否则保持不设，交给 accelerate / torch 自动
if [ -n "$DEVICES_ENV" ]; then
  export CUDA_VISIBLE_DEVICES="$DEVICES_ENV"
fi

# --- 4. 计算 GPU 数量 -------------------------------------------------------
gpu_count() {
  # 优先使用 nvidia-smi, 若不可用尝试 python (已在环境中)
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | awk '{print $1}'
  else
    python - <<'PY'
import torch, os
print(torch.cuda.device_count())
PY
  fi
}

GPUS=$(gpu_count || echo 0)

# --- 5. 基础固定参数 --------------------------------------------------------
BASE_ARGS=(
  --data_mode conversation
  --freeze_whisper
  --use_lora
  --train_qwen3_layers 0
  --learning_rate ${LR}
  --gradient_accumulation_steps ${ACC_STEPS}
  --num_train_epochs ${EPOCHS}
  --per_device_train_batch_size ${BATCH}
)

# 允许用户在命令行追加更多参数 (优先级更高)
EXTRA_ARGS=("$@")

echo "[fine.sh] 使用 GPU 数量: ${GPUS}"
echo "[fine.sh] 基础参数: ${BASE_ARGS[*]}"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  echo "[fine.sh] 追加参数: ${EXTRA_ARGS[*]}"
fi

# --- 6. 运行模式: 多卡 -> accelerate, 单卡 -> python -------------------------
if [ "$GPUS" -ge 2 ]; then
  # 如果用户选择了 device_map auto + 量化 (4bit/8bit)，需要单进程才能训练
  NEED_SINGLE=0
  for a in "${EXTRA_ARGS[@]}"; do
    case "$a" in
      --load_in_4bit|--load_in_8bit)
        NEED_SINGLE=1 ;;
      --device_map)
        NEED_SINGLE=1 ;;
      --manual_loss)
        NEED_SINGLE=1 ;;
    esac
  done
  echo "[fine.sh] 检测到多 GPU，准备使用 accelerate。NEED_SINGLE=${NEED_SINGLE}"
  if ! python -c "import os;print(os.path.exists(os.path.expanduser('~/.cache/huggingface/accelerate/default_config.yaml')))" 2>/dev/null | grep -q True; then
    echo "[fine.sh] 未发现 accelerate 默认配置，使用临时参数" >&2
  fi
  # 自动推断 mixed precision 模式：如用户传 --fp16 / --bf16 才启用对应模式，否则关闭 (no)
  MP_MODE=no
  for a in "${EXTRA_ARGS[@]}"; do
    case "$a" in
      --fp16) MP_MODE=fp16 ;;
      --bf16) MP_MODE=bf16 ;;
    esac
  done
  echo "[fine.sh] mixed_precision=${MP_MODE}"
  if [ "$NEED_SINGLE" -eq 1 ]; then
    echo "[fine.sh] 启用 device_map/量化，强制单进程训练以兼容 device_map=auto"
    # 方案1: 进一步限制只用第一块 GPU，避免 DataParallel 在模型内部复制导致跨设备 hidden_states
    if [ -z "${DEVICES_ENV:-}" ]; then
      ORIG_DEV_LIST=$(python - <<'PY'
import torch
print(','.join(str(i) for i in range(torch.cuda.device_count())))
PY
)
      export CUDA_VISIBLE_DEVICES=0
      echo "[fine.sh] 已自动将 CUDA_VISIBLE_DEVICES 从 [${ORIG_DEV_LIST}] 缩减为 [0] (量化+device_map=auto)"
    else
      # 如果用户自己传了多卡但又量化+device_map，仍然只取第一个
      FIRST=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d',' -f1)
      if echo "$CUDA_VISIBLE_DEVICES" | grep -q ','; then
        echo "[fine.sh] 用户指定多卡 (${CUDA_VISIBLE_DEVICES}) 与量化 device_map=auto 冲突，自动改为单卡 ${FIRST}" >&2
        export CUDA_VISIBLE_DEVICES=${FIRST}
      fi
    fi
    accelerate launch \
      --mixed_precision ${MP_MODE} \
      --num_processes 1 \
      --main_process_port 29500 \
      finetune_audio.py "${BASE_ARGS[@]}" "${EXTRA_ARGS[@]}"
  else
    accelerate launch \
      --mixed_precision ${MP_MODE} \
      --num_processes ${GPUS} \
      --main_process_port 29500 \
      finetune_audio.py "${BASE_ARGS[@]}" "${EXTRA_ARGS[@]}"
  fi
else
  echo "[fine.sh] 单 GPU / CPU 模式，直接运行 python (Plan A)"
  python finetune_audio.py "${BASE_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi

echo "[fine.sh] 任务完成"