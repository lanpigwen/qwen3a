#!/usr/bin/env bash
# Minimal launcher for projector-only fine-tuning

CUDA_VISIBLE_DEVICES=0 python finetune_audio.py \
  --whisper_model /root/autodl-tmp/whisper-large-v3-turbo \
  --qwen3_model /root/autodl-tmp/qwen3 \
  --train_json /root/autodl-tmp/qwen_auido_train.json \
  --output_dir ./outputs_projector_only \
  --learning_rate 5e-5 \
  --num_train_epochs 2