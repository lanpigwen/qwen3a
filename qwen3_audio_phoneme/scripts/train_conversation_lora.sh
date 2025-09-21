#!/usr/bin/env bash
set -e
RAW_DATA=qwen3_audio_phoneme/data/raw_conversations.json
EXT_TOKENIZER=qwen3_audio_phoneme/output/extended_tokenizer
BASE_MODEL=Qwen/Qwen3-7B-Instruct
WHISPER_MODEL=openai/whisper-large-v3
OUT_DIR=qwen3_audio_phoneme/output/qwen3_audio_phoneme_lora
AUDIO_TOKEN="<|audio|>"

python qwen3_audio_phoneme/scripts/build_phoneme_vocab_and_extend.py \
  --data $RAW_DATA \
  --base_tokenizer $BASE_MODEL \
  --save_dir $EXT_TOKENIZER

swift run \
  --custom_modeling_file qwen3_audio_phoneme/qwen3_audio/modeling_qwen3_audio.py \
  --custom_register_file qwen3_audio_phoneme/qwen3_audio/register_model.py \
  --train_dataset $RAW_DATA \
  --val_dataset $RAW_DATA \
  --output_dir $OUT_DIR \
  --model_id_or_path $BASE_MODEL \
  --whisper_model_id $WHISPER_MODEL \
  --extended_tokenizer_dir $EXT_TOKENIZER \
  --audio_token $AUDIO_TOKEN \
  --freeze_audio_encoder true \
  --replace_strategy prefix \
  --use_lora true \
  --lora_rank 64 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --lora_target_modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --learning_rate 1e-4 \
  --warmup_ratio 0.03 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_grad_norm 1.0 \
  --dtype bfloat16 \
  --gradient_checkpointing true \
  --save_steps 500 \
  --logging_steps 50 \
  --train_collator qwen3_audio_phoneme/qwen3_audio/data/collator_conversations_phoneme.py \
  --train_dataset_loader qwen3_audio_phoneme/qwen3_audio/data/dataset_conversations.py
