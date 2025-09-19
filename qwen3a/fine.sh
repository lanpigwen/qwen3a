python finetune_audio.py \
  --data_mode conversation \
  --freeze_whisper \
  --use_lora \
  --train_qwen3_layers 4 \
  --learning_rate 2e-4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --fp16