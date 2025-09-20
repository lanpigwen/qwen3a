# python inference_projector.py \
#   --whisper_model /root/autodl-tmp/whisper-large-v3-turbo \
#   --qwen3_model /root/autodl-tmp/qwen3 \
#   --projector ./outputs_projector_only/projector.pt \
#   --audio /root/autodl-tmp/test.wav \
#   --prompt "请你做语音评测" \
#   --max_new_tokens 1024 \
#   --min_new_tokens 8 \
#   --temperature 0.8 \
#   --top_p 0.95


python -m qwen3_audio.pipeline \
  --model_dir ./outputs_projector_only \
  --audio /root/autodl-tmp/test.wav \
  --prompt "<|audio_bos|><|AUDIO|><|audio_eos|>请你做语音评测" \
  --max_length 256