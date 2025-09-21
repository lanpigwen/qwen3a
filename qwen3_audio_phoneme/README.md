# Qwen3 Audio Phoneme Finetune (Whisper Encoder + Qwen3 LLM, Prefix Strategy)

Maintained by lanpigwen

本子项目演示如何将 Whisper (large-v3) 作为音频编码器，接入 Qwen3 文本 LLM，通过单层线性投影(1280->4096)实现“语音→音素序列”生成（语音评测 / 音素级转写）。

## 特性
- Whisper encoder (冻结) + 简单 Linear Projector
- Prefix 拼接多模态：音频向量置于文本前
- 自动扩展 tokenizer：添加数据集中音素 token
- LoRA 微调
- 直接支持含 `<audio>绝对路径</audio>` 的 conversation JSON/JSONL

## 数据格式
```json
{
  "conversations": [
    {
      "from": "user",
      "value": "<audio>/abs/path/sa1.wav</audio>请你做语音评测"
    },
    {
      "from": "assistant",
      "value": "sil sh iy hh ae ..."
    }
  ]
}
```

## 快速开始
```bash
conda create -n qwen3audio python=3.10 -y
conda activate qwen3audio
pip install -r qwen3_audio_phoneme/requirements.txt
```

### 扩展音素词表
```bash
python qwen3_audio_phoneme/scripts/build_phoneme_vocab_and_extend.py \
  --data qwen3_audio_phoneme/data/raw_conversations.json \
  --base_tokenizer Qwen/Qwen3-7B-Instruct \
  --save_dir qwen3_audio_phoneme/output/extended_tokenizer
```

### Sanity Check
```bash
python qwen3_audio_phoneme/scripts/quick_sanity_check.py
```

### 训练
```bash
bash qwen3_audio_phoneme/scripts/train_conversation_lora.sh
```

### 推理
```bash
python qwen3_audio_phoneme/inference/run_infer_conversation.py \
  --model_dir qwen3_audio_phoneme/output/qwen3_audio_phoneme_lora \
  --whisper_model openai/whisper-large-v3 \
  --audio_file /abs/path/sa1.wav \
  --prompt "请你做语音评测" \
  --extended_tokenizer_dir qwen3_audio_phoneme/output/extended_tokenizer
```

### 简单音素准确率评估
```bash
python qwen3_audio_phoneme/scripts/accuracy_eval.py \
  --pred_file preds.txt \
  --ref_file refs.txt \
  --ignore_sil true
```
(preds.txt / refs.txt 每行对应一条样本的空格分隔音素序列)

## 多 GPU / 显存说明
- 单 A100 40G：batch_size=2, grad_accum=8 可轻松运行
- 若想多卡：可使用 torchrun 或 accelerate；LoRA 的梯度较小，通信成本低
- 减少显存：可尝试 QLoRA (未内置脚本) 或降低 lora_rank

## 常见问题
| 问题 | 处理 |
|------|------|
| loss 不降 | 检查 tokenizer 是否扩展；学习率降至 5e-5 |
| 输出夹杂中文 | 训练步数不足或音素 token 频率低；多训或扩大数据 |
| OOM | 减小 max_audio_seconds / batch / 使用 4bit 量化 |
| NaN | 确认 bfloat16 支持；必要时改 float16 并减小 lr |

## 未来扩展
- QLoRA 脚本
- <PH_START>/<PH_END> 边界标记
- CTC 辅助头
- 多任务评分与置信度输出

## License
MIT (此子目录示例)