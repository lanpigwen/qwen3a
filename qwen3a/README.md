# Qwen3 Audio Fine-tuning

这是一个将 Whisper（音频编码器）与 Qwen3 因果语言模型（Causal LM）拼接以做音频到对话/文本微调的仓库。项目目标是低成本地将语音信息注入大型语言模型（LLM），支持 LoRA、projector、以及 4/8-bit 量化友好的训练流程，并提供一系列稳定性与调试策略以减少 NaN / loss=0 / 梯度爆炸等问题。

## 关键特性
- 将 Whisper encoder 的输出通过可配置的 projector 投影到 Qwen3 的隐藏向量空间。
- 支持 Conversation-style 数据（带 <audio>... </audio> 标签的对话样例）和简单的音频-文本样例。
- 支持 LoRA 注入以低成本微调 Qwen3。
- 支持 4-bit / 8-bit 量化加载（依赖 bitsandbytes 与 transformers 的 BitsAndBytesConfig）。
- 包含若干训练时稳定性措施（projector fp32、LoRA fp32、early LR 缩放、单独 LoRA 梯度裁剪、NaN 清理回调等）。

## 仓库结构

```
.
├── fine.sh                 # 示例/快捷运行脚本（项目根）
├── finetune_audio.py       # 主训练脚本，包含数据加载、模型组装、Trainer 配置与回调
├── SUMMARY.md              # 项目运行/调试总结与建议（包含大量稳定性经验）
└── qwen3_audio/
    ├── __init__.py         # 导出模型/数据工具
    ├── data.py             # 数据集与 collator：AudioTextDataset, ConversationAudioDataset, 对齐/掩码策略
    └── modeling_qwen3_audio.py
                           # 模型定义：Qwen3WhisperForConditionalGeneration、AudioProjection
```

## 快速开始

1. 环境准备（建议）

- Python 3.10+，安装常用依赖：torch、transformers、torchaudio、peft、bitsandbytes（若使用 4/8-bit）等。
- 建议使用虚拟环境或 conda 管理包。

示例（仅供参考）：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchaudio transformers peft accelerate
# 若需 4bit/8bit 量化支持：
pip install bitsandbytes
```

2. 数据格式

- flat（AudioTextDataset）: 每个样本为 JSON 对象 {"audio": "path_or_array", "text": "...", "prompt": optional}
- conversation（ConversationAudioDataset）: 每个样本带 `conversations` 字段，包含多轮对话。需在某个 user/assistant 文本中通过 `<audio>/path/to/file.wav</audio>` 嵌入音频路径；数据加载器会找到首个 audio tag 并加载音频。

训练数据示例（conversation 模式）：

{
  "conversations": [
    {"from": "user", "value": "<audio>/data/samples/1.wav</audio> 请帮我总结这段音频内容"},
    {"from": "assistant", "value": "好的，下面是..."}
  ]
}

3. 运行训练

仓库提供主脚本 `finetune_audio.py`。一个最小的本地运行示例：

```bash
CUDA_VISIBLE_DEVICES=0 python finetune_audio.py \
  --data_mode conversation \
  --freeze_whisper \
  --use_lora \
  --train_qwen3_layers 0 \
  --num_train_epochs 0.01 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --projector_fp32 \
  --freeze_projector_steps 5 \
  --max_grad_norm 1.0 \
  --projector_scale 1.0 \
  --lora_fp32 \
  --lora_init_scale 0.1 \
  --lora_grad_clip 5.0 \
  --early_lr_factor 0.2 \
  --early_lr_steps 20 \
  --train_json /path/to/train.json \
  --qwen3_model /path/to/qwen3 \
  --whisper_model /path/to/whisper
```

或者使用仓库根目录的 `fine.sh`（如果已配置好）来运行常见配置。

注意：`fine.sh` 中的示例命令为最小示例，实际运行时通常需要至少指定训练数据与模型路径，例如：

```bash
# 在 fine.sh 的命令基础上，确保添加以下参数（示例）：
--train_json /path/to/train.json \
--qwen3_model /path/to/qwen3 \
--whisper_model /path/to/whisper
```

此外：
- 若使用 4bit/8bit 量化加载，需要安装 `bitsandbytes` 并传入 `--load_in_4bit` 或 `--load_in_8bit`（并可使用 `--device_map auto`）。
- 若 GPU 内存紧张，尝试加上 `--whisper_cpu` 将 Whisper encoder 放到 CPU（trade-off: 速度）。

## 常用参数说明

- `--whisper_model`：Whisper 模型路径/ID（encoder 用于提取音频特征）。
- `--qwen3_model`：Qwen3 模型路径/ID。
- `--data_mode`：`flat` 或 `conversation`（推荐 conversation 用于对话式标注）。
- `--use_lora`：是否注入 LoRA。
- `--lora_fp32` / `--lora_init_scale` / `--lora_grad_clip`：LoRA 稳定性相关参数。
- `--projector_fp32` / `--projector_scale` / `--freeze_projector_steps`：projector 的数值稳定性选项。
- `--load_in_4bit` / `--load_in_8bit`：使用 bitsandbytes 进行 k-bit 量化加载（需安装 bitsandbytes 并使 transformers 支持）。

更多参数请参考 `finetune_audio.py` 中的 argparse 配置。

## 稳定性与调试建议（来自 SUMMARY.md）

- 若出现 NaN 或 loss=0：优先查看 `NaNMonitorCallback` 日志（脚本中实现了 NaN 清理逻辑与回调）。
- 常见优先级配置（推荐稳定优先）示例：

  --freeze_whisper --use_lora --lora_fp32 --lora_init_scale 0.1 --lora_grad_clip 5.0 \
  --projector_fp32 --freeze_projector_steps 20 --early_lr_factor 0.2 --early_lr_steps 100 \
  --learning_rate 3e-5 --max_grad_norm 1.0 --gradient_accumulation_steps 4 --per_device_train_batch_size 1

- 使用 4-bit/8-bit 时：保留 projector 与 LoRA 的 fp32（--projector_fp32、--lora_fp32），并将基础学习率初始值降 30% 左右。
- 若发现 valid_label_count 很低或全部为 0，请检查数据中的 role span、tokenizer 前缀（`<|assistant|>` 等）是否与脚本一致；conversation collator 会在所有 token 被 mask 时回退为保留末尾有效 tokens 以避免 loss=0。

## 进阶/可选

- 可以使用 `--device_map auto` 搭配量化加载将模型分布到多卡。
- `manual_loss` 模式可以逐步调试 cross-entropy 计算与 logits 值（脚本支持手动 loss 计算以便诊断）。
- README 与 SUMMARY.md 中包含大量实验经验与调参建议，请以 `SUMMARY.md` 为稳定性问题排查首要参考。

## 许可证与作者

请在发布之前补充合适的许可证与作者信息（仓库中当前未包含 LICENSE 文件）。

----

如果你希望我把 README 翻译为英文、补充运行脚本示例（如 Dockerfile / requirements.txt / examples），或者把 `fine.sh` 标准化为可运行脚本，我可以接着实现这些改进。 
