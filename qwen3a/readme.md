qwen3 + Whisper Audio 对齐微调示例

## 概述
本目录演示如何给 Qwen3 Causal LM 添加一个 Whisper 编码器 (仅 encoder) + 投影层 (projector)，实现类似 Qwen2-Audio 的语音理解 -> 文本对话能力。思路：
1. 语音波形 -> Whisper FeatureExtractor -> log-mel
2. Whisper encoder 输出隐层 (B, T_a, C_w)
3. 通过 projector (线性/多层感知器) 投影到 Qwen3 hidden size (C_q)
4. 将投影后的音频序列作为“前缀 token”拼接到文本 prompt 前面，输入 Qwen3 继续自回归生成。

## 目录结构
```
src/qwen3_audio/modeling_qwen3_audio.py   # 模型包装 (Whisper encoder + projector + Qwen3)
src/qwen3_audio/data.py                   # 数据集与 collator
scripts/finetune_audio.py                 # 微调脚本 (HF Trainer)
requirements.txt                          # 依赖
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 准备模型
假设你已有 Qwen3 基座 (例如 qwen/Qwen3-7B) 和 Whisper (例如 openai/whisper-small)。

## 数据格式
`train.json` 列表格式：
```json
[
	{"audio": "path/to/audio1.wav", "text": "转写或回答文本1", "prompt": "可选的提示指令："},
	{"audio": "path/to/audio2.wav", "text": "转写或回答文本2"}
]
```
说明：
- `audio` 可以是本地 wav 路径（支持多通道，自动转 mono & 重采样）。
- `text` 是你希望模型生成的目标文本（可包含对话格式，如："用户: ...\n助手: ..."）。
- `prompt` 可选，放在目标文本前面，用于额外指令或系统提示。

### 构造对话式数据的建议
如果你做语音问答，可以把样本组织成：
```
prompt = "<|system|>你是一个智能助手。\n<|user|>"
text   = "(这里放该语音对应的用户意图的文字)\n<|assistant|> (这里放期望的回答)"
```
或训练纯转写：`text = "转写结果"`。

## Special Tokens
如果你想加入 `<|audio|>` 之类占位符：
```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(PATH_OR_ID)
tok.add_special_tokens({"additional_special_tokens": ["<|audio|>"]})
tok.save_pretrained("./tok_with_audio")
```
然后在 prompt 中使用 `<|audio|>` 标记音频位置。当前实现将音频前缀直接拼接，不强制需要该 token，但可用于可读性或下游指令格式。

### 使用 <audio>path</audio> 占位符的会话数据
如果你的原始标注像：
```json
{
	"conversations": [
		{"from": "user", "value": "<audio>/root/xxx/test.wav</audio>请你做语音评测"},
		{"from": "assistant", "value": "sil sh iy ..."}
	]
}
```
可以组成一个列表保存为 `conv_train.json`，使用 `ConversationAudioDataset + ConversationCollator`。

线性化方式：每个 turn 前面加前缀（默认 `<|user|>` `<|assistant|>`），音频只读取第一次出现的 `<audio>...</audio>`，其余同条样本内如再出现会被忽略。

训练时只对 assistant 段计算 loss，user 段与 role 前缀被 mask 为 -100。

示例代码：
```python
from transformers import WhisperFeatureExtractor, AutoTokenizer
from qwen3_audio.data import ConversationAudioDataset, ConversationCollator

fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-small')
tok = AutoTokenizer.from_pretrained(PATH_TO_QWEN3)
if tok.pad_token is None:
		tok.pad_token = tok.eos_token

import json
data = json.load(open('conv_train.json'))  # List[ {"conversations": [...] } ]
ds = ConversationAudioDataset(data, fe, tok)
collator = ConversationCollator(tok)

batch = [ds[0], ds[1]]
out = collator(batch)
print(out['input_features'].shape, out['input_ids'].shape, out['labels'].shape)
```

在微调脚本中可以仿照 `AudioTextDataset` 自行替换为上述 Dataset/Collator 组合，或写一个开关参数。

## 微调 (全参数 or 部分参数 + LoRA)
脚本核心选项：
- `--freeze_whisper`: 冻结 Whisper encoder 参数 (常用)。
- `--train_qwen3_layers N`: 只解冻 Qwen3 顶部 N 层 (减少显存 & 加快训练)。0 表示不解冻主干层，只训练项目 projector + lm_head。
- `--use_lora`: 对解冻的 Qwen3 可训练线性层再套 LoRA。

### 运行示例
```bash
python scripts/finetune_audio.py \
	--whisper_model openai/whisper-small \
	--qwen3_model path_or_hub_id_to_qwen3 \
	--train_json ./train.json \
	--output_dir ./outputs \
	--freeze_whisper \
	--train_qwen3_layers 4 \
	--projector_layers 2 \
	--learning_rate 2e-4 \
	--per_device_train_batch_size 2 \
	--gradient_accumulation_steps 8 \
	--num_train_epochs 3 \
	--fp16
```

### 会话占位符数据微调示例
```bash
python scripts/finetune_audio.py \
	--data_mode conversation \
	--whisper_model openai/whisper-small \
	--qwen3_model path_or_hub_id_to_qwen3 \
	--train_json ./conv_train.json \
	--output_dir ./outputs_conv \
	--freeze_whisper \
	--train_qwen3_layers 4 \
	--learning_rate 2e-4 \
	--gradient_accumulation_steps 8 \
	--num_train_epochs 3 \
	--fp16 \
	--user_prefix '<|user|>' \
	--assistant_prefix '<|assistant|>'
```

### 典型超参建议
- batch_size * grad_accumulation 尽量 >= 32 (有效批大小)
- lr: 1e-4 ~ 5e-4 之间调试；若使用 LoRA，可略高
- warmup_ratio: 0.03 ~ 0.1
- projector_layers: 1 (线性) 或 2 (带中间隐藏层)
- projector_hidden: 若使用 2 层，可设为与输出同维或 2x 输出维 (如 8192)

## 推理示例
```python
import torch
from transformers import AutoTokenizer, WhisperFeatureExtractor
from qwen3_audio.modeling_qwen3_audio import Qwen3WhisperForConditionalGeneration

device = 'cuda'
model = Qwen3WhisperForConditionalGeneration(
		whisper_model_name='openai/whisper-small',
		qwen3_model_name='./outputs',  # 微调后目录
		freeze_whisper=True,
)
state_dict = torch.load('./outputs/projector.pt', map_location='cpu')
model.projector.load_state_dict(state_dict)
model.to(device).eval()

fe = WhisperFeatureExtractor.from_pretrained('openai/whisper-small')
tokenizer = AutoTokenizer.from_pretrained('./outputs')
import torchaudio
wav, sr = torchaudio.load('test.wav')
if sr != 16000:
		wav = torchaudio.functional.resample(wav, sr, 16000)
wav = wav.mean(0, keepdim=True)
features = fe(wav.squeeze(0).numpy(), sampling_rate=16000, return_tensors='pt').input_features.to(device)
prompt = "请根据上面语音内容回答："
prompt_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(device)
texts = model.generate(features, tokenizer, prompt_ids=prompt_ids, generation_kwargs={"max_new_tokens":128})
print(texts[0])
```

## 训练策略讨论
1. 先只训练 projector (冻结 Whisper + Qwen3) 让语音嵌入对齐 -> 观测 loss 收敛速度。
2. 如能力不足，再解冻 Qwen3 顶层 2~4 层，或开启 LoRA 针对注意力/MLP 大层。
3. 若语音域差异大（口音/噪声），可在低 lr (1e-5) 下部分解冻 Whisper 后段层数 (如最后 2~4 层)。代码可扩展：遍历 whisper.encoder.layers[-K:] 设置 requires_grad=True。

## 扩展方向
- 支持多轮对话：拼接历史轮文本至 prompt，控制最大长度截断。
- 引入时间戳：保留 Whisper 中间特征 (需要修改 projector 与对齐策略)。
- 多模态联合 (图文+音频)：可再加入图像 Tower。

## 已知限制 & Todo
- 当前 generate 简化：不做 streaming；可扩充 incremental state / cache。
- 未添加自动混合精度内置的梯度检查；若显存紧张可用 bitsandbytes + 4bit/8bit 加载 Qwen3。

## 参考
- Whisper: https://huggingface.co/openai/whisper-small
- Qwen3: 官方仓库 / Hugging Face Hub

欢迎根据自己数据格式调整 `data.py`。
