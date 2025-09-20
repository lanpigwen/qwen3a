import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

try:
    from transformers import WhisperFeatureExtractor, AutoTokenizer
except ImportError as e:  # pragma: no cover
    raise ImportError("Please install transformers to use data utilities") from e


class AudioTextDataset(Dataset):
    """Simple JSON-like list dataset.

    Each item: {"audio": path_or_array, "text": str, "prompt": optional str}
    Pre-loads metadata; loads audio lazily via torchaudio.load
    """
    def __init__(self, data: List[Dict[str, Any]], feature_extractor: WhisperFeatureExtractor, tokenizer, sampling_rate: int = 16000, max_audio_seconds: Optional[float] = None, text_field: str = "text"):
        self.data = data
        self.fe = feature_extractor
        self.tokenizer = tokenizer
        self.sampling_rate = sampling_rate
        self.max_audio_samples = int(max_audio_seconds * sampling_rate) if max_audio_seconds else None
        self.text_field = text_field

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        import torchaudio  # local import
        item = self.data[idx]
        audio = item["audio"]
        if isinstance(audio, str):
            wav, sr = torchaudio.load(audio)
        else:
            wav = audio
            sr = self.sampling_rate
        if sr != self.sampling_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sampling_rate)
        wav = wav.mean(0, keepdim=True)  # mono
        if self.max_audio_samples:
            wav = wav[..., : self.max_audio_samples]
        features = self.fe(wav.squeeze(0).numpy(), sampling_rate=self.sampling_rate, return_tensors="pt")
        text = item.get(self.text_field, "")
        prompt = item.get("prompt", None)
        return {
            "input_features": features.input_features[0],
            "text": text,
            "prompt": prompt,
        }


@dataclass
class AudioCollator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = None
    add_eos: bool = True

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Pad audio features along time axis
        input_features = [b["input_features"] for b in batch]
        feat_lens = [f.shape[-1] for f in input_features]
        max_len = max(feat_lens)
        if self.pad_to_multiple_of:
            if max_len % self.pad_to_multiple_of != 0:
                max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
        padded = []
        audio_mask = []
        for f, l in zip(input_features, feat_lens):
            if l < max_len:
                pad = torch.zeros(f.shape[0], max_len - l)
                padded.append(torch.cat([f, pad], dim=-1))
                am = torch.cat([torch.ones(l), torch.zeros(max_len - l)])
            else:
                padded.append(f)
                am = torch.ones(l)
            audio_mask.append(am)
        input_features = torch.stack(padded)  # (B, feat, T)
        audio_mask = torch.stack(audio_mask)  # (B, T)

        # Tokenize text (prompt + text) -> input_ids & labels
        input_texts = []
        for b in batch:
            if b["prompt"]:
                input_texts.append(b["prompt"] + b["text"])
            else:
                input_texts.append(b["text"])
        enc = self.tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        if self.add_eos and (self.tokenizer.eos_token is not None):
            # Append eos manually
            eos_id = self.tokenizer.eos_token_id
            input_ids = []
            for ids in enc.input_ids:
                if ids[-1] != eos_id:
                    ids = torch.cat([ids, torch.tensor([eos_id])])
                input_ids.append(ids)
            # Re-pad
            max_t = max(x.size(0) for x in input_ids)
            new_ids = []
            attn = []
            for x in input_ids:
                if x.size(0) < max_t:
                    pad = torch.full((max_t - x.size(0),), self.tokenizer.pad_token_id, dtype=torch.long)
                    new_ids.append(torch.cat([x, pad]))
                    a = torch.cat([torch.ones(x.size(0)), torch.zeros(max_t - x.size(0))])
                else:
                    new_ids.append(x)
                    a = torch.ones(max_t)
                attn.append(a)
            enc["input_ids"] = torch.stack(new_ids)
            enc["attention_mask"] = torch.stack(attn)
        labels = enc.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_features": input_features,
            "audio_pad_mask": audio_mask,
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "labels": labels,
        }


############################################################
# Conversation dataset with <audio>path</audio> placeholder #
############################################################

class ConversationAudioDataset(Dataset):
    """Dataset for conversation style samples containing an audio placeholder.

    Expected each item structure:
    {
      "conversations": [
          {"from": "user", "value": "<audio>/path.wav</audio>文本..."},
          {"from": "assistant", "value": "回复..."},
          ... (multi turns) ...
      ]
    }

    Strategy:
      - Only the FIRST encountered <audio>...</audio> in a sample is used to load audio.
      - All user & assistant texts (with placeholder stripped) form a linearized dialogue prompt+target.
      - Collator will build labels such that only assistant segments are optimized.
    """
    AUDIO_TAG_RE = re.compile(r"<audio>(.*?)</audio>")

    def __init__(
        self,
        data: List[Dict[str, Any]],
        feature_extractor: WhisperFeatureExtractor,
        tokenizer,
        sampling_rate: int = 16000,
        max_audio_seconds: Optional[float] = None,
        user_prefix: str = "<|user|>",
        assistant_prefix: str = "<|assistant|>",
        system_prefix: str = "<|system|>",
    ):
        self.data = data
        self.fe = feature_extractor
        self.tokenizer = tokenizer
        self.sampling_rate = sampling_rate
        self.max_audio_samples = int(max_audio_seconds * sampling_rate) if max_audio_seconds else None
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix
        self.system_prefix = system_prefix

    def __len__(self):
        return len(self.data)

    def _extract_audio_path(self, text: str) -> Tuple[str, Optional[str]]:
        match = self.AUDIO_TAG_RE.search(text)
        if not match:
            return text, None
        path = match.group(1).strip()
        # remove entire tag
        cleaned = self.AUDIO_TAG_RE.sub("", text)
        return cleaned, path

    def __getitem__(self, idx: int):
        import torchaudio
        sample = self.data[idx]
        conversations: List[Dict[str, str]] = sample["conversations"]
        audio_loaded = False
        audio_features = None

        dialogue_text_parts = []
        role_spans = []  # (start_char, end_char, role)
        cursor = 0

        for turn in conversations:
            role = turn.get("from")
            val = turn.get("value", "")
            # Extract audio path only once (prefer first user turn containing it)
            if not audio_loaded:
                cleaned, audio_path = self._extract_audio_path(val)
                if audio_path:
                    wav, sr = torchaudio.load(audio_path)
                    if sr != self.sampling_rate:
                        wav = torchaudio.functional.resample(wav, sr, self.sampling_rate)
                    wav = wav.mean(0, keepdim=True)
                    if self.max_audio_samples:
                        wav = wav[..., : self.max_audio_samples]
                    feats = self.fe(wav.squeeze(0).numpy(), sampling_rate=self.sampling_rate, return_tensors="pt")
                    audio_features = feats.input_features[0]
                    audio_loaded = True
                val = cleaned
            else:
                # remove possible residual tags if appear again
                val = self.AUDIO_TAG_RE.sub("", val)
            # Append with role prefix
            if role == "user":
                prefix = self.user_prefix
            elif role == "assistant":
                prefix = self.assistant_prefix
            else:
                prefix = self.system_prefix
            piece = prefix + val
            start = cursor
            cursor += len(piece)
            role_spans.append((start, cursor, role))
            dialogue_text_parts.append(piece)

        if not audio_loaded:
            raise ValueError(f"Sample {idx} does not contain an <audio>...</audio> tag")

        linear_text = "".join(dialogue_text_parts)

        return {
            "input_features": audio_features,  # (feat, T)
            "text": linear_text,
            "role_spans": role_spans,  # char spans to later mask
        }


@dataclass
class ConversationCollator:
    tokenizer: Any
    pad_to_multiple_of: Optional[int] = None
    add_eos: bool = True
    user_prefix: str = "<|user|>"
    assistant_prefix: str = "<|assistant|>"
    no_mask: bool = False
    debug: bool = False  # 输出调试统计

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Audio padding (single audio per sample already processed)
        input_features = [b["input_features"] for b in batch]
        feat_lens = [f.shape[-1] for f in input_features]
        max_len = max(feat_lens)
        padded = []
        audio_mask = []
        for f, l in zip(input_features, feat_lens):
            if l < max_len:
                pad = torch.zeros(f.shape[0], max_len - l)
                padded.append(torch.cat([f, pad], dim=-1))
                am = torch.cat([torch.ones(l), torch.zeros(max_len - l)])
            else:
                padded.append(f)
                am = torch.ones(l)
            audio_mask.append(am)
        input_features = torch.stack(padded)
        audio_mask = torch.stack(audio_mask)

        texts = [b["text"] for b in batch]
        enc = self.tokenizer(texts, padding=True, return_tensors="pt")
        if self.add_eos and self.tokenizer.eos_token is not None:
            eos = self.tokenizer.eos_token_id
            new_ids = []
            attn = []
            for ids in enc.input_ids:
                if ids[-1] != eos:
                    ids = torch.cat([ids, torch.tensor([eos])])
                new_ids.append(ids)
            max_t = max(x.size(0) for x in new_ids)
            final_ids = []
            final_attn = []
            for x in new_ids:
                if x.size(0) < max_t:
                    pad = torch.full((max_t - x.size(0),), self.tokenizer.pad_token_id, dtype=torch.long)
                    final_ids.append(torch.cat([x, pad]))
                    a = torch.cat([torch.ones(x.size(0)), torch.zeros(max_t - x.size(0))])
                else:
                    final_ids.append(x)
                    a = torch.ones(max_t)
                final_attn.append(a)
            enc["input_ids"] = torch.stack(final_ids)
            enc["attention_mask"] = torch.stack(final_attn)

        labels = enc.input_ids.clone()

        if not self.no_mask:
            # 精确基于每个样本的 role_spans（字符级） -> token 级映射
            # 我们需要原始字符序列: 已经在 dataset 中构建并传入 text
            for i, sample in enumerate(batch):
                text = sample["text"]
                spans = sample["role_spans"]  # list[(start, end, role)]
                # 使用 tokenizer 的 offset mapping 来映射 token -> char span
                # 为避免再次批处理，这里单独 encode 一次 (不会影响主批性能过多)
                tokenized = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
                offsets = tokenized["offset_mapping"]
                # 建立一个数组: token 是否属于 assistant 需要被学习 (预测) 的内容
                # 策略: assistant span 内全部 token(除其前缀标记) 设置可学习; 其它 token -> -100
                learn_mask = [False] * len(offsets)
                for (s, e, role) in spans:
                    if role != "assistant":
                        continue
                    for tidx, (ts, te) in enumerate(offsets):
                        if te <= s or ts >= e:
                            continue
                        learn_mask[tidx] = True
                    # 掉过前缀 <|assistant|> 本身: 如果它被单独 tokenize 出现在开始区域
                    # 通过再次 token 化前缀找出对应 token ids 序列
                # 找出 assistant 前缀 token 序列 (同一 tokenizer 下)
                prefix_ids = self.tokenizer(self.assistant_prefix, add_special_tokens=False).input_ids
                # 在 tokenized.input_ids 中搜索 prefix_ids 的任意出现，并将其 learn_mask 置 False
                ids_seq = tokenized.input_ids
                Lp = len(prefix_ids)
                for start in range(0, len(ids_seq) - Lp + 1):
                    if ids_seq[start:start+Lp] == prefix_ids:
                        for k in range(Lp):
                            learn_mask[start + k] = False

                # 现在需要把 learn_mask 映射到当前批里 pad 后的 enc.input_ids 对应位置
                # enc 与 tokenized 在 token 数可能不同 (因为 enc 是对 texts 批量处理, 但应一致除去 padding)
                # 这里假设 tokenizer 对同一 text 的 tokenization 稳定一致
                seq_len = enc.input_ids[i].size(0)
                copy_len = min(seq_len, len(learn_mask))
                for j in range(copy_len):
                    if not learn_mask[j]:
                        labels[i, j] = -100
                # 剩余 (padding) 稍后统一置 -100
                # 如果全部被屏蔽，fallback 到所有非 pad token 参与训练（避免 loss=0）
                if (labels[i] != -100).sum() == 0:
                    base_ids = enc.input_ids[i]
                    valid_pos = base_ids != self.tokenizer.pad_token_id
                    labels[i][valid_pos] = base_ids[valid_pos]
                    print(f"[ConversationCollator][fallback] sample {i} had 0 valid labels; reverted to full supervision")

        labels[labels == self.tokenizer.pad_token_id] = -100

        valid_labels = (labels != -100).sum(dim=1)
        if self.debug:
            for i, v in enumerate(valid_labels.tolist()):
                print(f"[ConversationCollator][debug] sample {i} valid_label_tokens={v}/{labels.size(1)}")
        if (valid_labels == 0).any():
            print('[ConversationCollator] WARNING: Some samples have 0 valid labels; loss contribution=0.')

        return {
            "input_features": input_features,
            "audio_pad_mask": audio_mask,
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "labels": labels,
            # 供调试 callback 统计使用
            "valid_label_count": valid_labels,
        }
