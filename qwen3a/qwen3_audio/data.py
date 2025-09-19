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
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
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
        # Mask everything that is NOT assistant content
        # 简化：直接把 user_prefix 开头到下一个 assistant_prefix 之前 mask。
        # 更精细可以利用 role_spans + 再 token-level 对齐，这里提供基本功能。
        input_texts = texts
        for i, text in enumerate(input_texts):
            # 重新 tokenize 逐 token 掩码
            # 这里简单策略：如果一个 token 解码后不包含 assistant_prefix 且在第一个 assistant_prefix 出现前 -> mask
            decoded_tokens = [self.tokenizer.decode([tid], skip_special_tokens=False) for tid in enc.input_ids[i]]
            seen_assistant = False
            for j, tok_str in enumerate(decoded_tokens):
                if self.assistant_prefix in tok_str:
                    seen_assistant = True
                    # 把含有前缀自身的 token 也 mask，防止学习前缀，或保留也可以按需调整
                    labels[i, j] = -100
                else:
                    if not seen_assistant:
                        labels[i, j] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_features": input_features,
            "audio_pad_mask": audio_mask,
            "input_ids": enc.input_ids,
            "attention_mask": enc.attention_mask,
            "labels": labels,
        }
