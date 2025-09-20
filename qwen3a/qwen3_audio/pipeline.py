"""HF-style pipeline wrappers for projector-only Qwen3 + Whisper audio model.

Usage example (after finetuning saved to ./outputs_projector_only):

    from qwen3_audio.pipeline import Qwen3AudioForConditionalGeneration, Qwen3AudioProcessor
    import librosa

    model_path = './outputs_projector_only'  # directory containing tokenizer files, projector.pt, finetune_meta.json
    processor = Qwen3AudioProcessor.from_pretrained(model_path)
    model = Qwen3AudioForConditionalGeneration.from_pretrained(model_path)

    prompt = '<|audio_bos|><|AUDIO|><|audio_eos|>请你做语音评测'
    wav_path = '/root/autodl-tmp/test.wav'
    audio, sr = librosa.load(wav_path, sr=processor.feature_extractor.sampling_rate)
    inputs = processor(text=prompt, audios=audio, return_tensors='pt')
    generated_ids = model.generate(**inputs, max_length=256)
    # Remove prompt part similar to user example
    prompt_len = inputs['input_ids'].size(1)
    new_tokens = generated_ids[:, prompt_len:]
    response = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
    print('Response:', response)

Limitations:
  - This is a lightweight local wrapper, not fully integrated with transformers hub auto classes.
  - Only supports single-audio batch for now.
"""

from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer, WhisperFeatureExtractor

from .modeling_qwen3_audio import Qwen3WhisperForConditionalGeneration


SPECIAL_AUDIO_TOKENS = ["<|audio_bos|>", "<|audio_eos|>", "<|AUDIO|>"]


def _maybe_add_special_tokens(tokenizer):
    added = []
    for tok in SPECIAL_AUDIO_TOKENS:
        if tok not in tokenizer.get_vocab():
            added.append(tok)
    if added:
        tokenizer.add_tokens(added, special_tokens=True)
        # NOTE: Underlying LM embeddings are NOT resized here since we never train LM; these new tokens
        # will map to random embeddings unless user decides to fine-tune LM later.
    return tokenizer


class Qwen3AudioProcessor:
    """Combines WhisperFeatureExtractor + Qwen tokenizer, mimicking an AutoProcessor style API.

    from_pretrained(path) expects directory with:
      - tokenizer files (saved via tokenizer.save_pretrained)
      - finetune_meta.json (to locate whisper base) OR user passes whisper_model explicitly.
    """

    def __init__(self, tokenizer, feature_extractor):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor

    @classmethod
    def from_pretrained(cls, path: str, whisper_model: Optional[str] = None):
        meta_path = os.path.join(path, 'finetune_meta.json')
        meta = {}
        if os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path))
            except Exception:
                meta = {}
        if whisper_model is None:
            whisper_model = meta.get('whisper_model')
        if whisper_model is None:
            raise ValueError('whisper_model not provided and not found in finetune_meta.json')

        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer = _maybe_add_special_tokens(tokenizer)
        fe = WhisperFeatureExtractor.from_pretrained(whisper_model)
        return cls(tokenizer, fe)

    # For compatibility with example code
    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def __call__(self,
                 text: Union[str, List[str]],
                 audios: Union[np.ndarray, torch.Tensor, List[np.ndarray]],
                 return_tensors: str = 'pt') -> Dict[str, Any]:
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text
        if isinstance(audios, list):
            if len(audios) != 1:
                raise ValueError('Only single-audio batch currently supported.')
            audio_arr = audios[0]
        else:
            audio_arr = audios
        if isinstance(audio_arr, torch.Tensor):
            audio_np = audio_arr.squeeze().cpu().numpy()
        else:
            audio_np = np.asarray(audio_arr).squeeze()

        # Feature extraction
        feats = self.feature_extractor(audio_np, sampling_rate=self.feature_extractor.sampling_rate, return_tensors='pt')
        toks = self.tokenizer(texts, return_tensors='pt', padding=False)

        out = {
            'input_features': feats.input_features,  # (1, feat_dim, frames)
            'input_ids': toks.input_ids,
            'attention_mask': toks.attention_mask if 'attention_mask' in toks else None,
        }
        if return_tensors != 'pt':
            raise ValueError('Only return_tensors="pt" is supported.')
        return out


class Qwen3AudioForConditionalGeneration(torch.nn.Module):
    """High-level wrapper exposing a HF-like generate API compatible with Qwen3AudioProcessor output."""

    def __init__(self, core_model: Qwen3WhisperForConditionalGeneration, tokenizer):
        super().__init__()
        self.core = core_model
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, path: str, trust_remote_code: bool = True):  # trust_remote_code kept for API similarity
        meta_path = os.path.join(path, 'finetune_meta.json')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f'Expected finetune_meta.json in {path}')
        meta = json.load(open(meta_path))
        whisper_model = meta['whisper_model']
        qwen3_model = meta['qwen3_model']

        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer = _maybe_add_special_tokens(tokenizer)

        core = Qwen3WhisperForConditionalGeneration(
            whisper_model_name=whisper_model,
            qwen3_model_name=qwen3_model,
            freeze_whisper=meta.get('freeze_whisper', True),
        )
        # Load projector
        projector_path = os.path.join(path, 'projector.pt')
        if not os.path.exists(projector_path):
            raise FileNotFoundError(f'projector.pt not found in {path}')
        try:
            state = torch.load(projector_path, map_location='cpu', weights_only=True)
        except TypeError:
            state = torch.load(projector_path, map_location='cpu')
        core.projector.load_state_dict(state, strict=False)
        core.eval()

        return cls(core, tokenizer)

    @property
    def device(self):
        return self.core.device

    def to(self, *args, **kwargs):  # type: ignore
        self.core.to(*args, **kwargs)
        return self

    @torch.no_grad()
    def generate(self, input_features: torch.Tensor, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, max_length: int = 256, **gen_kwargs):
        # Map to core.generate interface: treat input_ids as prompt_ids
        gen_kwargs = dict(gen_kwargs)
        gen_kwargs.setdefault('max_new_tokens', max_length)
        # Align dtype with whisper encoder first conv weight to avoid float/half mismatch
        ref = next(self.core.whisper.encoder.parameters())
        if input_features.dtype != ref.dtype or input_features.device != ref.device:
            input_features = input_features.to(device=ref.device, dtype=ref.dtype)
        texts = self.core.generate(
            input_features=input_features,
            tokenizer=self.tokenizer,
            prompt_ids=input_ids.to(self.core.device),
            generation_kwargs=gen_kwargs,
        )
        # Convert back to token ids by re-tokenizing output (best-effort). Core wrapper currently returns text only.
        # To mimic HF generate returning token ids we re-encode. This may not preserve exact ids for special tokens.
        batch_texts = texts
        encoded = self.tokenizer(batch_texts, return_tensors='pt', padding=False)
        return encoded.input_ids


__all__ = [
    'Qwen3AudioProcessor',
    'Qwen3AudioForConditionalGeneration',
]


def _cli():  # simple command line helper
    import argparse, librosa
    ap = argparse.ArgumentParser(description='Qwen3 Audio Projector Inference')
    ap.add_argument('--model_dir', required=True, help='Directory containing projector.pt + finetune_meta.json + tokenizer files')
    ap.add_argument('--audio', required=True, help='Path to wav')
    ap.add_argument('--prompt', default='<|audio_bos|><|AUDIO|><|audio_eos|>', help='Prompt (include audio tokens if desired)')
    ap.add_argument('--max_length', type=int, default=256)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--top_p', type=float, default=0.9)
    ap.add_argument('--greedy', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    processor = Qwen3AudioProcessor.from_pretrained(args.model_dir)
    model = Qwen3AudioForConditionalGeneration.from_pretrained(args.model_dir)
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    audio, _ = librosa.load(args.audio, sr=processor.feature_extractor.sampling_rate)
    inputs = processor(text=args.prompt, audios=audio, return_tensors='pt')
    # Move tensors
    inputs['input_features'] = inputs['input_features'].to(device)
    inputs['input_ids'] = inputs['input_ids'].to(device)
    do_sample = not args.greedy
    gen_ids = model.generate(
        input_features=inputs['input_features'],
        input_ids=inputs['input_ids'],
        max_length=args.max_length,
        do_sample=do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    # Slice new tokens after prompt
    prompt_len = inputs['input_ids'].size(1)
    new_ids = gen_ids[:, prompt_len:]
    text = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
    print(text)


if __name__ == '__main__':
    _cli()
