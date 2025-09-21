import os
import torch
from transformers import WhisperModel, AutoTokenizer
from .modeling_qwen3_audio import Qwen3AudioForCausalLM

def build_model_and_tokenizer(
    base_model: str,
    whisper_model_id: str,
    audio_token: str,
    freeze_audio_encoder: bool = True,
    replace_strategy: str = "prefix",
    dtype: str = "bfloat16",
    extended_tokenizer_dir: str = None
):
    if extended_tokenizer_dir and os.path.isdir(extended_tokenizer_dir):
        tokenizer = AutoTokenizer.from_pretrained(extended_tokenizer_dir, use_fast=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)

    whisper = WhisperModel.from_pretrained(whisper_model_id)
    if dtype == "bfloat16":
        whisper.to(torch.bfloat16)
    elif dtype == "float16":
        whisper.to(torch.float16)

    model = Qwen3AudioForCausalLM(
        base_model_name=base_model,
        whisper_model=whisper,
        audio_token=audio_token,
        freeze_audio_encoder=freeze_audio_encoder,
        replace_strategy=replace_strategy
    )

    if extended_tokenizer_dir:
        model.tokenizer = tokenizer
        model.lm.resize_token_embeddings(len(tokenizer))
        model.audio_token_id = tokenizer.convert_tokens_to_ids(audio_token)

    if dtype == "bfloat16":
        model.to(torch.bfloat16)
    elif dtype == "float16":
        model.to(torch.float16)

    return model, tokenizer
