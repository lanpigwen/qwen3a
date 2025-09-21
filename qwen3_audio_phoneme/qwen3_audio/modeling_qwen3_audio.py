import torch
import torch.nn as nn
from transformers import QwenForCausalLM, WhisperModel, AutoTokenizer

class SimpleProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)

class AudioEncoderWrapper(nn.Module):
    def __init__(self, whisper_model: WhisperModel, target_dim: int, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = whisper_model.encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        d_audio = whisper_model.config.d_model
        if d_audio != 1280:
            raise ValueError(f"Expect Whisper large hidden=1280, got {d_audio}")
        self.projector = SimpleProjector(d_audio, target_dim)

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor = None):
        enc_out = self.encoder(input_features, attention_mask=attention_mask)
        hidden = enc_out.last_hidden_state
        proj = self.projector(hidden)
        if attention_mask is None:
            am = torch.ones(proj.size(0), proj.size(1), dtype=torch.long, device=proj.device)
        else:
            am = torch.ones(proj.size(0), proj.size(1), dtype=torch.long, device=proj.device)
        return proj, am

class Qwen3AudioForCausalLM(nn.Module):
    def __init__(self,
        base_model_name: str,
        whisper_model: WhisperModel,
        audio_token: str = "<|audio|>",
        freeze_audio_encoder: bool = True,
        replace_strategy: str = "prefix"
    ):
        super().__init__()
        if replace_strategy != "prefix":
            raise ValueError("Only prefix strategy implemented in this file.")
        self.lm = QwenForCausalLM.from_pretrained(base_model_name)
        self.hidden_size = self.lm.config.hidden_size
        self.audio_token = audio_token
        self.replace_strategy = replace_strategy

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
        if self.audio_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.audio_token])
            self.lm.resize_token_embeddings(len(self.tokenizer))
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)

        self.audio_encoder = AudioEncoderWrapper(
            whisper_model=whisper_model,
            target_dim=self.hidden_size,
            freeze_encoder=freeze_audio_encoder
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        audio_input_features=None,
        audio_feature_attention_mask=None,
        return_dict=True
    ):
        if input_ids is None:
            raise ValueError("input_ids required")
        if audio_input_features is not None:
            audio_embeds, _ = self.audio_encoder(audio_input_features, audio_feature_attention_mask)
        else:
            audio_embeds = None

        text_embeds = self.lm.model.embed_tokens(input_ids)

        if audio_embeds is not None:
            inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1)
            B = input_ids.size(0)
            attn_audio = torch.ones(B, audio_embeds.size(1), device=text_embeds.device, dtype=torch.long)
            final_attention_mask = torch.cat([attn_audio, attention_mask], dim=1) if attention_mask is not None else None
            if labels is not None:
                pad_labels = torch.full((B, audio_embeds.size(1)), -100, device=labels.device, dtype=labels.dtype)
                final_labels = torch.cat([pad_labels, labels], dim=1)
            else:
                final_labels = None
        else:
            inputs_embeds = text_embeds
            final_attention_mask = attention_mask
            final_labels = labels

        outputs = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=final_attention_mask,
            labels=final_labels,
            return_dict=return_dict
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        audio_input_features=None,
        max_new_tokens=128,
        **gen_kwargs
    ):
        if input_ids is None:
            raise ValueError("input_ids required for generation")
        if audio_input_features is not None:
            audio_embeds, _ = self.audio_encoder(audio_input_features)
            text_embeds = self.lm.model.embed_tokens(input_ids)
            inputs_embeds = torch.cat([audio_embeds, text_embeds], dim=1)
            B = input_ids.size(0)
            attn_audio = torch.ones(B, audio_embeds.size(1), device=text_embeds.device, dtype=torch.long)
            attention_mask = torch.cat([attn_audio, attention_mask], dim=1) if attention_mask is not None else None
        else:
            inputs_embeds = self.lm.model.embed_tokens(input_ids)

        return self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **gen_kwargs
        )
