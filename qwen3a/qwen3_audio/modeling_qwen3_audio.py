import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Union

try:
    from transformers import (
        WhisperModel,
        AutoModelForCausalLM,
        AutoTokenizer,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError("Please install transformers to use Qwen3WhisperForConditionalGeneration") from e


class AudioProjection(nn.Module):
    """Project Whisper encoder hidden states to Qwen3 embedding dimension."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, num_layers: int = 1, activation: str = "gelu"):
        super().__init__()
        layers = []
        last = in_dim
        hidden_dim = hidden_dim or out_dim
        act_layer: nn.Module
        if activation == "gelu":
            act_layer = nn.GELU()
        elif activation == "silu":
            act_layer = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        for i in range(num_layers - 1):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(act_layer)
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C)
        return self.proj(x)


class Qwen3WhisperForConditionalGeneration(nn.Module):
    """A lightweight wrapper gluing a (frozen) Whisper encoder to a Qwen3 causal LM via a projector.

    Forward contract:
      Inputs:
        input_features: preprocessed log-mel (B, feat, frames) for Whisper encoder
        input_ids / attention_mask: textual prompt fed to Qwen3 (optional)
        audio_pad_mask: optional mask for audio time steps after projection (1 = keep)
        labels: optional labels for LM loss (shifted inside Qwen3)
        mode: "prefill" | "generate" for potential future optimization
      Behavior:
        1. Run Whisper encoder -> hidden states (B, T_a, C_w)
        2. Project to Qwen3 hidden size -> (B, T_a, C_q)
        3. Create audio prefix tokens appended before textual tokens.
        4. Feed concatenated embeddings & extended attention mask to Qwen3 LM.
    """

    def __init__(
        self,
        whisper_model_name: str,
        qwen3_model_name: str,
        projector_hidden: Optional[int] = None,
        projector_layers: int = 1,
        projector_activation: str = "gelu",
        freeze_whisper: bool = True,
        train_qwen3_layers: Optional[int] = None,
        add_audio_token_type: bool = False,
    ) -> None:
        super().__init__()
        # Load models
        self.whisper = WhisperModel.from_pretrained(whisper_model_name)
        self.qwen3 = AutoModelForCausalLM.from_pretrained(qwen3_model_name)
        q_config = self.qwen3.config

        # Determine dims
        self.audio_hidden = self.whisper.config.d_model
        self.q_hidden = q_config.hidden_size

        self.projector = AudioProjection(
            in_dim=self.audio_hidden,
            out_dim=self.q_hidden,
            hidden_dim=projector_hidden,
            num_layers=projector_layers,
            activation=projector_activation,
        )

        if freeze_whisper:
            for p in self.whisper.parameters():
                p.requires_grad = False

        # Optionally freeze most Qwen3 layers except top N (train_qwen3_layers)
        if train_qwen3_layers is not None and train_qwen3_layers >= 0:
            total_layers = len(self.qwen3.model.layers)
            train_from = max(0, total_layers - train_qwen3_layers)
            for i, layer in enumerate(self.qwen3.model.layers):
                requires = i >= train_from
                for p in layer.parameters():
                    p.requires_grad = requires
        # Always allow lm_head to adapt
        for p in self.qwen3.lm_head.parameters():
            p.requires_grad = True

        self.add_audio_token_type = add_audio_token_type
        if add_audio_token_type:
            self.audio_type_embedding = nn.Embedding(2, self.q_hidden)
        else:
            self.audio_type_embedding = None

    @property
    def device(self):  # convenience
        return next(self.parameters()).device

    def forward(
        self,
        input_features: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        audio_pad_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        # 1. Whisper encoder (returns last_hidden_state: (B, T_a, C_w))
        with torch.set_grad_enabled(any(p.requires_grad for p in self.whisper.encoder.parameters())):
            enc_out = self.whisper.encoder(input_features)
        audio_hidden = enc_out.last_hidden_state  # (B, T_a, C_w)

        # 2. Project to Qwen3 hidden size
        audio_proj = self.projector(audio_hidden)  # (B, T_a, C_q)

        # 3. Build audio attention mask
        if audio_pad_mask is None:
            audio_mask = torch.ones(audio_proj.size()[:-1], dtype=torch.long, device=audio_proj.device)
        else:
            audio_mask = audio_pad_mask.long()

        # 4. Text tokens -> embeddings via Qwen3 embedding layer
        if input_ids is not None:
            text_embeds = self.qwen3.model.model.embed_tokens(input_ids)
            if self.add_audio_token_type:
                text_type = self.audio_type_embedding(torch.ones_like(input_ids))
                text_embeds = text_embeds + text_type
        else:
            text_embeds = None

        if self.add_audio_token_type:
            audio_type = self.audio_type_embedding(torch.zeros(audio_proj.size(0), audio_proj.size(1), dtype=torch.long, device=audio_proj.device))
            audio_proj = audio_proj + audio_type

        # 5. Concatenate along time dimension -> inputs_embeds
        if text_embeds is not None:
            inputs_embeds = torch.cat([audio_proj, text_embeds], dim=1)
            # attention mask
            if attention_mask is None:
                if input_ids is not None:
                    attention_mask = torch.ones_like(input_ids)
            full_attention_mask = torch.cat([audio_mask, attention_mask], dim=1)
        else:
            inputs_embeds = audio_proj
            full_attention_mask = audio_mask

        outputs = self.qwen3(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels,
        )
        return outputs

    def generate(
        self,
        input_features: torch.Tensor,
        tokenizer,
        prompt_ids: Optional[torch.Tensor] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        generation_kwargs = generation_kwargs or {}
        with torch.no_grad():
            enc_out = self.whisper.encoder(input_features)
            audio_proj = self.projector(enc_out.last_hidden_state)
            audio_mask = torch.ones(audio_proj.size()[:-1], dtype=torch.long, device=audio_proj.device)
            if prompt_ids is not None:
                text_embeds = self.qwen3.model.model.embed_tokens(prompt_ids.to(audio_proj.device))
                inputs_embeds = torch.cat([audio_proj, text_embeds], dim=1)
                attn_mask = torch.cat([audio_mask, torch.ones_like(prompt_ids)], dim=1)
            else:
                inputs_embeds = audio_proj
                attn_mask = audio_mask
            gen_ids = self.qwen3.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                **generation_kwargs
            )
        return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
