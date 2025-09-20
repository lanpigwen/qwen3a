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
        # Load Qwen3 (allow subclass override for quantization / device_map)
        if hasattr(self, '_load_qwen'):
            self.qwen3 = self._load_qwen(qwen3_model_name)
            # 确保 dtype 一致 (某些 8bit/4bit 权重不会有 .dtype 统一成 fp16 供 projector 使用)
        else:
            self.qwen3 = AutoModelForCausalLM.from_pretrained(
                qwen3_model_name,
                dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        self.whisper = WhisperModel.from_pretrained(
            whisper_model_name,
            dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        q_config = self.qwen3.config

        # Resolve inner model wrapper (some HF models wrap actual modules in .model)
        qwen_inner = getattr(self.qwen3, 'model', self.qwen3)
        self.qwen_inner = qwen_inner

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
        # 自定义稳定初始化（Xavier uniform * 0.1）
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.weight.data.mul_(0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if freeze_whisper:
            for p in self.whisper.parameters():
                p.requires_grad = False

        # Optionally freeze most Qwen3 layers except top N (train_qwen3_layers)
        if train_qwen3_layers is not None and train_qwen3_layers >= 0:
            # Try common layer containers on HF models
            if hasattr(qwen_inner, 'layers'):
                layers_list = getattr(qwen_inner, 'layers')
            elif hasattr(qwen_inner, 'encoder') and hasattr(qwen_inner.encoder, 'layers'):
                layers_list = getattr(qwen_inner.encoder, 'layers')
            elif hasattr(qwen_inner, 'transformer') and hasattr(qwen_inner.transformer, 'layers'):
                layers_list = getattr(qwen_inner.transformer, 'layers')
            else:
                layers_list = []

            if len(layers_list) > 0:
                total_layers = len(layers_list)
                train_from = max(0, total_layers - train_qwen3_layers)
                for i, layer in enumerate(layers_list):
                    requires = i >= train_from
                    for p in layer.parameters():
                        p.requires_grad = requires
        # Always allow lm_head to adapt (try both top-level and inner model)
        lm_head = getattr(self.qwen3, 'lm_head', None)
        if lm_head is None:
            lm_head = getattr(qwen_inner, 'lm_head', None)
        if lm_head is not None:
            for p in lm_head.parameters():
                p.requires_grad = True

        self.add_audio_token_type = add_audio_token_type
        if add_audio_token_type:
            self.audio_type_embedding = nn.Embedding(2, self.q_hidden)
        else:
            self.audio_type_embedding = None

        # Ensure projector and embeddings match Qwen3 dtype for consistency
        self.projector.to(dtype=self.qwen3.dtype)
        if self.audio_type_embedding is not None:
            self.audio_type_embedding.to(dtype=self.qwen3.dtype)

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
        **kwargs,
    ) -> Dict[str, Any]:
        # Ensure dtype/device match Whisper parameters (half precision likely)
        # Whisper expects (batch, feature, time) float/half matching conv1 weight
        # Robustly fetch a reference parameter for dtype/device alignment
        if hasattr(self.whisper.encoder, 'conv1') and hasattr(self.whisper.encoder.conv1, 'weight'):
            ref_param = self.whisper.encoder.conv1.weight
        else:
            ref_iter = list(self.whisper.encoder.parameters())
            if len(ref_iter) == 0:
                raise RuntimeError("Whisper encoder has no parameters to infer dtype/device from.")
            ref_param = ref_iter[0]
        if input_features.dtype != ref_param.dtype or input_features.device != ref_param.device:
            input_features = input_features.to(device=ref_param.device, dtype=ref_param.dtype)
        # 1. Whisper encoder (returns last_hidden_state: (B, T_a, C_w))
        with torch.set_grad_enabled(any(p.requires_grad for p in self.whisper.encoder.parameters())):
            enc_out = self.whisper.encoder(input_features)
        audio_hidden = enc_out.last_hidden_state  # (B, T_a, C_w)

        # 2. Project to Qwen3 hidden size
        if getattr(self, 'projector_fp32', False):
            # 一次性将 projector 转为 fp32，提升数值稳定性
            if not getattr(self, '_projector_fp32_converted', False):
                self.projector.to(torch.float32)
                self._projector_fp32_converted = True
                self._printed_projector_stats = False
            proj_in = audio_hidden.float()  # (B, T_a, C_w) -> fp32
            # removed projector stats print
            audio_proj = self.projector(proj_in)  # already fp32 output
            # removed projector NaN warning
            audio_proj = audio_proj.to(self.qwen3.dtype)
        else:
            audio_proj = self.projector(audio_hidden)  # (B, T_a, C_q)
        # 可能缩放以防激活过大
        scale = getattr(self, 'projector_scale', 1.0)
        if scale != 1.0:
            audio_proj = audio_proj * scale

        # 3. Build audio attention mask
        if audio_pad_mask is None:
            audio_mask = torch.ones(audio_proj.size()[:-1], dtype=torch.long, device=audio_proj.device)
        else:
            audio_mask = audio_pad_mask.long()

        # 4. Text tokens -> embeddings via Qwen3 embedding layer
        if input_ids is not None:
            # Resolve embedding layer robustly
            emb_layer = None
            if hasattr(self.qwen_inner, 'get_input_embeddings'):
                emb_layer = self.qwen_inner.get_input_embeddings()
            elif hasattr(self.qwen_inner, 'embed_tokens'):
                emb_layer = getattr(self.qwen_inner, 'embed_tokens')
            elif hasattr(self.qwen3, 'get_input_embeddings'):
                emb_layer = self.qwen3.get_input_embeddings()
            else:
                raise RuntimeError('Could not find embedding layer on qwen3 model.')

            # emb_layer is typically an nn.Embedding or callable
            text_embeds = emb_layer(input_ids)
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

        # Extend labels to match inputs_embeds length (audio prefix + text)
        if labels is not None and text_embeds is not None:
            audio_seq_len = audio_proj.size(1)
            extended_labels = torch.cat([
                torch.full((labels.size(0), audio_seq_len), -100, dtype=labels.dtype, device=labels.device),
                labels
            ], dim=1)
            labels = extended_labels

        use_manual = getattr(self, 'manual_loss', False) and labels is not None and input_ids is not None
        if use_manual:
            raw_outputs = self.qwen3(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            logits = raw_outputs.logits
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            if not hasattr(self, '_manual_loss_steps'):
                self._manual_loss_steps = 0
            if self._manual_loss_steps < 6:
                self._manual_loss_steps += 1  # silent increment
            # 记录 logits 供回调监控
            self.last_logits = logits.detach()
            return {"loss": loss, "logits": logits}
        else:
            outputs = self.qwen3(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                labels=labels,
            )
            self.last_logits = outputs.logits.detach() if hasattr(outputs, 'logits') else None
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
                # Resolve embedding for generation path
                if hasattr(self.qwen_inner, 'get_input_embeddings'):
                    gen_emb = self.qwen_inner.get_input_embeddings()
                elif hasattr(self.qwen_inner, 'embed_tokens'):
                    gen_emb = getattr(self.qwen_inner, 'embed_tokens')
                elif hasattr(self.qwen3, 'get_input_embeddings'):
                    gen_emb = self.qwen3.get_input_embeddings()
                else:
                    raise RuntimeError('Could not find embedding layer on qwen3 model for generation.')
                text_embeds = gen_emb(prompt_ids.to(audio_proj.device))
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

    def gradient_checkpointing_enable(self, **kwargs):
        # Forward to Qwen3 model
        if hasattr(self.qwen3, 'gradient_checkpointing_enable'):
            self.qwen3.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.qwen3, 'gradient_checkpointing_disable'):
            self.qwen3.gradient_checkpointing_disable()
