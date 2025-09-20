import argparse
import json
import os
from typing import List
import torch
from torch.utils.data import DataLoader
from transformers import (
    WhisperFeatureExtractor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_callback import TrainerCallback
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None
from peft import LoraConfig, get_peft_model
try:
    from peft import prepare_model_for_kbit_training
except ImportError:
    prepare_model_for_kbit_training = None

from qwen3_audio import Qwen3WhisperForConditionalGeneration
from qwen3_audio import (
    AudioTextDataset,
    AudioCollator,
    ConversationAudioDataset,
    ConversationCollator,
)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--whisper_model', type=str, default='/root/autodl-tmp/whisper-large-v3-turbo')
    ap.add_argument('--qwen3_model', type=str,default='/root/autodl-tmp/qwen3', help='Base Qwen3 model id/path')
    ap.add_argument('--train_json', type=str, default='/root/autodl-tmp/qwen_auido_train.json', help='List[ {"audio": path, "text": str, "prompt": optional str} ]')
    ap.add_argument('--output_dir', type=str, default='./outputs')
    ap.add_argument('--freeze_whisper', action='store_true')
    ap.add_argument('--train_qwen3_layers', type=int, default=0, help='Number of top Qwen3 layers to unfreeze (0=none)')
    ap.add_argument('--projector_layers', type=int, default=1)
    ap.add_argument('--projector_hidden', type=int, default=None)
    ap.add_argument('--per_device_train_batch_size', type=int, default=2)
    ap.add_argument('--per_device_eval_batch_size', type=int, default=2)
    ap.add_argument('--learning_rate', type=float, default=2e-4)
    ap.add_argument('--num_train_epochs', type=float, default=3.0)
    ap.add_argument('--warmup_ratio', type=float, default=0.03)
    ap.add_argument('--gradient_accumulation_steps', type=int, default=4)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--use_lora', action='store_true')
    ap.add_argument('--lora_r', type=int, default=16)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--lora_patterns', type=str, default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,proj', help='Comma separated substrings to match module names for LoRA injection')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data_mode', type=str, default='conversation', choices=['flat','conversation'], help='flat: AudioTextDataset, conversation: ConversationAudioDataset')
    ap.add_argument('--user_prefix', type=str, default='<|user|>')
    ap.add_argument('--assistant_prefix', type=str, default='<|assistant|>')
    ap.add_argument('--system_prefix', type=str, default='<|system|>')
    ap.add_argument('--no_mask', action='store_true', help='Do not mask user tokens (debug: train on all tokens)')
    ap.add_argument('--debug_collator', action='store_true', help='Print per-sample valid label token stats for conversation mode')
    # Memory / quantization options
    ap.add_argument('--device_map', type=str, default=None, help="Pass 'auto' to let accelerate dispatch layers across GPUs")
    ap.add_argument('--load_in_8bit', action='store_true', help='Load Qwen3 in 8-bit (requires bitsandbytes)')
    ap.add_argument('--load_in_4bit', action='store_true', help='Load Qwen3 in 4-bit NF4 (requires bitsandbytes)')
    ap.add_argument('--whisper_cpu', action='store_true', help='Force Whisper encoder to stay on CPU (projector+LM still on GPU)')
    ap.add_argument('--manual_loss', action='store_true', help='Use manual cross-entropy over shifted logits (diagnostic mode)')
    ap.add_argument('--projector_scale', type=float, default=1.0, help='Scale factor applied to audio projected embeddings to prevent overflow')
    ap.add_argument('--projector_fp32', action='store_true', help='Force projector forward in fp32 and cast back to model dtype')
    ap.add_argument('--freeze_projector_steps', type=int, default=0, help='Keep projector frozen for first N optimizer steps (grad clipped to 0)')
    ap.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping value (override default 0.0)')
    # LoRA 稳定化相关
    ap.add_argument('--lora_fp32', action='store_true', help='Cast LoRA A/B params to fp32 for stability')
    ap.add_argument('--lora_init_scale', type=float, default=1.0, help='Scale factor applied to initial LoRA A/B weights (e.g., 0.1)')
    ap.add_argument('--lora_grad_clip', type=float, default=0.0, help='If >0, apply separate grad norm clipping to LoRA params only')
    ap.add_argument('--early_lr_factor', type=float, default=0.2, help='Scale base LR by this factor during early warm steps for stability')
    ap.add_argument('--early_lr_steps', type=int, default=0, help='Number of initial steps to apply early_lr_factor (0=disable)')
    return ap.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load feature extractor & tokenizer
    fe = WhisperFeatureExtractor.from_pretrained(args.whisper_model)
    tokenizer = AutoTokenizer.from_pretrained(args.qwen3_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    with open(args.train_json, 'r') as f:
        data_list = json.load(f)

    if args.data_mode == 'flat':
        dataset = AudioTextDataset(data_list, fe, tokenizer, max_audio_seconds=None)
        collator = AudioCollator(tokenizer)
    else:
        dataset = ConversationAudioDataset(
            data_list,
            fe,
            tokenizer,
            max_audio_seconds=None,
            user_prefix=args.user_prefix,
            assistant_prefix=args.assistant_prefix,
            system_prefix=args.system_prefix,
        )
        collator = ConversationCollator(
            tokenizer,
            user_prefix=args.user_prefix,
            assistant_prefix=args.assistant_prefix,
            no_mask=args.no_mask,
            debug=args.debug_collator,
        )

    # Build model
    # Prepare kwargs for model loading (quantization / device map)
    qwen_extra = {}
    if args.load_in_8bit or args.load_in_4bit:
        if BitsAndBytesConfig is None:
            raise ImportError("transformers version missing BitsAndBytesConfig; upgrade transformers to use 4bit/8bit quantization.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            bnb_4bit_quant_type='nf4' if args.load_in_4bit else None,
            bnb_4bit_use_double_quant=True if args.load_in_4bit else None,
            bnb_4bit_compute_dtype=torch.float16 if args.load_in_4bit else None,
        )
        qwen_extra['quantization_config'] = bnb_config
        if args.device_map:
            qwen_extra['device_map'] = args.device_map
        else:
            # 默认单进程情况下自动放置
            qwen_extra['device_map'] = 'auto'
    else:
        if args.device_map:
            qwen_extra['device_map'] = args.device_map

    # Monkey-patch: pass through via environment variables consumed inside model class if needed.
    # For simplicity, we modify model class to accept **qwen_extra by temporarily attaching to torch hub (not modifying class signature here).
    # Instead, we will dynamically subclass if extra kwargs exist.

    if qwen_extra:
        from qwen3_audio.modeling_qwen3_audio import Qwen3WhisperForConditionalGeneration as BaseCls
        class PatchedQwen(BaseCls):
            def __init__(self, *a, **kw):
                extra = kw.pop('qwen_extra')
                self._qwen_extra = extra
                super().__init__(*a, **kw)
            def _load_qwen(self, name):
                from transformers import AutoModelForCausalLM
                return AutoModelForCausalLM.from_pretrained(name, low_cpu_mem_usage=True, **self._qwen_extra)
        # Rebuild model using patched class
        model = PatchedQwen(
            whisper_model_name=args.whisper_model,
            qwen3_model_name=args.qwen3_model,
            projector_hidden=args.projector_hidden,
            projector_layers=args.projector_layers,
            freeze_whisper=args.freeze_whisper,
            train_qwen3_layers=args.train_qwen3_layers,
            qwen_extra=qwen_extra,
        )
    else:
        model = Qwen3WhisperForConditionalGeneration(
            whisper_model_name=args.whisper_model,
            qwen3_model_name=args.qwen3_model,
            projector_hidden=args.projector_hidden,
            projector_layers=args.projector_layers,
            freeze_whisper=args.freeze_whisper,
            train_qwen3_layers=args.train_qwen3_layers,
        )

    if args.whisper_cpu:
        # 强制把 whisper 放到 cpu，减少 GPU 显存 (编码开销较小)
        model.whisper.to('cpu')

    # 标记是否使用手动 loss 计算
    model.manual_loss = getattr(args, 'manual_loss', False)
    model.projector_scale = args.projector_scale
    model.projector_fp32 = args.projector_fp32
    model.freeze_projector_steps = args.freeze_projector_steps

    # 如果使用 k-bit 量化，需要先调用 prepare_model_for_kbit_training
    if (args.load_in_8bit or args.load_in_4bit) and prepare_model_for_kbit_training is not None:
        model.qwen3 = prepare_model_for_kbit_training(model.qwen3, use_gradient_checkpointing=True)

    if args.use_lora:
        patterns = [p.strip() for p in args.lora_patterns.split(',') if p.strip()]
        target_modules = []
        for name, module in model.qwen3.named_modules():
            if isinstance(module, torch.nn.Linear):
                if any(p in name for p in patterns):
                    target_modules.append(name)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias='none',
            task_type='CAUSAL_LM'
        )
        model.qwen3 = get_peft_model(model.qwen3, lora_config)
        # LoRA 稳定化处理：缩放初始化 & fp32
        lora_params = []
        for n,p in model.qwen3.named_parameters():
            if 'lora_A' in n or 'lora_B' in n:
                lora_params.append(p)
                if args.lora_init_scale != 1.0:
                    with torch.no_grad():
                        p.mul_(args.lora_init_scale)
                if args.lora_fp32 and p.dtype != torch.float32:
                    # 重新注册为 fp32 参数：创建新 Parameter 替换
                    with torch.no_grad():
                        new_p = torch.nn.Parameter(p.data.float(), requires_grad=True)
                    parent = model.qwen3
                    # 通过名称路径找到父模块并替换属性
                    name_parts = n.split('.')
                    obj = parent
                    for part in name_parts[:-1]:
                        obj = getattr(obj, part)
                    setattr(obj, name_parts[-1], new_p)
        model._lora_param_names = [n for n,_ in model.qwen3.named_parameters() if 'lora_A' in n or 'lora_B' in n]

    # Do not move to GPU manually, accelerate will handle it
    # model = model.to('cuda')

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=1,
        save_strategy='epoch',
        fp16=args.fp16,  # 仅在用户显式传入 --fp16 时启用 scaler，避免 grad unscale 错误
        bf16=args.bf16,
        dataloader_num_workers=0,  # reduce worker complexity while debugging
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_grad_norm=args.max_grad_norm,
        ddp_find_unused_parameters=False,
    )

    # Trainer expects model to return dict with loss when labels provided.
    # For DeepSpeed, use the model directly without wrapper
    class LossDebugCallback(TrainerCallback):
        def __init__(self):
            self.step = 0
        def on_step_end(self, args, state, control, **kwargs):
            self.step += 1
            # batch may contain custom fields via kwargs["inputs"]
            batch = kwargs.get('inputs', {})
            vlc = batch.get('valid_label_count', None)
            if vlc is not None:
                try:
                    total_tokens = batch['labels'].size(1)
                    vlc_list = vlc.tolist()
                    print(f"[Debug][step={self.step}] valid_label_count per sample: {vlc_list} / seq_len={total_tokens}")
                except Exception as e:
                    print(f"[Debug] valid_label_count print error: {e}")
            # Logits statistics (only compute lightweight stats to avoid overhead)
            # Access last model outputs from state.log_history? Not directly; we use callback on_log below.
            return control
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and 'loss' in logs:
                print(f"[Debug][log] step={state.global_step} loss={logs['loss']}")
            return control

    class NaNMonitorCallback(TrainerCallback):
        def __init__(self, model_ref, save_dir: str, max_snapshots: int = 2):
            self.model_ref = model_ref
            self.snapshots = 0
            self.max_snapshots = max_snapshots
            self.save_dir = save_dir
            os.makedirs(save_dir, exist_ok=True)
            self.optimizer = None
        def on_train_begin(self, args, state, control, **kwargs):
            # 获取 optimizer 引用
            self.optimizer = kwargs.get('optimizer', None)
        def on_step_end(self, args, state, control, **kwargs):
            model = self.model_ref
            took_snapshot = False
            with torch.no_grad():
                # 检查可训练参数是否出现 NaN
                has_nan = False
                for name, p in model.named_parameters():
                    if p.requires_grad and torch.isnan(p.data).any():
                        print(f"[NaNMonitor] Detected NaN in parameter: {name}")
                        has_nan = True
                        break
                if hasattr(model, 'last_logits') and model.last_logits is not None:
                    if torch.isnan(model.last_logits).any():
                        print('[NaNMonitor] last_logits contains NaN')
                        has_nan = True
                if has_nan and self.snapshots < self.max_snapshots:
                    snap = {
                        'step': state.global_step,
                        'loss_log': state.log_history[-1] if state.log_history else None,
                    }
                    try:
                        torch.save(snap, os.path.join(self.save_dir, f'nan_snapshot_{self.snapshots}.pt'))
                        print(f"[NaNMonitor] Saved snapshot {self.snapshots}")
                        self.snapshots += 1
                        took_snapshot = True
                    except Exception as e:
                        print(f"[NaNMonitor] Failed to save snapshot: {e}")
            if took_snapshot:
                control.should_training_stop = True
            return control

    class ProjectorFreezeHook(TrainerCallback):
        def __init__(self, model_ref):
            self.model_ref = model_ref
        def on_step_begin(self, args, state, control, **kwargs):
            if getattr(self.model_ref, 'freeze_projector_steps', 0) > 0 and state.global_step < self.model_ref.freeze_projector_steps:
                for n,p in self.model_ref.projector.named_parameters():
                    if p.grad is not None:
                        p.grad.zero_()
            return control

    class LoRAStabilizeCallback(TrainerCallback):
        def __init__(self, model_ref, early_steps: int, early_factor: float, lora_grad_clip: float):
            self.model_ref = model_ref
            self.early_steps = early_steps
            self.early_factor = early_factor
            self.lora_grad_clip = lora_grad_clip
            self.base_lrs = None
            self.lora_params_cached = None
        def on_train_begin(self, args, state, control, **kwargs):
            # Cache optimizer param groups initial lr
            self.optimizer = kwargs.get('optimizer', None)
            if self.optimizer:
                self.base_lrs = [g['lr'] for g in self.optimizer.param_groups]
            # Collect LoRA params
            lora_params = []
            for n,p in self.model_ref.named_parameters():
                if 'lora_A' in n or 'lora_B' in n:
                    lora_params.append(p)
            self.lora_params_cached = lora_params
            return control
        def on_step_begin(self, args, state, control, **kwargs):
            if self.optimizer and self.early_steps > 0 and state.global_step < self.early_steps:
                for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                    g['lr'] = base_lr * self.early_factor
            elif self.optimizer and self.base_lrs and self.early_steps > 0 and state.global_step == self.early_steps:
                # restore
                for g, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                    g['lr'] = base_lr
            return control
        def on_step_end(self, args, state, control, **kwargs):
            # LoRA 专属梯度裁剪 & NaN guard
            if self.lora_grad_clip > 0 and self.lora_params_cached:
                import math
                total_norm = 0.0
                for p in self.lora_params_cached:
                    if p.grad is None: continue
                    param_norm = p.grad.data.float().norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                clip_coef = self.lora_grad_clip / (total_norm + 1e-6)
                if clip_coef < 1.0:
                    for p in self.lora_params_cached:
                        if p.grad is not None:
                            p.grad.data.mul_(clip_coef)
                if state.global_step <= 5:
                    print(f"[LoRAStabilize] step={state.global_step} lora_grad_norm={total_norm:.2f} clip_coef={min(1.0, clip_coef):.4f}")
            # Replace NaN grads
            if self.lora_params_cached:
                replaced = 0
                for p in self.lora_params_cached:
                    if p.grad is not None and torch.isnan(p.grad).any():
                        p.grad.data.zero_()
                        replaced += 1
                if replaced > 0:
                    print(f"[LoRAStabilize] Replaced NaN grads in {replaced} LoRA params")
            return control

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
        callbacks=[
            LossDebugCallback(),
            NaNMonitorCallback(model, os.path.join(args.output_dir, 'nan_snapshots')),
            ProjectorFreezeHook(model),
            LoRAStabilizeCallback(
                model,
                early_steps=args.early_lr_steps,
                early_factor=args.early_lr_factor,
                lora_grad_clip=args.lora_grad_clip,
            ),
        ],
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save projector separately for clarity
    torch.save(model.projector.state_dict(), os.path.join(args.output_dir, 'projector.pt'))

if __name__ == '__main__':
    main()
