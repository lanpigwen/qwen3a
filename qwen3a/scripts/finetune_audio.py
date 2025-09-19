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
from peft import LoraConfig, get_peft_model

from src.qwen3_audio.modeling_qwen3_audio import Qwen3WhisperForConditionalGeneration
from src.qwen3_audio.data import (
    AudioTextDataset,
    AudioCollator,
    ConversationAudioDataset,
    ConversationCollator,
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--whisper_model', type=str, default='openai/whisper-small')
    ap.add_argument('--qwen3_model', type=str, required=True, help='Base Qwen3 model id/path')
    ap.add_argument('--train_json', type=str, required=True, help='List[ {"audio": path, "text": str, "prompt": optional str} ]')
    ap.add_argument('--output_dir', type=str, default='./outputs')
    ap.add_argument('--freeze_whisper', action='store_true')
    ap.add_argument('--train_qwen3_layers', type=int, default=0, help='Number of top Qwen3 layers to unfreeze (0=none)')
    ap.add_argument('--projector_layers', type=int, default=1)
    ap.add_argument('--projector_hidden', type=int, default=None)
    ap.add_argument('--per_device_train_batch_size', type=int, default=2)
    ap.add_argument('--per_device_eval_batch_size', type=int, default=2)
    ap.add_argument('--learning_rate', type=float, default=2e-4)
    ap.add_argument('--num_train_epochs', type=int, default=3)
    ap.add_argument('--warmup_ratio', type=float, default=0.03)
    ap.add_argument('--gradient_accumulation_steps', type=int, default=4)
    ap.add_argument('--fp16', action='store_true')
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--max_audio_seconds', type=float, default=None)
    ap.add_argument('--use_lora', action='store_true')
    ap.add_argument('--lora_r', type=int, default=16)
    ap.add_argument('--lora_alpha', type=int, default=32)
    ap.add_argument('--lora_dropout', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--data_mode', type=str, default='flat', choices=['flat','conversation'], help='flat: AudioTextDataset, conversation: ConversationAudioDataset')
    ap.add_argument('--user_prefix', type=str, default='<|user|>')
    ap.add_argument('--assistant_prefix', type=str, default='<|assistant|>')
    ap.add_argument('--system_prefix', type=str, default='<|system|>')
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
        dataset = AudioTextDataset(data_list, fe, tokenizer, max_audio_seconds=args.max_audio_seconds)
        collator = AudioCollator(tokenizer)
    else:
        dataset = ConversationAudioDataset(
            data_list,
            fe,
            tokenizer,
            max_audio_seconds=args.max_audio_seconds,
            user_prefix=args.user_prefix,
            assistant_prefix=args.assistant_prefix,
            system_prefix=args.system_prefix,
        )
        collator = ConversationCollator(
            tokenizer,
            user_prefix=args.user_prefix,
            assistant_prefix=args.assistant_prefix,
        )

    # Build model
    model = Qwen3WhisperForConditionalGeneration(
        whisper_model_name=args.whisper_model,
        qwen3_model_name=args.qwen3_model,
        projector_hidden=args.projector_hidden,
        projector_layers=args.projector_layers,
        freeze_whisper=args.freeze_whisper,
        train_qwen3_layers=args.train_qwen3_layers,
    )

    if args.use_lora:
        target_modules = []
        # choose typical linear layers in attention & MLP of Qwen3
        for name, module in model.qwen3.named_modules():
            if isinstance(module, torch.nn.Linear) and module.weight.requires_grad:
                if module.weight.shape[0] >= 1024:  # simple heuristic
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

    # Collect trainable params for logging
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({100*trainable/total:.2f}%)")

    # Trainer expects model to return dict with loss when labels provided.
    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, **batch):
            return self.m(**batch)

    wrapper = Wrapper(model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=10,
        save_strategy='epoch',
        fp16=args.fp16,
        bf16=args.bf16,
        dataloader_num_workers=2,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=wrapper,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save projector separately for clarity
    torch.save(model.projector.state_dict(), os.path.join(args.output_dir, 'projector.pt'))

if __name__ == '__main__':
    main()
