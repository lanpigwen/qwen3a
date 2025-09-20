import argparse
import argparse
import json
import os
import torch
from transformers import WhisperFeatureExtractor, AutoTokenizer, Trainer, TrainingArguments
from qwen3_audio import Qwen3WhisperForConditionalGeneration, ConversationAudioDataset, ConversationCollator

# Optional SwanLab integration
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:  # pragma: no cover
    swanlab = None
    SWANLAB_AVAILABLE = False

    if SWANLAB_AVAILABLE:
        from transformers import TrainerCallback

        class SwanLabCallback(TrainerCallback):
            """Minimal SwanLab integration for HuggingFace Trainer.

            Logs metrics coming from Trainer's logging dict. Avoids re-logging step keys.
            """
            def __init__(self, run):
                self.run = run
                self._logged_steps = set()

            def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
                if not logs:
                    return
                # Remove internal keys not needed
                clean = {}
                for k, v in logs.items():
                    if k.startswith('_'):  # internal
                        continue
                    # HF adds 'total_flos' etc.
                    if isinstance(v, (int, float)):
                        clean[k] = v
                if clean:
                    swanlab.log(clean)
                return

            def on_train_end(self, args, state, control, **kwargs):  # noqa: D401
                # Mark training finished
                swanlab.log({'train/finished': 1})
                return


def parse_args():
    ap = argparse.ArgumentParser(description='Simplified projector-only finetune script')
    ap.add_argument('--whisper_model', type=str, required=True, help='Path or id for Whisper feature extractor')
    ap.add_argument('--qwen3_model', type=str, required=True, help='Path or id for Qwen3 base model')
    ap.add_argument('--train_json', type=str, required=True, help='Training json path')
    ap.add_argument('--output_dir', type=str, default='./outputs')
    
    ap.add_argument('--per_device_train_batch_size', type=int, default=1)
    ap.add_argument('--gradient_accumulation_steps', type=int, default=1)
    ap.add_argument('--learning_rate', type=float, default=5e-5)
    ap.add_argument('--num_train_epochs', type=float, default=1.0)
    ap.add_argument('--freeze_whisper',default=True, action='store_true', help='Freeze Whisper encoder')
    ap.add_argument('--freeze_projector_steps', type=int, default=0, help='Freeze projector grads for first N steps')
    ap.add_argument('--projector_fp32', default=True,action='store_true', help='Run projector in fp32 during forward pass')
    ap.add_argument('--projector_scale', type=float, default=1.0, help='Scale factor for projector outputs')
    # SwanLab logging
    ap.add_argument('--use_swanlab', action='store_true', help='Enable SwanLab experiment tracking')
    ap.add_argument('--swanlab_project', type=str, default='qwen3a', help='SwanLab project name')
    ap.add_argument('--swanlab_run_name', type=str, default='1epoch', help='SwanLab run name (optional)')
    # Learning rate scheduler
    ap.add_argument('--lr_scheduler_type', type=str, default='cosine', help='Learning rate scheduler type (e.g., linear, cosine)')
    ap.add_argument('--warmup_ratio', type=float, default=0.03, help='Warmup ratio for learning rate scheduler (0 to disable warmup)')
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(42)

    swanlab_run = None
    if args.use_swanlab:
        if not SWANLAB_AVAILABLE:
            print('[WARN] --use_swanlab 指定但未安装 swanlab: pip install swanlab')
        else:
            swanlab_run = swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_run_name,
                config={k: v for k, v in vars(args).items() if k not in ['use_swanlab']},
                # auto detect offline mode via env if needed
            )
            print('[INFO] SwanLab run initialized.')

    # Load feature extractor and tokenizer
    fe = WhisperFeatureExtractor.from_pretrained(args.whisper_model)
    tokenizer = AutoTokenizer.from_pretrained(args.qwen3_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    with open(args.train_json, 'r') as f:
        data_list = json.load(f)

    dataset = ConversationAudioDataset(
        data_list,
        fe,
        tokenizer,
        max_audio_seconds=None,
    )
    collator = ConversationCollator(tokenizer)

    # Build model (uses underlying class that wraps Whisper + Qwen3)
    model = Qwen3WhisperForConditionalGeneration(
        whisper_model_name=args.whisper_model,
        qwen3_model_name=args.qwen3_model,
        freeze_whisper=args.freeze_whisper,
    )

    # Freeze all params except projector
    for n, p in model.named_parameters():
        p.requires_grad = False
    if hasattr(model, 'projector'):
        for p in model.projector.parameters():
            p.requires_grad = True
    else:
        raise RuntimeError('Model has no projector module')

    # Apply projector runtime options
    model.projector_fp32 = args.projector_fp32
    model.projector_scale = args.projector_scale
    model.freeze_projector_steps = args.freeze_projector_steps

    # Print trainable params
    print('\nTrainable parameters:')
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(' ', n, tuple(p.shape))

    # Note: default Trainer saving (safetensors) fails here because qwen3 and qwen_inner modules
    # share storage (weight tying). We'll disable automatic epoch saving and perform manual saves.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=1,
        max_grad_norm=1.0,
        save_strategy='no',  # manual save only
        save_safetensors=False,  # ensure torch.save usage when calling save_model
        remove_unused_columns=False,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        callbacks=([SwanLabCallback(swanlab_run)] if (args.use_swanlab and SWANLAB_AVAILABLE) else None),
    )

    trainer.train()

    # Minimal manual saving: only save projector + tokenizer to avoid duplicating tied base weights
    # If full model save is desired later, a custom untie or reference pruning step is needed.
    tokenizer.save_pretrained(args.output_dir)
    torch.save(model.projector.state_dict(), os.path.join(args.output_dir, 'projector.pt'))
    # Also persist a small config metadata for reproducibility
    meta = {
        'whisper_model': args.whisper_model,
        'qwen3_model': args.qwen3_model,
        'freeze_whisper': args.freeze_whisper,
        'projector_scale': args.projector_scale,
        'projector_fp32': args.projector_fp32,
        'freeze_projector_steps': args.freeze_projector_steps,
        'train_epochs': args.num_train_epochs,
        'learning_rate': args.learning_rate,
    }
    with open(os.path.join(args.output_dir, 'finetune_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('Saved projector weights and metadata to', args.output_dir)

    if swanlab_run is not None:
        # Log artifact paths (not uploading large binaries by default)
        swanlab.log({
            'artifact/projector_state_dict': os.path.join(args.output_dir, 'projector.pt'),
            'artifact/meta_json': os.path.join(args.output_dir, 'finetune_meta.json')
        })
        swanlab.finish()
        print('[INFO] SwanLab run finished.')


if __name__ == '__main__':
    main()