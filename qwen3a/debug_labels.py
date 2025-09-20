#!/usr/bin/env python
"""Quick script to inspect valid label token counts for first N batches.
Usage:
  python debug_labels.py --train_json path.json [--num_batches 5]
Assumes conversation mode.
"""
import argparse, json, torch
from transformers import WhisperFeatureExtractor, AutoTokenizer
from qwen3_audio import ConversationAudioDataset, ConversationCollator


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--whisper_model', type=str, default='/root/autodl-tmp/whisper-large-v3-turbo')
    ap.add_argument('--qwen3_model', type=str, default='/root/autodl-tmp/qwen3')
    ap.add_argument('--train_json', type=str, required=True)
    ap.add_argument('--num_batches', type=int, default=5)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--no_mask', action='store_true')
    return ap.parse_args()


def main():
    args = parse_args()
    fe = WhisperFeatureExtractor.from_pretrained(args.whisper_model)
    tokenizer = AutoTokenizer.from_pretrained(args.qwen3_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    with open(args.train_json, 'r') as f:
        data_list = json.load(f)
    ds = ConversationAudioDataset(data_list, fe, tokenizer, max_audio_seconds=None)
    collator = ConversationCollator(tokenizer, no_mask=args.no_mask, debug=False)

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    for bi, batch in enumerate(dl):
        vlc = batch.get('valid_label_count')
        if vlc is None:
            print('Batch missing valid_label_count; update ConversationCollator.')
            break
        print(f"[debug_labels] batch={bi} valid_label_count={vlc.tolist()} seq_len={batch['labels'].size(1)}")
        if bi + 1 >= args.num_batches:
            break

if __name__ == '__main__':
    main()
