import json
import re
import argparse
from transformers import AutoTokenizer

AUDIO_TAG_RE = re.compile(r"<audio>(.*?)</audio>")

def iter_objs(path):
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            arr = json.load(f)
            for obj in arr:
                yield obj
        else:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                yield json.loads(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--base_tokenizer", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--min_freq", type=int, default=1)
    args = parser.parse_args()

    phoneme_freq = {}
    for obj in iter_objs(args.data):
        conv = obj.get("conversations", [])
        if len(conv) < 2:
            continue
        asst = conv[1].get("value", "").strip()
        for p in asst.split():
            if not p:
                continue
            phoneme_freq[p] = phoneme_freq.get(p, 0) + 1

    phonemes = [p for p,c in phoneme_freq.items() if c >= args.min_freq]

    tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=False)
    existing = set(tokenizer.get_vocab().keys())
    new_tokens = [p for p in phonemes if p not in existing]

    print(f"Collected phoneme types: {len(phonemes)}; New to add: {len(new_tokens)}")
    if new_tokens:
        tokenizer.add_tokens(new_tokens)
        tokenizer.save_pretrained(args.save_dir)
        print(f"Extended tokenizer saved to {args.save_dir}")
    else:
        print("No new tokens added.")

if __name__ == "__main__":
    main()
