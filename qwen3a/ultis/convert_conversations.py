"""Utility to convert conversation style JSON (with audio placeholder) into flat list if needed.
Example input (conv_train.json):
[
  {
    "conversations": [
      {"from": "user", "value": "<audio>/path/a.wav</audio>请你做语音评测"},
      {"from": "assistant", "value": "sil sh iy ..."}
    ]
  }
]
This script can optionally filter or validate audio paths.
"""
import json, argparse, os, sys

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--check-audio', action='store_true', help='Warn if audio file not found')
    return ap.parse_args()


def main():
    args = parse_args()
    data = json.load(open(args.input, 'r'))
    cleaned = []
    missing = 0
    for ex in data:
        conv = ex.get('conversations', [])
        cleaned.append({'conversations': conv})
        if args.check_audio:
            # naive search first occurrence of <audio>path</audio>
            import re
            for turn in conv:
                m = re.search(r'<audio>(.*?)</audio>', turn.get('value',''))
                if m:
                    path = m.group(1).strip()
                    if not os.path.exists(path):
                        missing += 1
                    break
    json.dump(cleaned, open(args.output, 'w'), ensure_ascii=False, indent=2)
    if args.check_audio:
        print(f'Missing audio files: {missing}')

if __name__ == '__main__':
    main()
