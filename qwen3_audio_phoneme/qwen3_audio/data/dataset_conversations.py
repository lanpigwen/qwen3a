import json
import re
from torch.utils.data import Dataset

AUDIO_TAG_RE = re.compile(r"<audio>(.*?)</audio>")

class ConversationAudioDataset(Dataset):
    def __init__(self, path: str):
        self.samples = []
        self._load(path)
        if not self.samples:
            raise ValueError("No valid samples parsed.")

    def _load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            head = f.read(1)
            f.seek(0)
            if head == "[":
                arr = json.load(f)
                for obj in arr:
                    parsed = self._parse_obj(obj)
                    if parsed:
                        self.samples.append(parsed)
            else:
                for line in f:
                    line=line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    parsed = self._parse_obj(obj)
                    if parsed:
                        self.samples.append(parsed)

    def _parse_obj(self, obj):
        conv = obj.get("conversations", [])
        if len(conv) < 2:
            return None
        user = conv[0].get("value", "")
        asst = conv[1].get("value", "")
        m = AUDIO_TAG_RE.search(user)
        if not m:
            return None
        audio_path = m.group(1).strip()
        prompt_text = AUDIO_TAG_RE.sub("", user).strip()
        if not prompt_text:
            prompt_text = "请对该音频进行语音评测"
        return {
            "audio_path": audio_path,
            "prompt": prompt_text,
            "response": asst.strip()
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
