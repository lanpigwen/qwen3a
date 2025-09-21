import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

class ConversationPhonemeCollator:
    def __init__(self,
        tokenizer,
        audio_token="<|audio|>",
        sample_rate=16000,
        n_mels=80,
        max_audio_seconds=30
    ):  
        self.tokenizer = tokenizer
        self.audio_token = audio_token
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_audio_len = sample_rate * max_audio_seconds
        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels
        )

    def _wav_to_log_mel(self, wav: torch.Tensor, sr: int):
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        if wav.size(0) > self.max_audio_len:
            wav = wav[:self.max_audio_len]
        spec = self.mel_extractor(wav)
        spec = torch.log(torch.clamp(spec, min=1e-5))
        return spec

    def __call__(self, batch):
        input_ids_list, label_ids_list, audio_feats_list = [], [], []
        for sample in batch:
            audio_path = sample["audio_path"]
            prompt = sample["prompt"]
            response = sample["response"]

            wav, sr = torchaudio.load(audio_path)
            wav = wav[0]
            mel = self._wav_to_log_mel(wav, sr)
            audio_feats_list.append(mel)

            text_prompt = prompt.rstrip() + "\n" + self.audio_token + "\n"
            enc_prompt = self.tokenizer(text_prompt, add_special_tokens=True)
            enc_resp = self.tokenizer(response, add_special_tokens=False)

            ids = enc_prompt["input_ids"] + enc_resp["input_ids"] + [self.tokenizer.eos_token_id]
            labels = [-100]*len(enc_prompt["input_ids"]) + enc_resp["input_ids"] + [self.tokenizer.eos_token_id]

            input_ids_list.append(torch.tensor(ids, dtype=torch.long))
            label_ids_list.append(torch.tensor(labels, dtype=torch.long))

        input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(label_ids_list, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        max_T = max(m.size(1) for m in audio_feats_list)
        feat_batch = []
        for m in audio_feats_list:
            pad_T = max_T - m.size(1)
            if pad_T > 0:
                m = torch.cat([m, torch.zeros(self.n_mels, pad_T)], dim=1)
            feat_batch.append(m)
        audio_input_features = torch.stack(feat_batch, dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "audio_input_features": audio_input_features
        }
