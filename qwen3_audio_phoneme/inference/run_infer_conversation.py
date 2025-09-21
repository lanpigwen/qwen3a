import argparse
import torch
import torchaudio
from qwen3_audio_phoneme.qwen3_audio.register_model import build_model_and_tokenizer

def build_mel(wav, sr, target_sr=16000):
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    mel_extractor = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    )
    spec = mel_extractor(wav)
    spec = torch.log(torch.clamp(spec, min=1e-5))
    return spec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--whisper_model", default="openai/whisper-large-v3")
    parser.add_argument("--audio_file", required=True)
    parser.add_argument("--prompt", default="请你做语音评测")
    parser.add_argument("--audio_token", default="<|audio|>")
    parser.add_argument("--extended_tokenizer_dir", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokenizer = build_model_and_tokenizer(
        base_model=args.model_dir,
        whisper_model_id=args.whisper_model,
        audio_token=args.audio_token,
        freeze_audio_encoder=True,
        replace_strategy="prefix",
        dtype="bfloat16",
        extended_tokenizer_dir=args.extended_tokenizer_dir
    )
    model.to(device)
    model.eval()

    wav, sr = torchaudio.load(args.audio_file)
    wav = wav[0]
    spec = build_mel(wav, sr).unsqueeze(0).to(device)

    full_prompt = args.prompt.rstrip() + "\n" + args.audio_token + "\n"
    batch = tokenizer(full_prompt, return_tensors="pt")
    input_ids = batch["input_ids"].to(device)
    attn = batch["attention_mask"].to(device)

    with torch.no_grad():
        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            audio_input_features=spec,
            max_new_tokens=args.max_new_tokens
        )
    text = tokenizer.decode(gen[0], skip_special_tokens=True)
    print("=== Generated ===")
    print(text)

if __name__ == "__main__":
    main()
