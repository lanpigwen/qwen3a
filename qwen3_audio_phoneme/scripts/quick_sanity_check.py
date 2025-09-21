import torch
from torch.utils.data import DataLoader
from qwen3_audio_phoneme.qwen3_audio.data.dataset_conversations import ConversationAudioDataset
from qwen3_audio_phoneme.qwen3_audio.data.collator_conversations_phoneme import ConversationPhonemeCollator
from qwen3_audio_phoneme.qwen3_audio.register_model import build_model_and_tokenizer

RAW_DATA = "qwen3_audio_phoneme/data/raw_conversations.json"
BASE = "Qwen/Qwen3-7B-Instruct"
WHISPER = "openai/whisper-large-v3"
EXT_TOKENIZER = "qwen3_audio_phoneme/output/extended_tokenizer"
AUDIO_TOKEN = "<|audio|>"

def main():
    dataset = ConversationAudioDataset(RAW_DATA)
    model, tokenizer = build_model_and_tokenizer(
        base_model=BASE,
        whisper_model_id=WHISPER,
        audio_token=AUDIO_TOKEN,
        freeze_audio_encoder=True,
        replace_strategy="prefix",
        dtype="bfloat16",
        extended_tokenizer_dir=EXT_TOKENIZER
    )
    collator = ConversationPhonemeCollator(tokenizer)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collator)
    batch = next(iter(loader))
    for k, v in batch.items():
        print(k, v.shape)
    model.eval()
    with torch.no_grad():
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            audio_input_features=batch["audio_input_features"]
        )
        print("Loss:", out.loss.item())

if __name__ == "__main__":
    main()
