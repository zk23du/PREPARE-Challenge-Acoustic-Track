from lib import *

class UniSpeechEmbeddingExtractor:
    def __init__(self, model_name="microsoft/unispeech-large-1500h-cv", device="cuda:1"):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = UniSpeechModel.from_pretrained(model_name).to(device)
        self.device = device
        self.target_length = 16000 * 30  # 30 seconds at 16kHz
        print(f"Loaded UniSpeech model '{model_name}' for embedding extraction on {device}.")

    def extract_embedding(self, audio_path):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.numel() == 0:
                raise ValueError(f"Waveform is empty for {audio_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load {audio_path}: {e}")
        
        if sample_rate != 16000:
            waveform[torch.isnan(waveform)] = 0  # Remove NaNs before resampling
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            waveform[torch.isnan(waveform)] = 0  # Remove NaNs after resampling
        
        max_val = waveform.abs().max()
        if max_val == 0:
            raise ValueError(f"Max of waveform is zero for {audio_path}")
        waveform = waveform / (max_val + 1e-8)  # Avoid division by zero

        waveform = waveform.squeeze(0).numpy()
        inputs = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            pooled_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)

            if torch.isnan(pooled_embedding).any():
                print(f"NaN detected in embeddings from UniSpeech for {audio_path}")
                self.model = UniSpeechModel.from_pretrained("microsoft/unispeech-large-1500h-cv").to(self.device)

                pooled_embedding = torch.zeros_like(pooled_embedding)  # Return zeros instead of NaNs

        return pooled_embedding


