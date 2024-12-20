from lib import *
class WavLMEmbeddingExtractor:
    def __init__(self, model_name="microsoft/wavlm-large", device="cuda:1"):
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = WavLMModel.from_pretrained(model_name).to(device)
        self.device = device

        print("Loaded WavLM Model")
        print(f"Model name: {model_name}")
        print(f"Model architecture:\n{self.model}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")

    def extract_embedding(self, audio_path):
        # Load and preprocess the audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        waveform = waveform / waveform.abs().max()
        waveform = waveform.squeeze(0).numpy()

        # Extract features instead of using a tokenizer
        inputs = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use the last hidden state of the WavLM encoder as the embedding
        hidden_states = outputs.last_hidden_state
        embedding = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()  # Mean pool over the time dimension
        return embedding


