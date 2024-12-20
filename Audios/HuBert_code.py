from lib import *
class HubertEmbeddingExtractor: #facebook/hubert-base-ls960
    def __init__(self, model_name="facebook/hubert-large-ls960-ft", device="cuda:0"):
        self.model_name = model_name
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = HubertModel.from_pretrained(model_name).to(device)
        # self.feature_extractor = AutoProcessor.from_pretrained(model_name)  # Use feature extractor
        # self.model = HubertModel.from_pretrained(model_name).to(device)

        self.model.eval()  # Set model to evaluation mode
        self.device = device
        print(f"Loaded {model_name} for embedding extraction on {device}.")

    def extract_embedding(self, audio_path):
        # Load audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        #print(f"\nProcessing: {audio_path}")
        #print(f"Original waveform shape: {waveform.shape}, Sample rate: {sample_rate}")
        
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            #print(f"Resampled waveform shape: {waveform.shape}")
        
        # Normalize waveform
        if waveform.abs().max() == 0:
            #print(f"Warning: Invalid waveform detected in {audio_path}, replacing with zeros.")
            waveform = torch.zeros_like(waveform)
        else:
            waveform = waveform / waveform.abs().max()
        #print(f"Normalized waveform shape: {waveform.shape}")

        # Process input
        inputs = self.feature_extractor(waveform.squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        #print(f"Processed inputs: {inputs}")

        # Extract embeddings
        with torch.no_grad():
            outputs = self.model(**inputs).last_hidden_state
            #print(f"Raw embeddings shape: {outputs.shape}")

        # Mean pooling
        pooled_embedding = outputs.mean(dim=1).squeeze(0)
        #print(f"Pooled embedding shape: {pooled_embedding.shape}")
        return pooled_embedding



