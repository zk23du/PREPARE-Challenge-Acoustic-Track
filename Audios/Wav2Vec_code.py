
from lib import *
class Wav2Vec2ModelForClassification(nn.Module): #facebook/wav2vec2-large-960h-lv60-self
    def __init__(self, model_name="facebook/wav2vec2-base", num_labels=3, device="cuda:2"):
        super(Wav2Vec2ModelForClassification, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.num_labels = num_labels

        # Adding a classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.wav2vec2_model.config.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_labels)
        )

    def forward(self, audio, sampling_rate):
        if sampling_rate != 16000:
            raise ValueError(f"Sampling rate must be 16000, but got {sampling_rate}")

        # Ensure audio is on the CPU and converted to NumPy
        if audio.is_cuda:
            audio = audio.cpu()
        audio = audio.numpy()

        # Preprocess audio
        inputs = self.processor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}

        # Extract features from Wav2Vec2 model
        with torch.no_grad():  # Ensure no gradients are calculated
            features = self.wav2vec2_model(**inputs)

        # Pool Wav2Vec2 outputs over the time dimension
        pooled_output = features.last_hidden_state.mean(dim=1)

        # Classification head
        return self.classifier(pooled_output)
    
    def extract_embedding(self, audio_path):
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load {audio_path}: {e}")
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        waveform = waveform / waveform.abs().max()
        
        if waveform.size(1) < self.target_length:
            padding = self.target_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.target_length]
        
        waveform = waveform.squeeze(0).numpy()
        
        inputs = self.feature_extractor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Extract embeddings from last_hidden_state
            pooled_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # Take mean over sequence dimension

            #print(f"Shape of last_hidden_state: {outputs.last_hidden_state.shape}")  # Debug line
            #print(f"Shape of pooled_embedding: {pooled_embedding.shape}")  # Debug line

        return pooled_embedding


