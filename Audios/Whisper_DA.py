from lib import *
torch.manual_seed(42)
np.random.seed(42)

# Data Normalization Functions
def normalize_waveform(waveform):
    # DC-Offset Removal
    waveform = waveform - waveform.mean()
    # Z-Score Standardization
    waveform = waveform / (waveform.std() + 1e-8)
    return waveform

def augment_waveform(waveform, sample_rate=16000):
    # Add Gaussian noise
    noise = torch.randn(waveform.shape) * 0.005
    waveform = waveform + noise
    return waveform

class WhisperEmbeddingExtractorDA:
    def __init__(self, model_name="large", device="cuda:0"):
        self.model = whisper.load_model(model_name).to(device)
        self.device = device
        self.target_length = 16000 * 30  # 30 seconds at 16 kHz

    def extract_embedding(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            
        waveform = normalize_waveform(waveform)
        waveform = augment_waveform(waveform)
        
        if waveform.size(1) < self.target_length:
            padding = self.target_length - waveform.size(1)
            waveform = F.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.target_length]
            
        audio = waveform.squeeze(0).numpy()
        
        # Convert to mel-spectrogram with 128 channels
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        
        # Ensure the mel spectrogram has the required 128 channels
        if mel.shape[0] < 128:  
            mel = F.pad(mel, (0, 0, 0, 128 - mel.shape[0]), mode='constant', value=0)
        elif mel.shape[0] > 128:  
            mel = mel[:128, :]
        
        #print(f"Mel Spectrogram Shape: {mel.shape}")  # Debug info
        
        with torch.no_grad():
            encoded = self.model.encoder(mel.unsqueeze(0).to(self.device))
            embedding = encoded.mean(dim=1).squeeze(0).cpu().numpy()
        return embedding


