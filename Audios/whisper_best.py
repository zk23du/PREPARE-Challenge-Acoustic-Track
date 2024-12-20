from lib import *
class WhisperEmbeddingExtractor:
    def __init__(self, model_name="large", device="cuda:1"):
        self.model = whisper.load_model(model_name).to(device)
        self.device = device
        self.target_length = 16000 * 30  # 30 seconds at 16 kHz

        # Print model details
        print("Loaded Whisper Model")
        print(f"Model name: {model_name}")
        print(f"Model architecture:\n{self.model}")
        print(f"Encoder parameters: {sum(p.numel() for p in self.model.encoder.parameters()):,}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Device: {self.device}")

    def extract_embedding(self, audio_path):
        # Load and preprocess the audio
        waveform, sample_rate = torchaudio.load(audio_path)
        #print(f"Original waveform shape: {waveform.shape}, Sample rate: {sample_rate}")
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        waveform = waveform / waveform.abs().max()
        #print(f"Resampled waveform shape: {waveform.shape}")
        
        # Pad or truncate the waveform to the target length
        if waveform.size(1) < self.target_length:
            padding = self.target_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.target_length]
        #print(f"Padded/Truncated waveform shape: {waveform.shape}")
        # Generate the log-mel spectrogram
        audio = waveform.squeeze(0).numpy()
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        #print(f"Log-mel spectrogram shape: {mel.shape}")
        
        # Ensure the mel spectrogram has the required number of channels
        if mel.shape[0] < 128:  # Pad channels if they are fewer than 128
            mel = torch.nn.functional.pad(mel, (0, 0, 0, 128 - mel.shape[0]), mode='constant', value=0)
        elif mel.shape[0] > 128:  # Truncate channels if they exceed 128
            mel = mel[:128, :]

        # Pass the mel spectrogram to the Whisper encoder
        with torch.no_grad():
            encoded = self.model.encoder(mel.unsqueeze(0).to(self.device))
            #print(f"Encoder output shape: {encoded.shape}")
            embedding = encoded.mean(dim=1).squeeze(0).cpu().numpy()
            #print(f"Final embedding shape: {embedding.shape}")
            
        return embedding






