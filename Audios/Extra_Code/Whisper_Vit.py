from lib import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class SpectrogramDataset(Dataset):
    def __init__(self, csv_file, audio_dir):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.data.iloc[idx, 0] + ".mp3")
        
        # Extract label (convert one-hot to class index)
        label = torch.tensor(self.data.iloc[idx, 1:].values.astype(float), dtype=torch.float)
        label = torch.argmax(label).long()  # Convert to class index (0, 1, 2)
        
        # Extract log-mel spectrogram using Whisper's function
        log_mel = self.convert_to_whisper_log_mel_spectrogram(audio_path)
        
        return log_mel, label

    def convert_to_whisper_log_mel_spectrogram(self, audio_path):
        """
        Use whisper's log-mel spectrogram function with fixed padding/truncation.
        """
        waveform, sample_rate = torchaudio.load(audio_path)

        if waveform.size(0) > 1:  # Convert stereo to mono
            waveform = waveform.mean(0, keepdim=True)

        # Convert waveform to numpy for whisper.log_mel_spectrogram
        waveform = waveform.squeeze(0).numpy()

        # Use Whisper's internal log-mel spectrogram function
        log_mel = whisper.log_mel_spectrogram(waveform)

        # Pad or truncate the spectrogram to a fixed size (80 x 9000)
        target_length = 9000
        if log_mel.size(1) < target_length:  # Pad if shorter
            padding = target_length - log_mel.size(1)
            log_mel = torch.nn.functional.pad(log_mel, (0, padding))
        else:  # Truncate if longer
            log_mel = log_mel[:, :target_length]

        return log_mel


class WhisperClassifier(nn.Module):
    def __init__(self):
        super(WhisperClassifier, self).__init__()
        self.encoder = whisper.load_model("small").encoder  # Load the encoder only
        for param in self.encoder.parameters():
            param.requires_grad = True  # Fine-tune Whisper encoder

        # Classifier on top of the encoder
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128), 
            nn.ReLU(),
            nn.Linear(128, 3)  # Assuming 3 output classes (Control, MCI, ADRD)
        )

    def forward(self, x):
        # Input shape: [batch_size, 80, frames]
        batch_size = x.size(0)  # Extract batch size
        encoded = self.encoder(x)  # Pass through Whisper encoder

        # Shape of `encoded` is [batch_size, frames, 1024]
        pooled_embedding = encoded.mean(dim=1)  # Global average pooling along the time dimension
        output = self.classifier(pooled_embedding)  # Pass through the classifier
        return output



def train_whisper_classifier(model, dataloader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for mel, label in dataloader:
            mel, label = mel.to(device), label.to(device)

            optimizer.zero_grad()
            
            # Forward pass
            output = model(mel)  # Remove any unsqueeze here
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(output, dim=1)
            correct_predictions += (preds == label).sum().item()
            total_samples += label.size(0)
        
        accuracy = correct_predictions / total_samples
        avg_loss = total_loss / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


def evaluate_whisper_classifier(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for mel, label in dataloader:
            mel, label = mel.to(device), label.to(device)
            
            output = model(mel)
            loss = criterion(output, label)
            total_loss += loss.item()

            preds = torch.argmax(output, dim=1)
            correct_predictions += (preds == label).sum().item()
            total_samples += label.size(0)

    accuracy = correct_predictions / total_samples
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    train_csv = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_labels.csv"
    train_audio_dir = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_audios"
    test_csv = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission_format.csv"
    test_audio_dir = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/test_audios"
    output_csv = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/predictions.csv"
    

    train_dataset = SpectrogramDataset(train_csv, train_audio_dir)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    test_dataset = SpectrogramDataset(test_csv, test_audio_dir)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WhisperClassifier().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    print("\nTraining Whisper-based model...")
    train_whisper_classifier(model, train_loader, criterion, optimizer, device, epochs=10)
    
    print("\nEvaluating Whisper-based model...")
    evaluate_whisper_classifier(model, test_loader, criterion, device)




