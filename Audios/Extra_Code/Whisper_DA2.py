from lib import *
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Data Augmentation
def spec_augment(mel):
    """Apply time masking and frequency masking to the Mel spectrogram."""
    time_mask = TimeMasking(time_mask_param=80)
    freq_mask = FrequencyMasking(freq_mask_param=30)
    return time_mask(freq_mask(mel))

# Dataset
class AudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, target_length=16000*30):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.target_length = target_length  # 30 seconds at 16kHz
        self.resampler = Resample(orig_freq=48000, new_freq=16000)

    def __len__(self):
        return len(self.data)

    def load_audio(self, audio_path):
        """Load and normalize the audio file."""
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = self.resampler(waveform)
        waveform = waveform / waveform.abs().max()  # Normalize
        if waveform.size(1) < self.target_length:
            pad = self.target_length - waveform.size(1)
            waveform = F.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :self.target_length]
        return waveform

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.data.iloc[idx, 0] + ".mp3")
        label = torch.tensor(self.data.iloc[idx, 1:].values.astype(float), dtype=torch.float)
        waveform = self.load_audio(audio_path)
        mel = whisper.log_mel_spectrogram(waveform.squeeze(0).numpy())
        mel = spec_augment(mel)  # Apply SpecAugment
        return mel, label


# Whisper-based Embedding Extractor
class WhisperEmbeddingExtractor:
    def __init__(self, model_name="openai/whisper-small", device="cuda:0"):
        self.whisper = WhisperModel.from_pretrained(model_name).to(device)
        self.device = device

    def extract_embedding(self, mel):
        """Extract embeddings from Whisper encoder."""
        mel = mel.unsqueeze(0).to(self.device)
        with torch.no_grad():
            encoded = self.whisper.encoder(mel).last_hidden_state
            embedding = encoded.mean(dim=1).squeeze(0)  # Pooling the last hidden state
        return embedding


# CNN-based Classifier
class WhisperClassifier(nn.Module):
    def __init__(self, model_name="openai/whisper-small", num_classes=3):
        super(WhisperClassifier, self).__init__()
        self.whisper = WhisperModel.from_pretrained(model_name)
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=self.whisper.config.hidden_size, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_features):
        outputs = self.whisper.encoder(input_features).last_hidden_state
        outputs = outputs.permute(0, 2, 1)  
        features = self.cnn_layers(outputs) 
        features = features.view(features.size(0), -1)  
        logits = self.fc(features)  
        return logits

# Training function
def train_model(model, dataloader, criterion, optimizer, scheduler, device, epochs=25):
    model.train()
    best_loss = float("inf")
    for epoch in range(epochs):
        total_loss = 0
        for mel, labels in dataloader:
            mel, labels = mel.to(device), labels.to(device)
            optimizer.zero_grad()
            #mel, labels = mixup_data(mel, labels)  # Data augmentation
            outputs = model(mel)  # Pass mel directly (no unsqueeze)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_whisper_model.pth")
            print("Best model saved.")
    return model

# Evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_labels, all_probs = [], []
    with torch.no_grad():
        for mel, labels in dataloader:
            mel, labels = mel.to(device), labels.to(device)
            outputs = model(mel)  # Pass mel directly (no unsqueeze)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == labels).all(dim=1).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    logloss = log_loss(all_labels, all_probs)
    accuracy = correct / total
    print(f"Eval Loss: {total_loss / len(dataloader):.4f}, Log Loss: {logloss:.4f}, Accuracy: {accuracy:.4f}")
    return accuracy

# Predict and save predictions
def predict_and_save(model, dataloader, device, output_csv):
    model.eval()
    predictions = []
    with torch.no_grad():
        for mel, _ in dataloader:
            mel = mel.to(device)
            outputs = model(mel)  # Pass mel directly (no unsqueeze)
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.cpu().numpy())
    # Save predictions
    df = pd.DataFrame(predictions, columns=["diagnosis_control", "diagnosis_mci", "diagnosis_adrd"])
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

# Main Pipeline
def main_pipeline(train_csv, train_audio_dir, test_csv, test_audio_dir, output_csv, epochs=25, batch_size=8):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets and Dataloaders
    train_dataset = AudioDataset(train_csv, train_audio_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = AudioDataset(test_csv, test_audio_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = WhisperClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    print("Starting Training...")
    model = train_model(model, train_loader, criterion, optimizer, scheduler, device, epochs)

    print("Evaluating model...")
    evaluate_model(model, train_loader, criterion, device)

    print("Predicting on test set...")
    predict_and_save(model, test_loader, device, output_csv)

if __name__ == "__main__":
    TRAIN_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_labels.csv"
    TRAIN_AUDIO_DIR = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_audios"
    TEST_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission_format.csv"
    TEST_AUDIO_DIR = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/test_audios"
    OUTPUT_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/whisper_testing_medium.csv"
    main_pipeline(TRAIN_CSV, TRAIN_AUDIO_DIR, TEST_CSV, TEST_AUDIO_DIR, OUTPUT_CSV)
