from lib import *

torch.cuda.empty_cache()
class WhisperConformerModel(nn.Module):
    def __init__(self, whisper_model_name="openai/whisper-large", num_labels=3, device="cuda:0"):
        super(WhisperConformerModel, self).__init__()
        self.device = device
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_model_name).to(device)

        # Conformer Block
        self.conformer = nn.Sequential(
        nn.Conv1d(1500, 256, kernel_size=3, padding=1),  # Adjusted input channels to match Whisper encoder output
        nn.ReLU(),
        nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=4, dim_feedforward=512, activation="relu", batch_first=True),
            num_layers=2
        ),
        nn.Linear(256, num_labels)  # Final output layer
    )

    def forward(self, audio, sampling_rate):
        if sampling_rate != 16000:
            raise ValueError(f"Expected sampling rate 16000, but got {sampling_rate}")

        # Preprocess audio: Move to CPU for NumPy conversion
        inputs = self.whisper_processor(audio.cpu().numpy(), sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract embeddings
        with torch.no_grad():
            encoder_outputs = self.whisper_model.model.encoder(inputs["input_features"])

        # Confirm encoder output shape
        print(f"Encoder output shape: {encoder_outputs.last_hidden_state.shape}")

        # Permute dimensions to match Conv1d input requirements
        embeddings = encoder_outputs.last_hidden_state.permute(0, 2, 1)  # (batch, embedding_dim, seq_len)

        # Process through Conformer
        outputs = self.conformer(embeddings)
        return outputs



class TrainAudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, max_length=480000):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uid = self.data.iloc[idx, 0]
        audio_path = os.path.join(self.audio_dir, f"{uid}.mp3")

        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()

        if waveform.size(1) < self.max_length:
            padding = self.max_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.max_length]

        label = torch.tensor(self.data.iloc[idx, 1:].values.astype(float), dtype=torch.float)
        return waveform.squeeze(0), 16000, label, uid


class TestAudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, max_length=480000):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        uid = self.data.iloc[idx, 0]
        audio_path = os.path.join(self.audio_dir, f"{uid}.mp3")

        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)

        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()

        if waveform.size(1) < self.max_length:
            padding = self.max_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.max_length]

        return waveform.squeeze(0), uid


def train_whisper_model(model, dataloader, criterion, optimizer, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for waveforms, sampling_rates, labels, _ in dataloader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(waveforms, sampling_rates[0])
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
    return model


def evaluate_whisper_model(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_samples, correct_predictions = 0, 0, 0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for waveforms, sampling_rates, labels, _ in dataloader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            outputs = model(waveforms, sampling_rates[0])
            probs = torch.sigmoid(outputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = (probs > 0.5).float()
            correct_predictions += (preds == labels).all(dim=1).sum().item()
            total_samples += labels.size(0)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct_predictions / total_samples
    avg_loss = total_loss / len(dataloader)
    multiclass_log_loss = log_loss(all_labels, all_probs)
    print(f"Validation Loss: {avg_loss:.4f}, Log Loss: {multiclass_log_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, multiclass_log_loss, accuracy


def predict_labels(test_loader, model, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for waveforms, _, uids in test_loader:
            waveforms = waveforms.to(device)
            outputs = model(waveforms, 16000)
            probs = torch.sigmoid(outputs)
            predictions.extend(zip(uids, probs.cpu().numpy()))
    return predictions


def save_predictions(predictions, output_csv):
    results = [{"uid": uid, "diagnosis_control": probs[0], "diagnosis_mci": probs[1], "diagnosis_adrd": probs[2]}
               for uid, probs in predictions]
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


def main_pipeline(train_csv, train_audio_dir, test_csv, test_audio_dir, output_csv, model_name, num_labels, epochs=15, batch_size=16):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model = WhisperConformerModel(whisper_model_name=model_name, num_labels=num_labels, device=device).to(device)
    train_loader = DataLoader(TrainAudioDataset(train_csv, train_audio_dir), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TestAudioDataset(test_csv, test_audio_dir), batch_size=batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model = train_whisper_model(model, train_loader, criterion, optimizer, device, epochs)
    avg_loss, log_loss, accuracy = evaluate_whisper_model(model, test_loader, criterion, device)
    predictions = predict_labels(test_loader, model, device)
    save_predictions(predictions, output_csv)


if __name__ == "__main__":
    main_pipeline(
        train_csv="/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_labels.csv",
        train_audio_dir="/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_audios",
        test_csv="/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission_format.csv",
        test_audio_dir="/home/hiddenrock/DDS/DataDrivenCompetition/Data/test_audios",
        output_csv="/home/hiddenrock/DDS/DataDrivenCompetition/Data/predicted_whisper_conformer_labels.csv",
        model_name="openai/whisper-large",
        num_labels=3
    )
