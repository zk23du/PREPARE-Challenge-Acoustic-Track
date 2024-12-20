from lib import *
from classifier import CNNClassifier
# ==========================
# 1. Custom Embedding Extractor (ZFF + LPC + Segment Filtering)
# ==========================
class SpeechFeatureExtractor:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.target_length = 16000 * 4  # 4 seconds of audio at 16 kHz

    def zero_frequency_filtering(self, signal):
        """Apply zero-frequency filtering (ZFF) to the audio signal."""
        zff_signal = torch.zeros_like(signal)
        for i in range(2, len(signal)):
            zff_signal[i] = 2 * zff_signal[i-1] - zff_signal[i-2] + signal[i] - 2 * signal[i-1] + signal[i-2]
        return zff_signal

    def compute_autocorrelation(self, signal, lag):
        """Compute autocorrelation of the signal for a given lag."""
        autocorr = np.correlate(signal, signal, mode='full')
        mid = len(autocorr) // 2
        return autocorr[mid:mid + lag + 1]

    def levinson_durbin(self, r, order):
        """Compute LPC coefficients using Levinson-Durbin algorithm."""
        a = np.zeros(order + 1)
        e = r[0]
        a[0] = 1
        for i in range(1, order + 1):
            acc = sum([a[j] * r[i - j] for j in range(1, i + 1)])
            k = -acc / e
            new_a = a.copy()
            for j in range(1, i + 1):
                new_a[j] += k * a[i - j]
            a = new_a
            e *= (1 - k ** 2)
        return a

    def extract_linear_prediction_residual(self, signal, order=12):
        """Extract LPC residual from signal."""
        signal_np = signal.numpy()  # Convert to NumPy for autocorrelation and Levinson-Durbin
        autocorr = self.compute_autocorrelation(signal_np, order)
        lpc_coeffs = self.levinson_durbin(autocorr, order)
        lpc_coeffs = torch.tensor(lpc_coeffs, dtype=torch.float32)  # Convert to torch tensor
        
        # Calculate the residual signal
        signal_torch = torch.tensor(signal_np, dtype=torch.float32)
        residual = signal_torch.clone()
        for i in range(order, len(signal_torch)):
            predicted_value = torch.sum(lpc_coeffs[1:] * signal_torch[i - order:i])
            residual[i] -= predicted_value
        return residual

    def extract_features(self, audio_path):
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

        zff_signal = self.zero_frequency_filtering(waveform.squeeze(0))
        residual = self.extract_linear_prediction_residual(zff_signal)
        return residual.unsqueeze(0)



class AudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, feature_extractor):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.data.iloc[idx, 0] + ".mp3")
        label = torch.tensor(self.data.iloc[idx, 1:].values.astype(float), dtype=torch.float)
        features = self.feature_extractor.extract_features(audio_path)
        return features, label


def train_model(model, dataloader, criterion, optimizer, device, epochs=25):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {total_loss / len(dataloader):.4f}")
    return model

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    print(f"Evaluation Loss: {total_loss / len(dataloader):.4f}")

def predict_labels(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, _ in dataloader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.cpu().numpy())
    return predictions


def main_pipeline(train_csv, train_audio_dir, test_csv, test_audio_dir, output_csv, epochs=25, batch_size=16):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_extractor = SpeechFeatureExtractor(device=device)

    train_dataset = AudioDataset(train_csv, train_audio_dir, feature_extractor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = AudioDataset(test_csv, test_audio_dir, feature_extractor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    classifier = CNNClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    classifier = train_model(classifier, train_loader, criterion, optimizer, device, epochs)
       
    print("\nEvaluating model on test data...")
    evaluate_model(classifier, test_loader, criterion, device)

    print("\nPredicting test data...")
    predictions = predict_labels(classifier, test_loader, device)

    results = pd.DataFrame(predictions, columns=["diagnosis_control", "diagnosis_mci", "diagnosis_adrd"])
    results.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    TRAIN_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_labels.csv"
    TRAIN_AUDIO_DIR = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_audios"
    TEST_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission_format.csv"
    TEST_AUDIO_DIR = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/test_audios"
    OUTPUT_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/predicted_whisper_large_test_labels.csv"
    
    main_pipeline(TRAIN_CSV, TRAIN_AUDIO_DIR, TEST_CSV, TEST_AUDIO_DIR, OUTPUT_CSV)
