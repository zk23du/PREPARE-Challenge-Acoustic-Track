from lib import *
from Whisper_Best import WhisperEmbeddingExtractor
from classifier import Classifier
from predict_save import predict_labels, save_predictions

class TrainAudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, embedding_extractor, spectrogram_transform):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.embedding_extractor = embedding_extractor
        self.spectrogram_transform = spectrogram_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.data.iloc[idx, 0] + ".mp3")
        label = torch.tensor(self.data.iloc[idx, 1:].values.astype(float), dtype=torch.float)

        # Extract Whisper embeddings
        embedding = self.embedding_extractor.extract_embedding(audio_path)

        # Generate spectrogram for AST/CNN
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        spectrogram = self.spectrogram_transform(waveform)

        return embedding, spectrogram, label


class ASTClassifier(nn.Module):
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593", num_classes=3):
        super(ASTClassifier, self).__init__()
        self.model = ASTForAudioClassification.from_pretrained(model_name)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)  # Update output layer

    def forward(self, spectrogram):
        outputs = self.model(spectrogram)
        return outputs.logits
    
class CNNClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(CNNClassifier, self).__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=num_classes)

    def forward(self, spectrogram):
        spectrogram = spectrogram.unsqueeze(1)  # Add channel dimension
        return self.model(spectrogram)

class EnsembleModel(nn.Module):
    def __init__(self, whisper_model, ast_model, cnn_model, num_classes=3):
        super(EnsembleModel, self).__init__()
        self.whisper_model = whisper_model
        self.ast_model = ast_model
        self.cnn_model = cnn_model
        self.fc = nn.Linear(num_classes * 3, num_classes)  # Combine predictions from all models

    def forward(self, embedding, spectrogram):
        whisper_logits = self.whisper_model(embedding)
        ast_logits = self.ast_model(spectrogram)
        cnn_logits = self.cnn_model(spectrogram)

        combined_logits = torch.cat([whisper_logits, ast_logits, cnn_logits], dim=1)
        return self.fc(combined_logits)


def train_ensemble_model(model, dataloader, criterion, optimizer, device, epochs=25):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for embeddings, spectrograms, labels in dataloader:
            embeddings, spectrograms, labels = embeddings.to(device), spectrograms.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(embeddings, spectrograms)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {total_loss / len(dataloader):.4f}")
    return model


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for i, (embeddings, labels) in enumerate(dataloader):
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            # Model prediction
            outputs = model(embeddings)
            probs = torch.sigmoid(outputs)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Collect predictions and labels for log loss
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Calculate batch accuracy
            preds = (probs > 0.5).float()
            correct_predictions += (preds == labels).all(dim=1).sum().item()
            total_samples += labels.size(0)

    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    multiclass_log_loss = log_loss(all_labels, all_probs)
    accuracy = correct_predictions / total_samples

    print(f"[EVAL] Avg Loss: {avg_loss:.4f} | Multiclass Log Loss: {multiclass_log_loss:.4f} | Accuracy: {accuracy:.4f}")
    return avg_loss, multiclass_log_loss, accuracy


   
def main_pipeline(model_name, train_csv, train_audio_dir, test_csv, test_audio_dir, output_csv, trained_model_path, epochs=25, batch_size=16):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Whisper extractor and spectrogram transform
    embedding_extractor = WhisperEmbeddingExtractor(model_name, device=device)
    spectrogram_transform = T.MelSpectrogram(sample_rate=16000, n_mels=128)

    # Prepare datasets and dataloaders
    train_dataset = TrainAudioDataset(train_csv, train_audio_dir, embedding_extractor, spectrogram_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    whisper_model = Classifier(input_dim=128, output_dim=3).to(device)
    ast_model = ASTClassifier(num_classes=3).to(device)
    cnn_model = CNNClassifier(num_classes=3).to(device)
    ensemble_model = EnsembleModel(whisper_model, ast_model, cnn_model, num_classes=3).to(device)

    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(ensemble_model.parameters(), lr=1e-4)

    # Train the ensemble model
    print("Training ensemble model...")
    ensemble_model = train_ensemble_model(ensemble_model, train_loader, criterion, optimizer, device, epochs)

    # Save the trained model
    torch.save(ensemble_model.state_dict(), trained_model_path)
    print(f"Trained ensemble model saved to {trained_model_path}")

    # Predict and save test results
    print("Evaluating model on test data...")
    test_dataset = TrainAudioDataset(test_csv, test_audio_dir, embedding_extractor, spectrogram_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    predictions = predict_labels(test_loader, ensemble_model, device)

    save_predictions(predictions, output_csv)


if __name__ == "__main__":
    # Define paths
    MODEL_NAME = "large"  # Whisper model variant
    TRAIN_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_labels.csv"
    TRAIN_AUDIO_DIR = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_audios"
    TEST_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission_format.csv"
    TEST_AUDIO_DIR = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/test_audios"
    OUTPUT_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/predicted_whisper_large_test_labels.csv"
    TRAIN_MODEL_PATH = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/trained_whisper_large_model.pth"

    main_pipeline(MODEL_NAME, TRAIN_CSV, TRAIN_AUDIO_DIR, TEST_CSV, TEST_AUDIO_DIR, OUTPUT_CSV, TRAIN_MODEL_PATH)


