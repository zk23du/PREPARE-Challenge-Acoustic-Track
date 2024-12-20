from lib import *
from classifier import DynamicClassifier
from Dataloader import TestAudioDataset
from predict_save import predict_labels, save_predictions
from Hypernet import Hypernetwork
from Whisper_Best import WhisperEmbeddingExtractor
from predict_save import predict_labels, save_predictions

class AugmentedTrainAudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, embedding_extractor):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.embedding_extractor = embedding_extractor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.data.iloc[idx, 0] + ".mp3")
        label = torch.tensor(self.data.iloc[idx, 1:].values.astype(float), dtype=torch.float)
        
        # Load waveform
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        
        waveform = waveform / waveform.abs().max()
        
        # Data Augmentation
        # 1. Add Gaussian noise
        noise = torch.randn_like(waveform) * 0.005  
        waveform += noise

        # 2. Apply time-stretch
        if torch.rand(1).item() > 0.5:
            try:
                # Convert waveform to STFT (complex-valued tensor)
                spectrogram = torchaudio.functional.spectrogram(
                    waveform, 
                    pad=0, 
                    window=torch.hann_window(400), 
                    n_fft=400, 
                    hop_length=160, 
                    win_length=400, 
                    power=None, 
                    normalized=False
                )
                
                # Apply TimeStretch
                time_stretch = T.TimeStretch()
                rate = torch.empty(1).uniform_(0.8, 1.2).item()  # Random stretch rate between 0.8x and 1.2x
                spectrogram = time_stretch(spectrogram, rate)  # Corrected positional argument
                
                # Convert back to waveform using torch.istft
                waveform = torch.istft(
                    spectrogram, 
                    n_fft=400, 
                    hop_length=160, 
                    win_length=400, 
                    window=torch.hann_window(400), 
                    center=True, 
                    normalized=False, 
                    onesided=True
                )
            except Exception as e:
                print(f"Time stretch failed: {e}")
        
        # 3. Random shift
        shift_amount = torch.randint(low=-200, high=200, size=(1,)).item()  
        waveform = torch.roll(waveform, shifts=shift_amount, dims=1)
        
        # Extract embedding from the path
        embedding = self.embedding_extractor.extract_embedding(audio_path)
        
        return embedding, label


def train_model(hypernetwork, dynamic_classifier, dataloader, criterion, optimizer, scheduler, device, epochs=25):
    best_loss = float('inf')
    patience = 5
    no_improvement_count = 0
    
    for epoch in range(epochs):  
        hypernetwork.train()
        total_loss = 0
        
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            generated_weights = hypernetwork(embeddings)
            outputs = dynamic_classifier(embeddings, generated_weights)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        epoch_loss = total_loss / len(dataloader)
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improvement_count = 0
            torch.save(hypernetwork.state_dict(), 'best_model.pth')
        else:
            no_improvement_count += 1

        if no_improvement_count >= patience:
            print("Early stopping...")
            break
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {epoch_loss:.4f}")
    return hypernetwork


def evaluate_model(hypernetwork, dynamic_classifier, dataloader, criterion, device):
    hypernetwork.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for i, (embeddings, labels) in enumerate(dataloader):
            embeddings, labels = embeddings.to(device), labels.to(device)  
            generated_weights = hypernetwork(embeddings)
            outputs = dynamic_classifier(embeddings, generated_weights)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct_predictions += (preds == labels).all(dim=1).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples
    print(f"[EVAL] Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy



def main_pipeline(model_name, train_csv, train_audio_dir, test_csv, test_audio_dir, output_csv, epochs=25, batch_size=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embedding_extractor = WhisperEmbeddingExtractor(model_name, device=device)
    train_dataset = AugmentedTrainAudioDataset(train_csv, train_audio_dir, embedding_extractor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    sample_embedding, _ = next(iter(train_loader))
    input_dim = sample_embedding.size(1)
    output_dim = len(pd.read_csv(train_csv).columns) - 1 
    hypernetwork = Hypernetwork(input_dim, output_dim).to(device)
    dynamic_classifier = DynamicClassifier(input_dim, output_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(hypernetwork.parameters(), lr=0.001, weight_decay=1e-5) 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  

    hypernetwork = train_model(hypernetwork, dynamic_classifier, train_loader, criterion, optimizer, scheduler, device, epochs)
    test_dataset = TestAudioDataset(test_csv, test_audio_dir, embedding_extractor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    predictions = predict_labels(test_loader, hypernetwork, dynamic_classifier, device)
    save_predictions(predictions, output_csv)


if __name__ == "__main__":
    MODEL_NAME = "large"
    TRAIN_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_labels.csv"
    TRAIN_AUDIO_DIR = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_audios"
    TEST_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission_format.csv"
    TEST_AUDIO_DIR = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/test_audios"
    OUTPUT_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/predicted_whisper_large_hypernet_test_labels_better.csv"
    
    main_pipeline(MODEL_NAME, TRAIN_CSV, TRAIN_AUDIO_DIR, TEST_CSV, TEST_AUDIO_DIR, OUTPUT_CSV)
