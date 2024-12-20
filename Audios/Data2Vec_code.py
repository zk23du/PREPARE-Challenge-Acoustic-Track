from lib import *
class Data2VecEmbeddingExtractor:
    def __init__(self, model_name="facebook/data2vec-audio-large-960h", device="cuda:1"):
        """
        Extract embeddings from audio files using Facebook's Data2Vec model.
        """
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = Data2VecAudioModel.from_pretrained(model_name).to(device)  # Use Data2VecAudioModel
        self.device = device
        self.target_length = 16000 * 30  # 30 seconds at 16kHz

        print(f"Loaded Data2Vec model '{model_name}' for embedding extraction on {device}.")

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



########### hypernetwoork approach 
class Data2VecEmbeddingExtractor:
    def __init__(self, model_name="facebook/data2vec-audio-large-960h", device="cuda:1"):
        """
        Extract embeddings from audio files using Facebook's Data2Vec model.
        """
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = Data2VecAudioModel.from_pretrained(model_name).to(device)
        self.device = device
        self.target_length = 16000 * 30  # 30 seconds at 16kHz
        
        # ⚠️ Move the projection layer to the GPU (cuda:1)
        self.projection_layer = nn.Linear(1024, 768).to(device)  # Reduce from 1024 to 768
        print(f"Loaded Data2Vec model '{model_name}' for embedding extraction on {device}.")

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
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move inputs to CUDA device

        with torch.no_grad():
            outputs = self.model(**inputs)
            pooled_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)  # Mean over sequence dimension
            
            # ⚠️ Make sure the Linear layer is on the same device as the embedding (both on cuda:1)
            projected_embedding = self.projection_layer(pooled_embedding)  # Reduce 1024 to 768
        return projected_embedding



class TrainAudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, embedding_extractor):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.embedding_extractor = embedding_extractor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.data.iloc[idx, 0] + ".mp3")
        label = torch.tensor(self.data.iloc[idx, 1:].values.astype(float), dtype=torch.float)
        embedding = self.embedding_extractor.extract_embedding(audio_path)
        return embedding, label


class TestAudioDataset(Dataset):
    def __init__(self, csv_file, audio_dir, embedding_extractor):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.embedding_extractor = embedding_extractor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.data.iloc[idx, 0] + ".mp3")
        embedding = self.embedding_extractor.extract_embedding(audio_path)
        return embedding, self.data.iloc[idx, 0] 
    
    
class Hypernetwork(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim=1024):  # Corrected to 1024 to match Data2Vec embeddings
        super(Hypernetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 256),  # 1024 to 256
            nn.ReLU(),
            nn.Linear(256, input_dim * output_dim)  # Generate input_dim * output_dim weights
        )

    def forward(self, speaker_embedding):
        weights = self.fc(speaker_embedding)  # Shape: (batch_size, input_dim * output_dim)
        weights = weights.view(-1, self.input_dim, self.output_dim)  # Reshape to (batch_size, input_dim, output_dim)
        return weights



class DynamicClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DynamicClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x, generated_weights):
        """
        x: Embedding from Data2Vec (batch_size, input_dim)
        generated_weights: Weights from the Hypernetwork (batch_size, input_dim, output_dim)
        """
        # Apply weights to the embeddings (batch_size, input_dim) x (batch_size, input_dim, output_dim)
        out = torch.bmm(x.unsqueeze(1), generated_weights).squeeze(1)  # Batch matrix multiplication
        return out


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


def train_model(hypernetwork, dynamic_classifier, dataloader, criterion, optimizer, device, epochs=25):
    hypernetwork.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (embeddings, labels) in enumerate(dataloader):
            embeddings, labels = embeddings.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Generate weights for the classifier using the Hypernetwork
            generated_weights = hypernetwork(embeddings)
            
            # Pass embeddings and weights to the classifier
            outputs = dynamic_classifier(embeddings, generated_weights)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {total_loss / len(dataloader):.4f}")
    return hypernetwork


def evaluate_model(hypernetwork, dynamic_classifier, dataloader, criterion, device):
    hypernetwork.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    correct_predictions = 0
    total_samples = 0
    with torch.no_grad():
        for i, (embeddings, labels) in enumerate(dataloader):
            embeddings, labels = embeddings.to(device), labels.to(device)  
            
            # Generate weights for the classifier using the Hypernetwork
            generated_weights = hypernetwork(embeddings)
            
            # Pass embeddings and weights to the classifier
            outputs = dynamic_classifier(embeddings, generated_weights)
            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            probs = torch.sigmoid(outputs)            
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            preds = (probs > 0.5).float()
            correct_predictions += (preds == labels).all(dim=1).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_samples

    print(f"[EVAL] Avg Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy



def predict_labels(test_loader, hypernetwork, dynamic_classifier, device):
    hypernetwork.eval()
    dynamic_classifier.eval()
    predictions = []
    with torch.no_grad():
        for i, (embeddings, uids) in enumerate(test_loader):
            embeddings = embeddings.to(device)
            # Generate weights using the Hypernetwork
            generated_weights = hypernetwork(embeddings)  # Shape: (batch_size, input_dim, output_dim)
            # Pass embeddings and weights to the Dynamic Classifier
            outputs = dynamic_classifier(embeddings, generated_weights)
            probs = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
            predictions.extend(zip(uids, probs.cpu().numpy()))
    return predictions




def save_predictions(predictions, output_csv):
    results = []
    for uid, probs in predictions:
        results.append({
            "uid": uid,
            "diagnosis_control": probs[0],
            "diagnosis_mci": probs[1],
            "diagnosis_adrd": probs[2]
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")
    

def main_pipeline(model_name, train_csv, train_audio_dir, test_csv, test_audio_dir, output_csv, epochs=25, batch_size=32):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    embedding_extractor = Data2VecEmbeddingExtractor(model_name, device=device)

    train_dataset = TrainAudioDataset(train_csv, train_audio_dir, embedding_extractor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    sample_embedding, _ = next(iter(train_loader))
    input_dim = sample_embedding.shape[-1]  # Infer input dimensions
    output_dim = len(pd.read_csv(train_csv).columns) - 1

    # Instantiate Hypernetwork + Dynamic Classifier
    hypernetwork = Hypernetwork(input_dim, output_dim, embedding_dim=768).to(device)
    dynamic_classifier = DynamicClassifier(input_dim, output_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(hypernetwork.parameters(), lr=0.001)

    hypernetwork = train_model(hypernetwork, dynamic_classifier, train_loader, criterion, optimizer, device, epochs)

    print("Evaluating model...")
    evaluate_model(hypernetwork, dynamic_classifier, train_loader, criterion, device)

    test_dataset = TestAudioDataset(test_csv, test_audio_dir, embedding_extractor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Pass both hypernetwork and dynamic_classifier
    predictions = predict_labels(test_loader, hypernetwork, dynamic_classifier, device)

    save_predictions(predictions, output_csv)




if __name__ == "__main__":
    MODEL_NAME = "facebook/data2vec-audio-large-960h" 
    TRAIN_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_labels.csv"
    TRAIN_AUDIO_DIR = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_audios"
    TEST_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission_format.csv"
    TEST_AUDIO_DIR = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/test_audios"
    OUTPUT_CSV = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/predicted_data2vec_hypernet_labels.csv"
    main_pipeline(MODEL_NAME, TRAIN_CSV, TRAIN_AUDIO_DIR, TEST_CSV, TEST_AUDIO_DIR, OUTPUT_CSV)



