from lib import *

def load_labels(csv_file):
    label_df = pd.read_csv(csv_file)
    
    # Map one-hot encoding to integer labels
    def map_one_hot_to_label(row):
        if row['diagnosis_control'] == 1.0:
            return 0
        elif row['diagnosis_mci'] == 1.0:
            return 1
        elif row['diagnosis_adrd'] == 1.0:
            return 2
        else:
            raise ValueError("Invalid one-hot encoding in the labels.")
    
    label_df['label'] = label_df.apply(map_one_hot_to_label, axis=1)
    label_mapping = dict(zip(label_df['uid'], label_df['label']))  # Map UID to integer label
    print(f"Loaded {len(label_mapping)} labels from {csv_file}")
    return label_mapping


class SpectrogramDataset(Dataset):
    def __init__(self, csv_file, audio_dir, label_mapping=None, is_train=True, target_length=3000):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.is_train = is_train
        self.target_length = target_length
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_id = self.data.iloc[idx, 0]
        audio_path = os.path.join(self.audio_dir, audio_id + ".mp3")
        
        uid = self.data.iloc[idx, 0]  # Unique ID for the file

        label = -1  # Default for test set
        if self.is_train and self.label_mapping is not None:
            label = self.label_mapping.get(audio_id, -1)  # Get label from mapping
            label = torch.tensor(label).long()
        
        log_mel = self.convert_to_log_mel_spectrogram(audio_path)
        log_mel = self.pad_or_truncate(log_mel)
        log_mel = log_mel.repeat(3, 1, 1)  # ResNet expects 3 channels (like RGB)
        
        return log_mel, label, uid

    def convert_to_log_mel_spectrogram(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.mean(0, keepdim=True)  # Convert to mono
        
        # Adjust parameters to resolve the warning
        n_fft = 1024  # Increased FFT size for more frequency bins
        hop_length = 256  # Overlap between frames
        n_mels = 80  # Reduced mel bands to fit within frequency range

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )(waveform)
        
        log_mel = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        return log_mel


    def pad_or_truncate(self, mel_spectrogram):
        current_length = mel_spectrogram.size(2)
        if current_length < self.target_length:
            padding = self.target_length - current_length
            mel_spectrogram = torch.nn.functional.pad(mel_spectrogram, (0, padding))
        else:
            mel_spectrogram = mel_spectrogram[:, :, :self.target_length]
        return mel_spectrogram


class TrainableCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(TrainableCNN, self).__init__()
        self.resnet = resnet50(pretrained=True) 
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes) 

    def forward(self, x):
        return self.resnet(x)


class FeatureExtractorCNN(nn.Module):
    def __init__(self):
        super(FeatureExtractorCNN, self).__init__()
        self.resnet = resnet50(pretrained=True)  
        self.resnet.fc = nn.Identity() 

    def forward(self, x):
        return self.resnet(x)



def extract_features(model, dataloader, device, is_train=True):
    model.eval()
    features, labels, uids = [], [], []
    with torch.no_grad():
        for images, label, uids_batch in dataloader:
            images = images.to(device)
            feature = model(images)
            features.append(feature.cpu().numpy())
            uids.extend(uids_batch)
            if is_train:
                labels.extend(label.cpu().numpy().tolist())
    features = np.vstack(features)
    if is_train:
        labels = np.array(labels)
    return features, labels, uids


# 6️⃣ Train CNN
def train_cnn(model, dataloader, device, epochs=30, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader):.4f}")


# 7️⃣ Train XGBoost Classifier
def train_xgboost(X_train, y_train):
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss")
    xgb_model.fit(X_train, y_train)
    return xgb_model


# 8️⃣ Predict with XGBoost
def predict_with_xgboost(xgb_model, X_test, uids, save_csv_path):
    probabilities = xgb_model.predict_proba(X_test)
    predictions = np.argmax(probabilities, axis=1)
    prediction_data = []
    for i, uid in enumerate(uids):
        prediction_data.append({
            'uid': uid,
            'diagnosis_control': probabilities[i][0],
            'diagnosis_mci': probabilities[i][1],
            'diagnosis_adrd': probabilities[i][2],
            'predicted_class': predictions[i]
        })
    pd.DataFrame(prediction_data).to_csv(save_csv_path, index=False)
    print(f"Predictions saved to {save_csv_path}")


def split_train_dev(csv_file, dev_size=0.2):
    label_df = pd.read_csv(csv_file)
    
    def map_one_hot_to_label(row):
        if row['diagnosis_control'] == 1.0:
            return 0
        elif row['diagnosis_mci'] == 1.0:
            return 1
        elif row['diagnosis_adrd'] == 1.0:
            return 2
        else:
            raise ValueError("Invalid one-hot encoding in the labels.")
    
    label_df['label'] = label_df.apply(map_one_hot_to_label, axis=1)  # Add a new 'label' column
    
    train_df, dev_df = train_test_split(label_df, test_size=dev_size, random_state=42, stratify=label_df['label'])
    print(f"Train split: {len(train_df)} samples, Dev split: {len(dev_df)} samples")
    return train_df, dev_df



def evaluate_cnn(cnn_model, dataloader, device):
    cnn_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = cnn_model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def evaluate_xgboost(xgb_model, X_dev, y_dev):
    predictions = xgb_model.predict(X_dev)
    return accuracy_score(y_dev, predictions)

if __name__ == "__main__":
    train_csv = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_labels.csv"
    train_audio_dir = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_audios"
    test_csv = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission_format.csv"
    test_audio_dir = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/test_audios"
    output_csv = "/home/hiddenrock/DDS/DataDrivenCompetition/Data/predictions_xgboost_resnet50.csv"
    
     
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df, dev_df = split_train_dev(train_csv, dev_size=0.2)
    
    train_df.to_csv("train_split.csv", index=False)
    dev_df.to_csv("dev_split.csv", index=False)

    train_label_mapping = dict(zip(train_df['uid'], train_df['label']))
    dev_label_mapping = dict(zip(dev_df['uid'], dev_df['label']))
    
    train_dataset = SpectrogramDataset("train_split.csv", train_audio_dir, train_label_mapping, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    dev_dataset = SpectrogramDataset("dev_split.csv", train_audio_dir, dev_label_mapping, is_train=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, shuffle=False)
    
    print("Training CNN")
    cnn_model = TrainableCNN().to(device)
    train_cnn(cnn_model, train_loader, device, epochs=30)
    
    print("Evaluating CNN")
    cnn_accuracy = evaluate_cnn(cnn_model, dev_loader, device)
    print(f"CNN Dev Accuracy: {cnn_accuracy:.4f}")

    feature_extractor = FeatureExtractorCNN().to(device)
    train_features, train_labels, _ = extract_features(feature_extractor, train_loader, device, is_train=True)
    dev_features, dev_labels, _ = extract_features(feature_extractor, dev_loader, device, is_train=True)
    
    print("Training XGBoost")
    xgb_model = train_xgboost(train_features, train_labels)
    
    print("evaluating XGboost")
    xgb_accuracy = evaluate_xgboost(xgb_model, dev_features, dev_labels)
    print(f"XGBoost Dev Accuracy: {xgb_accuracy:.4f}")

    test_dataset = SpectrogramDataset(test_csv, test_audio_dir, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    test_features, _, test_uids = extract_features(feature_extractor, test_loader, device, is_train=False)
    predict_with_xgboost(xgb_model, test_features, test_uids, save_csv_path=output_csv)

