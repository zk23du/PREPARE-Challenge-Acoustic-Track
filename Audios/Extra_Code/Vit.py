import os
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import ViTForImageClassification
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class SpectrogramDataset(Dataset):
    def __init__(self, csv_file, audio_dir):
        self.data = pd.read_csv(csv_file)
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.data.iloc[idx, 0] + ".mp3")
        
        label = torch.tensor(self.data.iloc[idx, 1:].values.astype(float), dtype=torch.float)
        label = torch.argmax(label).long()  # Convert to class index (0, 1, 2)
        
        log_mel = self.convert_to_log_mel_spectrogram(audio_path)
        
        if log_mel.ndim == 2:
            log_mel = log_mel.unsqueeze(0)  # Shape (1, H, W)
        
        log_mel = log_mel.repeat(3, 1, 1)  # Shape (3, H, W)
        
        log_mel = torch.nn.functional.interpolate(log_mel.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
        
        log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min())  # Normalize [0, 1]
        log_mel = log_mel * 2 - 1  # Normalize to [-1, 1]
        
        return log_mel, label

    def convert_to_log_mel_spectrogram(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        
        if waveform.size(0) > 1:
            waveform = waveform.mean(0, keepdim=True)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, 
            n_fft=1024, 
            n_mels=80
        )(waveform)
        
        log_mel = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
        return log_mel


class ViTClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ViTClassifier, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
        
        self.vit.classifier = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 128), 
            nn.BatchNorm1d(128), 
            nn.ReLU(),
            nn.Linear(128, num_classes)  # Assuming 3 classes
        )

    def forward(self, x):
        return self.vit(x)


def train_vit(model, dataloader, criterion, optimizer, scheduler, device, epochs=30):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(images)
            logits = outputs.logits
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
        
        accuracy = correct_predictions / total_samples
        avg_loss = total_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")


def evaluate_vit(model, dataloader, criterion, device, save_csv=None):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    predictions = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if save_csv:
                for j in range(images.size(0)):
                    uid = dataloader.dataset.data.iloc[i * images.size(0) + j, 0]  # Extract UID
                    predictions.append({
                        "uid": uid,
                        "diagnosis_control": probs[j][0].item(),
                        "diagnosis_mci": probs[j][1].item(),
                        "diagnosis_adrd": probs[j][2].item(),
                    })
    
    if save_csv:
        df = pd.DataFrame(predictions)
        df.to_csv(save_csv, index=False)
        print(f"Predictions saved to {save_csv}")

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
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_dataset = SpectrogramDataset(test_csv, test_audio_dir)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ViTClassifier(num_classes=3).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

    print("\nTraining ViT model...")
    train_vit(model, train_loader, criterion, optimizer, scheduler, device, epochs=30)
    
    print("\nEvaluating ViT model...")
    evaluate_vit(model, test_loader, criterion, device, save_csv=output_csv)

