
from lib import *
class AudioDataset(Dataset):
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