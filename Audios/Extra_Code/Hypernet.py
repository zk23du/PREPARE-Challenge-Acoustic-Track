from lib import *
class Hypernetwork(nn.Module):
    def __init__(self, input_dim, output_dim, embedding_dim=1280): 
        super(Hypernetwork, self).__init__()
        self.input_dim = input_dim  
        self.output_dim = output_dim  
        self.fc = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, input_dim * output_dim)
        )

    def forward(self, speaker_embedding):
        weights = self.fc(speaker_embedding)
        weights = weights.view(weights.size(0), self.input_dim, self.output_dim)
        return weights
