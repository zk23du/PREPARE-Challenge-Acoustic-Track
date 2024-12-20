from lib import *

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
    
    
class DynamicClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DynamicClassifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x, generated_weights):
        out = torch.bmm(x.unsqueeze(1), generated_weights).squeeze(1)
        return out
    
    
class Quadric(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    qweight: Tensor
    lweight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # quadratic weights
        self.qweight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # linear weights
        self.lweight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # bias
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.qweight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lweight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in_q, _ = nn.init._calculate_fan_in_and_fan_out(self.qweight)
            fan_in_l, _ = nn.init._calculate_fan_in_and_fan_out(self.lweight)
            bound = 1 / math.sqrt(fan_in_l) if fan_in_l > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        input_sqr = torch.mul(input, input)
        qi = nn.functional.linear(input_sqr, self.qweight, None)
        wib = nn.functional.linear(input, self.lweight, self.bias)
        return torch.add(qi, wib)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
        
class Classifier2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            Quadric(input_dim, 256),  # Increased output dimensions
            nn.BatchNorm1d(256),      # Added Batch Normalization
            nn.GELU(),                # Replaced ReLU with GELU
            nn.Dropout(0.3),          # Added Dropout
            Quadric(256, 128),        # Additional Quadratic layer
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            Quadric(128, output_dim)  # Final layer for output
        )

    def forward(self, x):
        return self.fc(x)
    
    

class Classifier3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Classifier, self).__init__()
        self.embedding_layer = nn.Sequential(
            nn.Linear(input_dim, 128),  # Use 128 as embedding dimension
            nn.ReLU()
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x, return_embeddings=False):
        embeddings = self.embedding_layer(x)
        logits = self.fc(embeddings)
        if return_embeddings:
            return embeddings
        return logits, embeddings


# Classifier Model
class ConvolutionalClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ConvolutionalClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension for Conv1d
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x
    
    
class CNNClassifier(nn.Module):
    def __init__(self, input_channels=1, output_dim=3):
        super(CNNClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # Calculate the size of the flattened feature vector after CNN
        self._calculate_flattened_size()

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 128),  # Use calculated size
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def _calculate_flattened_size(self):
        """Calculate the size of the flattened feature vector after the CNN layers."""
        with torch.no_grad():
            sample_input = torch.zeros(1, 1, 64000)  # Assume a 4-second signal (1 channel, 16kHz x 4s = 64000 samples)
            x = self.conv_layers(sample_input)
            self.flattened_size = x.view(1, -1).size(1)
            print(f"Flattened size: {self.flattened_size}")

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = self.fc(x)
        return x