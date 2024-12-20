from requirement import *
from DataloaderCSV import load_data, prepare_features, standardize_features
from predict_save_csv import generate_submission


train_data, test_features, submission_format = load_data(
        "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_features_new.csv",
        "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_labels.csv",
        "/home/hiddenrock/DDS/DataDrivenCompetition/Data/test_features_new.csv",
        "/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission_format.csv"
    )

X, y, X_test, feature_columns = prepare_features(train_data, test_features)

# # Feature engineering: Add polynomial features
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X = poly.fit_transform(X)
# X_test = poly.transform(X_test)


X, X_test = standardize_features(X, X_test)
    


# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define neural network
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

# Training loop
def train_model(model, optimizer, criterion, train_loader, val_loader, epochs=50, device="cpu"):
    model.to(device)
    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_pred = model(X_val)
                val_loss += criterion(y_pred, y_val).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

# Perform cross-validation
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = NeuralNetwork(input_dim=X_train.shape[1], output_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Train model
    train_model(model, optimizer, criterion, train_loader, val_loader, epochs=50, device="cpu")

# Load the best model for prediction
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Predict probabilities for the test set
with torch.no_grad():
    test_probs = model(X_test_tensor).numpy()

generate_submission(
        model, test_features, submission_format, 
        X_test, "/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission_lightgbm_data_preprocessed.csv"
    )
