from requirement import *
from DataloaderCSV import load_data, prepare_features, standardize_features
from predict_save_csv import generate_submission

# Check GPU availability
print("Is GPU available?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device Name:", torch.cuda.get_device_name(0))
else:
    print("No GPU available, using CPU.")

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_data, test_features, submission_format = load_data(
        "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_features_new.csv",
        "/home/hiddenrock/DDS/DataDrivenCompetition/Data/train_labels.csv",
        "/home/hiddenrock/DDS/DataDrivenCompetition/Data/test_features_new.csv",
        "/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission_format.csv"
    )

X, y, X_test, feature_columns = prepare_features(train_data, test_features)
X, X_test = standardize_features(X, X_test)

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Step 1: Feature Importance
print("Calculating feature importance with XGBoost...")
xgb_model = xgb.XGBClassifier(objective="multi:softmax", num_class=3, random_state=42)
xgb_model.fit(X_train, y_train)
feature_importances = xgb_model.feature_importances_

# Select top-k features based on importance
k = 100  # Number of features to keep
top_k_indices = np.argsort(feature_importances)[-k:]
X_train_reduced = X_train[:, top_k_indices]
X_val_reduced = X_val[:, top_k_indices]
X_test_reduced = X_test[:, top_k_indices]

# Step 2: Autoencoder for Dimensionality Reduction
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Initialize autoencoder
input_dim = X_train_reduced.shape[1]
latent_dim = 32
autoencoder = Autoencoder(input_dim=input_dim, latent_dim=latent_dim).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# Convert data to PyTorch tensors
train_tensor = torch.tensor(X_train_reduced, dtype=torch.float32).to(device)
val_tensor = torch.tensor(X_val_reduced, dtype=torch.float32).to(device)
test_tensor = torch.tensor(X_test_reduced, dtype=torch.float32).to(device)
train_loader = DataLoader(train_tensor, batch_size=32, shuffle=True)

# Train autoencoder
epochs = 50
print("Training autoencoder...")
for epoch in range(epochs):
    autoencoder.train()
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        _, reconstructed = autoencoder(batch)
        loss = criterion(reconstructed, batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# Get reduced features
autoencoder.eval()
with torch.no_grad():
    X_train_encoded = autoencoder.encoder(train_tensor).cpu().numpy()
    X_val_encoded = autoencoder.encoder(val_tensor).cpu().numpy()
    X_test_encoded = autoencoder.encoder(test_tensor).cpu().numpy()

# Step 3: Classification on Reduced Features
print("Training classifier on reduced features...")
classifier = nn.Sequential(
    nn.Linear(latent_dim, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 3),
    nn.Softmax(dim=1)
).to(device)

criterion_class = nn.CrossEntropyLoss()
optimizer_class = torch.optim.Adam(classifier.parameters(), lr=0.001)

# Prepare DataLoader for classification
train_dataset = TensorDataset(torch.tensor(X_train_encoded, dtype=torch.float32).to(device),
                               torch.tensor(y_train, dtype=torch.long).to(device))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train classifier
epochs = 50
for epoch in range(epochs):
    classifier.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer_class.zero_grad()
        outputs = classifier(X_batch)
        loss = criterion_class(outputs, y_batch)
        loss.backward()
        optimizer_class.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# Predict on test set
classifier.eval()
with torch.no_grad():
    test_probs = classifier(torch.tensor(X_test_encoded, dtype=torch.float32).to(device)).cpu().numpy()

# Adjust probabilities to ensure sum = 1 and round to 2 decimal places
test_probs = np.round(test_probs, 2)
test_probs[:, -1] = 1 - np.sum(test_probs[:, :-1], axis=1)
test_probs = np.clip(test_probs, 0, 1)

# Group predictions by UID and average probabilities
test_predictions = pd.DataFrame({
    'uid': test_features['uid'],
    'diagnosis_control': test_probs[:, 0],
    'diagnosis_mci': test_probs[:, 1],
    'diagnosis_adrd': test_probs[:, 2]
})
aggregated_predictions = test_predictions.groupby('uid', as_index=False).mean()

# Prepare submission file
submission = pd.DataFrame({
    'uid': submission_format['uid'],
    'diagnosis_control': aggregated_predictions['diagnosis_control'],
    'diagnosis_mci': aggregated_predictions['diagnosis_mci'],
    'diagnosis_adrd': aggregated_predictions['diagnosis_adrd']
})

# Save submission
submission.to_csv("/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission.csv", index=False)
print("Submission saved as submission.csv")
