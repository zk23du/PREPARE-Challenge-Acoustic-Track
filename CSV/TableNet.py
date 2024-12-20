
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

# Convert data to TabNet format (numpy arrays)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

# Initialize TabNet model
print("Initializing TabNet model...")
tabnet_model = TabNetClassifier(
    n_d=32, n_a=32, n_steps=5, gamma=1.5,
    lambda_sparse=1e-3, optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
    mask_type="entmax",  # Use "sparsemax" if needed
    scheduler_params={"step_size":10, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    verbose=10,
    device_name=device  # Explicitly set device to GPU or CPU
)

# Train TabNet model
print("Training TabNet model...")
tabnet_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_name=["val"],
    eval_metric=["logloss"],  # Multiclass log-loss metric
    max_epochs=200,
    patience=20,
    batch_size=256,
    virtual_batch_size=64,
    num_workers=0,
    drop_last=False
)

# Evaluate on validation set
val_probs = tabnet_model.predict_proba(X_val)
val_loss = log_loss(y_val, val_probs)  # mlogloss calculation
print(f"Validation Multiclass Log Loss (mlogloss): {val_loss}")

# # Adjust probabilities to ensure rounding and sum to 1
# test_probs = np.round(test_probs, 2)
# for i, row in enumerate(test_probs):
#     diff = 1.0 - np.sum(row)
#     max_idx = np.argmax(row)
#     row[max_idx] += diff

generate_submission(
        tabnet_model, test_features, submission_format, 
        X_test, "/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission_lightgbm_data_preprocessed.csv"
    )
