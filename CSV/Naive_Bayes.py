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
X, X_test = standardize_features(X, X_test)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Evaluate on validation data
y_val_pred_proba = nb_model.predict_proba(X_val)
y_val_pred = np.argmax(y_val_pred_proba, axis=1)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_log_loss = log_loss(y_val, y_val_pred_proba)

print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Log Loss: {val_log_loss:.4f}")

# Predict on test data
test_probs = nb_model.predict_proba(X_test)

generate_submission(
        nb_model, test_features, submission_format, 
        X_test, "/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission_lightgbm_data_preprocessed.csv"
    )

