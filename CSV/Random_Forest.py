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
    

# Split training data for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Hyperparameter tuning for Random Forest
def tune_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'random_state': 42
    }
    rf_model = RandomForestClassifier(**params)
    rf_model.fit(X_train, y_train)
    preds = rf_model.predict_proba(X_val)
    return log_loss(y_val, preds)

print("Tuning Random Forest hyperparameters...")
study_rf = optuna.create_study(direction="minimize")
study_rf.optimize(tune_rf, n_trials=20)
rf_best_params = study_rf.best_params

# Train Random Forest with best parameters
print("Training Random Forest...")
rf_model = RandomForestClassifier(**rf_best_params, random_state=42)
rf_model.fit(X_train, y_train)

# Predict probabilities for test set
test_probs = rf_model.predict_proba(X_test)

# Adjust probabilities
test_probs = np.round(test_probs, 2)
test_probs[:, -1] = 1 - np.sum(test_probs[:, :-1], axis=1)
test_probs = np.clip(test_probs, 0, 1)

# Prepare submission file
submission = pd.DataFrame({
    'uid': submission_format['uid'],
    'diagnosis_control': test_probs[:, 0],
    'diagnosis_mci': test_probs[:, 1],
    'diagnosis_adrd': test_probs[:, 2]
})

# Save submission
submission.to_csv("/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission.csv", index=False)
print("Submission saved as submission.csv")