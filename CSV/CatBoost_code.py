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
    

# Cross-validation setup
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Hyperparameter optimization with Optuna
def objective(trial):
    param = {
        "iterations": trial.suggest_int("iterations", 500, 1500),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 10),
        "bagging_temperature": trial.suggest_uniform("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_uniform("random_strength", 0.0, 1.0),
        "verbose": 0,
        "loss_function": "MultiClass",
        "eval_metric": "MultiClass"
    }
    
    # Perform cross-validation
    logloss_scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_pool = Pool(X_train, y_train)
        val_pool = Pool(X_val, y_val)
        
        catboost_model = CatBoostClassifier(**param)
        catboost_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50, use_best_model=True)
        preds = catboost_model.predict_proba(X_val)
        logloss_scores.append(log_loss(y_val, preds))
    
    return np.mean(logloss_scores)

print("Tuning CatBoost hyperparameters with Optuna...")
study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)
best_params = study.best_params
print("Best hyperparameters:", best_params)

# Train the final CatBoost model with the best parameters
print("Training CatBoost with optimized parameters...")
final_params = {
    **best_params,
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "verbose": 100
}

train_pool = Pool(X, y)
catboost_model = CatBoostClassifier(**final_params)
catboost_model.fit(train_pool)

# Predict probabilities for the test set
print("Generating predictions for the test set...")
test_pool = Pool(X_test)
test_probs = catboost_model.predict_proba(test_pool)

# Create a DataFrame with predictions and associated UIDs
test_predictions = pd.DataFrame({
    "uid": test_features["uid"],  # UIDs from test_features
    "diagnosis_control": test_probs[:, 0],
    "diagnosis_mci": test_probs[:, 1],
    "diagnosis_adrd": test_probs[:, 2],
})

# Group by UID and calculate the mean probabilities for each UID
aggregated_predictions = test_predictions.groupby("uid", as_index=False).mean()

# Align aggregated predictions with the submission format
submission = pd.DataFrame({
    "uid": submission_format["uid"],  # Ensure alignment with submission format UIDs
    "diagnosis_control": aggregated_predictions.set_index("uid").reindex(submission_format["uid"])["diagnosis_control"].values,
    "diagnosis_mci": aggregated_predictions.set_index("uid").reindex(submission_format["uid"])["diagnosis_mci"].values,
    "diagnosis_adrd": aggregated_predictions.set_index("uid").reindex(submission_format["uid"])["diagnosis_adrd"].values,
})

# Save the submission file
submission.to_csv("/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission.csv", index=False)
print("Submission saved at /home/hiddenrock/DDS/DataDrivenCompetition/Data/submission.csv")