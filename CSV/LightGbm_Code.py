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

# Train LightGBM with extracted features
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

# Hyperparameter optimization with Optuna
def objective(trial):
    param = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "device": "gpu",  # Use GPU for training
        "num_leaves": trial.suggest_int("num_leaves", 10, 200),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.2),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "feature_fraction": trial.suggest_uniform("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.6, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-3, 10.0),
        "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-3, 10.0),
        "min_gain_to_split": trial.suggest_loguniform("min_gain_to_split", 1e-3, 1.0),
        # Optional: Specify the GPU device if you have multiple GPUs
        "gpu_platform_id": 0,  # Default GPU platform (use if you have multiple GPUs)
        "gpu_device_id": 0,    # Use GPU 0 (or change as needed)
    }
    
    # Perform cross-validation
    logloss_scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        gbm = lgb.train(
            param, train_data, valid_sets=[val_data], num_boost_round=1000
        )
        preds = gbm.predict(X_val, num_iteration=gbm.best_iteration)
        logloss_scores.append(log_loss(y_val, preds))
    
    return np.mean(logloss_scores)

print("Tuning LightGBM hyperparameters with Optuna...")
study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)
best_params = study.best_params
print("Best hyperparameters:", best_params)

# Train the final LightGBM model with the best parameters
print("Training LightGBM with optimized parameters...")
# Add `force_col_wise=True` to the final parameters
final_params = {
    **best_params,
    "objective": "multiclass",
    "metric": "multi_logloss",
    "num_class": 3,
    "force_col_wise": True,  # Enable column-wise parallelization
    "device": "gpu",  # Ensure GPU usage
}

train_data = lgb.Dataset(X, label=y)
gbm = lgb.train(
    final_params, train_data, num_boost_round=1000
)

# Feature importance analysis using SHAP
print("Analyzing feature importance...")
explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, feature_names=feature_columns)

# Evaluate Combined Model
y_val_pred_proba = gbm.predict(X_val)
y_val_pred = np.argmax(y_val_pred_proba, axis=1)

val_loss = log_loss(y_val, y_val_pred_proba)
val_accuracy = accuracy_score(y_val, y_val_pred)
print(f"Validation Log Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")

generate_submission(
        gbm, test_features, submission_format, 
        X_test, "/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission_lightgbm_data_preprocessed.csv"
    )
