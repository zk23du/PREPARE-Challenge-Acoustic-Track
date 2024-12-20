from requirement import *
from DataloaderCSV import load_data, prepare_features, standardize_features
from predict_save_csv import generate_submission



# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

# Hyperparameter Tuning for XGBoost
def tune_xgb(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'objective': "multi:softprob",
        'eval_metric': "mlogloss",
        'num_class': 3,
        'tree_method': "hist", # Use GPU
        'device': "cuda:0"   
    }
    xgb_model = xgb.XGBClassifier(**params, random_state=42)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    preds = xgb_model.predict_proba(X_val)
    return log_loss(y_val, preds)

print("Tuning XGBoost hyperparameters...")
study = optuna.create_study(direction="minimize")
study.optimize(tune_xgb, n_trials=5)
xgb_params = study.best_params
xgb_params.update({'tree_method': "hist", 'device': "cuda"})# Add GPU support

# Train tuned XGBoost
print("Training tuned XGBoost...")
xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42)
xgb_model.fit(X_train, y_train)

# Train LightGBM
print("Training LightGBM...")
lgb_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=3,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    device="gpu",  # Use GPU
    random_state=42
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="multi_logloss")

# Train CatBoost
print("Training CatBoost...")
cb_model = cb.CatBoostClassifier(
    loss_function="MultiClass",
    iterations=200,
    depth=6,
    learning_rate=0.1,
    random_state=42,
    task_type="GPU" # Use GPU
)
cb_model.fit(X_train, y_train)

# Predict probabilities for test set
print("Predicting probabilities with individual models...")
xgb_probs_test = xgb_model.predict_proba(X_test)
lgb_probs_test = lgb_model.predict_proba(X_test)
cb_probs_test = cb_model.predict_proba(X_test)

# Stacking Ensemble
print("Stacking ensemble...")
stack_train = np.hstack((xgb_model.predict_proba(X_val), lgb_model.predict_proba(X_val), cb_model.predict_proba(X_val)))
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(stack_train, y_val)

# Predict final probabilities
stack_test = np.hstack((xgb_probs_test, lgb_probs_test, cb_probs_test))
final_probs = meta_model.predict_proba(stack_test)

# Adjust probabilities
final_probs = np.round(final_probs, 2)
final_probs[:, -1] = 1 - np.sum(final_probs[:, :-1], axis=1)
final_probs = np.clip(final_probs, 0, 1)

generate_submission(
        meta_model, test_features, submission_format, 
        X_test, "/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission_lightgbm_data_preprocessed.csv"
    )
