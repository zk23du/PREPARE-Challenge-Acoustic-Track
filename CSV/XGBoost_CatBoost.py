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

# Train XGBoost model
print("Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=3,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=True
)

# Train LightGBM model
print("Training LightGBM model...")
lgb_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=3,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="multi_logloss")

# Train CatBoost model
print("Training CatBoost model...")
cb_model = cb.CatBoostClassifier(
    loss_function="MultiClass",
    iterations=200,
    depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=10
)
cb_model.fit(X_train, y_train, eval_set=(X_val, y_val))

# Predict probabilities for test set
xgb_probs_test = xgb_model.predict_proba(X_test)
lgb_probs_test = lgb_model.predict_proba(X_test)
cb_probs_test = cb_model.predict_proba(X_test)

# Weighted ensemble for test set
test_probs = 0.4 * xgb_probs_test + 0.3 * lgb_probs_test + 0.3 * cb_probs_test

# Adjust probabilities to ensure they are rounded to 2 decimals and sum to 1
adjusted_probs = test_probs.round(2)  # Round all values to 2 decimals
adjusted_probs[:, -1] = 1 - np.sum(adjusted_probs[:, :-1], axis=1)  # Adjust the last column
adjusted_probs[:, -1] = np.round(adjusted_probs[:, -1], 2)  # Round the last column again to 2 decimals

# Ensure no probabilities are negative and enforce sum = 1
adjusted_probs = np.where(adjusted_probs < 0, 0, adjusted_probs)  # Set negatives to 0
for i, row in enumerate(adjusted_probs):
    total = row.sum()
    if total != 1.0:
        max_idx = np.argmax(row)  # Find the column with the max probability
        row[max_idx] += 1.0 - total  # Adjust the max column to make the sum exactly 1

# Convert adjusted_probs to a DataFrame
adjusted_probs = pd.DataFrame(adjusted_probs, columns=['diagnosis_control', 'diagnosis_mci', 'diagnosis_adrd'])

# Prepare submission file
submission = pd.DataFrame({
    'uid': submission_format['uid'],
    'diagnosis_control': adjusted_probs['diagnosis_control'],
    'diagnosis_mci': adjusted_probs['diagnosis_mci'],
    'diagnosis_adrd': adjusted_probs['diagnosis_adrd']
})

# Save submission
submission.to_csv("/home/hiddenrock/DDS/DataDrivenCompetition/Data/submission.csv", index=False)
print("Submission saved as submission.csv")
