from req import *
# ======================= Data Loading Functions =======================
def load_data(train_features_path, train_labels_path, test_features_path, submission_format_path):
    """Load train, test, and submission files."""
    train_features = pd.read_csv(train_features_path)
    train_labels = pd.read_csv(train_labels_path)
    test_features = pd.read_csv(test_features_path)
    submission_format = pd.read_csv(submission_format_path)
    
    train_data = train_features.merge(train_labels, on="uid")
    return train_data, test_features, submission_format

def prepare_features(train_data, test_features):
    """Prepare feature columns and labels for training and testing."""
    feature_columns = test_features.columns.difference(["uid"])
    X = train_data[feature_columns].values
    y = train_data[['diagnosis_control', 'diagnosis_mci', 'diagnosis_adrd']].idxmax(axis=1).apply(
        lambda x: {'diagnosis_control': 0, 'diagnosis_mci': 1, 'diagnosis_adrd': 2}[x]
    ).values  # Map class labels to integers
    X_test = test_features[feature_columns].values
    
    return X, y, X_test, feature_columns

def standardize_features(X, X_test):
    """Standardize the features for training and testing."""
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_test = scaler.transform(X_test)
    return X, X_test