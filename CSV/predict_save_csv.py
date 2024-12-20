from requirement import *
def generate_submission(model, test_features, submission_format, X_test, output_path):
    """Generate a CSV submission file for the competition."""
    test_probs = model.predict(X_test)
    test_predictions = pd.DataFrame({
        "uid": test_features["uid"],
        "diagnosis_control": test_probs[:, 0],
        "diagnosis_mci": test_probs[:, 1],
        "diagnosis_adrd": test_probs[:, 2],
    })
    
    aggregated_predictions = test_predictions.groupby("uid", as_index=False).mean()
    
    submission = pd.DataFrame({
        "uid": submission_format["uid"],
        "diagnosis_control": aggregated_predictions.set_index("uid").reindex(submission_format["uid"])["diagnosis_control"].values,
        "diagnosis_mci": aggregated_predictions.set_index("uid").reindex(submission_format["uid"])["diagnosis_mci"].values,
        "diagnosis_adrd": aggregated_predictions.set_index("uid").reindex(submission_format["uid"])["diagnosis_adrd"].values,
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission saved at {output_path}")