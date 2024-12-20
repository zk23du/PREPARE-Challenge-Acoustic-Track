from lib import *
def predict_labels(test_loader, model, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i, (embeddings, uids) in enumerate(test_loader):
            embeddings = embeddings.to(device)
            #print(f"\n[PREDICT] Batch {i+1} | Embeddings Shape: {embeddings.shape}")

            outputs = model(embeddings)
            probs = torch.sigmoid(outputs)
            #print(f"[PREDICT] Outputs Shape: {outputs.shape} | Probs: {probs[:3]}")

            predictions.extend(zip(uids, probs.cpu().numpy()))
    return predictions



def save_predictions(predictions, output_csv):
    results = []
    for uid, probs in predictions:
        results.append({
            "uid": uid,
            "diagnosis_control": probs[0],
            "diagnosis_mci": probs[1],
            "diagnosis_adrd": probs[2]
        })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")