from lib import *
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for i, (embeddings, labels) in enumerate(dataloader):
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            # Model prediction
            outputs = model(embeddings)
            probs = torch.sigmoid(outputs)

            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Collect predictions and labels for log loss
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Calculate batch accuracy
            preds = (probs > 0.5).float()
            correct_predictions += (preds == labels).all(dim=1).sum().item()
            total_samples += labels.size(0)

    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    multiclass_log_loss = log_loss(all_labels, all_probs)
    accuracy = correct_predictions / total_samples

    print(f"[EVAL] Avg Loss: {avg_loss:.4f} | Multiclass Log Loss: {multiclass_log_loss:.4f} | Accuracy: {accuracy:.4f}")
    return avg_loss, multiclass_log_loss, accuracy
