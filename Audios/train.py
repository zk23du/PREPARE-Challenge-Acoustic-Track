from lib import *
def train_model(model, dataloader, criterion, optimizer, device, epochs=25):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, (embeddings, labels) in enumerate(dataloader):
            if torch.isnan(embeddings).any():
                raise ValueError(f"NaN detected in embeddings for batch {i}")
            
            embeddings, labels = embeddings.to(device), labels.to(device)
            #print(f"\n[TRAIN] Batch {i+1} | Embeddings Shape: {embeddings.shape} | Labels Shape: {labels.shape}")

            if torch.isnan(labels).any():
                raise ValueError(f"NaN detected in labels for batch {i}")
            
            optimizer.zero_grad()
            outputs = model(embeddings)
            #print(f"[TRAIN] Outputs Shape: {outputs.shape} | Outputs: {outputs[:3]}")

            loss = criterion(outputs, labels)
            # Check for NaNs in loss
            if torch.isnan(loss):
                print(f"NaN Loss Detected at Epoch {epoch}, Batch {i}")
                print(f"Outputs: {outputs}")
                print(f"Labels: {labels}")
                print(f"Embedding: {embeddings}")
                break
            
            loss.backward()
            
            # # Clip gradients to avoid exploding gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            
            optimizer.step()

            total_loss += loss.item()
            #print(f"[TRAIN] Batch {i+1} Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {total_loss / len(dataloader):.4f}")
    return model