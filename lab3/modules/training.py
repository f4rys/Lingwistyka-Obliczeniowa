import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .eval import evaluate_classifier

def train_classifier(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 5,
    save_path: str = "model.pt",
    scheduler = None,
    max_grad_norm: float = 1.0,
    patience: int = 3, # Early stopping patience
):
    model.to(device)
    best_acc = 0.0
    history = []
    no_improve_epochs = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Handle call signature difference
            # HF models use 'labels', custom uses 'targets'
            # We can try passing both or checking type
            # But simpler: try-except or check attribute
            
            if hasattr(model, 'config') and hasattr(model.config, 'architectures'): # Likely HF
                 outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            else: # Custom
                 outputs = model(input_ids, attention_mask=attention_mask, targets=labels)

            loss = None
            logits = None

            if isinstance(outputs, tuple):
                if outputs[0].dim() == 0: # HF (loss, logits)
                    loss = outputs[0]
                    logits = outputs[1]
                else: # Custom (logits, loss)
                    logits = outputs[0]
                    loss = outputs[1]
            elif hasattr(outputs, 'loss'):
                loss = outputs.loss
                logits = outputs.logits
            
            loss.backward()
            
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1), 'acc': correct / total})
            
        train_acc = correct / total
        train_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        # Validation
        val_acc, val_loss = evaluate_classifier(model, val_loader, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Time={epoch_time:.2f}s")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'time': epoch_time
        })

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with Acc {best_acc:.4f}")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs.")
            
        if patience > 0 and no_improve_epochs >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
            
    return history
