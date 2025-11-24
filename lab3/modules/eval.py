import torch

def evaluate_classifier(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Handle both custom model and HF model
            # Custom: returns (logits, loss)
            # HF: returns Output object with .loss and .logits, or tuple (loss, logits) depending on return_dict
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels) if hasattr(model, 'config') and hasattr(model.config, 'use_return_dict') else model(input_ids, attention_mask=attention_mask, targets=labels)

            loss = None
            logits = None

            if isinstance(outputs, tuple):
                # Custom model: (logits, loss)
                # HF model (return_dict=False): (loss, logits)
                # This is ambiguous.
                # Let's check the shape or type.
                if outputs[0].dim() == 0: # Scalar loss -> HF
                    loss = outputs[0]
                    logits = outputs[1]
                else: # Logits first -> Custom
                    logits = outputs[0]
                    loss = outputs[1]
            elif hasattr(outputs, 'loss'): # HF Output object
                loss = outputs.loss
                logits = outputs.logits
            else:
                # Assume logits only? No, we passed targets/labels so we expect loss.
                # Fallback for custom model if it returns just logits (unlikely with targets)
                logits = outputs
            
            if loss is not None:
                total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    acc = correct / total if total > 0 else 0.0
    return acc, avg_loss
