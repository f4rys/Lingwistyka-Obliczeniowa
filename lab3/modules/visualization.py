import matplotlib.pyplot as plt
import os

def plot_history(history_scratch, history_ft=None, save_dir="outputs"):
    """
    Plots training and validation metrics (Loss and Accuracy) for one or two models.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epochs = [h['epoch'] for h in history_scratch]
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [h['train_loss'] for h in history_scratch], label='Scratch Train Loss', linestyle='--')
    plt.plot(epochs, [h['val_loss'] for h in history_scratch], label='Scratch Val Loss', linestyle='-')
    
    if history_ft:
        # Handle case where FT might have different number of epochs (though in this lab they are same)
        epochs_ft = [h['epoch'] for h in history_ft]
        plt.plot(epochs_ft, [h['train_loss'] for h in history_ft], label='FT Train Loss', linestyle='--')
        plt.plot(epochs_ft, [h['val_loss'] for h in history_ft], label='FT Val Loss', linestyle='-')
        
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.show()
    
    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, [h['train_acc'] for h in history_scratch], label='Scratch Train Acc', linestyle='--')
    plt.plot(epochs, [h['val_acc'] for h in history_scratch], label='Scratch Val Acc', linestyle='-')
    
    if history_ft:
        epochs_ft = [h['epoch'] for h in history_ft]
        plt.plot(epochs_ft, [h['train_acc'] for h in history_ft], label='FT Train Acc', linestyle='--')
        plt.plot(epochs_ft, [h['val_acc'] for h in history_ft], label='FT Val Acc', linestyle='-')
        
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.show()
