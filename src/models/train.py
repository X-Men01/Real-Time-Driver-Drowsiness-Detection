import torch
from torch import nn
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter  
from datetime import datetime
import os
from typing import Dict, Optional
import sys
from pathlib import Path
import matplotlib.pyplot as plt


def create_run_name(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> str:
    """Create a descriptive run name with readable date format"""
    timestamp = datetime.now().strftime("(%Y-%m-%d)-(%H:%M:%S)")
    model_name = model.__class__.__name__
    optimizer_name = optimizer.__class__.__name__
    lr = optimizer.param_groups[0]['lr']
    
    return f"{model_name}_{optimizer_name}_lr{lr:.0e}_{timestamp}"



def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    Performs one epoch of training.
    
    Returns:
        tuple of (train_loss, train_acc)
    """
    model.train()
    train_loss, train_acc = 0, 0
    
    # Add progress bar for batches
    progress_bar = tqdm(enumerate(dataloader), 
                       total=len(dataloader),
                       desc="Training",
                       leave=False)
    
    for batch_idx, (X, y) in progress_bar:
        # Move data to device
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        y_pred_class = torch.argmax(y_pred, dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        train_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{(y_pred_class == y).sum().item() / len(y_pred):.4f}"
        })

    # Calculate average loss and accuracy
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc



def val_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """
    Performs evaluation on the validation/test set.
    
    Returns:
        tuple of (test_loss, test_acc)
    """
    model.eval()
    test_loss, test_acc = 0, 0
     # Add progress bar for batches
    progress_bar = tqdm(enumerate(dataloader), 
                       total=len(dataloader),
                       desc="Validation",
                       leave=False)
    
    with torch.inference_mode():
        for batch_idx,(X, y) in progress_bar:
            # Move data to device
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            test_pred = model(X)
            
            # Calculate loss
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            
            # Calculate accuracy
            test_pred_labels = test_pred.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{(test_pred_labels == y).sum().item() / len(test_pred):.4f}"
            })
            
    # Calculate average loss and accuracy
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    
    return test_loss, test_acc


def test_step(
    model: torch.nn.Module,
    test_dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
    writer: SummaryWriter,
    class_names: list = ['Close_Eye', 'Open_Eye']
) -> tuple[float, float]:
    """
    Performs final evaluation on the test set and logs results to TensorBoard.
    
    Args:
        model: Trained PyTorch model
        test_dataloader: DataLoader for test data
        loss_fn: Loss function
        device: Device to test on
        writer: TensorBoard SummaryWriter
        class_names: List of class names for confusion matrix
    
    Returns:
        tuple of (test_loss, test_acc)
    """
    model.eval()
    test_loss, correct_predictions, total_predictions = 0, 0, 0
    
    # For confusion matrix
    all_predictions = []
    all_labels = []
    
    # For per-class accuracy
    class_correct = {classname: 0 for classname in class_names}
    class_total = {classname: 0 for classname in class_names}
    
    progress_bar = tqdm(enumerate(test_dataloader), 
                       total=len(test_dataloader),
                       desc="Testing",
                       leave=False)
    
    with torch.inference_mode():
        for batch_idx, (X, y) in progress_bar:
            X, y = X.to(device), y.to(device)
            
            # Forward pass
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            
            # Calculate accuracy
            test_pred_labels = test_pred.argmax(dim=1)
            correct_predictions += (test_pred_labels == y).sum().item()
            total_predictions += len(y)
            
            # Store predictions and labels for confusion matrix
            all_predictions.extend(test_pred_labels.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            # Per-class accuracy
            for label, pred in zip(y, test_pred_labels):
                label_idx = label.item()
                class_name = class_names[label_idx]
                class_total[class_name] += 1
                if label == pred:
                    class_correct[class_name] += 1
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{correct_predictions/total_predictions:.4f}"
            })
    
    # Calculate final metrics
    test_loss = test_loss / len(test_dataloader)
    test_acc = correct_predictions / total_predictions
    
    # Log per-class accuracy to TensorBoard
    for class_name in class_names:
        class_acc = class_correct[class_name] / class_total[class_name]
        writer.add_scalar(f'Test/Accuracy_{class_name}', class_acc, 0)
    
    # Log confusion matrix to TensorBoard
    try:
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        cm = confusion_matrix(all_labels, all_predictions)
        fig, ax = plt.subplots(figsize=(8, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        writer.add_figure('Test/Confusion_Matrix', fig, 0)
        plt.close()
        
        # Calculate and log precision, recall, and F1 score
        from sklearn.metrics import classification_report
        report = classification_report(all_labels, all_predictions, 
                                     target_names=class_names, 
                                     output_dict=False)
        writer.add_text("classification_report", report)
        # for class_name in class_names:
        #     metrics = report[class_name]
        #     writer.add_scalar(f'Test/Precision_{class_name}', metrics['precision'], 0)
        #     writer.add_scalar(f'Test/Recall_{class_name}', metrics['recall'], 0)
        #     writer.add_scalar(f'Test/F1_{class_name}', metrics['f1-score'], 0)
            
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
    
    return test_loss, test_acc

def training(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,    
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
    epochs: int = 5,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    early_stopping_patience: int = 5,
    experiment_name: Optional[str] = None,
    test_dataloader: torch.utils.data.DataLoader = None,
) -> Dict:
    """
    Trains the model and returns training history.
    
    Args:
        model: PyTorch model to train
        train_dataloader: DataLoader for training data
        test_dataloader: DataLoader for validation/test data
        optimizer: PyTorch optimizer
        device: Device to train on
        loss_fn: Loss function
        epochs: Number of epochs to train for
        checkpoint_path: Path to save best model
        scheduler: Optional learning rate scheduler
        early_stopping_patience: Number of epochs to wait before early stopping
    
    Returns:
       Best model
    """
    
    model = torch.compile(model)
    print("\033[32mModel compiled successfully using torch.compile\033[0m")
   # Create run name and directories
    if experiment_name is None:
        experiment_name = create_run_name(model, optimizer)
    
    # Create directories for this run
    run_dir = Path("../logs/runs") / experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint path specific to this run
    checkpoint_path = run_dir / "best_model.pt"
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(run_dir)
    
    hparams = {
        "model_type": model.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "batch_size": train_dataloader.batch_size,
        "epochs": epochs,
        "train_dataset": train_dataloader.dataset,
        "test_dataset": val_dataloader.dataset,
        
    }
    writer.add_text("hyperparameters", str(hparams))
   
    best_acc = 0.0
    patience_counter = 0
    
    
   
    # # Try to log model graph
    # try:
    #     sample_batch = next(iter(train_dataloader))
    #     sample_images = sample_batch[0].to(device)
    #     writer.add_graph(model, sample_images)
    # except Exception as e:
    #     print(f"Could not log model graph: {e}")
    
    # Training loop
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        try:
            # Training phase
            train_loss, train_acc = train_step(model=model,dataloader=train_dataloader,loss_fn=loss_fn,optimizer=optimizer,device=device)
            
            # Testing phase
            val_loss, val_acc = val_step(model=model,dataloader=val_dataloader,loss_fn=loss_fn,device=device)
            
            # Get current learning rate
            current_lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr']
            
             # Log metrics to TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/val', val_acc, epoch)
            writer.add_scalar('Learning_rate', current_lr, epoch)
            
            # Print progress
            print(
                f"\033[34m \nEpoch: {epoch:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            
           
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                checkpoint = {'model_state_dict': model.state_dict(),'best_acc': best_acc,}
                torch.save(checkpoint, checkpoint_path)
                print(f"\033[32m Saved best model with accuracy: {best_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping check
            if patience_counter >= early_stopping_patience:
                print(f"\033[91m\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            # Step scheduler if provided
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                    
        except KeyboardInterrupt:
            print("\033[31m \nTraining interrupted by user!")
            break
        except Exception as e:
            print(f"\033[31m \nError during training: {str(e)}")
            sys.exit()
            
    # Cleanup
    try:
      
        # Load best model
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\033[32m Loaded best model with accuracy: {checkpoint['best_acc']:.4f}")
            
            
            if test_dataloader is not None:
                print("\033[34m Running final evaluation on test set...")
                final_test_loss, final_test_acc = test_step(
                    model=model,
                    test_dataloader=test_dataloader,
                    loss_fn=loss_fn,
                    device=device,
                    writer=writer
                )
                print(f"\033[32m Final Test Loss: {final_test_loss:.4f}")
                print(f"\033[32m Final Test Accuracy: {final_test_acc:.4f}")
            writer.close()
            
            
    except Exception as e:
        print(f"\033[31m Error during cleanup: {str(e)}")
        
    return model