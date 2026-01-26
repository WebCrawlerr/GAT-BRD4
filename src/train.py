import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from src.utils import calculate_metrics, plot_loss_curve, plot_val_ap_curve, plot_confusion_matrix, plot_roc_curve, plot_pr_curve
from src.loss import FocalLoss
from src.config import *
import os
import numpy as np
import pandas as pd

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out = model(data)
        loss = criterion(out.view(-1), data.y.view(-1))
        
        loss.backward()
        
        # Gradient Clipping (Prevent exploding gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred_logits = []
    
    for data in loader:
        data = data.to(device)
        out = model(data)
        
        y_true.extend(data.y.cpu().numpy())
        y_pred_logits.extend(out.view(-1).cpu().numpy())
        
    metrics = calculate_metrics(np.array(y_true), np.array(y_pred_logits))
    return metrics, np.array(y_true), np.array(y_pred_logits)

def run_training(train_dataset, val_dataset, test_dataset=None, config=None, fold_idx=None, target_name='BRD4', plot=True, verbose=True):
    # Config overrides
    hidden_dim = config.get('hidden_dim', GAT_HIDDEN_DIM) if config else GAT_HIDDEN_DIM
    heads = config.get('heads', GAT_HEADS) if config else GAT_HEADS
    layers = config.get('layers', GAT_LAYERS) if config else GAT_LAYERS
    dropout = config.get('dropout', DROPOUT) if config else DROPOUT
    lr = config.get('lr', LEARNING_RATE) if config else LEARNING_RATE
    epochs = config.get('epochs', EPOCHS) if config else EPOCHS
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        if fold_idx is not None:
            print(f"Using device: {device} for Fold {fold_idx}")
        else:
            print(f"Using device: {device}")
    
    # DataLoaders
    # Use num_workers to parallelize graph conversion (CPU) while GPU trains
    # pin_memory speeds up transfer to GPU
    num_workers = 4
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory) if test_dataset else None
    
    # Model
    # Get num features from first data object
    num_node_features = train_dataset[0].x.shape[1]
    num_edge_features = train_dataset[0].edge_attr.shape[1]
    
    from src.model import GATModel
    model = GATModel(num_node_features, num_edge_features, hidden_dim, heads, layers, dropout).to(device)
    
    # Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    
    # Calculate pos_weight for imbalance
    # Calculate pos_weight for imbalance
    # Optimization: If dataset has backing dataframe, use it to avoid slow iteration
    try:
        # Check if it's a Subset (from split)
        # Fix: Check 'dataset' attr first to distinguish form Dataset with 'indices' method
        if hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'df'):
            indices = train_dataset.indices
            labels = train_dataset.dataset.df.iloc[indices]['Label']
            num_pos = labels.sum()
            num_neg = len(labels) - num_pos
        # Check if it's the base dataset
        elif hasattr(train_dataset, 'df'):
             # Caution: This counts WHOLE dataset if not subset. But usually run_training gets a subset.
             labels = train_dataset.df['Label']
             num_pos = labels.sum()
             num_neg = len(labels) - num_pos
        else:
            # Fallback to slow iteration
            print("Warning: calculating pos_weight via iteration (slow)...")
            y_train = [data.y.item() for data in train_dataset]
            num_pos = sum(y_train)
            num_neg = len(y_train) - num_pos
    except Exception as e:
        print(f"Warning: Error in fast label counting ({e}). Falling back to default weight.")
        num_pos = 1
        num_neg = 1

    print(f"Class Distribution in Train: Pos={num_pos}, Neg={num_neg}")
    # pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float).to(device) if num_pos > 0 else torch.tensor([1.0]).to(device)
    
    # Focal Loss (Alpha strategy: inverse class frequency, similar to pos_weight but normalized to [0,1])
    # alpha should be weight for class 1. Simple heuristic: num_neg / (num_pos + num_neg)
    if num_pos > 0:
        alpha = num_neg / (num_pos + num_neg)
    else:
        alpha = 0.5
        
    print(f"Focal Loss Alpha (Dynamic): {alpha:.4f}")
    
    criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA)
    
    # Training Loop
    best_val_ap = 0
    patience_counter = 0
    train_losses = []
    val_aps = []
    
    save_prefix = f"fold_{fold_idx}_" if fold_idx is not None else ""
    
    # Dynamic naming
    train_size_str = f"{len(train_dataset)//1000}k" if hasattr(train_dataset, '__len__') else "unknown"
    model_save_name = f"{save_prefix}{target_name}_gat_h{hidden_dim}_bs{BATCH_SIZE}_lr{lr}_data{train_size_str}_best.pth"
    print(f"Model will be saved as: {model_save_name}")
    
    # Logging history
    history = []
    model_saved = False

    # Time Safety
    import time
    start_time = time.time()
    MAX_TIME_SECONDS = 11.6 * 3600 # 11.5 hours safety margin for 12h limit

    for epoch in range(1, epochs + 1):
        # Time Check
        elapsed = time.time() - start_time
        if elapsed > MAX_TIME_SECONDS:
            print(f"⚠️ SAFETY STOP: Approaching time limit ({elapsed/3600:.1f}h). Stopping training to save results.")
            break

        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _, _ = evaluate(model, val_loader, device)
        val_ap = val_metrics['AP']
        
        train_losses.append(loss)
        val_aps.append(val_ap)
        
        # Log metrics
        history.append({
            'Epoch': epoch,
            'Train_Loss': loss,
            'Val_AP': val_metrics['AP'],
            'Val_AUC': val_metrics['AUC'],
            'Val_F1': val_metrics['F1']
        })
        
        if verbose:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AP: {val_ap:.4f}, Val AUC: {val_metrics["AUC"]:.4f}')
        
        # Early Stopping
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            patience_counter = 0
            
            # 1. Save Detailed Version (Archival)
            detailed_name = f"{save_prefix}{target_name}_gat_h{hidden_dim}_ep{epoch:02d}_ap{val_ap:.4f}.pth"
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, detailed_name))
            
            # 2. Save Operational 'Best' Model
            # We use a consistent name for loading back, including save_prefix for CV support
            best_model_name = f"{save_prefix}{target_name}_best_model.pth"
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, best_model_name))
            
            print(f" -> New Best Model Saved! ({best_model_name})")
            model_saved = True
        else:
            patience_counter += 1
            
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break
            
    # Save training log to CSV
    if plot:
        # Determine log directory (same as plots)
        if fold_idx is not None:
            run_log_dir = os.path.join(PLOTS_DIR, f'fold_{fold_idx}')
        else:
            run_log_dir = os.path.join(PLOTS_DIR, 'single_run')
            
        os.makedirs(run_log_dir, exist_ok=True)
        log_df = pd.DataFrame(history)
        log_csv_path = os.path.join(run_log_dir, 'training_log.csv')
        log_df.to_csv(log_csv_path, index=False)
        print(f"Training log saved to {log_csv_path}")
            
    # Final Evaluation
    if model_saved:
        best_model_name = f"{save_prefix}{target_name}_best_model.pth"
        print(f"Loading best model for final evaluation ({best_model_name})...")
        model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, best_model_name)))
    else:
        print("Warning: No best model was saved (did not improve). Using final model state.")
    
    # If test dataset is provided, evaluate on it. Otherwise evaluate on Val set (for CV)
    eval_loader = test_loader if test_loader else val_loader
    eval_name = "Test" if test_loader else "Validation"
    
    test_metrics, y_true_test, y_pred_test = evaluate(model, eval_loader, device)
    print(f"{eval_name} Metrics: {test_metrics}")
    
    # Plots
    if plot:
        # Create plot directory for this run
        if fold_idx is not None:
            run_plot_dir = os.path.join(PLOTS_DIR, f'fold_{fold_idx}')
        else:
            run_plot_dir = os.path.join(PLOTS_DIR, 'single_run')
            
        os.makedirs(run_plot_dir, exist_ok=True)
        
        plot_loss_curve(train_losses, os.path.join(run_plot_dir, 'loss_curve.png'))
        plot_val_ap_curve(val_aps, os.path.join(run_plot_dir, 'val_ap_curve.png'))
        plot_confusion_matrix(y_true_test, y_pred_test, os.path.join(run_plot_dir, 'confusion_matrix.png'))
        plot_roc_curve(y_true_test, y_pred_test, os.path.join(run_plot_dir, 'roc_curve.png'))
        plot_pr_curve(y_true_test, y_pred_test, os.path.join(run_plot_dir, 'pr_curve.png'))
    
    return test_metrics
