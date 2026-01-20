import optuna
import torch
from src.dataset import building_block_split
from src.train import run_training
from src.config import *

def objective(trial, dataset):
    # Define search space
    config = {
        'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
        'heads': trial.suggest_categorical('heads', [2, 4, 8]),
        'layers': trial.suggest_int('layers', 2, 4),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    }
    
    # Split data (Building Block split for valid generalization metric)
    # This aligns the optimization objective with the competition goal.
    train_dataset, val_dataset, _ = building_block_split(dataset)
    
    # Run training
    # We suppress plotting and use the trial config
    metrics = run_training(train_dataset, val_dataset, test_dataset=None, config=config, plot=False)
    
    # We optimize for Average Precision (AP)
    return metrics['AP']

def run_optimization(dataset, n_trials=20):
    print(f"Starting Hyperparameter Optimization with {n_trials} trials...")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, dataset), n_trials=n_trials)
    
    print("Optimization Complete.")
    print("Best trial:")
    trial = study.best_trial
    
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    return trial.params
