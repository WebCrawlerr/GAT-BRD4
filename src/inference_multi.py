import argparse
import torch
import pandas as pd
import os
import glob
from tqdm import tqdm
import numpy as np
from torch_geometric.loader import DataLoader

from src.config import *
from src.dataset import BRD4Dataset
from src.model import GATModel

def run_target_inference(target_name, test_file, model_path, batch_size, limit):
    print(f"\n--- Processing Target: {target_name} ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init Dataset (now filters by target_name in test_mode)
    # We must allow raw_file to be the test_file
    # Note: dataset expects raw_file or filtered_file. We use filtered_file arg for the test csv path for simplicity in existing init
    dataset = BRD4Dataset(
        root='.', 
        filtered_file=test_file, 
        limit=limit, 
        test_mode=True, 
        target_name=target_name
    )
    
    if len(dataset) == 0:
        print(f"No samples found for {target_name}. Skipping.")
        return [], []
        
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=torch.cuda.is_available()
    )
    
    # Load Model
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    
    model = GATModel(num_node_features, num_edge_features, 
                     hidden_dim=GAT_HIDDEN_DIM, 
                     heads=GAT_HEADS, 
                     num_layers=GAT_LAYERS, 
                     dropout=DROPOUT).to(device)
                     
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return [], []
        
    print(f"Loading model: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    ids = []
    probs = []
    
    with torch.no_grad():
        for data in tqdm(loader, desc=f"Inference {target_name}"):
            data = data.to(device)
            logits = model(data)
            p = torch.sigmoid(logits).view(-1).cpu().numpy()
            
            if hasattr(data, 'id'):
                batch_ids = data.id.cpu().numpy() if torch.is_tensor(data.id) else np.array(data.id)
                ids.extend(batch_ids)
            else:
                 # Should not happen in updated dataset
                 pass
            
            probs.extend(p)
            
    return ids, probs

def main():
    parser = argparse.ArgumentParser(description="Multi-Target Inference")
    parser.add_argument('--test_file', type=str, default='data/raw/leash-BELKA/test.csv')
    parser.add_argument('--output_file', type=str, default='submission.csv')
    parser.add_argument('--model_dir', type=str, default=MODEL_SAVE_DIR, help='Directory to search for models if explicit paths not provided')
    
    # Explicit model paths
    parser.add_argument('--model_path_BRD4', type=str, default=None, help='Explicit path to BRD4 model')
    parser.add_argument('--model_path_HSA', type=str, default=None, help='Explicit path to HSA model')
    parser.add_argument('--model_path_sEH', type=str, default=None, help='Explicit path to sEH model')
    
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--limit', type=int, default=None)
    
    args = parser.parse_args()
    
    targets = ['BRD4', 'HSA', 'sEH']
    
    all_ids = []
    all_probs = []
    
    for target in targets:
        # Determine model path
        # 1. Check explicit argument
        explicit_arg = getattr(args, f"model_path_{target}", None)
        
        if explicit_arg:
             model_path = explicit_arg
        else:
             # 2. Key generic path
             model_path = os.path.join(args.model_dir, f"{target}_best_model.pth")
             
             # 3. Fallback search
             if not os.path.exists(model_path):
                print(f"Warning: {model_path} not found. Searching for alternatives in {args.model_dir}...")
                candidates = glob.glob(os.path.join(args.model_dir, f"*{target}*.pth"))
                if candidates:
                    model_path = candidates[0]
                    print(f"Found alternative: {model_path}")
        
        # Final check
        if not os.path.exists(model_path):
             print(f"Error: No model found for {target} (Checked: {model_path}). Skipping.")
             continue
                
        ids, probs = run_target_inference(target, args.test_file, model_path, args.batch_size, args.limit)
        
        all_ids.extend(ids)
        all_probs.extend(probs)
        
    print(f"\nCombining results... Total predictions: {len(all_ids)}")
    
    df = pd.DataFrame({'id': all_ids, 'binds': all_probs})
    
    # Sort by ID is usually good practice for Kaggle
    df.sort_values('id', inplace=True)
    
    df.to_csv(args.output_file, index=False)
    print(f"Submission saved to {args.output_file}")

if __name__ == "__main__":
    main()
