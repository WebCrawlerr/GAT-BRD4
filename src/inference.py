import argparse
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

from src.config import *
from src.dataset import BRD4Dataset
from src.model import GATModel

def run_inference(model_path, test_file, output_file, batch_size=None, limit=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Config overrides
    batch_size = batch_size if batch_size else BATCH_SIZE
    
    # 1. Load Data
    # test_mode=True is critical here
    print(f"Loading test data from {test_file}...")
    dataset = BRD4Dataset(root=os.path.dirname(test_file), filtered_file=test_file, limit=limit, test_mode=True)
    
    # Optimization: Multi-process loading
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=4, pin_memory=torch.cuda.is_available())
    
    # 2. Load Model
    # We need to know input features
    num_node_features = dataset[0].x.shape[1]
    num_edge_features = dataset[0].edge_attr.shape[1]
    
    print(f"Initializing model (Features: Node={num_node_features}, Edge={num_edge_features})...")
    # Assuming default config for now. If you changed config, make sure to update strict args here.
    model = GATModel(num_node_features, num_edge_features, 
                     hidden_dim=GAT_HIDDEN_DIM, 
                     heads=GAT_HEADS, 
                     num_layers=GAT_LAYERS, 
                     dropout=DROPOUT).to(device)
                     
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
    print(f"Loading checkpoint: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 3. Inference Loop
    print("Starting inference...")
    ids = []
    probs = []
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Inference"):
            data = data.to(device)
            
            # Forward pass
            logits = model(data)
            
            # Sigmoid to get probability (Test Mode Requirement!)
            p = torch.sigmoid(logits).view(-1).cpu().numpy()
            
            # Collect IDs
            # data.id is a batch
            if hasattr(data, 'id'):
                batch_ids = data.id.cpu().numpy() if torch.is_tensor(data.id) else np.array(data.id)
                ids.extend(batch_ids)
            else:
                 # Fallback if id attribute issue (shouldn't happen with updated dataset)
                 print("Warning: Missing IDs in batch")
                 
            probs.extend(p)
            
    # 4. Save Submission
    print(f"Saving submission to {output_file}...")
    
    # Create DF
    submission = pd.DataFrame({
        'id': ids,
        'binds': probs
    })
    
    submission.to_csv(output_file, index=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT Inference for Kaggle Submission")
    parser.add_argument('--model_path', type=str, default=os.path.join(MODEL_SAVE_DIR, 'best_model.pth'),
                        help='Path to the trained model checkpoint')
    parser.add_argument('--test_file', type=str, default='data/raw/leash-BELKA/test.csv',
                        help='Path to the test CSV file')
    parser.add_argument('--output_file', type=str, default='submission.csv',
                        help='Output submission CSV file')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for inference')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of test samples (for dry run)')
                        
    args = parser.parse_args()
    
    run_inference(args.model_path, args.test_file, args.output_file, args.batch_size, args.limit)
