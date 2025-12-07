import pandas as pd
import numpy as np
from rdkit import Chem
from src.config import THRESHOLD_NM, TARGET_NAMES

def load_and_filter_data(filepath, chunksize=100000):
    """
    Loads BindingDB data from TSV file in chunks and filters for BRD4 on the fly.
    This drastically reduces memory usage.
    """
    print(f"Loading and filtering data from {filepath} in chunks of {chunksize}...")
    
    # Target Name is usually the column name in BindingDB
    target_col = 'Target Name'
    
    filtered_chunks = []
    total_processed = 0
    
    try:
        # First, peek at columns to ensure we have the right target column
        # We read just the header
        header_df = pd.read_csv(filepath, sep='\t', nrows=0)
        if target_col not in header_df.columns:
             # Try to find a similar column
            candidates = [c for c in header_df.columns if 'Target Name' in c]
            if candidates:
                target_col = candidates[0]
                print(f"Using column '{target_col}' for filtering.")
            else:
                print("Could not find Target Name column.")
                return None

        # Iterate in chunks
        with pd.read_csv(filepath, sep='\t', on_bad_lines='skip', low_memory=False, chunksize=chunksize) as reader:
            for chunk in reader:
                # Filter
                mask = chunk[target_col].apply(lambda x: any(name in str(x) for name in TARGET_NAMES))
                filtered_chunk = chunk[mask].copy()
                
                if not filtered_chunk.empty:
                    filtered_chunks.append(filtered_chunk)
                
                total_processed += len(chunk)
                if total_processed % (chunksize * 10) == 0:
                    print(f"Processed {total_processed} rows...")
                    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
        
    if not filtered_chunks:
        print("No matching records found.")
        return pd.DataFrame()
        
    print(f"Finished processing {total_processed} rows.")
    return pd.concat(filtered_chunks, ignore_index=True)

def clean_and_label_data(df):
    """
    Cleans data, prioritizes Ki > Kd > IC50, and creates binary labels.
    """
    # Columns of interest (BindingDB names are specific)
    # Usually: 'Ki (nM)', 'IC50 (nM)', 'Kd (nM)', 'Ligand SMILES'
    
    # Helper to parse numeric
    def parse_activity(val):
        try:
            # Remove > or < signs
            val = str(val).replace('>', '').replace('<', '').strip()
            return float(val)
        except:
            return np.nan

    # Create a consolidated 'Activity_nM' column
    # Priority: Ki > Kd > IC50
    # Note: BindingDB column names might vary slightly, checking common ones
    
    ki_col = 'Ki (nM)'
    kd_col = 'Kd (nM)'
    ic50_col = 'IC50 (nM)'
    smiles_col = 'Ligand SMILES'
    
    # Ensure columns exist
    for col in [ki_col, kd_col, ic50_col]:
        if col not in df.columns:
            df[col] = np.nan
            
    # Create new columns in a separate dataframe to avoid fragmentation
    new_cols = pd.DataFrame(index=df.index)
    new_cols['Ki_val'] = df[ki_col].apply(parse_activity)
    new_cols['Kd_val'] = df[kd_col].apply(parse_activity)
    new_cols['IC50_val'] = df[ic50_col].apply(parse_activity)
    
    # Coalesce
    new_cols['Activity_nM'] = new_cols['Ki_val'].fillna(new_cols['Kd_val']).fillna(new_cols['IC50_val'])
    
    # Concatenate back
    df = pd.concat([df, new_cols], axis=1)
    
    # Drop rows with no activity
    df = df.dropna(subset=['Activity_nM'])
    
    # Drop rows with no SMILES
    if smiles_col in df.columns:
        df = df.dropna(subset=[smiles_col])
    else:
        print("SMILES column not found!")
        return pd.DataFrame()
        
    # Binary Labeling
    # Active (1) if Activity <= THRESHOLD (1000 nM)
    # Inactive (0) if Activity > THRESHOLD
    
    df['Label'] = (df['Activity_nM'] <= THRESHOLD_NM).astype(int)
    
    # Calculate and print statistics
    total_molecules = len(df)
    active_molecules = df['Label'].sum()
    active_percentage = (active_molecules / total_molecules) * 100 if total_molecules > 0 else 0
    
    print(f"\n--- Data Statistics ---")
    print(f"Total Molecules: {total_molecules}")
    print(f"Active Molecules (Label=1): {active_molecules}")
    print(f"Active Percentage: {active_percentage:.2f}%")
    print(f"-----------------------\n")
    
    # Optional: Remove grey zone? User mentioned it as optional.
    # Let's keep it simple for now, or maybe add a flag.
    
    return df[[smiles_col, 'Activity_nM', 'Label']]

def sanitize_smiles(smiles):
    """
    Sanitizes a SMILES string using RDKit.
    Returns canonical SMILES or None if invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Remove salts/solvents (simple version: keep largest fragment)
            # For now just canonicalize
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None
